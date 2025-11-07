from typing import Sequence

from json.encoder import JSONEncoder
from json.decoder import JSONDecoder
import time

from functools import partial
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

import ffsim
import scipy

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CPhaseGate, XGate, XXPlusYYGate
from qiskit.quantum_info import Pauli, SparsePauliOp


def exact_ground_state_energy(H_spo: SparsePauliOp, eigvals: np.ndarray) -> float:
    """Calculates the exact ground state energy of the given Hamiltonian.

    Args:
        H_spo: Hamiltonian as a SparsePauliOp
        eigvals: numpy array of eigenvalues

    Returns:
        Ground state energy of the Hamiltonian.
    """
    H_dense = H_spo.to_matrix()
    exact_eigvals = np.linalg.eigvalsh(H_dense)
    return np.min(exact_eigvals).real


def dt_from_spectral_norm(H_spo: SparsePauliOp) -> float:
    """Calculate dt based on Hamiltonian spectral norm.

    Note that this is a theoretical optimal dt. Heuristically, the optimal can
    be 6-10x the theoretical value.

    Args:
        H_spo: Hamiltonian represented as a SparsePauliOp

    Returns:
        An optimal Trotter timestep
    """
    num_qubits = H_spo.num_qubits
    single_particle_H = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1):
            for p, coeff in H_spo.to_list():
                p_x = Pauli(p).x
                p_z = Pauli(p).z
                if all(
                    p_x[k] == ((i == k) + (j == k)) % 2 for k in range(num_qubits)
                ):
                    sgn = (
                        (-1j) ** sum(p_z[k] and p_x[k] for k in range(num_qubits))
                    ) * ((-1) ** p_z[i])
                else:
                    sgn = 0
                single_particle_H[i, j] += sgn * coeff
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            single_particle_H[i, j] = np.conj(single_particle_H[j, i])

    # dt based on spectral norm
    return np.pi / np.linalg.norm(single_particle_H, ord=2)


def siam_hamiltonian(
    num_orbs: int,
    hopping: float,
    onsite: float,
    hybridization: float,
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hamiltonian for the single-impurity Anderson model.

    Args:
        num_orbs: Number of spatial orbitals
        hopping: Hopping term
        onsite: On-site repulsion
        hybrid: Hybridization interaction
        mu: Chemical potential

    Returns:
        One- and two-body terms of Hamiltonian.
    """
    # Place the impurity on the first site
    impurity_orb = 0

    # One body matrix elements in the "position" basis
    h1e = np.zeros((num_orbs, num_orbs))
    np.fill_diagonal(h1e[:, 1:], -hopping)
    np.fill_diagonal(h1e[1:, :], -hopping)
    h1e[impurity_orb, impurity_orb + 1] = -hybridization
    h1e[impurity_orb + 1, impurity_orb] = -hybridization
    h1e[impurity_orb, impurity_orb] = chemical_potential

    # Two body matrix elements in the "position" basis
    h2e = np.zeros((num_orbs, num_orbs, num_orbs, num_orbs))
    h2e[impurity_orb, impurity_orb, impurity_orb, impurity_orb] = onsite

    return h1e, h2e

def _canonicalize_real_eigvec_signs(V: np.ndarray, anchor_row: int = 0) -> np.ndarray:
    V = V.copy()
    signs = np.sign(V[anchor_row, :])
    # fallback for exact zeros: use the largest-magnitude element as anchor
    zero = np.isclose(signs, 0.0)
    if np.any(zero):
        js = np.where(zero)[0]
        i_max = np.argmax(np.abs(V[:, js]), axis=0)
        signs[js] = np.sign(V[i_max, js])
    signs[signs == 0] = 1.0
    return V * signs

def momentum_basis(norb: int) -> np.ndarray:
    """Get the orbital rotation to change from the position to the momentum basis.

    Args:
        norb: Number of spatial orbitals

    Returns:
        Matrix corresponding to orbital rotation for position -> momentum basis.
    """
    n_bath = norb - 1

    # Orbital rotation that diagonalizes the bath (non-interacting system)
    hopping_matrix = np.zeros((n_bath, n_bath))
    np.fill_diagonal(hopping_matrix[:, 1:], -1)
    np.fill_diagonal(hopping_matrix[1:, :], -1)
    _, vecs = np.linalg.eigh(hopping_matrix)

    # >>> deterministic sign convention <<<
    vecs = _canonicalize_real_eigvec_signs(vecs, anchor_row=0)

    # Expand to include impurity
    orbital_rotation = np.zeros((norb, norb))
    # Impurity is on the first site
    orbital_rotation[0, 0] = 1
    orbital_rotation[1:, 1:] = vecs

    # Move the impurity to the center
    new_index = n_bath // 2
    perm = np.r_[1: (new_index + 1), 0, (new_index + 1): norb]
    orbital_rotation = orbital_rotation[:, perm]

    return orbital_rotation

def rotated(
    h1e: np.ndarray, h2e: np.ndarray, orbital_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the orbital basis of a Hamiltonian.

    Args:
        h1e: One-body term
        h2e: Two-body term
        orbital_rotation: Position-to-momentum basis rotation matrix

    Returns:
        One- and two-body Hamiltonian rotated from position to momentum basis.
    """
    h1e_rotated = np.einsum(
        "ab,Aa,Bb->AB",
        h1e,
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    h2e_rotated = np.einsum(
        "abcd,Aa,Bb,Cc,Dd->ABCD",
        h2e,
        orbital_rotation,
        orbital_rotation.conj(),
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    return h1e_rotated, h2e_rotated


def prepare_initial_state(qubits: Sequence[Qubit], num_orbs: int, occ_num: int):
    """Prepare initial state.

    Args:
        qubits: Ordered collection of qubits
        num_orbs: Number of spatial orbitals
        occ_num: Occupancy number

    Returns:
        Initial state generator.
    """
    x_gate = XGate()
    rot = XXPlusYYGate(0.5 * np.pi, -0.5 * np.pi)
    for i in range(occ_num):
        yield CircuitInstruction(x_gate, [qubits[i]])
        yield CircuitInstruction(x_gate, [qubits[num_orbs + i]])
    for i in range(3):
        for j in range(occ_num - i - 1, occ_num + i, 2):
            yield CircuitInstruction(rot, [qubits[j], qubits[j + 1]])
            yield CircuitInstruction(
                rot, [qubits[num_orbs + j], qubits[num_orbs + j + 1]]
            )
    yield CircuitInstruction(rot, [qubits[j + 1], qubits[j + 2]])
    yield CircuitInstruction(
        rot, [qubits[num_orbs + j + 1], qubits[num_orbs + j + 2]]
    )


def trotter_step(
    qubits: Sequence[Qubit],
    time_step: float,
    hamiltonian: tuple[np.ndarray, np.ndarray],
    impurity_index: int,
    num_orbs: int,
):
    """A Trotter step.

    Args:
        qubits: Ordered collection of qubits
        time_step: Time step
        hamiltonian: One- and two-body Hamiltonian terms,
        impurity_index: Index of impurity
        num_orbs: Number of spatial orbitals

    Returns:
        Trotter step generator.
    """
    # Assume the two-body interaction is just the on-site interaction of the impurity
    onsite = hamiltonian[1][
        impurity_index, impurity_index, impurity_index, impurity_index
    ]
    # Two-body evolution for half the time
    yield CircuitInstruction(
        CPhaseGate(-0.5 * time_step * onsite),
        [qubits[impurity_index], qubits[num_orbs + impurity_index]],
    )
    # One-body evolution for the full time
    yield CircuitInstruction(
        ffsim.qiskit.OrbitalRotationJW(num_orbs, hamiltonian[0]), qubits
    )
    # Two-body evolution for half the time
    yield CircuitInstruction(
        CPhaseGate(-0.5 * time_step * onsite),
        [qubits[impurity_index], qubits[num_orbs + impurity_index]],
    )


def plot_comparison(
    history: list[SCIResult], result: SCIResult, hamiltonian: tuple[np.ndarray, np.ndarray]
):
    """Plot SKQD energy per iteration and reference (DMRG) energy.

    Args:
        history: List of SCIResults per iteration
        result: Result from diagonalize_fermionic_hamiltonian
        hamil
    """
    ref_energy = -28.70659686

    min_es = [
        min(result, key=lambda res: res.energy).energy
        for result in history
    ]
    min_id, min_e = min(enumerate(min_es), key=lambda x: x[1])

    # Data for energies plot
    x1 = range(len(history))

    # Data for avg spatial orbital occupancy
    y2 = np.sum(result.orbital_occupancies, axis=0)
    x2 = range(len(y2))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot energies
    axs[0].plot(x1, min_es, label="energy", marker="o")
    axs[0].set_xticks(x1)
    axs[0].set_xticklabels(x1)
    axs[0].axhline(
        y=ref_energy, color="#BF5700", linestyle="--", label="DMRG energy"
    )
    axs[0].set_title("SKQD Approximated Ground State Energy")
    axs[0].set_xlabel("Iteration", fontdict={"fontsize": 12})
    axs[0].set_ylabel("Approximate Energy", fontdict={"fontsize": 12})
    axs[0].legend()

    # Plot orbital occupancy
    axs[1].bar(x2, y2, width=0.8)
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels(x2)
    axs[1].set_title("Average Occupancy")
    axs[1].set_xlabel("Spatial Orbital", fontdict={"fontsize": 12})
    axs[1].set_ylabel("Average Occupancy", fontdict={"fontsize": 12})

    print(f"Reference (DMRG) Energy: {ref_energy:.5f}")
    print(f"SQD Energy: {min_e:.5f}")
    print(f"Absolute Error: {abs(min_e - ref_energy):.5f}")
    plt.tight_layout()
    plt.show()

    rdm1 = result.sci_state.rdm(rank=1, spin_summed=True)
    rdm2 = result.sci_state.rdm(rank=2, spin_summed=True)
    energy = np.sum(hamiltonian[0] * rdm1) + 0.5 * np.sum(hamiltonian[1] * rdm2)
    print(f"Verified recomputed energy: {energy:.5f}")


def classically_diagonalize(bit_array=None,
                            hcore=None,
                            eri=None,
                            num_orbitals=None,
                            nelec=None,
                            num_elec_a=None,
                            num_elec_b=None,
                            job_id=None,
                            client=None,
                            energy_tol=1e-4,
                            occupancies_tol=1e-3,
                            max_iterations=12,
                            num_batches=8,
                            samples_per_batch=300,
                            symmetrize_spin=False,
                            carryover_threshold=1e-5,
                            max_cycle=200,
                            mem=64,
                            local=True
                            ):
    """Classical Diagonalization Engine sent to HPC.

    Args:
        bit_array: Bit string array; only needed if locally processing data
        hcore: 1-electron hamiltonian integrals
        eri: 2-electron hamiltonian integrals
        num_orbitals: Number of spatial orbitals
        nelec: Number of electrons
        num_elec_a: Alpha orbitals
        num_elec_b: Beta orbitals
        job_id: QPU bitstring Job ID
        client: Diagonalization engine worker
        energy_tol: SQD option
        occupancies_tol: SQD option
        max_iterations: SQD option
        num_batches: Eigenstate solver option
        samples_per_batch: Eigenstate solver option
        symmetrize_spin: Eigenstate solver option
        carryover_threshold: Eigenstate solver option
        max_cycle: Eigenstate solver option
        mem: distributed task memory, in GiB
        local: True to run locally, False for remote

    Returns:
        Serverless job result outputs as a tuple.
    """

    print(">>>>> Starting Diagonalization Engine...")
    # Pass options to the built-in eigensolver. If you just want to use the defaults,
    # you can omit this step, in which case you would not specify the sci_solver argument
    # in the call to diagonalize_fermionic_hamiltonian below.
    if local:
        sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

        # List to capture intermediate results
        result_history = []

        def callback(results: list[SCIResult]):
            result_history.append(results)
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, result in enumerate(results):
                print(f"\tSubsample {i}")
                print(f"\t\tEnergy: {result.energy}")
                print(
                        f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
                )

        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=samples_per_batch,
            norb=num_orbitals,
            nelec=(nelec//2, nelec//2),
            num_batches=num_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=max_iterations,
            sci_solver=sci_solver,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            callback=callback,
            seed=12345,
        )

        # Get the energies at each iteration
        min_es = [
            min(result, key=lambda res: res.energy).energy
            for result in result_history
        ]
        energy_history = [float(e) for e in min_es]

        result = energy_history

    else:
        # Serverless Logic
        print(f">>>>> Sending job {job_id} to Serverless...")

        data = [
            job_id,
            hcore.tolist(),
            eri.tolist(),
            int(num_orbitals),
            int(num_elec_a),
            int(num_elec_b)
        ]

        # Encode the execution dependencies with the JSONEncoder
        data_e = JSONEncoder().encode(data)

        # Send to Serverless
        worker = client.load("diagonalization_engine")
        serverless_job = worker.run(
            data=data_e,
            mem=mem,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=max_iterations,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            num_batches=num_batches,
            samples_per_batch=samples_per_batch,
            max_cycle=max_cycle,
        )

        # Wait for the job to execute
        print(f">>>>> Serverless status: {serverless_job.job_id}")
        timer = 0
        while (timer < 10000):
            if serverless_job.status() == "QUEUED" \
                or serverless_job.status() == "INITIALIZING" \
                    or serverless_job.status() == "RUNNING":
                print(f">>>>> [{timer}s] Serverless job {serverless_job.job_id}: \
                    {serverless_job.status()}")
                time.sleep(10)
                timer += 10
                result = None

            elif serverless_job.status() == "ERROR":
                print(f">>>>> Serverless job {serverless_job.job_id}: {serverless_job.status()}")
                print(">>>>> Logs:")
                print(serverless_job.logs())
                result = None
                break

            elif serverless_job.status() == "DONE":
                print(f">>>>> Serverless job {serverless_job.job_id}: {serverless_job.status()}")
                o_data = JSONDecoder().decode(serverless_job.result()["outputs"])
                result = o_data[0]

                print(f">>>>>>>>>> Energies/iteration: {result}")
                break

            else:
                break

    return result
