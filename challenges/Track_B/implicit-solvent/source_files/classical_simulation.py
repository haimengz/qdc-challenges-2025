#!/usr/bin/env python3
from json.encoder import JSONEncoder
from json.decoder import JSONDecoder

from qiskit_serverless import get_arguments, save_result

import pyscf
from pyscf import gto, scf
from pyscf.solvent import pcm
from pyscf.mcscf import avas 

### Argument retrieval
args                      = get_arguments() 
data                      = args["data"]      # Chemistry Data

i_data = JSONDecoder().decode(data)

[mol_geo, eps, ao_labels] = i_data

print(f">>>>> DEFINING MOLECULE")
mol = gto.M()
mol.atom = mol_geo
mol.basis = "<XXX>"
mol.unit= "<XXX>"
mol.charge= "<XXX>"
mol.spin= "<XXX>"
mol.verbose=0

print(f">>>>> BUILDING MOLECULE")
# FILL THIS CODE IN (only 1 line is missing)

print(f">>>>> DEFINING PCM")
# FILL THIS CODE IN (3 lines are missing)
cm = "<XXX>"

print(f">>>>> BUILDING RESTRICTED HARTREE FOCK")
# FILL THIS CODE IN (1 line is missing)
mf = "<XXX>"

# Atomic Valence Active Spave, constructs Molecular Orbitals from Atomic Orbitals
print(f">>>>> RUNNING AVAS") 
avas_ = avas.AVAS(mf, ao_labels, with_iao=True, canonicalize=True, verbose=0)
avas_.kernel()
norb, ne_act, mo_avas = avas_.ncas, avas_.nelecas, avas_.mo_coeff

print(f">>>>> STARTING CASCI")
mc_pcm = pyscf.mcscf.CASCI(mf, norb, ne_act).PCM(cm) # This is how CASCI is run (the classical-baseline method)
mc_pcm.mo_coeff = mo_avas

(CASCI_E, _, _, _, _) = mc_pcm.kernel(verbose=0)

print(f">>>>> CASCI_E: {CASCI_E}")
o_data = JSONEncoder().encode([float(CASCI_E)] )

# JSON-safe package
save_result({"outputs": o_data})  # single JSON blob returned to client

