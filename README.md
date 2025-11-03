# IBM Quantum Developer Conference 2025

Welcome to the IBM Quantum Developer Conference 2025! These challenges are designed to test your quantum computing skills using the Qiskit product features announced during QDC. The competition spans three days and features two challenge tracks.

## Installation and Setup

To participate, please complete the setup steps below. Remember to bring a laptop with internet access, as no devices will be provided at the venue.

### Step 1: Set up Your IBM Quantum Platform Account
1. [Create an IBM Quantum account](https://quantum.cloud.ibm.com/signin) if you haven't already.
2. **Sign up using the same email address you registered with for this event**. Your account will be linked to a specific [instance](https://quantum.cloud.ibm.com/instances) for IBM Quantum services.

### Step 2: Create a Python Environment
1. Follow [this guide](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit) to create a Python environment and install the Qiskit SDK and Runtime. Ensure Jupyter is installed (Step 4 in the guide).
2. Install additional packages:
   - Qiskit Functions catalog client ([instructions](https://quantum.cloud.ibm.com/docs/en/guides/functions#install-qiskit-functions-catalog-client))

> **Alternative:** If setting up Python or Jupyter locally is difficult, consider using Google Colab or qBraid. See [this guide](https://quantum.cloud.ibm.com/docs/en/guides/online-lab-environments) for details.


## Preparing for the Challenge

To ensure all tools are up to date, please update your packages right before the challenge begins.

**If using a local environment:**
1. Activate your Python environment.
2. Download the [requirements file](./requirements.txt) to your workspace.
3. Run this command to install/update packages:

    ```bash
    pip install -r path/to/requirements.txt --upgrade
    ```

   > **Note:** Replace `path/to/requirements.txt` with your file's path.

**If using an online lab environment:**  
Use the same `requirements.txt` file and pip command to install/update packages. For qBraid, activate the "QDC 2024" environment under the [Environment tab](https://docs.qbraid.com/lab/user-guide/environments), which comes pre-configured.
You can launch the  by clicking the button below Launch on qBraid to clone and open this gitHub link on qBraid Lab.

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/qiskit-community/qdc-challenges-2024)

## Challenge Notebooks

Each day, the challenge notebooks will be available in their respective "Day" folders within the GitHub repository. **Please download the notebooks for each day, open them, and run them to proceed with the challenges.**


## Challenges Topics

Every afternoon during QDC, attendees will have the opportunity to participate in a hands-on coding challenges, where you will be able to practice using the latest tools and features highlighted in the morning seminars. You will be using Jupyter notebooks to complete the challenges, which can be found in this dedicated GitHub repository. To ensure you have the best challenge experience possible it is very important to follow the preparations steps outlined in the GitHub README prior to the event.

Every day there will be two challenges happening in parallel, and you may choose which one you would like to participate in based on your skills and interests. See below a summary of topics covered by each challenge:

### Track A: Simulation
- Scattering and particle creation in 1D Ising Field Theory
- Hadron Dynamics in the Schwinger Model
- Real-Time Dynamics in a (2+1)-D Gauge Theory
- Sample-based Krylov Quantum Diagonalization

### Track B: Optimization
- Quantum Approximate Multi-Objective Optimization (QMOO)
- Computing Solute-Solvent Interactions using Sample-based Quantum Diagonalization


## Learning Resources

The seminars and challenges at QDC are designed to cover a broad range of advanced topics featuring Qiskit tools and capabilities. While participants are not expected to have a deep understanding of everything that will be covered we do recommend familiarizing yourself with the basics before attending the event. 

See the [cheat sheet](./challenges_cheat_sheet.md) for more information and links to various resources which may help get you started.

