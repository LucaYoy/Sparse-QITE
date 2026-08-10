"""Run adaptive sparse-QITE"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# Allow this script to import modules from the project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from modules import Evolution
from modules import Hamiltonian


# Parameters shared by both simulations.
n_qubits = 8
n_steps = 100
dt = 0.05
T = 2

# Phase-specific parameters and matching output-folder names.
cases = [
    {
        "phase": "ordered",
        "D": 6,
        "J": 1.0,
        "h": 0.5,
        "epsilon": 0.01,
        "folder": "tfim_n8_D6_N100_dt0p05_T2_J1_h0p5",
    },
    {
        "phase": "disordered",
        "D": 4,
        "J": 0.5,
        "h": 1.0,
        "epsilon": 0.01,
        "folder": "tfim_n8_D4_N100_dt0p05_T2_J0p5_h1",
    },
]


# Initial state |00000000>.
initial_state = np.zeros((2**n_qubits, 1), dtype=complex)
initial_state[0, 0] = 1.0
initial_state = sp.csc_matrix(initial_state)


for case in cases:
    print(f"Running the {case['phase']} phase...")

    H, H_trot = Hamiltonian.TFIM(
        case["J"],
        case["h"],
        n_qubits,
        T=T,
    )

    # No fixed lambda is passed. The default objective tolerance is also used.
    energy, states, coefficients = Evolution.QITE(
        n_qubits,
        H,
        H_trot,
        case["D"],
        initial_state,
        n_steps,
        dt,
        method="adaptive_lasso_v1",
        objective_tol=case["epsilon"],
    )

    output_dir = (
        PROJECT_DIR
        / "outputs"
        / "qite_runs"
        / case["folder"]
        / "qite_adaptive_lasso"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "trajectory.npz",
        Energy=energy,
        a=coefficients,
    )
    sp.save_npz(
        output_dir / "states.npz",
        states.tocsc(),
        compressed=True,
    )

    print(f"Saved results in {output_dir}")
