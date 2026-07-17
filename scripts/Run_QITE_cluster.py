#!/usr/bin/env python3
"""Run one QITE job described by a TOML configuration file.

Usage from the ``Sparse-QITE-via-LASSO-main`` directory::

    python scripts/Run_QITE_cluster.py configs/qite_job.toml

The script uses standard QITE when ``method`` is ``"default"`` (or ``"LU"``)
and sparse-QITE when it is ``"vanilla_lasso"``. Results are written as two NPZ
files: ``trajectory.npz`` contains ``Energy`` and ``a``, while ``states.npz``
contains the sparse state trajectory ``Psi``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

# One BLAS thread per Condor process avoids oversubscribing a worker node.
# Set these variables in the Condor submit file if more CPUs are requested.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp

# Imports work independently of the directory from which Condor starts the job.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modules import Evolution as evolution
from modules import Hamiltonian as hamiltonian


ALLOWED_METHODS = {
    "default": "LU",
    "qite": "LU",
    "LU": "LU",
    "lstsq": "lstsq",
    "pinv": "pinv",
    "vanilla_lasso": "vanilla_lasso",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QITE or sparse-QITE from one TOML configuration."
    )
    parser.add_argument("config", type=Path, help="Path to the job's TOML configuration file")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load TOML and merge its sections into one flat configuration."""
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    config: dict[str, Any] = {}
    for section in ("simulation", "hamiltonian", "lasso", "output"):
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"[{section}] must be a TOML table.")
        config.update(values)

    config["config_path"] = path
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("model", "n_qubits", "D", "N", "dt", "method", "output_dir")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    if config["method"] not in ALLOWED_METHODS:
        choices = ", ".join(ALLOWED_METHODS)
        raise ValueError(f"Unknown method {config['method']!r}. Choose one of: {choices}.")
    if config["model"] not in {"tfim", "cluster_ising", "heisenberg_obc", "heisenberg_pbc"}:
        raise ValueError(f"Unknown Hamiltonian model: {config['model']!r}.")
    if config["n_qubits"] < 2:
        raise ValueError("n_qubits must be at least 2.")
    if not 1 <= config["D"] <= config["n_qubits"]:
        raise ValueError("D must be between 1 and n_qubits.")
    if config["N"] < 1 or config["dt"] <= 0:
        raise ValueError("N and dt must be positive.")
    basis_state = config.get("initial_basis_state", 0)
    if not 0 <= basis_state < 2 ** config["n_qubits"]:
        raise ValueError("initial_basis_state must be in [0, 2**n_qubits).")
    if config["method"] == "vanilla_lasso" and config.get("lasso_lambda", 1e-3) < 0:
        raise ValueError("lasso_lambda cannot be negative.")


def build_hamiltonian(config: dict[str, Any]):
    """Build the requested full Hamiltonian and its Trotter decomposition."""
    model = config["model"]
    n = config["n_qubits"]
    trotter_size = config.get("T", 2)

    if model == "tfim":
        return hamiltonian.TFIM(config["J"], config["h"], n, T=trotter_size)
    if model == "cluster_ising":
        return hamiltonian.ClusterIsing(config["cluster_lambda"], n, T=trotter_size)
    if model == "heisenberg_obc":
        return hamiltonian.Heisenberg_OBC(config["J"], n)
    if model == "heisenberg_pbc":
        return hamiltonian.Heisenberg_PBC(config["J"], n, T=trotter_size)
    raise AssertionError("The model was checked by validate_config().")


def build_initial_state(n_qubits: int, basis_state: int) -> sp.csc_matrix:
    """Create |basis_state> as a normalized sparse column vector."""
    state = np.zeros((2**n_qubits, 1), dtype=complex)
    state[basis_state, 0] = 1.0
    return sp.csc_matrix(state)


def run_qite(config: dict[str, Any]):
    """Run the configured evolution and return aligned trajectory data."""
    H, H_trot = build_hamiltonian(config)
    psi_0 = build_initial_state(config["n_qubits"], config.get("initial_basis_state", 0))
    method = ALLOWED_METHODS[config["method"]]

    energy, states, coefficients = evolution.QITE(
        config["n_qubits"],
        H,
        H_trot,
        config["D"],
        psi_0,
        config["N"],
        config["dt"],
        vervose=config.get("verbose", False),
        method=method,
        lasso_lambda=config.get("lasso_lambda", 1e-3),
        lasso_max_iter=config.get("lasso_max_iter", 500),
        lasso_tol=config.get("lasso_tol", 1e-8),
    )
    return energy, states, coefficients


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Resolve a relative result directory from the configuration file's folder."""
    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = config["config_path"].parent / output_dir
    return output_dir.resolve()


def save_results_atomically(
    output_dir: Path,
    energy: np.ndarray,
    states: sp.spmatrix,
    coefficients: np.ndarray,
) -> None:
    """Save numerical trajectories without publishing partially written files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "trajectory.npz"
    temporary_trajectory = output_dir / "trajectory.npz.tmp"
    with temporary_trajectory.open("wb") as handle:
        np.savez_compressed(handle, Energy=energy, a=coefficients)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_trajectory.replace(trajectory_path)

    states_path = output_dir / "states.npz"
    temporary_states = output_dir / "states.npz.tmp"
    with temporary_states.open("wb") as handle:
        sp.save_npz(handle, states.tocsc(), compressed=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_states.replace(states_path)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    validate_config(config)
    output_dir = resolve_output_dir(config)

    method_label = "sparse-QITE" if config["method"] == "vanilla_lasso" else "QITE"
    print(f"Starting {method_label} from {config['config_path']}", flush=True)
    print(
        f"model={config['model']}, n={config['n_qubits']}, D={config['D']}, "
        f"N={config['N']}, dt={config['dt']}, method={config['method']}",
        flush=True,
    )

    energy, states, coefficients = run_qite(config)
    save_results_atomically(output_dir, energy, states, coefficients)

    print(f"Completed {len(energy) - 1} steps; final energy={energy[-1]:.16g}", flush=True)
    print(f"Trajectory written to {output_dir / 'trajectory.npz'}", flush=True)
    print(f"States written to {output_dir / 'states.npz'}", flush=True)


if __name__ == "__main__":
    main()
