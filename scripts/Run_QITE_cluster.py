#!/usr/bin/env python3
"""Run one QITE job described by a TOML configuration file.

Usage from the ``Sparse-QITE-via-LASSO-main`` directory::

    python scripts/Run_QITE_cluster.py configs/qite_job.toml

The script uses standard QITE when ``method`` is ``"default"`` (or ``"LU"``),
sparse-QITE when it is ``"vanilla_lasso"``, and cumulative-support freezing
when it is ``"Jaccard_lasso"``. Results are written as two NPZ files:
``trajectory.npz`` contains ``Energy`` and ``a``, while ``states.npz`` contains
the sparse state trajectory ``Psi``.
"""

from __future__ import annotations

import argparse
import hashlib
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
    "Jaccard_lasso": "Jaccard_lasso",
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
    for section in (
        "simulation",
        "hamiltonian",
        "lasso",
        "jaccard",
        "checkpoint",
        "output",
    ):
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
    if config["method"] in {"vanilla_lasso", "Jaccard_lasso"} and config.get(
        "lasso_lambda", 1e-3
    ) < 0:
        raise ValueError("lasso_lambda cannot be negative.")
    if config["method"] == "Jaccard_lasso":
        jaccard_min = config.get("jaccard_cumulative_min", 0.99)
        if not 0.0 <= jaccard_min <= 1.0:
            raise ValueError("jaccard_cumulative_min must lie between 0 and 1.")
        if config.get("jaccard_patience", 3) < 1:
            raise ValueError("jaccard_patience must be at least 1.")
        if config.get("support_tol", 1e-14) < 0:
            raise ValueError("support_tol must be non-negative.")
    if config.get("checkpoint_enabled", True) and config.get(
        "checkpoint_every_steps", 5
    ) < 1:
        raise ValueError("checkpoint_every_steps must be at least 1.")


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


def run_qite(
    config: dict[str, Any],
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback=None,
):
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
        jaccard_cumulative_min=config.get("jaccard_cumulative_min", 0.99),
        jaccard_patience=config.get("jaccard_patience", 3),
        support_tol=config.get("support_tol", 1e-14),
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
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


def config_fingerprint(config_path: Path) -> str:
    """Identify the exact TOML contents associated with a checkpoint."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def checkpoint_path(output_dir: Path) -> Path:
    """Return the single atomic checkpoint file used by one result folder."""
    return output_dir / "checkpoint" / "checkpoint.npz"


def save_checkpoint_atomically(
    output_dir: Path,
    state: dict[str, Any],
    fingerprint: str,
) -> None:
    """Persist all resume state in one atomically published NPZ file."""
    path = checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    freeze_step = state["freeze_step"]
    current_state = state["current_state"]
    if not (sp.isspmatrix_csc(current_state) or sp.isspmatrix_csr(current_state)):
        current_state = current_state.tocsc()
    current_state_format = "csc" if sp.isspmatrix_csc(current_state) else "csr"
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            checkpoint_version=np.asarray(2, dtype=np.int64),
            config_fingerprint=np.asarray(fingerprint),
            next_step=np.asarray(state["next_step"], dtype=np.int64),
            Energy=np.asarray(state["energy"]),
            Psi=np.asarray(state["states"].toarray()),
            current_state_data=current_state.data,
            current_state_indices=current_state.indices,
            current_state_indptr=current_state.indptr,
            current_state_shape=np.asarray(current_state.shape, dtype=np.int64),
            current_state_format=np.asarray(current_state_format),
            a=np.asarray(state["coefficients"]),
            allowed_mask=np.asarray(state["allowed_mask"], dtype=bool),
            cumulative_support=np.asarray(
                state["cumulative_support"], dtype=np.int64
            ),
            jaccard_streak=np.asarray(state["jaccard_streak"], dtype=np.int64),
            frozen=np.asarray(state["frozen"], dtype=bool),
            freeze_step=np.asarray(
                -1 if freeze_step is None else freeze_step, dtype=np.int64
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)
    print(
        f"Checkpoint saved after update {state['next_step']} to {path}",
        flush=True,
    )


def load_checkpoint(
    output_dir: Path,
    expected_fingerprint: str,
) -> dict[str, Any] | None:
    """Load and validate a checkpoint, returning ``None`` when absent."""
    path = checkpoint_path(output_dir)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        version = int(saved["checkpoint_version"])
        if version != 2:
            raise ValueError(f"Unsupported checkpoint version {version} in {path}.")
        saved_fingerprint = str(saved["config_fingerprint"].item())
        if saved_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Checkpoint {path} belongs to different TOML contents. "
                "Remove it or restore the original configuration."
            )
        freeze_value = int(saved["freeze_step"])
        current_shape = tuple(int(value) for value in saved["current_state_shape"])
        current_parts = (
            np.asarray(saved["current_state_data"]),
            np.asarray(saved["current_state_indices"]),
            np.asarray(saved["current_state_indptr"]),
        )
        current_format = str(saved["current_state_format"].item())
        if current_format == "csc":
            current_state = sp.csc_matrix(current_parts, shape=current_shape)
        elif current_format == "csr":
            current_state = sp.csr_matrix(current_parts, shape=current_shape)
        else:
            raise ValueError(f"Unsupported sparse state format {current_format!r}.")
        return {
            "next_step": int(saved["next_step"]),
            "energy": np.asarray(saved["Energy"]),
            "states": sp.csc_matrix(np.asarray(saved["Psi"])),
            "current_state": current_state,
            "coefficients": np.asarray(saved["a"]),
            "allowed_mask": np.asarray(saved["allowed_mask"], dtype=bool),
            "cumulative_support": np.asarray(
                saved["cumulative_support"], dtype=np.int64
            ),
            "jaccard_streak": int(saved["jaccard_streak"]),
            "frozen": bool(saved["frozen"]),
            "freeze_step": None if freeze_value < 0 else freeze_value,
        }


def remove_completed_checkpoint(output_dir: Path) -> None:
    """Discard a checkpoint only after both final result files are published."""
    path = checkpoint_path(output_dir)
    if path.is_file():
        path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass


def write_checkpoint_progress(
    output_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Atomically print the latest checkpoint values to a readable text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients = np.asarray(state["coefficients"])
    energy = np.asarray(state["energy"])
    support_tol = float(config.get("support_tol", 1e-14))
    active = np.abs(coefficients) > support_tol
    latest_active = int(np.count_nonzero(active[-1])) if len(active) else 0
    cumulative = np.any(active, axis=0) if len(active) else np.zeros(0, dtype=bool)
    cumulative_terms = int(np.count_nonzero(cumulative))

    if len(active) > 1:
        previous_cumulative = np.any(active[:-1], axis=0)
        previous_terms = int(np.count_nonzero(previous_cumulative))
        new_terms = cumulative_terms - previous_terms
        cumulative_jaccard = (
            previous_terms / cumulative_terms if cumulative_terms else 1.0
        )
    else:
        new_terms = cumulative_terms
        cumulative_jaccard = 0.0 if cumulative_terms else 1.0

    completed = int(state["next_step"])
    requested = int(config["N"])
    freeze_step = state["freeze_step"]

    lines = [
        "QITE checkpoint progress",
        "========================",
        f"configuration: {config['config_path']}",
        f"model: {config['model']}",
        f"method: {config['method']}",
        f"completed updates: {completed} / {requested}",
        f"progress: {100.0 * completed / requested:.2f}%",
        f"imaginary time: {completed * float(config['dt']):.16g}",
        f"latest energy: {float(energy[-1]):.16g}",
        f"active terms in latest update: {latest_active}",
        f"new cumulative terms in latest update: {new_terms}",
        f"cumulative terms: {cumulative_terms}",
        f"latest cumulative Jaccard: {cumulative_jaccard:.16g}",
        f"support frozen: {bool(state['frozen'])}",
        f"freeze decision update: {freeze_step + 1 if freeze_step is not None else 'not frozen'}",
        (
            f"first frozen-pool update: {freeze_step + 2}"
            if freeze_step is not None
            else "first frozen-pool update: not frozen"
        ),
        f"checkpoint file: {checkpoint_path(output_dir)}",
        "",
    ]

    progress_path = output_dir / "checkpoint_progress.txt"
    temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(progress_path)
    print(f"Checkpoint progress written to {progress_path}", flush=True)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    validate_config(config)
    output_dir = resolve_output_dir(config)
    fingerprint = config_fingerprint(config["config_path"])
    checkpoint_enabled = bool(config.get("checkpoint_enabled", True))
    checkpoint_every = int(config.get("checkpoint_every_steps", 5))
    resume_state = (
        load_checkpoint(output_dir, fingerprint) if checkpoint_enabled else None
    )

    method_labels = {
        "vanilla_lasso": "sparse-QITE",
        "Jaccard_lasso": "Jaccard sparse-QITE",
    }
    method_label = method_labels.get(config["method"], "QITE")
    print(f"Starting {method_label} from {config['config_path']}", flush=True)
    print(
        f"model={config['model']}, n={config['n_qubits']}, D={config['D']}, "
        f"N={config['N']}, dt={config['dt']}, method={config['method']}",
        flush=True,
    )

    if resume_state is not None:
        print(
            f"Resuming from checkpoint after update {resume_state['next_step']}; "
            f"next update is {resume_state['next_step'] + 1}.",
            flush=True,
        )

    def checkpoint_callback(state: dict[str, Any]) -> None:
        next_step = int(state["next_step"])
        if checkpoint_enabled and (
            next_step % checkpoint_every == 0 or next_step == config["N"]
        ):
            save_checkpoint_atomically(output_dir, state, fingerprint)
            write_checkpoint_progress(output_dir, state, config)

    energy, states, coefficients = run_qite(
        config,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback if checkpoint_enabled else None,
    )
    save_results_atomically(output_dir, energy, states, coefficients)
    remove_completed_checkpoint(output_dir)

    print(f"Completed {len(energy) - 1} steps; final energy={energy[-1]:.16g}", flush=True)
    print(f"Trajectory written to {output_dir / 'trajectory.npz'}", flush=True)
    print(f"States written to {output_dir / 'states.npz'}", flush=True)


if __name__ == "__main__":
    main()
