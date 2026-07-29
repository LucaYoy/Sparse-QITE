#!/usr/bin/env python3
"""Transpile the circuit associated with one completed QITE run.

The positional argument is the same TOML file used by ``Run_QITE_cluster.py``.
The script reads the stored coefficients from ``trajectory.npz``, rebuilds the
same Pauli pool, creates and transpiles the QITE circuit, and stores:

* ``circuit.qpy``: Qiskit's machine-readable circuit representation.
* ``summary.csv``: a human-readable vertical metric/value table.
* ``metadata.json``: a human-readable, structured record of the run.

Example, from the project directory::

    python scripts/Run_Transpilation_cluster.py \
        configs/generated/tfim_n8_D4_N100_dt0p05_T2_J0p5_h1/qite_default.toml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Avoid oversubscribing a Condor worker. A submit file may override these.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from qiskit import qpy
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modules import PauliStrings
from modules import Transpile
from Run_QITE_cluster import (
    build_hamiltonian,
    build_initial_state,
    load_config,
    resolve_output_dir,
    validate_config,
)


def parse_arguments() -> argparse.Namespace:
    """Read the QITE configuration and the transpiler choices."""
    parser = argparse.ArgumentParser(
        description="Transpile the circuit from one completed QITE run."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="TOML configuration originally passed to Run_QITE_cluster.py",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(4),
        default=1,
        help="Qiskit transpiler optimization level, from 0 to 3 (default: 1)",
    )
    parser.add_argument(
        "--basis-gates",
        nargs="+",
        default=["u3", "cx"],
        help="Native gate set of the generic backend (default: u3 cx)",
    )
    parser.add_argument(
        "--hamiltonian-gate",
        action="store_true",
        help="Use HamiltonianGate instead of PauliEvolutionGate",
    )
    parser.add_argument(
        "--compute-unitary-error",
        action="store_true",
        help=(
            "Construct full-system unitaries and calculate their Frobenius "
            "error (expensive in memory and time)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing transpilation with identical settings",
    )
    return parser.parse_args()


def reconstruct_pauli_pool(config: dict[str, Any]):
    """Rebuild exactly the Pauli-string pool used by the QITE evolution."""
    hamiltonian, h_trot = build_hamiltonian(config)
    initial_state = build_initial_state(
        config["n_qubits"], config.get("initial_basis_state", 0)
    )
    D = config["D"]
    n_qubits = config["n_qubits"]

    if h_trot.BC == "OBC":
        if D == n_qubits:
            num_paulis, pauli_pool, failed = PauliStrings.OBC_DN(
                h_trot, n_qubits, PDstr=True
            )
        elif np.isreal(hamiltonian.data).all() and np.isreal(
            initial_state.data
        ).all():
            num_paulis, pauli_pool, failed = PauliStrings.real_OBC(
                h_trot, D, n_qubits, PDstr=True
            )
        else:
            num_paulis, pauli_pool, failed = PauliStrings.general_OBC(
                h_trot, D, n_qubits, PDstr=True
            )
    elif np.isreal(hamiltonian.data).all() and np.isreal(initial_state.data).all():
        num_paulis, pauli_pool, failed = PauliStrings.real(
            h_trot, D, n_qubits, PDstr=True
        )
    else:
        num_paulis, pauli_pool, failed = PauliStrings.general(
            h_trot, D, n_qubits, PDstr=True
        )

    if failed:
        raise RuntimeError("The Pauli-string constructor reported a failed pool.")
    return h_trot, num_paulis, pauli_pool


def settings_name(
    basis_gates: list[str],
    optimization_level: int,
    pauli_evolution: bool,
    steps: int,
    compute_unitary_error: bool,
) -> str:
    """Return a readable output-folder name for this transpilation."""
    gate_tag = "-".join(basis_gates)
    construction = "pauli_evolution" if pauli_evolution else "hamiltonian_gate"
    error_tag = "with_unitary_error" if compute_unitary_error else "no_unitary_error"
    return (
        f"generic_{gate_tag}_opt{optimization_level}_{construction}_"
        f"steps{steps}_{error_tag}"
    )


def write_text_atomically(path: Path, text: str) -> None:
    """Publish a text file only after its complete contents are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def pure_state_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized pure-state fidelity |<left|right>|**2."""
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    denominator = np.vdot(left, left).real * np.vdot(right, right).real
    if denominator <= 0.0:
        raise ValueError("Cannot calculate fidelity for a zero-norm state.")
    return float(np.clip(np.abs(np.vdot(left, right)) ** 2 / denominator, 0.0, 1.0))


def save_outputs(
    output_dir: Path,
    circuit,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    """Save the circuit, structured metadata, and vertical CSV summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "circuit.qpy",
        output_dir / "metadata.json",
        output_dir / "summary.csv",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Transpilation output already exists ({names}). "
            "Use --overwrite to replace it."
        )

    circuit_path = output_dir / "circuit.qpy"
    temporary_circuit = circuit_path.with_suffix(".qpy.tmp")
    with temporary_circuit.open("wb") as handle:
        qpy.dump(circuit, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_circuit.replace(circuit_path)

    write_text_atomically(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )

    qite = metadata["qite"]
    transpilation = metadata["transpilation"]
    fidelity = metadata["fidelity"]
    rows = [
        ("model", qite["model"]),
        ("method", qite["method"]),
        ("n_qubits", qite["n_qubits"]),
        ("D", qite["D"]),
        ("dt", qite["dt"]),
        ("steps", qite["steps"]),
        ("num_trotter_pieces", qite["num_trotter_pieces"]),
        ("num_paulis_per_piece", qite["num_paulis_per_piece"]),
        ("nonzero_coefficients", qite["nonzero_coefficients"]),
        ("active_coefficients_at_support_tol", qite["active_coefficients"]),
        (
            "pauli_rotations_before_transpilation",
            transpilation["pauli_rotations_before_transpilation"],
        ),
        (
            "nonempty_evolution_blocks_before_transpilation",
            transpilation["nonempty_evolution_blocks_before_transpilation"],
        ),
        ("backend", transpilation["backend"]),
        ("basis_gates", ";".join(transpilation["basis_gates"])),
        ("optimization_level", transpilation["optimization_level"]),
        ("circuit_construction", transpilation["circuit_construction"]),
        ("compiled_depth", transpilation["depth"]),
        ("compiled_size", transpilation["size"]),
        ("compiled_nonlocal_gates", transpilation["nonlocal_gates"]),
        (
            "fidelity_uncompiled_vs_compiled",
            fidelity["uncompiled_vs_compiled"],
        ),
        (
            "fidelity_uncompiled_vs_ground_state",
            fidelity["uncompiled_vs_ground_state"],
        ),
        (
            "fidelity_compiled_vs_ground_state",
            fidelity["compiled_vs_ground_state"],
        ),
        (
            "ground_state_fidelity_degradation",
            fidelity["ground_state_fidelity_degradation"],
        ),
        ("ground_state_energy", fidelity["ground_state_energy"]),
        ("unitary_frobenius_error", transpilation["unitary_frobenius_error"]),
        ("transpilation_elapsed_seconds", transpilation["elapsed_seconds"]),
    ]
    for gate, count in sorted(transpilation["gate_counts"].items()):
        rows.append((f"compiled_gate_count_{gate}", count))

    csv_path = output_dir / "summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_csv.replace(csv_path)


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    validate_config(config)

    qite_output_dir = resolve_output_dir(config)
    trajectory_path = qite_output_dir / "trajectory.npz"
    states_path = qite_output_dir / "states.npz"
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Completed QITE trajectory not found: {trajectory_path}"
        )
    if not states_path.is_file():
        raise FileNotFoundError(f"Completed QITE states not found: {states_path}")

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        if "a" not in trajectory:
            raise KeyError(f"{trajectory_path} does not contain coefficients 'a'.")
        coefficients = np.asarray(trajectory["a"])
    states = sp.load_npz(states_path).tocsc()

    steps = coefficients.shape[0]
    if not 1 <= steps <= config["N"]:
        raise ValueError(
            "The stored coefficient trajectory has an invalid number of updates: "
            f"{steps}; the configuration requests N={config['N']}."
        )
    if states.shape != (2 ** config["n_qubits"], steps + 1):
        raise ValueError(
            "Stored state trajectory is not aligned with the coefficients: "
            f"states={states.shape}, expected={(2 ** config['n_qubits'], steps + 1)}."
        )

    h_trot, num_paulis, pauli_pool = reconstruct_pauli_pool(config)
    expected_shape = (h_trot.Nk, num_paulis)
    if coefficients.shape[1:] != expected_shape:
        raise ValueError(
            "Stored coefficient shape does not match the reconstructed pool: "
            f"{coefficients.shape[1:]} != {expected_shape}."
        )

    pauli_evolution = not args.hamiltonian_gate
    requested_basis_gates = list(args.basis_gates)
    backend = GenericBackendV2(
        num_qubits=config["n_qubits"],
        # GenericBackendV2 extends the supplied list with built-in operations,
        # so pass a copy and preserve the user's original gate selection.
        basis_gates=requested_basis_gates.copy(),
        seed=40,
    )

    print(f"Reading coefficients from {trajectory_path}")
    print(
        f"Transpiling {steps} update(s), optimization level "
        f"{args.optimization_level}, basis gates {requested_basis_gates}"
    )
    started = time.perf_counter()
    circuit, _target_unitary, unitary_error = Transpile.transpile_circuit_qite(
        config["n_qubits"],
        config["D"],
        config.get("T", 2),
        pauli_pool,
        coefficients,
        config["dt"],
        steps,
        backend,
        args.optimization_level,
        PEvolution=pauli_evolution,
        compute_unitary_error=args.compute_unitary_error,
        coefficient_threshold=config.get("support_tol", 1e-14),
    )
    elapsed = time.perf_counter() - started

    initial_state = np.asarray(
        build_initial_state(
            config["n_qubits"], config.get("initial_basis_state", 0)
        ).todense()
    ).reshape(-1)
    compiled_state = np.asarray(
        Statevector(initial_state).evolve(circuit).data
    ).reshape(-1)
    uncompiled_state = np.asarray(states[:, -1].todense()).reshape(-1)

    hamiltonian, _ = build_hamiltonian(config)
    ground_energies, ground_vectors = eigsh(
        hamiltonian.tocsc(), k=1, which="SA", tol=1e-10
    )
    ground_index = int(np.argmin(ground_energies))
    ground_state = np.asarray(ground_vectors[:, ground_index]).reshape(-1)
    fidelity_uncompiled_compiled = pure_state_fidelity(
        uncompiled_state, compiled_state
    )
    fidelity_uncompiled_ground = pure_state_fidelity(
        uncompiled_state, ground_state
    )
    fidelity_compiled_ground = pure_state_fidelity(compiled_state, ground_state)

    gate_counts = {str(gate): int(count) for gate, count in circuit.count_ops().items()}
    support_tol = config.get("support_tol", 1e-14)
    nonzero_coefficients = int(np.count_nonzero(coefficients[:steps]))
    active_coefficients = int(
        np.count_nonzero(np.abs(coefficients[:steps]) > support_tol)
    )
    nonempty_evolution_blocks = int(
        np.count_nonzero(
            np.any(np.abs(coefficients[:steps]) > support_tol, axis=2)
        )
    )
    source_hash = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    output_dir = (
        qite_output_dir
        / "transpilation"
        / settings_name(
            requested_basis_gates,
            args.optimization_level,
            pauli_evolution,
            steps,
            args.compute_unitary_error,
        )
    )
    metadata = {
        "source": {
            "config_path": str(config["config_path"]),
            "trajectory_path": str(trajectory_path),
            "states_path": str(states_path),
            "trajectory_sha256": source_hash,
        },
        "qite": {
            "model": config["model"],
            "method": config["method"],
            "n_qubits": int(config["n_qubits"]),
            "D": int(config["D"]),
            "dt": float(config["dt"]),
            "steps": int(steps),
            "configured_steps": int(config["N"]),
            "num_trotter_pieces": int(h_trot.Nk),
            "num_paulis_per_piece": int(num_paulis),
            "support_tolerance": float(support_tol),
            "nonzero_coefficients": nonzero_coefficients,
            "active_coefficients": active_coefficients,
        },
        "transpilation": {
            "backend": "GenericBackendV2",
            "basis_gates": requested_basis_gates,
            "optimization_level": int(args.optimization_level),
            "seed_transpiler": 40,
            "circuit_construction": (
                "PauliEvolutionGate" if pauli_evolution else "HamiltonianGate"
            ),
            "depth": int(circuit.depth()),
            "size": int(circuit.size()),
            "width": int(circuit.width()),
            "nonlocal_gates": int(circuit.num_nonlocal_gates()),
            "gate_counts": gate_counts,
            # With PauliEvolutionGate's default one-repetition product
            # synthesis, every coefficient surviving support_tol produces
            # one conceptual Pauli rotation before native-gate transpilation.
            "pauli_rotations_before_transpilation": (
                active_coefficients if pauli_evolution else None
            ),
            "nonempty_evolution_blocks_before_transpilation": (
                nonempty_evolution_blocks
            ),
            # Transpile.py uses np.linalg.norm, which is the Frobenius norm.
            "unitary_frobenius_error": (
                None if unitary_error is None else float(unitary_error)
            ),
            "elapsed_seconds": float(elapsed),
        },
        "fidelity": {
            "uncompiled_vs_compiled": fidelity_uncompiled_compiled,
            "uncompiled_vs_ground_state": fidelity_uncompiled_ground,
            "compiled_vs_ground_state": fidelity_compiled_ground,
            "ground_state_fidelity_degradation": (
                fidelity_uncompiled_ground - fidelity_compiled_ground
            ),
            "ground_state_energy": float(np.real(ground_energies[ground_index])),
            "ground_state_convention": (
                "one lowest-eigenvalue eigenvector; use a ground-space "
                "projector for degenerate Hamiltonians"
            ),
        },
    }
    save_outputs(output_dir, circuit, metadata, overwrite=args.overwrite)
    print(f"Saved transpilation results to {output_dir}")
    print(
        f"depth={circuit.depth()}, size={circuit.size()}, "
        f"nonlocal_gates={circuit.num_nonlocal_gates()}, elapsed={elapsed:.2f}s"
    )
    print(
        f"Pauli rotations before transpilation: "
        f"{metadata['transpilation']['pauli_rotations_before_transpilation']}"
    )
    print(
        f"F(uncompiled, compiled)={fidelity_uncompiled_compiled:.12g}, "
        f"F(uncompiled, ground)={fidelity_uncompiled_ground:.12g}, "
        f"F(compiled, ground)={fidelity_compiled_ground:.12g}"
    )


if __name__ == "__main__":
    main()
