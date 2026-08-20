#!/usr/bin/env python3
"""Compare best and final default-QITE fidelity versus domain size D.

The script scans the completed default-QITE trajectories for n=9,...,14 and
D=2,...,7. For each model it creates a six-panel PDF, with one panel per
system size, and writes the numerical values to a combined CSV file.

Incomplete trajectories are omitted rather than interpreted as final results.
Edit the parameter block in ``main()`` to change the scanned grid.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modules import Hamiltonian


def model_specifications() -> dict[str, dict[str, Any]]:
    """Describe how to build each Hamiltonian and locate its QITE results."""
    result_root = PROJECT_DIR / "outputs" / "qite_runs"
    return {
        "disordered_tfim": {
            "title": r"Disordered TFIM ($J=0.5,\ h=1$)",
            "hamiltonian": lambda n: Hamiltonian.TFIM(0.5, 1.0, n, T=2)[0],
            "result_dir": lambda n, D: (
                result_root
                / f"tfim_n{n}_D{D}_N100_dt0p05_T2_J0p5_h1"
                / "qite_default"
            ),
        },
        "ordered_tfim": {
            "title": r"Ordered TFIM ($J=1,\ h=0.5$)",
            "hamiltonian": lambda n: Hamiltonian.TFIM(1.0, 0.5, n, T=2)[0],
            "result_dir": lambda n, D: (
                result_root
                / f"tfim_n{n}_D{D}_N100_dt0p05_T2_J1_h0p5"
                / "qite_default"
            ),
        },
        "heisenberg_pbc": {
            "title": r"Heisenberg model (PBC, $J=1$)",
            "hamiltonian": lambda n: Hamiltonian.Heisenberg_PBC(1.0, n, T=2)[0],
            "result_dir": lambda n, D: (
                result_root
                / f"heisenberg_pbc_n{n}_D{D}_N100_dt0p05_T2_J1"
                / "qite_default"
            ),
        },
    }


def ground_state(hamiltonian: sp.spmatrix) -> np.ndarray:
    """Return one normalized eigenvector with the lowest eigenvalue."""
    eigenvalues, eigenvectors = eigsh(
        hamiltonian.tocsc(), k=1, which="SA", tol=1e-10
    )
    index = int(np.argmin(eigenvalues))
    vector = np.asarray(eigenvectors[:, index]).reshape(-1)
    return vector / np.linalg.norm(vector)


def fidelity_trajectory(states: sp.csc_matrix, reference: np.ndarray) -> np.ndarray:
    """Calculate pure-state fidelity with a reference for every saved state."""
    overlaps = np.asarray(states.getH() @ reference).reshape(-1)
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def collect_model_rows(
    model: str,
    specification: dict[str, Any],
    n_values: list[int],
    D_values: list[int],
    expected_updates: int,
) -> list[dict[str, Any]]:
    """Collect best and final fidelity for every completed trajectory."""
    rows: list[dict[str, Any]] = []
    hamiltonian_builder: Callable[[int], sp.spmatrix] = specification["hamiltonian"]
    result_dir_builder: Callable[[int, int], Path] = specification["result_dir"]

    for n_qubits in n_values:
        reference = ground_state(hamiltonian_builder(n_qubits))
        for D in D_values:
            result_dir = result_dir_builder(n_qubits, D)
            trajectory_path = result_dir / "trajectory.npz"
            states_path = result_dir / "states.npz"
            if not trajectory_path.is_file() or not states_path.is_file():
                continue

            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                coefficients = np.asarray(trajectory["a"])
            states = sp.load_npz(states_path).tocsc()

            # Only completed runs belong in a final scalability comparison.
            if (
                coefficients.shape[0] != expected_updates
                or states.shape[1] != expected_updates + 1
            ):
                continue

            fidelities = fidelity_trajectory(states, reference)
            best_step = int(np.argmax(fidelities))
            rows.append(
                {
                    "model": model,
                    "n_qubits": n_qubits,
                    "D": D,
                    "best_fidelity": float(fidelities[best_step]),
                    "best_step": best_step,
                    "final_fidelity": float(fidelities[-1]),
                    "fidelity_drop": float(fidelities[best_step] - fidelities[-1]),
                    "result_dir": str(result_dir),
                }
            )
    return rows


def plot_model(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    n_values: list[int],
) -> None:
    """Create one six-panel best-versus-final comparison for a model."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for axis, n_qubits in zip(axes.flat, n_values):
        subset = sorted(
            (row for row in rows if int(row["n_qubits"]) == n_qubits),
            key=lambda row: int(row["D"]),
        )
        if subset:
            domains = [int(row["D"]) for row in subset]
            axis.plot(
                domains,
                [float(row["best_fidelity"]) for row in subset],
                color="tab:blue",
                linestyle="-",
                marker="o",
                label="Best over time",
            )
            axis.plot(
                domains,
                [float(row["final_fidelity"]) for row in subset],
                color="tab:orange",
                linestyle="--",
                marker="s",
                label="Final update",
            )
        axis.axhline(1.0, color="black", linewidth=1.0)
        axis.set_title(rf"$n={n_qubits}$")
        axis.set_xticks(range(2, 8))
        axis.grid(alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xlabel("Domain size $D$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Ground-state fidelity")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write all plotted values to one machine-readable table."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    # ------------------------------------------------------------------
    # EDIT THIS BLOCK TO CHANGE THE DEFAULT-QITE SCALABILITY GRID.
    # ------------------------------------------------------------------
    n_values = [9, 10, 11, 12, 13, 14]
    D_values = [2, 3, 4, 5, 6, 7]
    expected_updates = 100
    output_dir = PROJECT_DIR / "plots" / "scalability" / "qite_n_D_scalability"
    # ------------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for model, specification in model_specifications().items():
        rows = collect_model_rows(
            model, specification, n_values, D_values, expected_updates
        )
        all_rows.extend(rows)
        plot_model(
            output_dir / f"{model}_best_vs_final_fidelity_vs_D.pdf",
            specification["title"],
            rows,
            n_values,
        )

    if not all_rows:
        raise FileNotFoundError("No completed default-QITE trajectories were found.")
    write_csv(output_dir / "best_vs_final_fidelity.csv", all_rows)

    print(
        "Note: odd-size Heisenberg rings have a degenerate ground space. "
        "Their plotted values use one ground-state eigenvector, matching the "
        "existing plotting convention."
    )


if __name__ == "__main__":
    main()
