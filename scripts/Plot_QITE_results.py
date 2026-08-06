#!/usr/bin/env python3
"""Plot the standard, sparse, and adaptive QITE results used in Figure 2.

Run:

    python scripts/Plot_QITE_results.py

The PDFs are written below ``plots/qite_adaptive``.  Each curve plot compares
standard QITE, fixed-lambda Sparse-QITE, and Adaptive Sparse-QITE.
"""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any

# Use a writable Matplotlib cache on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/qite-matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")  # Plot safely on Condor/login nodes without a display.

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from Generate_QITE_configs import method_file_tag, physical_folder_name
from Run_QITE_cluster import build_hamiltonian


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def curve_label(job: dict[str, Any]) -> str:
    """Return a readable legend label for one result trajectory."""
    if job["method"] == "vanilla_lasso":
        return rf"Sparse-QITE ($\lambda={job['lasso_lambda']:g}$)"
    if job["method"] == "adaptive_lasso_v1":
        return "Adaptive Sparse-QITE"
    if job["method"] in {"default", "LU"}:
        return "QITE (LU)"
    return f"QITE ({job['method']})"


def result_file_tag(job: dict[str, Any]) -> str:
    """Return the output-folder name used by each method."""
    if job["method"] == "adaptive_lasso_v1":
        return "qite_adaptive_lasso"
    return method_file_tag(job["method"], job["lasso_lambda"])


def load_result(result_dir: Path) -> tuple[np.ndarray, sp.csc_matrix, np.ndarray]:
    """Load energy, states and coefficients, checking their time dimensions."""
    trajectory_path = result_dir / "trajectory.npz"
    states_path = result_dir / "states.npz"
    if not trajectory_path.is_file() or not states_path.is_file():
        raise FileNotFoundError(f"Missing trajectory.npz or states.npz in {result_dir}")

    with np.load(trajectory_path) as trajectory:
        if set(trajectory.files) != {"Energy", "a"}:
            raise ValueError(f"Unexpected arrays in {trajectory_path}: {trajectory.files}")
        energy = np.asarray(trajectory["Energy"])
        coefficients = np.asarray(trajectory["a"])
    states = sp.load_npz(states_path).tocsc()

    if len(energy) != states.shape[1] or len(energy) != coefficients.shape[0] + 1:
        raise ValueError(
            f"Misaligned trajectory in {result_dir}: Energy={energy.shape}, "
            f"Psi={states.shape}, a={coefficients.shape}"
        )
    return energy, states, coefficients


def ground_state(job: dict[str, Any]) -> tuple[float, np.ndarray]:
    """Compute the lowest eigenvalue and normalized ground-state vector."""
    hamiltonian, _ = build_hamiltonian(job)
    eigenvalues, eigenvectors = eigsh(hamiltonian.tocsc(), k=1, which="SA")
    index = int(np.argmin(eigenvalues))
    vector = np.asarray(eigenvectors[:, index]).reshape(-1)
    vector /= np.linalg.norm(vector)
    return float(np.real(eigenvalues[index])), vector


def fidelity_with_vector(states: sp.csc_matrix, reference: np.ndarray) -> np.ndarray:
    """Return |<reference|psi(t)>|^2 for every state column."""
    overlaps = np.asarray(states.getH() @ reference).reshape(-1)
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def trajectory_fidelity(left: sp.csc_matrix, right: sp.csc_matrix) -> np.ndarray:
    """Return |<left(t)|right(t)>|^2 over their common time interval."""
    common_steps = min(left.shape[1], right.shape[1])
    fidelities = np.empty(common_steps, dtype=float)
    for step in range(common_steps):
        overlap = (left[:, step].getH() @ right[:, step]).toarray()[0, 0]
        fidelities[step] = float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))
    return fidelities


def coefficient_localities(
    job: dict[str, Any],
    coefficient_shape: tuple[int, ...],
) -> np.ndarray:
    """Map every (Trotter piece, basis index) coefficient to its Pauli weight."""
    hamiltonian, h_trot = build_hamiltonian(job)
    D = job["D"]
    n_qubits = job["n_qubits"]
    use_real_basis = bool(np.isreal(hamiltonian.data).all()) and not (
        h_trot.BC == "OBC" and D == n_qubits
    )

    local_words = []
    for word in product("IXYZ", repeat=D):
        if use_real_basis and sum(symbol == "Y" for symbol in word) % 2 == 0:
            continue
        local_words.append(word)

    expected_shape = (h_trot.Nk, len(local_words))
    if coefficient_shape[1:] != expected_shape:
        raise ValueError(
            f"Coefficient basis shape {coefficient_shape[1:]} does not match "
            f"the reconstructed Pauli basis {expected_shape}."
        )

    localities = np.zeros(expected_shape, dtype=int)
    displacement = int(np.floor((D - h_trot.T) / 2))
    for piece in range(h_trot.Nk):
        for pauli_index, word in enumerate(local_words):
            weight = 0
            for local_index, symbol in enumerate(word):
                if symbol == "I":
                    continue
                site = h_trot.indk[piece] + local_index - displacement
                if h_trot.BC != "OBC" or D == n_qubits or 0 <= site < n_qubits:
                    weight += 1
            localities[piece, pauli_index] = weight
    return localities


def cumulative_basis_count(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """Count distinct coefficient slots activated at or before every time step."""
    active = np.abs(coefficients) > threshold
    seen = np.logical_or.accumulate(active, axis=0)
    return np.count_nonzero(seen, axis=(1, 2))


def locality_counts_over_time(
    coefficients: np.ndarray,
    localities: np.ndarray,
    max_locality: int,
    threshold: float,
) -> np.ndarray:
    """Count active coefficient slots at localities 1 through max_locality."""
    active = np.abs(coefficients) > threshold
    counts = np.zeros((max_locality, coefficients.shape[0]), dtype=int)
    for locality in range(1, max_locality + 1):
        counts[locality - 1] = np.count_nonzero(
            active & (localities[None, :, :] == locality),
            axis=(1, 2),
        )
    return counts


def save_locality_heatmap(
    counts: np.ndarray,
    dt: float,
    label: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a locality-versus-time heatmap with zero-count cells in silver."""
    figure, axis = plt.subplots(figsize=(9, 5))
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("silver")
    masked_counts = np.ma.masked_equal(counts, 0)
    n_steps = counts.shape[1]
    image = axis.imshow(
        masked_counts,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        extent=(0.5 * dt, (n_steps + 0.5) * dt, 0.5, counts.shape[0] + 0.5),
    )
    axis.set_xlabel("Imaginary time")
    axis.set_ylabel("Pauli locality")
    axis.set_yticks(np.arange(1, counts.shape[0] + 1))
    axis.set_title(f"{title}\n{label}")
    color_bar = figure.colorbar(image, ax=axis)
    color_bar.set_label("Active terms")
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def save_curves(
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    horizontal_line: tuple[float, str] | None = None,
) -> None:
    """Save one consistently styled PDF containing the supplied curves."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for x_values, y_values, label in curves:
        axis.plot(x_values, y_values, marker="o", markersize=3, label=label)
    if horizontal_line is not None:
        value, label = horizontal_line
        axis.axhline(value, color="black", linestyle="-", linewidth=1.5, label=label)
    axis.set_xlabel("Imaginary time")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_configuration(
    folder_name: str,
    jobs: list[dict[str, Any]],
    results_root: Path,
    plots_root: Path,
    coefficient_threshold: float,
) -> int:
    """Plot every available method/lambda for one physical configuration."""
    loaded_results = []
    for job in jobs:
        file_tag = result_file_tag(job)
        result_dir = results_root / folder_name / file_tag
        try:
            energy, states, coefficients = load_result(result_dir)
        except FileNotFoundError as error:
            print(f"Skipping: {error}")
            continue
        loaded_results.append((job, energy, states, coefficients))

    if not loaded_results:
        print(f"No completed results found for {folder_name}")
        return 0

    exact_energy, exact_ground_state = ground_state(loaded_results[0][0])
    energy_curves = []
    count_curves = []
    percentage_curves = []
    cumulative_curves = []
    ground_fidelity_curves = []
    locality_heatmaps = []

    for job, energy, states, coefficients in loaded_results:
        label = curve_label(job)
        dt = job["dt"]
        energy_times = np.arange(len(energy)) * dt
        coefficient_times = (np.arange(coefficients.shape[0]) + 1) * dt

        active_count = np.count_nonzero(
            np.abs(coefficients) > coefficient_threshold,
            axis=(1, 2),
        )
        coefficients_per_step = coefficients.shape[1] * coefficients.shape[2]
        active_percentage = 100.0 * active_count / coefficients_per_step
        cumulative_count = cumulative_basis_count(coefficients, coefficient_threshold)
        localities = coefficient_localities(job, coefficients.shape)
        locality_counts = locality_counts_over_time(
            coefficients,
            localities,
            job["D"],
            coefficient_threshold,
        )
        ground_fidelity = fidelity_with_vector(states, exact_ground_state)

        energy_curves.append((energy_times, energy, label))
        count_curves.append((coefficient_times, active_count, label))
        percentage_curves.append((coefficient_times, active_percentage, label))
        cumulative_curves.append((coefficient_times, cumulative_count, label))
        ground_fidelity_curves.append((energy_times, ground_fidelity, label))
        locality_heatmaps.append((job, locality_counts, label))

    plot_dir = plots_root / folder_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_curves(
        energy_curves,
        plot_dir / "energy.pdf",
        folder_name,
        "Energy",
        horizontal_line=(exact_energy, "Exact ground energy"),
    )
    save_curves(
        count_curves,
        plot_dir / "active_coefficients.pdf",
        folder_name,
        "Active coefficients",
    )
    save_curves(
        percentage_curves,
        plot_dir / "active_coefficients_percentage.pdf",
        folder_name,
        "Active coefficients (%)",
        ylim=(-2, 102),
    )
    save_curves(
        cumulative_curves,
        plot_dir / "cumulative_basis_terms.pdf",
        folder_name,
        "Cumulative distinct basis terms",
    )
    for job, locality_counts, label in locality_heatmaps:
        file_tag = result_file_tag(job)
        save_locality_heatmap(
            locality_counts,
            job["dt"],
            label,
            folder_name,
            plot_dir / f"locality_heatmap_{file_tag}.pdf",
        )
    save_curves(
        ground_fidelity_curves,
        plot_dir / "fidelity_vs_ground_state.pdf",
        folder_name,
        "Fidelity with ground state",
        ylim=(-0.02, 1.02),
        horizontal_line=(1.0, "Unit fidelity"),
    )
    figures_written = 5 + len(locality_heatmaps)

    qite_result = next(
        (result for result in loaded_results if result[0]["method"] in {"default", "LU"}),
        None,
    )
    sparse_results = [
        result
        for result in loaded_results
        if result[0]["method"] in {"vanilla_lasso", "adaptive_lasso_v1"}
    ]
    if qite_result is not None and sparse_results:
        qite_job, _qite_energy, qite_states, _qite_coefficients = qite_result
        fidelity_curves = []
        for sparse_job, _energy, sparse_states, _coefficients in sparse_results:
            fidelity = trajectory_fidelity(qite_states, sparse_states)
            times = np.arange(len(fidelity)) * qite_job["dt"]
            fidelity_curves.append(
                (
                    times,
                    fidelity,
                    f"QITE vs {curve_label(sparse_job)}",
                )
            )
        save_curves(
            fidelity_curves,
            plot_dir / "fidelity_qite_vs_sparse_qite.pdf",
            folder_name,
            "QITE–Sparse-QITE fidelity",
            ylim=(-0.02, 1.02),
            horizontal_line=(1.0, "Unit fidelity"),
        )
        figures_written += 1
    else:
        print(f"Skipping QITE–Sparse-QITE fidelity for {folder_name}: both methods are required.")

    return figures_written


def main() -> None:
    results_root = PROJECT_DIR / "outputs" / "qite_runs"
    plots_root = PROJECT_DIR / "plots" / "qite_adaptive"
    coefficient_threshold = 1e-14

    configurations = [
        {"D": 6, "J": 1.0, "h": 0.5, "lasso_lambda": 0.002},
        {"D": 4, "J": 0.5, "h": 1.0, "lasso_lambda": 0.0015},
    ]

    figures_written = 0
    for configuration in configurations:
        common_job = {
            "model": "tfim",
            "n_qubits": 8,
            "D": configuration["D"],
            "N": 100,
            "dt": 0.05,
            "T": 2,
            "J": configuration["J"],
            "h": configuration["h"],
            "cluster_lambda": 0.5,
        }
        jobs = [
            {**common_job, "method": "default", "lasso_lambda": None},
            {
                **common_job,
                "method": "vanilla_lasso",
                "lasso_lambda": configuration["lasso_lambda"],
            },
            {
                **common_job,
                "method": "adaptive_lasso_v1",
                "lasso_lambda": None,
            },
        ]
        folder_name = physical_folder_name(common_job)
        figures_written += plot_configuration(
            folder_name,
            jobs,
            results_root.resolve(),
            plots_root.resolve(),
            coefficient_threshold,
        )

    print(f"Created {figures_written} figure(s) under {plots_root.resolve()}")


if __name__ == "__main__":
    main()
