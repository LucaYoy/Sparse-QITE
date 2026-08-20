#!/usr/bin/env python3
"""Plot one or more completed cumulative-Jaccard sparse-QITE trajectories.

Edit the parameter lists in ``main()`` and run from the project directory::

    python scripts/plotting/Plot_QITE_Jaccard_results.py

The Jaccard diagnostics are reconstructed from the saved coefficient tensor
using the support tolerance, cumulative-Jaccard threshold, and patience stored
in the corresponding configuration parameters. Every time-based plot marks
the support-freezing decision with a dashed black vertical line.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/qite-matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Generate_QITE_configs import method_file_tag, parameter_jobs, physical_folder_name
from Plot_QITE_results import (
    coefficient_localities,
    fidelity_with_vector,
    ground_state,
    load_result,
    locality_counts_over_time,
    trajectory_fidelity,
)


def reconstruct_jaccard_metrics(
    coefficients: np.ndarray,
    support_tol: float,
    jaccard_cumulative_min: float,
    jaccard_patience: int,
) -> dict[str, Any]:
    """Reconstruct the online cumulative-support freezing rule from ``a``."""
    n_steps = coefficients.shape[0]
    active_terms = np.zeros(n_steps, dtype=int)
    new_terms = np.zeros(n_steps, dtype=int)
    cumulative_terms = np.zeros(n_steps, dtype=int)
    cumulative_jaccard = np.ones(n_steps, dtype=float)
    jaccard_streak = np.zeros(n_steps, dtype=int)
    frozen = np.zeros(n_steps, dtype=bool)

    cumulative_support: set[int] = set()
    streak = 0
    freeze_step: int | None = None

    for step, coefficient_step in enumerate(coefficients):
        active_support = {
            int(index)
            for index in np.flatnonzero(
                np.abs(np.asarray(coefficient_step).reshape(-1)) > support_tol
            )
        }
        previous_size = len(cumulative_support)
        discovered = active_support - cumulative_support
        cumulative_support |= active_support

        if cumulative_support:
            overlap = previous_size / len(cumulative_support)
            stable = overlap >= jaccard_cumulative_min
        else:
            overlap = 1.0
            stable = False

        if freeze_step is None:
            streak = streak + 1 if stable else 0
            if streak >= jaccard_patience:
                freeze_step = step

        active_terms[step] = len(active_support)
        new_terms[step] = len(discovered)
        cumulative_terms[step] = len(cumulative_support)
        cumulative_jaccard[step] = overlap
        jaccard_streak[step] = streak
        frozen[step] = freeze_step is not None

    return {
        "active_terms": active_terms,
        "new_terms": new_terms,
        "cumulative_terms": cumulative_terms,
        "cumulative_jaccard": cumulative_jaccard,
        "jaccard_streak": jaccard_streak,
        "frozen": frozen,
        "freeze_step": freeze_step,
    }


def add_freeze_line(axis: plt.Axes, freeze_time: float | None) -> None:
    """Mark the time at which support freezing was decided."""
    if freeze_time is not None:
        axis.axvline(
            freeze_time,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Support frozen",
        )


def save_time_curve(
    times: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title: str,
    ylabel: str,
    freeze_time: float | None,
    curve_label: str,
    ylim: tuple[float, float] | None = None,
    horizontal_line: tuple[float, str, str] | None = None,
    baseline_times: np.ndarray | None = None,
    baseline_values: np.ndarray | None = None,
    baseline_label: str = "QITE",
) -> None:
    """Save one time-series PDF with consistent freeze-line styling."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(times, values, marker="o", markersize=3, label=curve_label)
    if baseline_times is not None and baseline_values is not None:
        axis.plot(
            baseline_times,
            baseline_values,
            color="tab:orange",
            linestyle="-",
            linewidth=1.4,
            marker="s",
            markersize=2.5,
            label=baseline_label,
        )
    if horizontal_line is not None:
        value, label, color = horizontal_line
        axis.axhline(value, color=color, linestyle=":", linewidth=1.2, label=label)
    add_freeze_line(axis, freeze_time)
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


def save_locality_heatmap(
    counts: np.ndarray,
    dt: float,
    freeze_time: float | None,
    title: str,
    output_path: Path,
) -> None:
    """Save locality counts with zero cells silver and the freeze time marked."""
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
    add_freeze_line(axis, freeze_time)
    axis.set_xlabel("Imaginary time")
    axis.set_ylabel("Pauli locality")
    axis.set_yticks(np.arange(1, counts.shape[0] + 1))
    axis.set_title(title)
    axis.legend()
    color_bar = figure.colorbar(image, ax=axis)
    color_bar.set_label("Active terms")
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def write_metrics_csv(
    output_path: Path,
    times: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Write reconstructed support diagnostics in a portable table."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step_index",
            "update",
            "time",
            "active_terms",
            "new_terms",
            "cumulative_terms",
            "cumulative_jaccard",
            "jaccard_streak",
            "frozen",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step, time in enumerate(times):
            writer.writerow(
                {
                    "step_index": step,
                    "update": step + 1,
                    "time": float(time),
                    "active_terms": int(metrics["active_terms"][step]),
                    "new_terms": int(metrics["new_terms"][step]),
                    "cumulative_terms": int(metrics["cumulative_terms"][step]),
                    "cumulative_jaccard": float(metrics["cumulative_jaccard"][step]),
                    "jaccard_streak": int(metrics["jaccard_streak"][step]),
                    "frozen": bool(metrics["frozen"][step]),
                }
            )
    print(f"Wrote {output_path}")


def write_support_measurement_summary(
    output_path: Path,
    job: dict[str, Any],
    qite_coefficients: np.ndarray,
    sparse_coefficients: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Compare total active support and evaluated-pool cost against QITE.

    A measurement is counted for every coefficient slot whose QITE equation is
    evaluated. QITE evaluates the full reconstructed pool at every update.
    Jaccard sparse-QITE evaluates that full pool through the freeze-decision
    update and evaluates only the frozen cumulative pool afterward.
    """
    n_steps = min(qite_coefficients.shape[0], sparse_coefficients.shape[0])
    qite = qite_coefficients[:n_steps]
    sparse = sparse_coefficients[:n_steps]
    support_tol = float(job["support_tol"])

    qite_total_active = int(np.count_nonzero(np.abs(qite) > support_tol))
    sparse_total_active = int(np.count_nonzero(np.abs(sparse) > support_tol))
    qite_slots_per_step = int(qite.shape[1] * qite.shape[2])
    sparse_slots_per_step = int(sparse.shape[1] * sparse.shape[2])
    if qite_slots_per_step != sparse_slots_per_step:
        raise ValueError(
            "QITE and Jaccard coefficient pools have different sizes: "
            f"{qite_slots_per_step} != {sparse_slots_per_step}."
        )

    qite_total_measurements = qite_slots_per_step * n_steps
    freeze_step = metrics["freeze_step"]
    if freeze_step is None or freeze_step >= n_steps:
        full_pool_updates = n_steps
        frozen_pool_updates = 0
        frozen_support_size = 0
        sparse_total_measurements = sparse_slots_per_step * n_steps
    else:
        # a[freeze_step] produced the decision and still used the full pool.
        full_pool_updates = freeze_step + 1
        frozen_pool_updates = n_steps - full_pool_updates
        frozen_support_size = int(metrics["cumulative_terms"][freeze_step])
        sparse_total_measurements = (
            sparse_slots_per_step * full_pool_updates
            + frozen_support_size * frozen_pool_updates
        )

    active_fraction = (
        sparse_total_active / qite_total_active
        if qite_total_active > 0
        else float("nan")
    )
    measurement_fraction = (
        sparse_total_measurements / qite_total_measurements
        if qite_total_measurements > 0
        else float("nan")
    )
    row = {
        "model": job["model"],
        "n_qubits": job["n_qubits"],
        "D": job["D"],
        "N_compared": n_steps,
        "dt": job["dt"],
        "lasso_lambda": job["lasso_lambda"],
        "jaccard_cumulative_min": job["jaccard_cumulative_min"],
        "jaccard_patience": job["jaccard_patience"],
        "support_tol": support_tol,
        "freeze_step_index": freeze_step,
        "freeze_decision_update": None if freeze_step is None else freeze_step + 1,
        "first_frozen_update": None if freeze_step is None else freeze_step + 2,
        "full_pool_size": qite_slots_per_step,
        "frozen_support_size": frozen_support_size,
        "full_pool_updates_sparse_qite": full_pool_updates,
        "frozen_pool_updates_sparse_qite": frozen_pool_updates,
        "total_active_terms_qite": qite_total_active,
        "total_active_terms_sparse_qite": sparse_total_active,
        "active_term_fraction_sparse_over_qite": active_fraction,
        "active_term_saving_fraction": 1.0 - active_fraction,
        "total_measurements_qite": qite_total_measurements,
        "total_measurements_sparse_qite": sparse_total_measurements,
        "measurement_fraction_sparse_over_qite": measurement_fraction,
        "measurement_saving_fraction": 1.0 - measurement_fraction,
        "measurement_definition": (
            "coefficient slots evaluated: full pool through freeze-decision "
            "update; frozen cumulative pool afterward"
        ),
    }
    write_vertical_csv(output_path, row)
    print(f"Wrote {output_path}")


def write_vertical_csv(output_path: Path, metrics: dict[str, Any]) -> None:
    """Write one summary as human-readable ``metric,value`` rows."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for name, value in metrics.items():
            writer.writerow([name, value])


def write_final_metrics(
    output_path: Path,
    job: dict[str, Any],
    energy: np.ndarray,
    exact_energy: float,
    ground_fidelity: np.ndarray,
    final_fidelity_qite: float,
    metrics: dict[str, Any],
) -> None:
    """Write final trajectory values and support sizes in vertical form."""
    freeze_step = metrics["freeze_step"]
    summary = {
        "model": job["model"],
        "n_qubits": job["n_qubits"],
        "D": job["D"],
        "completed_updates": len(energy) - 1,
        "final_time": (len(energy) - 1) * job["dt"],
        "lasso_lambda": job["lasso_lambda"],
        "jaccard_cumulative_min": job["jaccard_cumulative_min"],
        "jaccard_patience": job["jaccard_patience"],
        "support_tol": job["support_tol"],
        "freeze_step_index": freeze_step,
        "freeze_decision_update": None if freeze_step is None else freeze_step + 1,
        "freeze_time": (
            None
            if freeze_step is None
            else round((freeze_step + 1) * job["dt"], 12)
        ),
        "final_fidelity_vs_ground_state": float(ground_fidelity[-1]),
        "final_fidelity_vs_qite": final_fidelity_qite,
        "exact_ground_energy": exact_energy,
        "final_energy": float(energy[-1]),
        "final_absolute_energy_error": abs(float(energy[-1]) - exact_energy),
        "final_cumulative_terms": int(metrics["cumulative_terms"][-1]),
        "final_active_terms": int(metrics["active_terms"][-1]),
        "final_new_terms": int(metrics["new_terms"][-1]),
        "final_cumulative_jaccard": float(metrics["cumulative_jaccard"][-1]),
    }
    write_vertical_csv(output_path, summary)
    print(f"Wrote {output_path}")


def plot_job(
    job: dict[str, Any],
    results_root: Path,
    plots_root: Path,
) -> int:
    """Load and plot one completed Jaccard-LASSO result."""
    folder_name = physical_folder_name(job)
    file_tag = method_file_tag(
        job["method"],
        job["lasso_lambda"],
        job["jaccard_cumulative_min"],
        job["jaccard_patience"],
        job["support_tol"],
    )
    result_dir = results_root / folder_name / file_tag
    try:
        energy, states, coefficients = load_result(result_dir)
    except FileNotFoundError as error:
        print(f"Skipping: {error}")
        return 0

    metrics = reconstruct_jaccard_metrics(
        coefficients,
        job["support_tol"],
        job["jaccard_cumulative_min"],
        job["jaccard_patience"],
    )
    freeze_step = metrics["freeze_step"]
    freeze_time = None if freeze_step is None else (freeze_step + 1) * job["dt"]
    state_times = np.arange(len(energy)) * job["dt"]
    coefficient_times = (np.arange(coefficients.shape[0]) + 1) * job["dt"]
    label = (
        rf"Jaccard-LASSO ($\lambda={job['lasso_lambda']:g}$, "
        rf"$J_{{\rm cum}}={job['jaccard_cumulative_min']:g}$, "
        rf"$P={job['jaccard_patience']}$)"
    )

    plot_dir = plots_root / folder_name / file_tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(plot_dir / "jaccard_metrics.csv", coefficient_times, metrics)

    exact_energy, exact_state = ground_state(job)
    ground_fidelity = fidelity_with_vector(states, exact_state)

    # Load the matching standard-QITE trajectory once and reuse it in every
    # plot for which a direct baseline has a well-defined meaning.
    qite_dir = results_root / folder_name / "qite_default"
    qite_energy = None
    qite_states = None
    qite_coefficients = None
    qite_state_times = None
    qite_coefficient_times = None
    qite_ground_fidelity = None
    qite_metrics = None
    try:
        qite_energy, qite_states, qite_coefficients = load_result(qite_dir)
    except FileNotFoundError:
        print(f"No completed QITE baseline found in {qite_dir}")
    else:
        if qite_coefficients.shape[1:] != coefficients.shape[1:]:
            raise ValueError(
                "QITE and Jaccard coefficient pools have different shapes: "
                f"{qite_coefficients.shape[1:]} != {coefficients.shape[1:]}."
            )
        qite_state_times = np.arange(len(qite_energy)) * job["dt"]
        qite_coefficient_times = (
            np.arange(qite_coefficients.shape[0]) + 1
        ) * job["dt"]
        qite_ground_fidelity = fidelity_with_vector(qite_states, exact_state)
        qite_metrics = reconstruct_jaccard_metrics(
            qite_coefficients,
            job["support_tol"],
            job["jaccard_cumulative_min"],
            job["jaccard_patience"],
        )

    save_time_curve(
        state_times,
        energy,
        plot_dir / "energy.pdf",
        folder_name,
        "Energy",
        freeze_time,
        label,
        horizontal_line=(exact_energy, "Exact ground energy", "black"),
        baseline_times=qite_state_times,
        baseline_values=qite_energy,
    )
    save_time_curve(
        state_times,
        ground_fidelity,
        plot_dir / "fidelity_vs_ground_state.pdf",
        folder_name,
        "Fidelity with ground state",
        freeze_time,
        label,
        ylim=(-0.02, 1.02),
        horizontal_line=(1.0, "Unit fidelity", "gray"),
        baseline_times=qite_state_times,
        baseline_values=qite_ground_fidelity,
    )

    slots_per_step = coefficients.shape[1] * coefficients.shape[2]
    active_percentage = 100.0 * metrics["active_terms"] / slots_per_step
    qite_active_percentage = (
        None
        if qite_metrics is None
        else 100.0 * qite_metrics["active_terms"] / slots_per_step
    )
    support_plots = [
        (
            "active_terms.pdf",
            metrics["active_terms"],
            None if qite_metrics is None else qite_metrics["active_terms"],
            "Active coefficient slots",
        ),
        (
            "new_terms.pdf",
            metrics["new_terms"],
            None if qite_metrics is None else qite_metrics["new_terms"],
            "New coefficient slots",
        ),
        (
            "cumulative_terms.pdf",
            metrics["cumulative_terms"],
            None if qite_metrics is None else qite_metrics["cumulative_terms"],
            "Cumulative coefficient slots",
        ),
        (
            "active_terms_percentage.pdf",
            active_percentage,
            qite_active_percentage,
            "Active slots (%)",
        ),
    ]
    for filename, values, qite_values, ylabel in support_plots:
        save_time_curve(
            coefficient_times,
            values,
            plot_dir / filename,
            folder_name,
            ylabel,
            freeze_time,
            label,
            baseline_times=qite_coefficient_times,
            baseline_values=qite_values,
        )

    save_time_curve(
        coefficient_times,
        metrics["cumulative_jaccard"],
        plot_dir / "cumulative_jaccard.pdf",
        folder_name,
        "Cumulative Jaccard",
        freeze_time,
        label,
        ylim=(-0.02, 1.02),
        horizontal_line=(
            job["jaccard_cumulative_min"],
            "Jaccard threshold",
            "gray",
        ),
        baseline_times=qite_coefficient_times,
        baseline_values=(
            None if qite_metrics is None else qite_metrics["cumulative_jaccard"]
        ),
    )
    save_time_curve(
        coefficient_times,
        metrics["jaccard_streak"],
        plot_dir / "jaccard_streak.pdf",
        folder_name,
        "Consecutive stable steps",
        freeze_time,
        label,
        horizontal_line=(job["jaccard_patience"], "Required patience", "gray"),
    )

    localities = coefficient_localities(job, coefficients.shape)
    locality_counts = locality_counts_over_time(
        coefficients,
        localities,
        job["D"],
        job["support_tol"],
    )
    save_locality_heatmap(
        locality_counts,
        job["dt"],
        freeze_time,
        f"{folder_name}\n{label}",
        plot_dir / "locality_heatmap.pdf",
    )

    figures_written = 9
    final_fidelity_qite = float("nan")
    if qite_states is None or qite_coefficients is None:
        print(f"Skipping fidelity with QITE: no completed reference in {qite_dir}")
    else:
        fidelity = trajectory_fidelity(qite_states, states)
        final_fidelity_qite = float(fidelity[-1])
        fidelity_times = np.arange(len(fidelity)) * job["dt"]
        save_time_curve(
            fidelity_times,
            fidelity,
            plot_dir / "fidelity_vs_qite.pdf",
            folder_name,
            "Fidelity with QITE",
            freeze_time,
            label,
            ylim=(-0.02, 1.02),
            horizontal_line=(1.0, "Unit fidelity", "gray"),
        )
        figures_written += 1
        write_support_measurement_summary(
            plot_dir / "support_measurement_summary.csv",
            job,
            qite_coefficients,
            coefficients,
            metrics,
        )

    write_final_metrics(
        plot_dir / "final_metrics.csv",
        job,
        energy,
        exact_energy,
        ground_fidelity,
        final_fidelity_qite,
        metrics,
    )

    if freeze_step is None:
        print(f"Support did not freeze for {file_tag}.")
    else:
        print(
            f"Support froze after update {freeze_step + 1} at time {freeze_time:g}; "
            f"the reduced pool was first used in update {freeze_step + 2}."
        )
    return figures_written


def main() -> None:
    # -----------------------------------------------------------------------
    # EDIT THESE LISTS TO MATCH THE COMPLETED JACCARD CONFIGURATIONS TO PLOT.
    # -----------------------------------------------------------------------
    parameters = {
        "n_qubits": [8],
        "D": [6],
        "N": [100],
        "dt": [0.05],
        "method": ["Jaccard_lasso"],
        "model": ["tfim"],

        "J": [1.0],
        "h": [0.5],
        "T": [2],
        "cluster_lambda": [0.5],

        "lasso_lambda": [1e-3],
        "lasso_max_iter": 500,
        "lasso_tol": 1e-8,
        "jaccard_cumulative_min": [0.99],
        "jaccard_patience": [3],
        "support_tol": [1e-14],

        "initial_basis_state": 0,
        "verbose": False,
    }

    results_root = (PROJECT_DIR / "outputs" / "qite_runs").resolve()
    plots_root = (PROJECT_DIR / "plots" / "qite_jaccard").resolve()
    # -----------------------------------------------------------------------

    figures_written = 0
    jobs = list(parameter_jobs(parameters))
    for job in jobs:
        if job["method"] != "Jaccard_lasso":
            raise ValueError("This script only plots method='Jaccard_lasso'.")
        figures_written += plot_job(job, results_root, plots_root)

    print(f"Created {figures_written} figure(s) under {plots_root}")


if __name__ == "__main__":
    main()
