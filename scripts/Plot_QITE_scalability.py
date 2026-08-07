#!/usr/bin/env python3
"""Select and plot best-lambda QITE scalability results.

Edit the parameter and fidelity-threshold lists in ``main()``, then run::

    python scripts/Plot_QITE_scalability.py

For each D and fidelity threshold, the selected run is the largest tested
LASSO lambda that reaches the QITE time horizon and satisfies
F(QITE, sparse-QITE) >= threshold at that horizon.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/qite-matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from Generate_QITE_configs import method_file_tag, parameter_jobs, physical_folder_name
from Plot_QITE_results import (
    coefficient_localities,
    cumulative_basis_count,
    fidelity_with_vector,
    ground_state,
    load_result,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def phase_label(parameters: dict[str, Any]) -> str:
    """Infer the TFIM phase represented by the selected J and h values."""
    if set(parameters["model"]) != {"tfim"}:
        return "not_applicable_phase"
    phases = set()
    for J in parameters["J"]:
        for h in parameters["h"]:
            if abs(J) > abs(h):
                phases.add("ordered")
            elif abs(J) < abs(h):
                phases.add("disordered")
            else:
                phases.add("critical")
    return next(iter(phases)) if len(phases) == 1 else "mixed_phase"


def scalability_analysis_name(parameters: dict[str, Any]) -> str:
    """Build a model-aware folder name for a D-scalability analysis."""
    models = list(parameters["model"])
    if models == ["tfim"]:
        model_tag = f"{phase_label(parameters)}_tfim"
    elif len(models) == 1:
        model_tag = models[0]
    else:
        model_tag = "mixed_models_" + "_".join(models)
    qubit_tag = "_".join(f"n{n}" for n in parameters["n_qubits"])
    return f"{model_tag}_{qubit_tag}_D_scalability"


def state_fidelity(left, right) -> float:
    """Return pure-state fidelity for two sparse state columns."""
    overlap = (left.getH() @ right).toarray()[0, 0]
    return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))


def support_metrics(coefficients: np.ndarray, threshold: float) -> dict[str, float]:
    """Summarize active and cumulative coefficient support over time."""
    if coefficients.shape[0] == 0:
        return {
            "max_active_terms": 0.0,
            "total_active_terms": 0.0,
            "average_active_terms": 0.0,
            "final_cumulative_terms": 0.0,
        }
    active = np.count_nonzero(np.abs(coefficients) > threshold, axis=(1, 2))
    cumulative = cumulative_basis_count(coefficients, threshold)
    return {
        "max_active_terms": float(np.max(active)),
        "total_active_terms": float(np.sum(active)),
        "average_active_terms": float(np.mean(active)),
        "final_cumulative_terms": float(cumulative[-1]),
    }


def load_scan_rows(
    parameters: dict[str, Any],
    results_root: Path,
    coefficient_threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load every available lambda and its QITE reference for each selected D."""
    jobs = list(parameter_jobs(parameters))
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        groups.setdefault(physical_folder_name(job), []).append(job)

    scan_rows: list[dict[str, Any]] = []
    qite_rows: dict[int, dict[str, Any]] = {}
    for folder_name, group in groups.items():
        qite_job = next(
            (job for job in group if job["method"] in {"default", "LU"}),
            None,
        )
        if qite_job is None:
            print(f"Skipping {folder_name}: no QITE method was selected.")
            continue

        qite_dir = results_root / folder_name / method_file_tag(
            qite_job["method"], qite_job["lasso_lambda"]
        )
        try:
            qite_energy, qite_states, qite_coefficients = load_result(qite_dir)
        except FileNotFoundError as error:
            print(f"Skipping {folder_name}: {error}")
            continue

        D = int(qite_job["D"])
        qite_steps = qite_coefficients.shape[0]
        exact_energy, exact_state = ground_state(qite_job)
        qite_support = support_metrics(qite_coefficients, coefficient_threshold)
        qite_ground_fidelities = fidelity_with_vector(qite_states, exact_state)
        qite_best_fidelity_step = int(np.argmax(qite_ground_fidelities))
        qite_ground_fidelity = float(qite_ground_fidelities[-1])
        qite_rows[D] = {
            "D": D,
            "qite_steps": qite_steps,
            "qite_final_energy": float(qite_energy[-1]),
            "qite_ground_fidelity": qite_ground_fidelity,
            "qite_best_ground_fidelity": float(
                qite_ground_fidelities[qite_best_fidelity_step]
            ),
            "qite_best_fidelity_step": qite_best_fidelity_step,
            "qite_energy_abs_error": abs(float(qite_energy[-1]) - exact_energy),
            **{f"qite_{key}": value for key, value in qite_support.items()},
            "qite_result_dir": str(qite_dir),
        }

        sparse_jobs = [job for job in group if job["method"] == "vanilla_lasso"]
        for sparse_job in sparse_jobs:
            sparse_dir = results_root / folder_name / method_file_tag(
                sparse_job["method"], sparse_job["lasso_lambda"]
            )
            try:
                sparse_energy, sparse_states, sparse_coefficients = load_result(sparse_dir)
            except FileNotFoundError as error:
                print(f"Missing lambda={sparse_job['lasso_lambda']:g}, D={D}: {error}")
                continue

            sparse_steps = sparse_coefficients.shape[0]
            reaches_horizon = sparse_steps >= qite_steps
            comparison_step = min(qite_steps, sparse_steps)
            trajectory_fidelities_qite = np.asarray(
                [
                    state_fidelity(qite_states[:, step], sparse_states[:, step])
                    for step in range(comparison_step + 1)
                ]
            )
            comparison_fidelity = float(trajectory_fidelities_qite[-1])
            comparable_coefficients = sparse_coefficients[:comparison_step]
            sparse_support = support_metrics(comparable_coefficients, coefficient_threshold)
            comparison_energy = float(sparse_energy[comparison_step])
            ground_fidelities = fidelity_with_vector(
                sparse_states[:, : comparison_step + 1], exact_state
            )
            best_fidelity_step = int(np.argmax(ground_fidelities))
            ground_fidelity = float(ground_fidelities[-1])
            coefficient_slots = (
                sparse_coefficients.shape[1] * sparse_coefficients.shape[2]
            )
            # Validate that the saved tensor still matches the reconstructed basis.
            coefficient_localities(sparse_job, sparse_coefficients.shape)

            scan_rows.append(
                {
                    "D": D,
                    "lambda": float(sparse_job["lasso_lambda"]),
                    "qite_steps": qite_steps,
                    "sparse_steps": sparse_steps,
                    "reaches_qite_horizon": reaches_horizon,
                    "comparison_step": comparison_step,
                    "comparison_time": comparison_step * float(sparse_job["dt"]),
                    "fidelity_qite": comparison_fidelity,
                    "mean_fidelity_qite": float(
                        np.mean(trajectory_fidelities_qite)
                    ),
                    "infidelity_qite": 1.0 - comparison_fidelity,
                    "fidelity_ground_state": ground_fidelity,
                    "best_fidelity_ground_state": float(
                        ground_fidelities[best_fidelity_step]
                    ),
                    "best_fidelity_step": best_fidelity_step,
                    "energy": comparison_energy,
                    "energy_abs_error": abs(comparison_energy - exact_energy),
                    "coefficient_slots_per_step": coefficient_slots,
                    **sparse_support,
                    "result_dir": str(sparse_dir),
                }
            )
    return scan_rows, qite_rows


def select_best_lambdas(
    scan_rows: list[dict[str, Any]],
    qite_rows: dict[int, dict[str, Any]],
    fidelity_thresholds: list[float],
) -> list[dict[str, Any]]:
    """Select the largest horizon-valid lambda meeting each fidelity threshold."""
    selected_rows: list[dict[str, Any]] = []
    for D in sorted(qite_rows):
        group = [row for row in scan_rows if int(row["D"]) == D]
        for threshold in fidelity_thresholds:
            valid = [
                row
                for row in group
                if row["reaches_qite_horizon"]
                and np.isfinite(row["fidelity_qite"])
                and row["fidelity_qite"] >= threshold
            ]
            qite = qite_rows[D]
            if valid:
                best = max(valid, key=lambda row: float(row["lambda"]))
                average_fraction = (
                    best["average_active_terms"] / qite["qite_average_active_terms"]
                    if qite["qite_average_active_terms"] > 0
                    else float("nan")
                )
                total_fraction = (
                    best["total_active_terms"] / qite["qite_total_active_terms"]
                    if qite["qite_total_active_terms"] > 0
                    else float("nan")
                )
                cumulative_fraction = (
                    best["final_cumulative_terms"] / qite["qite_final_cumulative_terms"]
                    if qite["qite_final_cumulative_terms"] > 0
                    else float("nan")
                )
                selected_rows.append(
                    {
                        "D": D,
                        "fidelity_threshold": threshold,
                        "selected": True,
                        "best_lambda": best["lambda"],
                        "fidelity_qite": best["fidelity_qite"],
                        "infidelity_qite": max(1.0 - best["fidelity_qite"], 0.0),
                        "mean_fidelity_qite": best["mean_fidelity_qite"],
                        "fidelity_ground_state": best["fidelity_ground_state"],
                        "infidelity_ground_state": max(
                            1.0 - best["fidelity_ground_state"], 0.0
                        ),
                        "best_fidelity_ground_state": best[
                            "best_fidelity_ground_state"
                        ],
                        "best_infidelity_ground_state": max(
                            1.0 - best["best_fidelity_ground_state"], 0.0
                        ),
                        "best_fidelity_step": best["best_fidelity_step"],
                        "energy_abs_error": best["energy_abs_error"],
                        "average_active_terms": best["average_active_terms"],
                        "total_active_terms": best["total_active_terms"],
                        "final_cumulative_terms": best["final_cumulative_terms"],
                        "average_active_fraction_vs_qite": average_fraction,
                        "total_active_fraction_vs_qite": total_fraction,
                        "cumulative_fraction_vs_qite": cumulative_fraction,
                        "qite_ground_fidelity": qite["qite_ground_fidelity"],
                        "qite_ground_infidelity": max(
                            1.0 - qite["qite_ground_fidelity"], 0.0
                        ),
                        "qite_best_ground_fidelity": qite[
                            "qite_best_ground_fidelity"
                        ],
                        "qite_best_ground_infidelity": max(
                            1.0 - qite["qite_best_ground_fidelity"], 0.0
                        ),
                        "qite_self_infidelity": 0.0,
                        "qite_best_fidelity_step": qite[
                            "qite_best_fidelity_step"
                        ],
                        "qite_energy_abs_error": qite["qite_energy_abs_error"],
                        "qite_average_active_terms": qite["qite_average_active_terms"],
                        "qite_total_active_terms": qite["qite_total_active_terms"],
                        "qite_final_cumulative_terms": qite[
                            "qite_final_cumulative_terms"
                        ],
                        "result_dir": best["result_dir"],
                        "qite_result_dir": qite["qite_result_dir"],
                    }
                )
            else:
                selected_rows.append(
                    {
                        "D": D,
                        "fidelity_threshold": threshold,
                        "selected": False,
                        "best_lambda": float("nan"),
                        "fidelity_qite": float("nan"),
                        "infidelity_qite": float("nan"),
                        "mean_fidelity_qite": float("nan"),
                        "fidelity_ground_state": float("nan"),
                        "infidelity_ground_state": float("nan"),
                        "best_fidelity_ground_state": float("nan"),
                        "best_infidelity_ground_state": float("nan"),
                        "best_fidelity_step": float("nan"),
                        "energy_abs_error": float("nan"),
                        "average_active_terms": float("nan"),
                        "total_active_terms": float("nan"),
                        "final_cumulative_terms": float("nan"),
                        "average_active_fraction_vs_qite": float("nan"),
                        "total_active_fraction_vs_qite": float("nan"),
                        "cumulative_fraction_vs_qite": float("nan"),
                        "qite_ground_fidelity": qite["qite_ground_fidelity"],
                        "qite_ground_infidelity": max(
                            1.0 - qite["qite_ground_fidelity"], 0.0
                        ),
                        "qite_best_ground_fidelity": qite[
                            "qite_best_ground_fidelity"
                        ],
                        "qite_best_ground_infidelity": max(
                            1.0 - qite["qite_best_ground_fidelity"], 0.0
                        ),
                        "qite_self_infidelity": 0.0,
                        "qite_best_fidelity_step": qite[
                            "qite_best_fidelity_step"
                        ],
                        "qite_energy_abs_error": qite["qite_energy_abs_error"],
                        "qite_average_active_terms": qite[
                            "qite_average_active_terms"
                        ],
                        "qite_total_active_terms": qite["qite_total_active_terms"],
                        "qite_final_cumulative_terms": qite[
                            "qite_final_cumulative_terms"
                        ],
                        "result_dir": "",
                        "qite_result_dir": qite["qite_result_dir"],
                    }
                )
    return selected_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries with a stable header to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_domain_axis(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    """Use the physical domain sizes present in the loaded output data."""
    domains = sorted({int(row["D"]) for row in rows})
    if not domains:
        return
    axis.set_xticks(domains)
    axis.set_xlim(domains[0] - 0.25, domains[-1] + 0.25)


def plot_selected_metric(
    output_path: Path,
    rows: list[dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
    yscale: str | None = None,
    qite_key: str | None = None,
    unit_fidelity_line: bool = False,
    show_title: bool = True,
) -> None:
    """Plot one selected-lambda metric versus D for every threshold."""
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    def display_value(value: float) -> float:
        """Keep exact zeros visible when a logarithmic y-axis is requested."""
        return max(value, 1e-16) if yscale == "log" else value

    for threshold in sorted({float(row["fidelity_threshold"]) for row in rows}):
        subset = sorted(
            [
                row
                for row in rows
                if row["selected"]
                and float(row["fidelity_threshold"]) == threshold
                and np.isfinite(float(row[key]))
            ],
            key=lambda row: int(row["D"]),
        )
        if not subset:
            continue
        axis.plot(
            [row["D"] for row in subset],
            [display_value(float(row[key])) for row in subset],
            marker="o",
            label=rf"$F_{{QITE}} \geq {threshold:g}$",
        )
        plotted = True
    if qite_key is not None:
        qite_by_D: dict[int, float] = {}
        for row in rows:
            value = float(row[qite_key])
            if np.isfinite(value):
                qite_by_D[int(row["D"])] = display_value(value)
        if qite_by_D:
            domains = sorted(qite_by_D)
            axis.plot(
                domains,
                [qite_by_D[D] for D in domains],
                color="black",
                linestyle="--",
                marker="s",
                label="QITE",
            )
            plotted = True
    if unit_fidelity_line:
        axis.axhline(
            1.0,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="Unit fidelity",
        )
    if not plotted:
        plt.close(figure)
        return
    axis.set_xlabel("Domain size D")
    set_domain_axis(axis, rows)
    axis.set_ylabel(ylabel)
    if show_title:
        axis.set_title(title)
    if yscale:
        axis.set_yscale(yscale)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_lambda_scan(
    output_path: Path,
    scan_rows: list[dict[str, Any]],
    fidelity_thresholds: list[float],
) -> None:
    """Plot all available final fidelities versus lambda, one curve per D."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for D in sorted({int(row["D"]) for row in scan_rows}):
        subset = sorted(
            [row for row in scan_rows if int(row["D"]) == D],
            key=lambda row: float(row["lambda"]),
        )
        axis.plot(
            [row["lambda"] for row in subset],
            [row["fidelity_qite"] for row in subset],
            marker="o",
            label=f"D={D}",
        )
        invalid = [row for row in subset if not row["reaches_qite_horizon"]]
        if invalid:
            axis.scatter(
                [row["lambda"] for row in invalid],
                [row["fidelity_qite"] for row in invalid],
                marker="x",
                color="black",
                zorder=5,
            )
    for threshold in fidelity_thresholds:
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_xlabel(r"LASSO $\lambda$")
    axis.set_ylabel("Fidelity with QITE")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Lambda scan")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_support_comparison(
    output_path: Path,
    rows: list[dict[str, Any]],
    sparse_key: str,
    qite_key: str,
    ylabel: str,
    title: str,
    show_title: bool = True,
) -> None:
    """Plot selected sparse support and the corresponding QITE reference."""
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    selected_rows = [row for row in rows if row["selected"]]
    for threshold in sorted(
        {float(row["fidelity_threshold"]) for row in selected_rows}
    ):
        subset = sorted(
            [
                row
                for row in selected_rows
                if float(row["fidelity_threshold"]) == threshold
                and np.isfinite(float(row[sparse_key]))
            ],
            key=lambda row: int(row["D"]),
        )
        if not subset:
            continue
        axis.plot(
            [row["D"] for row in subset],
            [row[sparse_key] for row in subset],
            marker="o",
            label=rf"Sparse-QITE, $F_{{QITE}} \geq {threshold:g}$",
        )
        plotted = True

    # QITE does not depend on the fidelity threshold, so plot it only once.
    qite_by_D: dict[int, float] = {}
    for row in rows:
        value = float(row[qite_key])
        if np.isfinite(value):
            qite_by_D[int(row["D"])] = value
    if qite_by_D:
        domains = sorted(qite_by_D)
        axis.plot(
            domains,
            [qite_by_D[D] for D in domains],
            color="black",
            linestyle="--",
            marker="s",
            label="QITE",
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return
    axis.set_xlabel("Domain size D")
    set_domain_axis(axis, rows)
    axis.set_ylabel(ylabel)
    if show_title:
        axis.set_title(title)
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_best_final_ground_fidelity(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Compare best-over-time and final ground-state fidelity versus D."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            "best_fidelity_ground_state",
            "qite_best_ground_fidelity",
            "Best fidelity over time",
        ),
        (
            axes[1],
            "fidelity_ground_state",
            "qite_ground_fidelity",
            "Final-update fidelity",
        ),
    )

    selected_rows = [row for row in rows if row["selected"]]
    thresholds = sorted(
        {float(row["fidelity_threshold"]) for row in selected_rows}
    )
    for axis, sparse_key, qite_key, title in panels:
        for threshold in thresholds:
            subset = sorted(
                (
                    row
                    for row in selected_rows
                    if float(row["fidelity_threshold"]) == threshold
                    and np.isfinite(float(row[sparse_key]))
                ),
                key=lambda row: int(row["D"]),
            )
            if subset:
                axis.plot(
                    [int(row["D"]) for row in subset],
                    [float(row[sparse_key]) for row in subset],
                    marker="o",
                    label=rf"Sparse-QITE, $F_{{QITE}}\geq {threshold:g}$",
                )

        qite_by_D: dict[int, float] = {}
        for row in rows:
            value = float(row[qite_key])
            if np.isfinite(value):
                qite_by_D[int(row["D"])] = value
        if qite_by_D:
            domains = sorted(qite_by_D)
            axis.plot(
                domains,
                [qite_by_D[D] for D in domains],
                color="black",
                linestyle="--",
                marker="s",
                label="QITE",
            )

        axis.axhline(1.0, color="black", linewidth=1.0)
        axis.set_xlabel("Domain size D")
        set_domain_axis(axis, rows)
        axis.set_title(title)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Ground-state fidelity")
    axes[0].legend()
    figure.suptitle("Best versus final ground-state fidelity")
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def main() -> None:
    # -----------------------------------------------------------------------
    # EDIT THIS BLOCK TO CHOOSE THE SCALABILITY SCAN AND ACCEPTANCE LEVELS.
    # -----------------------------------------------------------------------
    parameters = {
        "n_qubits": [8],
        "D": [2, 3, 4, 5, 6,7],
        "N": [100],
        "dt": [0.05],
        "method": ["default", "vanilla_lasso"],
        #"model": ["tfim"],
        "model" : ["heisenberg_pbc"],
        "J": [1.0],
        "h": [1.0],
        "T": [2],
        "cluster_lambda": [0.5],
        "lasso_lambda": [
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            1.5e-3,
            2e-3,
            3e-3,
            5e-3,
            7e-3,
            1e-2,
            1.25e-2,
            1.5e-2,
            2e-2,
            2.5e-2,
            3e-2,
            5e-2,
            1e-1,
            1.5e-1,
            2e-1,
            2.5e-1,
            3e-1,
            4e-1,
            5e-1,
            6e-1,
            7e-1,
            8e-1,
            1.0,
        ],
        "lasso_max_iter": 500,
        "lasso_tol": 1e-8,
        "initial_basis_state": 0,
        "verbose": False,
    }
    fidelity_thresholds = [0.95, 0.99, 0.999, 0.9999, 0.99999]
    coefficient_threshold = 1e-14
    analysis_name = scalability_analysis_name(parameters)
    # -----------------------------------------------------------------------

    results_root = (PROJECT_DIR / "outputs" / "qite_runs").resolve()
    output_dir = (PROJECT_DIR / "plots" / "scalability" / analysis_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_rows, qite_rows = load_scan_rows(
        parameters, results_root, coefficient_threshold
    )
    selected_rows = select_best_lambdas(
        scan_rows, qite_rows, fidelity_thresholds
    )
    write_csv(output_dir / "lambda_scan.csv", scan_rows)
    write_csv(output_dir / "selected_lambdas.csv", selected_rows)

    if scan_rows:
        plot_lambda_scan(
            output_dir / "fidelity_vs_lambda.pdf", scan_rows, fidelity_thresholds
        )
    metrics = [
        ("best_lambda", "Selected lambda", "Largest acceptable lambda", "log"),
        ("fidelity_qite", "Fidelity with QITE", "Selected-run fidelity", None),
        (
            "mean_fidelity_qite",
            "Mean fidelity with QITE",
            "Mean trajectory fidelity with QITE",
            None,
        ),
        ("energy_abs_error", "Absolute energy error", "Selected-run energy error", "log"),
        (
            "final_cumulative_terms",
            "Cumulative basis terms",
            "Selected-run cumulative support",
            "log",
        ),
        (
            "total_active_fraction_vs_qite",
            "Total support / QITE",
            "Selected-run total support fraction",
            "log",
        ),
        (
            "average_active_fraction_vs_qite",
            "Average support / QITE",
            "Selected-run average support fraction",
            "log",
        ),
        (
            "cumulative_fraction_vs_qite",
            "Cumulative support / QITE",
            "Selected-run cumulative support fraction",
            "log",
        ),
    ]
    for key, ylabel, title, scale in metrics:
        plot_selected_metric(
            output_dir / f"{key}_vs_D.pdf",
            selected_rows,
            key,
            ylabel,
            title,
            yscale=scale,
        )

    plot_selected_metric(
        output_dir / "fidelity_ground_state_vs_D.pdf",
        selected_rows,
        "fidelity_ground_state",
        "Ground-state fidelity",
        "Selected-run ground-state fidelity",
        qite_key="qite_ground_fidelity",
        unit_fidelity_line=True,
    )
    plot_selected_metric(
        output_dir / "best_fidelity_ground_state_vs_D.pdf",
        selected_rows,
        "best_fidelity_ground_state",
        "Best observed ground-state fidelity",
        "Best observed ground-state fidelity",
        qite_key="qite_best_ground_fidelity",
        unit_fidelity_line=True,
    )
    plot_selected_metric(
        output_dir / "infidelity_qite_vs_D.pdf",
        selected_rows,
        "infidelity_qite",
        "Final infidelity with QITE, 1 - F",
        "Selected-run final infidelity with QITE",
        yscale="log",
        qite_key="qite_self_infidelity",
    )
    plot_selected_metric(
        output_dir / "infidelity_ground_state_vs_D.pdf",
        selected_rows,
        "infidelity_ground_state",
        "Final ground-state infidelity, 1 - F",
        "Selected-run final ground-state infidelity",
        yscale="log",
        qite_key="qite_ground_infidelity",
        show_title=False,
    )
    plot_selected_metric(
        output_dir / "best_infidelity_ground_state_vs_D.pdf",
        selected_rows,
        "best_infidelity_ground_state",
        "Best observed ground-state infidelity, 1 - F",
        "Best observed ground-state infidelity",
        yscale="log",
        qite_key="qite_best_ground_infidelity",
    )
    plot_best_final_ground_fidelity(
        output_dir / "fidelity_ground_state_best_vs_final_vs_D.pdf",
        selected_rows,
    )

    plot_support_comparison(
        output_dir / "average_active_terms_vs_D.pdf",
        selected_rows,
        "average_active_terms",
        "qite_average_active_terms",
        "Average active terms per step",
        "Average active terms: selected Sparse-QITE vs QITE",
    )
    plot_support_comparison(
        output_dir / "total_active_terms_vs_D.pdf",
        selected_rows,
        "total_active_terms",
        "qite_total_active_terms",
        "Total active terms over evolution",
        "Total active terms: selected Sparse-QITE vs QITE",
        show_title=False,
    )

    print(f"Loaded {len(scan_rows)} sparse-QITE result(s).")
    print(f"Wrote scalability analysis to {output_dir}")


if __name__ == "__main__":
    main()
