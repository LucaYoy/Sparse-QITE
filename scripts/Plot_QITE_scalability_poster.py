#!/usr/bin/env python3
"""Create n=8 poster scalability plots for final fidelity above 0.999."""

from __future__ import annotations

import csv
from pathlib import Path

from Plot_QITE_scalability import (
    PROJECT_DIR,
    plot_best_final_ground_fidelity,
    plot_selected_metric,
    plot_support_comparison,
    write_csv,
)


FIDELITY_CUTOFF = 0.999
CASES = (
    "disordered_tfim_n8_D_scalability",
    "ordered_tfim_n8_D_scalability",
    "heisenberg_pbc_n8_D_scalability",
)


def load_filtered_rows(path: Path) -> list[dict[str, object]]:
    """Load selected 0.999-threshold rows whose measured fidelity is > 0.999."""
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for source in source_rows:
        selected = source["selected"].strip().lower() == "true"
        threshold = float(source["fidelity_threshold"])
        fidelity = float(source["fidelity_qite"] or "nan")
        if not selected or threshold != FIDELITY_CUTOFF or fidelity <= FIDELITY_CUTOFF:
            continue

        row: dict[str, object] = dict(source)
        row["selected"] = True
        row["D"] = int(source["D"])
        row["fidelity_threshold"] = threshold
        numeric_fields = (
            "best_lambda",
            "fidelity_qite",
            "mean_fidelity_qite",
            "fidelity_ground_state",
            "best_fidelity_ground_state",
            "best_fidelity_step",
            "energy_abs_error",
            "average_active_terms",
            "total_active_terms",
            "final_cumulative_terms",
            "average_active_fraction_vs_qite",
            "total_active_fraction_vs_qite",
            "cumulative_fraction_vs_qite",
            "qite_ground_fidelity",
            "qite_best_ground_fidelity",
            "qite_best_fidelity_step",
            "qite_energy_abs_error",
            "qite_average_active_terms",
            "qite_total_active_terms",
            "qite_final_cumulative_terms",
        )
        for field in numeric_fields:
            row[field] = float(source[field])
        row["infidelity_qite"] = max(1.0 - fidelity, 0.0)
        row["infidelity_ground_state"] = max(
            1.0 - float(source["fidelity_ground_state"]), 0.0
        )
        row["best_infidelity_ground_state"] = max(
            1.0 - float(source["best_fidelity_ground_state"]), 0.0
        )
        row["qite_ground_infidelity"] = max(
            1.0 - float(source["qite_ground_fidelity"]), 0.0
        )
        row["qite_best_ground_infidelity"] = max(
            1.0 - float(source["qite_best_ground_fidelity"]), 0.0
        )
        row["qite_self_infidelity"] = 0.0
        rows.append(row)
    return rows


def plot_case(case: str, rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write the filtered CSV and the standard selected-run scalability figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_lambdas.csv", rows)

    metrics = (
        ("best_lambda", "Selected lambda", "Largest acceptable lambda", "log"),
        ("fidelity_qite", "Fidelity with QITE", "Selected-run fidelity", None),
        ("mean_fidelity_qite", "Mean fidelity with QITE", "Mean trajectory fidelity with QITE", None),
        ("energy_abs_error", "Absolute energy error", "Selected-run energy error", "log"),
        ("final_cumulative_terms", "Cumulative basis terms", "Selected-run cumulative support", "log"),
        ("total_active_fraction_vs_qite", "Total support / QITE", "Selected-run total support fraction", "log"),
        ("average_active_fraction_vs_qite", "Average support / QITE", "Selected-run average support fraction", "log"),
        ("cumulative_fraction_vs_qite", "Cumulative support / QITE", "Selected-run cumulative support fraction", "log"),
    )
    for key, ylabel, title, scale in metrics:
        plot_selected_metric(output_dir / f"{key}_vs_D.pdf", rows, key, ylabel, title, yscale=scale)

    plot_selected_metric(output_dir / "fidelity_ground_state_vs_D.pdf", rows, "fidelity_ground_state", "Ground-state fidelity", "Selected-run ground-state fidelity", qite_key="qite_ground_fidelity", unit_fidelity_line=True)
    plot_selected_metric(output_dir / "best_fidelity_ground_state_vs_D.pdf", rows, "best_fidelity_ground_state", "Best observed ground-state fidelity", "Best observed ground-state fidelity", qite_key="qite_best_ground_fidelity", unit_fidelity_line=True)
    plot_selected_metric(output_dir / "infidelity_qite_vs_D.pdf", rows, "infidelity_qite", "Final infidelity with QITE, 1 - F", "Selected-run final infidelity with QITE", yscale="log", qite_key="qite_self_infidelity")
    plot_selected_metric(output_dir / "infidelity_ground_state_vs_D.pdf", rows, "infidelity_ground_state", "Final ground-state infidelity, 1 - F", "Selected-run final ground-state infidelity", yscale="log", qite_key="qite_ground_infidelity")
    plot_selected_metric(output_dir / "best_infidelity_ground_state_vs_D.pdf", rows, "best_infidelity_ground_state", "Best observed ground-state infidelity, 1 - F", "Best observed ground-state infidelity", yscale="log", qite_key="qite_best_ground_infidelity")
    plot_best_final_ground_fidelity(output_dir / "fidelity_ground_state_best_vs_final_vs_D.pdf", rows)
    plot_support_comparison(output_dir / "average_active_terms_vs_D.pdf", rows, "average_active_terms", "qite_average_active_terms", "Average active terms per step", "Average active terms: selected Sparse-QITE vs QITE")
    plot_support_comparison(output_dir / "total_active_terms_vs_D.pdf", rows, "total_active_terms", "qite_total_active_terms", "Total active terms over evolution", "Total active terms: selected Sparse-QITE vs QITE")
    print(f"{case}: plotted {len(rows)} domain sizes in {output_dir}")


def main() -> None:
    scalability_dir = PROJECT_DIR / "plots" / "scalability"
    poster_dir = PROJECT_DIR / "plots" / "poster"
    for case in CASES:
        rows = load_filtered_rows(scalability_dir / case / "selected_lambdas.csv")
        if not rows:
            print(f"Skipping {case}: no selected rows have fidelity > {FIDELITY_CUTOFF}")
            continue
        plot_case(case, rows, poster_dir / case)


if __name__ == "__main__":
    main()
