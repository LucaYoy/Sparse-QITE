#!/usr/bin/env python3
"""Plot sparse-QITE fidelity/support Pareto fronts for selected experiments.

The input is the ``lambda_scan.csv`` produced by ``Plot_QITE_scalability.py``.
Edit the cases and plotting choices in ``main()``, then run:

    python scripts/plotting/Plot_QITE_pareto.py

For each case, two separate logarithmic-infidelity PDF figures are written:

1. final sparse-QITE infidelity with QITE versus the support cost;
2. final sparse-QITE infidelity with the ground state versus the support cost.

The default cost, ``total_active_terms``, counts every active coefficient at
every update. It is therefore also the number of Pauli rotations used to
construct the complete uncompiled evolution circuit.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/qite-matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


COST_LABELS = {
    "total_active_terms": "Active Pauli rotations over the trajectory",
    "average_active_terms": "Average active terms per update",
    "final_cumulative_terms": "Distinct cumulative basis terms",
}

INFIDELITY_FLOOR = 1e-16
LOG_AXIS_LOWER_PADDING_FACTOR = 2.0
LOG_AXIS_UPPER_PADDING_FACTOR = 5.0


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV file into a list of dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Required scalability file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth_value(value: str) -> bool:
    """Interpret the boolean text written by the scalability analysis."""
    return value.strip().lower() in {"true", "1", "yes"}


def finite_float(row: dict[str, str], key: str) -> float:
    """Read a required finite floating-point value from a CSV row."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {key!r} value in row: {row}") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} value in row: {row}")
    return value


def load_case_rows(
    scan_path: Path,
    D: int,
    cost_key: str,
) -> list[dict[str, Any]]:
    """Load horizon-valid lambda results for one domain size."""
    rows: list[dict[str, Any]] = []
    for raw in read_csv(scan_path):
        if int(raw["D"]) != D or not truth_value(raw["reaches_qite_horizon"]):
            continue
        rows.append(
            {
                "D": D,
                "lambda": finite_float(raw, "lambda"),
                "number_of_terms": finite_float(raw, cost_key),
                "fidelity_qite": finite_float(raw, "fidelity_qite"),
                "fidelity_ground_state": finite_float(
                    raw, "fidelity_ground_state"
                ),
                "result_dir": raw["result_dir"],
            }
        )
    if not rows:
        raise ValueError(
            f"No horizon-valid lambda results were found for D={D} in {scan_path}."
        )
    return sorted(rows, key=lambda row: (row["number_of_terms"], row["lambda"]))


def load_qite_reference(
    selected_path: Path,
    D: int,
    cost_key: str,
) -> dict[str, float]:
    """Load the standard-QITE cost and ground-state fidelity for one D."""
    qite_cost_key = f"qite_{cost_key}"
    candidates = [row for row in read_csv(selected_path) if int(row["D"]) == D]
    if not candidates:
        raise ValueError(f"No QITE reference for D={D} in {selected_path}.")
    row = candidates[0]
    return {
        "number_of_terms": finite_float(row, qite_cost_key),
        "fidelity_qite": 1.0,
        "fidelity_ground_state": finite_float(row, "qite_ground_fidelity"),
    }


def pareto_mask(
    rows: list[dict[str, Any]],
    fidelity_key: str,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Mark points not dominated when minimizing cost and maximizing fidelity.

    A point is dominated if another tested lambda uses no more terms and
    reaches at least the same fidelity, with one of those comparisons strict.
    """
    mask = np.zeros(len(rows), dtype=bool)
    for index, candidate in enumerate(rows):
        cost = float(candidate["number_of_terms"])
        fidelity = float(candidate[fidelity_key])
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_cost = float(other["number_of_terms"])
            other_fidelity = float(other[fidelity_key])
            no_worse = (
                other_cost <= cost + tolerance
                and other_fidelity >= fidelity - tolerance
            )
            strictly_better = (
                other_cost < cost - tolerance
                or other_fidelity > fidelity + tolerance
            )
            if no_worse and strictly_better:
                dominated = True
                break
        mask[index] = not dominated
    return mask


def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write all tested points and their two Pareto classifications."""
    output_rows = []
    qite_mask = pareto_mask(rows, "fidelity_qite")
    ground_mask = pareto_mask(rows, "fidelity_ground_state")
    for row, qite_pareto, ground_pareto in zip(
        rows, qite_mask, ground_mask, strict=True
    ):
        output_rows.append(
            {
                **row,
                "infidelity_qite": max(
                    1.0 - float(row["fidelity_qite"]), 0.0
                ),
                "infidelity_ground_state": max(
                    1.0 - float(row["fidelity_ground_state"]), 0.0
                ),
                "pareto_fidelity_qite": bool(qite_pareto),
                "pareto_fidelity_ground_state": bool(ground_pareto),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def plot_pareto(
    output_path: Path,
    rows: list[dict[str, Any]],
    qite_reference: dict[str, float],
    fidelity_key: str,
    ylabel: str,
    title: str,
    cost_key: str,
    color_limits: tuple[float, float],
    show_colorbar: bool = True,
) -> None:
    """Plot logarithmic infidelity, the Pareto front, and QITE reference."""
    costs = np.asarray([row["number_of_terms"] for row in rows], dtype=float)
    fidelities = np.asarray([row[fidelity_key] for row in rows], dtype=float)
    infidelities = np.maximum(1.0 - fidelities, INFIDELITY_FLOOR)
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    mask = pareto_mask(rows, fidelity_key)

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = np.log10(lambdas)
    points = axis.scatter(
        costs,
        infidelities,
        c=colors,
        cmap="viridis",
        vmin=color_limits[0],
        vmax=color_limits[1],
        s=50,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )

    front = sorted(
        (row for row, selected in zip(rows, mask, strict=True) if selected),
        key=lambda row: row["number_of_terms"],
    )
    axis.plot(
        [row["number_of_terms"] for row in front],
        [max(1.0 - row[fidelity_key], INFIDELITY_FLOOR) for row in front],
        color="tab:red",
        linewidth=1.6,
        zorder=2,
    )
    axis.scatter(
        [row["number_of_terms"] for row in front],
        [max(1.0 - row[fidelity_key], INFIDELITY_FLOOR) for row in front],
        s=95,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.6,
        label="Pareto front",
        zorder=4,
    )

    reference_infidelity: float | None = None
    if fidelity_key == "fidelity_qite":
        axis.axvline(
            qite_reference["number_of_terms"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=(
                "QITE term count "
                f"({qite_reference['number_of_terms']:,.0f})"
            ),
            zorder=2,
        )
    else:
        reference_infidelity = max(
            1.0 - qite_reference[fidelity_key], INFIDELITY_FLOOR
        )
        axis.scatter(
            [qite_reference["number_of_terms"]],
            [reference_infidelity],
            marker="D",
            s=65,
            color="black",
            label="QITE",
            zorder=5,
        )

    annotation_stride = max(1, math.ceil(len(front) / 8))
    annotated = {id(row) for row in front[::annotation_stride]}
    annotated.add(id(front[-1]))
    for row, selected in zip(rows, mask, strict=True):
        if selected and id(row) in annotated:
            annotation_offset = (5, 5)
            annotation_alignment = "left"
            if (
                fidelity_key == "fidelity_ground_state"
                and "Disordered TFIM" in title
                and np.isclose(float(row["lambda"]), 0.0015)
            ):
                annotation_offset = (-8, 5)
                annotation_alignment = "right"
            elif (
                "Heisenberg" in title
                and np.isclose(float(row["lambda"]), 1.0)
            ):
                annotation_offset = (9, 5)
            axis.annotate(
                rf"$\lambda={row['lambda']:g}$",
                (
                    row["number_of_terms"],
                    max(1.0 - row[fidelity_key], INFIDELITY_FLOOR),
                ),
                xytext=annotation_offset,
                textcoords="offset points",
                fontsize=7.5,
                horizontalalignment=annotation_alignment,
            )

    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    if show_colorbar:
        colorbar.set_label(r"$\log_{10}(\lambda)$")
    else:
        # Keep the color-bar column reserved so every panel has identical
        # data-axis dimensions when included at the same width.
        colorbar.ax.set_visible(False)
    # ``symlog`` retains a legitimate zero-term solution while behaving
    # logarithmically for the rest of the several-orders-of-magnitude range.
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_yscale("log")
    axis.set_xlabel(COST_LABELS[cost_key])
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, which="both")
    axis.legend()

    displayed = list(infidelities)
    if reference_infidelity is not None:
        displayed.append(reference_infidelity)
    positive_min = min(displayed)
    positive_max = max(displayed)
    lower_decade = 10.0 ** np.floor(np.log10(positive_min))
    upper_decade = 10.0 ** np.ceil(np.log10(positive_max))
    if lower_decade == upper_decade:
        lower_decade /= 10.0
        upper_decade *= 10.0
    axis.set_ylim(
        lower_decade / LOG_AXIS_LOWER_PADDING_FACTOR,
        upper_decade * LOG_AXIS_UPPER_PADDING_FACTOR,
    )

    # Explicit margins leave room for the color bar and logarithmic tick labels.
    figure.subplots_adjust(left=0.12, right=0.88, bottom=0.13, top=0.90)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)
    print(f"Wrote {output_path}")


def main() -> None:
    # ------------------------------------------------------------------
    # EDIT THIS BLOCK TO CHOOSE THE CASES AND THE DEFINITION OF "TERMS".
    # ------------------------------------------------------------------
    cases = [
        {
            "name": "ordered_tfim_n8_D6",
            "label": "Ordered TFIM, n=8, D=6",
            "analysis_folder": "ordered_tfim_n8_D_scalability",
            "D": 6,
        },
        {
            "name": "disordered_tfim_n8_D4",
            "label": "Disordered TFIM, n=8, D=4",
            "analysis_folder": "disordered_tfim_n8_D_scalability",
            "D": 4,
        },
        {
            "name": "heisenberg_pbc_n8_D6",
            "label": "Heisenberg PBC, n=8, D=6",
            "analysis_folder": "heisenberg_pbc_n8_D_scalability",
            "D": 6,
        },
    ]

    # This equals the number of Pauli rotations given to the full circuit.
    cost_key = "total_active_terms"
    output_root = PROJECT_DIR / "plots" / "pareto"
    # ------------------------------------------------------------------

    if cost_key not in COST_LABELS:
        choices = ", ".join(COST_LABELS)
        raise ValueError(f"Unknown cost_key {cost_key!r}. Choose one of: {choices}.")

    scalability_root = PROJECT_DIR / "plots" / "scalability"
    prepared_cases = []
    for case in cases:
        analysis_dir = scalability_root / case["analysis_folder"]
        rows = load_case_rows(
            analysis_dir / "lambda_scan.csv",
            int(case["D"]),
            cost_key,
        )
        qite_reference = load_qite_reference(
            analysis_dir / "selected_lambdas.csv",
            int(case["D"]),
            cost_key,
        )
        prepared_cases.append((case, rows, qite_reference))

    all_log_lambdas = [
        np.log10(float(row["lambda"]))
        for _case, rows, _reference in prepared_cases
        for row in rows
    ]
    color_limits = (min(all_log_lambdas), max(all_log_lambdas))

    for case, rows, qite_reference in prepared_cases:

        output_dir = output_root / case["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_points_csv(output_dir / "pareto_points.csv", rows)

        plot_pareto(
            output_dir / "infidelity_vs_qite_pareto.pdf",
            rows,
            qite_reference,
            "fidelity_qite",
            r"Infidelity with QITE, $1-F$",
            f"{case['label']}: infidelity with QITE",
            cost_key,
            color_limits,
            show_colorbar=case["name"] != "disordered_tfim_n8_D4",
        )
        plot_pareto(
            output_dir / "infidelity_vs_ground_state_pareto.pdf",
            rows,
            qite_reference,
            "fidelity_ground_state",
            r"Ground-state infidelity, $1-F$",
            f"{case['label']}: ground-state infidelity",
            cost_key,
            color_limits,
            show_colorbar=case["name"] != "disordered_tfim_n8_D4",
        )


if __name__ == "__main__":
    main()
