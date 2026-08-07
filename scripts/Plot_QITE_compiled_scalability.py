#!/usr/bin/env python3
"""Plot compiled-circuit scalability for the n=8 QITE experiments.

This script combines:

* ``selected_lambdas.csv`` from ``Plot_QITE_scalability.py``;
* ``summary.csv`` from ``Run_Transpilation_cluster.py``.

Edit the cases and fidelity thresholds in ``main()``, then run:

    python scripts/Plot_QITE_compiled_scalability.py

Each sparse-QITE curve uses the lambda selected at the stated fidelity
threshold. Standard QITE is plotted as a dashed black reference.
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TRANSPILATION_FOLDER = (
    "generic_u3-cx_opt1_pauli_evolution_steps100_no_unitary_error"
)
SYNTHESIS_FORMULA = "suzuki2"
ADAPTIVE_POLICY_GLOB = "adaptive_angle0p1_r*-r*_cap128"
COMPILED_ANALYSIS_FOLDER = "adaptive_suzuki2_angle0p1"
INFIDELITY_FLOOR = 1e-16


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth_value(value: str) -> bool:
    """Interpret a CSV boolean."""
    return value.strip().lower() in {"true", "1", "yes"}


def vertical_summary(path: Path) -> dict[str, str]:
    """Read the human-readable metric,value transpilation summary."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing transpilation summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][:2] != ["metric", "value"]:
        raise ValueError(f"Unexpected transpilation summary format: {path}")
    return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}


def summary_path(result_dir: str) -> Path:
    """Return the unique adaptive-Suzuki summary for one QITE result."""
    synthesis_root = (
        Path(result_dir)
        / "transpilation"
        / SYNTHESIS_FORMULA
    )
    matches = sorted(
        synthesis_root.glob(
            f"{ADAPTIVE_POLICY_GLOB}/{TRANSPILATION_FOLDER}/summary.csv"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one adaptive-Suzuki summary below "
            f"{synthesis_root}, found {len(matches)}."
        )
    return matches[0]


def metric(summary: dict[str, str], key: str) -> float:
    """Read one required finite numeric metric."""
    try:
        value = float(summary[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid transpilation metric {key!r}.") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite transpilation metric {key!r}.")
    return value


def compiled_metrics(result_dir: str) -> dict[str, float]:
    """Extract the compiled metrics used by the scalability analysis."""
    summary = vertical_summary(summary_path(result_dir))
    fidelity_compilation = metric(summary, "fidelity_uncompiled_vs_compiled")
    fidelity_ground_before = metric(
        summary, "fidelity_uncompiled_vs_ground_state"
    )
    fidelity_ground_after = metric(summary, "fidelity_compiled_vs_ground_state")
    return {
        "compiled_total_gates": metric(summary, "compiled_size"),
        "compiled_cx_gates": metric(summary, "compiled_nonlocal_gates"),
        "compiled_depth": metric(summary, "compiled_depth"),
        "pauli_rotations": metric(
            summary, "pauli_rotations_before_transpilation"
        ),
        "compilation_fidelity": fidelity_compilation,
        "compilation_infidelity": max(1.0 - fidelity_compilation, 0.0),
        "ground_fidelity_before_compilation": fidelity_ground_before,
        "ground_fidelity_after_compilation": fidelity_ground_after,
        "ground_infidelity_before_compilation": max(
            1.0 - fidelity_ground_before, 0.0
        ),
        "ground_infidelity_after_compilation": max(
            1.0 - fidelity_ground_after, 0.0
        ),
        "transpilation_elapsed_seconds": metric(
            summary, "transpilation_elapsed_seconds"
        ),
        "minimum_repetitions": metric(summary, "minimum_repetitions_used"),
        "mean_repetitions": metric(summary, "mean_repetitions_used"),
        "maximum_repetitions": metric(summary, "maximum_repetitions_used"),
        "generators_above_one_repetition": metric(
            summary, "generators_above_one_repetition"
        ),
        "maximum_generator_coefficient": metric(
            summary, "maximum_generator_coefficient"
        ),
        "maximum_realized_coefficient_angle": metric(
            summary, "maximum_realized_coefficient_angle"
        ),
    }


def load_compiled_rows(
    selected_path: Path,
    fidelity_thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Combine selected sparse-QITE rows with sparse and QITE summaries."""
    selected = [
        row
        for row in read_csv(selected_path)
        if truth_value(row["selected"])
        and any(
            np.isclose(float(row["fidelity_threshold"]), threshold)
            for threshold in fidelity_thresholds
        )
    ]
    if not selected:
        raise ValueError(f"No selected lambda rows found in {selected_path}.")

    sparse_rows: list[dict[str, Any]] = []
    qite_rows: dict[int, dict[str, Any]] = {}
    summary_cache: dict[str, dict[str, float]] = {}

    def cached_metrics(result_dir: str) -> dict[str, float]:
        if result_dir not in summary_cache:
            summary_cache[result_dir] = compiled_metrics(result_dir)
        return summary_cache[result_dir]

    for row in selected:
        D = int(row["D"])
        threshold = float(row["fidelity_threshold"])
        sparse_dir = row["result_dir"]
        qite_dir = row["qite_result_dir"]
        sparse_metrics = cached_metrics(sparse_dir)
        qite_metrics = cached_metrics(qite_dir)

        qite_rows[D] = {"D": D, **qite_metrics, "result_dir": qite_dir}
        sparse_rows.append(
            {
                "D": D,
                "fidelity_threshold": threshold,
                "lambda": float(row["best_lambda"]),
                **sparse_metrics,
                "total_gate_fraction_vs_qite": (
                    sparse_metrics["compiled_total_gates"]
                    / qite_metrics["compiled_total_gates"]
                ),
                "cx_gate_fraction_vs_qite": (
                    sparse_metrics["compiled_cx_gates"]
                    / qite_metrics["compiled_cx_gates"]
                ),
                "depth_fraction_vs_qite": (
                    sparse_metrics["compiled_depth"]
                    / qite_metrics["compiled_depth"]
                ),
                "rotation_fraction_vs_qite": (
                    sparse_metrics["pauli_rotations"]
                    / qite_metrics["pauli_rotations"]
                ),
                "result_dir": sparse_dir,
                "qite_result_dir": qite_dir,
            }
        )
    return sparse_rows, qite_rows


def threshold_label(threshold: float) -> str:
    """Return the common sparse-QITE legend label."""
    return rf"Sparse-QITE, $F_{{QITE}}\geq {threshold:g}$"


def plot_metric(
    output_path: Path,
    sparse_rows: list[dict[str, Any]],
    qite_rows: dict[int, dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
    yscale: str | None = None,
) -> None:
    """Plot one compiled metric against D."""
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    for threshold in sorted(
        {float(row["fidelity_threshold"]) for row in sparse_rows}
    ):
        subset = sorted(
            (
                row
                for row in sparse_rows
                if np.isclose(row["fidelity_threshold"], threshold)
            ),
            key=lambda row: row["D"],
        )
        axis.plot(
            [row["D"] for row in subset],
            [max(float(row[key]), INFIDELITY_FLOOR) for row in subset],
            marker="o",
            label=threshold_label(threshold),
        )

    domains = sorted(qite_rows)
    axis.plot(
        domains,
        [max(float(qite_rows[D][key]), INFIDELITY_FLOOR) for D in domains],
        color="black",
        linestyle="--",
        marker="s",
        label="QITE",
    )
    axis.set_xlabel("Domain size D")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if yscale:
        axis.set_yscale(yscale)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_compression_ratios(
    output_path: Path,
    sparse_rows: list[dict[str, Any]],
) -> None:
    """Plot total-gate, CX-gate, and depth fractions relative to QITE."""
    panels = (
        ("total_gate_fraction_vs_qite", "Total gates"),
        ("cx_gate_fraction_vs_qite", "CX gates"),
        ("depth_fraction_vs_qite", "Circuit depth"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    thresholds = sorted(
        {float(row["fidelity_threshold"]) for row in sparse_rows}
    )
    for axis, (key, title) in zip(axes, panels, strict=True):
        for threshold in thresholds:
            subset = sorted(
                (
                    row
                    for row in sparse_rows
                    if np.isclose(row["fidelity_threshold"], threshold)
                ),
                key=lambda row: row["D"],
            )
            axis.plot(
                [row["D"] for row in subset],
                [row[key] for row in subset],
                marker="o",
                label=threshold_label(threshold),
            )
        axis.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="QITE",
        )
        axis.set_yscale("log")
        axis.set_xlabel("Domain size D")
        axis.set_title(title)
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Sparse-QITE / QITE")
    axes[0].legend(fontsize=8)
    figure.suptitle("Compiled-circuit compression relative to QITE")
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def write_compiled_csv(
    path: Path,
    sparse_rows: list[dict[str, Any]],
    qite_rows: dict[int, dict[str, Any]],
) -> None:
    """Write one tidy table containing sparse-QITE and QITE metrics."""
    rows: list[dict[str, Any]] = []
    for D in sorted(qite_rows):
        rows.append(
            {
                "method": "QITE",
                "D": D,
                "fidelity_threshold": "",
                "lambda": "",
                **{
                    key: value
                    for key, value in qite_rows[D].items()
                    if key not in {"D"}
                },
            }
        )
    for row in sorted(
        sparse_rows, key=lambda item: (item["fidelity_threshold"], item["D"])
    ):
        rows.append({"method": "Sparse-QITE", **row})

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    # ------------------------------------------------------------------
    # EDIT THIS BLOCK TO CHOOSE THE n=8 COMPILED SCALABILITY ANALYSES.
    # ------------------------------------------------------------------
    cases = [
        {
            "folder": "disordered_tfim_n8_D_scalability",
            "label": "Disordered TFIM, n=8",
        },
        {
            "folder": "ordered_tfim_n8_D_scalability",
            "label": "Ordered TFIM, n=8",
        },
        {
            "folder": "heisenberg_pbc_n8_D_scalability",
            "label": "Heisenberg PBC, n=8",
        },
    ]
    fidelity_thresholds = [0.95, 0.99, 0.999, 0.9999, 0.99999]
    # ------------------------------------------------------------------

    scalability_root = PROJECT_DIR / "plots" / "scalability"
    for case in cases:
        analysis_dir = scalability_root / case["folder"]
        sparse_rows, qite_rows = load_compiled_rows(
            analysis_dir / "selected_lambdas.csv",
            fidelity_thresholds,
        )
        output_dir = analysis_dir / "compiled" / COMPILED_ANALYSIS_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)

        write_compiled_csv(
            output_dir / "compiled_scalability.csv", sparse_rows, qite_rows
        )
        plot_metric(
            output_dir / "compiled_total_gates_vs_D.pdf",
            sparse_rows,
            qite_rows,
            "compiled_total_gates",
            "Total compiled gates",
            f"{case['label']}: compiled gate count",
            "log",
        )
        plot_metric(
            output_dir / "compiled_cx_gates_vs_D.pdf",
            sparse_rows,
            qite_rows,
            "compiled_cx_gates",
            "Compiled CX gates",
            f"{case['label']}: compiled two-qubit gates",
            "log",
        )
        plot_metric(
            output_dir / "compiled_depth_vs_D.pdf",
            sparse_rows,
            qite_rows,
            "compiled_depth",
            "Compiled circuit depth",
            f"{case['label']}: compiled circuit depth",
            "log",
        )
        plot_compression_ratios(
            output_dir / "compiled_compression_ratios_vs_D.pdf",
            sparse_rows,
        )
        plot_metric(
            output_dir / "compilation_infidelity_vs_D.pdf",
            sparse_rows,
            qite_rows,
            "compilation_infidelity",
            r"Compilation infidelity, $1-F$",
            f"{case['label']}: uncompiled versus compiled state",
            "log",
        )
        plot_metric(
            output_dir / "compiled_ground_state_infidelity_vs_D.pdf",
            sparse_rows,
            qite_rows,
            "ground_infidelity_after_compilation",
            r"Compiled ground-state infidelity, $1-F$",
            f"{case['label']}: compiled ground-state accuracy",
            "log",
        )


if __name__ == "__main__":
    main()
