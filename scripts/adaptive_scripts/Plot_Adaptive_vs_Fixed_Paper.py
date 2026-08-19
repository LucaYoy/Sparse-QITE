#!/usr/bin/env python3
"""Create both adaptive-versus-fixed paper figures from NPZ run data only.

The script is intentionally independent of the repository's plotting,
Hamiltonian, and evolution modules.  It requires only NumPy, SciPy,
Matplotlib, and the ``trajectory.npz``/``states.npz`` files for each run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/qite-matplotlib-{os.getuid()}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_DATA_ROOT = PROJECT_DIR / "final_outputs" / "qite_runs"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "final_plots" / "qite_adaptive"
SUPPORT_TOL = 1e-14


@dataclass(frozen=True)
class Case:
    phase: str
    folder: str
    n_qubits: int
    D: int
    N: int
    dt: float
    J: float
    h: float
    fixed_lambda: float
    fixed_dir: str
    adaptive_epsilon: float
    adaptive_dir: str


@dataclass(frozen=True)
class Run:
    kind: str
    energy: np.ndarray
    states: sp.spmatrix
    coefficients: np.ndarray


CASES = (
    Case(
        phase="disordered",
        folder="tfim_n8_D4_N100_dt0p05_T2_J0p5_h1",
        n_qubits=8,
        D=4,
        N=100,
        dt=0.05,
        J=0.5,
        h=1.0,
        fixed_lambda=0.25,
        fixed_dir="sparse_qite_lambda_0p25",
        adaptive_epsilon=0.113,
        adaptive_dir="adaptive_lasso_epsilon_0p113",
    ),
    Case(
        phase="ordered",
        folder="tfim_n8_D5_N100_dt0p05_T2_J1_h0p5",
        n_qubits=8,
        D=5,
        N=100,
        dt=0.05,
        J=1.0,
        h=0.5,
        fixed_lambda=0.0083,
        fixed_dir="sparse_qite_lambda_0p0083",
        adaptive_epsilon=0.0049,
        adaptive_dir="adaptive_lasso_epsilon_0p0049",
    ),
)

RUN_DIRECTORIES = {
    "qite": "qite_default",
    "fixed": None,
    "adaptive": None,
}

COLORS = {
    "qite": "black",
    "fixed": "#0072B2",
    "adaptive": "#D55E00",
    "ground": "#009E73",
}

CURVE_STYLES = {
    "qite": {"linestyle": ":", "marker": None, "linewidth": 1.15},
    "fixed": {
        "linestyle": "-",
        "marker": "s",
        "linewidth": 0.8,
        "markersize": 1.7,
        "markevery": 1,
    },
    "adaptive": {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 0.8,
        "markersize": 1.7,
        "markevery": 1,
    },
}


def configure_style() -> None:
    """Configure legible typography for a REVTeX full-width figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def run_directory(case: Case, kind: str) -> str:
    if kind == "fixed":
        return case.fixed_dir
    if kind == "adaptive":
        return case.adaptive_dir
    return "qite_default"


def load_run(data_root: Path, case: Case, kind: str) -> Run:
    directory = data_root / case.folder / run_directory(case, kind)
    trajectory_path = directory / "trajectory.npz"
    states_path = directory / "states.npz"
    if not trajectory_path.is_file() or not states_path.is_file():
        raise FileNotFoundError(
            f"Run '{kind}' for '{case.phase}' requires both "
            f"{trajectory_path} and {states_path}."
        )

    with np.load(trajectory_path) as trajectory:
        energy = np.asarray(trajectory["Energy"], dtype=float)
        coefficients = np.asarray(trajectory["a"])
    states = sp.load_npz(states_path)

    if energy.shape != (case.N + 1,):
        raise ValueError(f"Unexpected energy shape {energy.shape} in {directory}")
    if coefficients.shape[0] != case.N:
        raise ValueError(
            f"Unexpected coefficient shape {coefficients.shape} in {directory}"
        )
    if states.shape != (2**case.n_qubits, case.N + 1):
        raise ValueError(f"Unexpected state shape {states.shape} in {directory}")
    return Run(kind, energy, states, coefficients)


def load_case(data_root: Path, case: Case) -> tuple[Run, Run, Run]:
    return tuple(load_run(data_root, case, kind) for kind in RUN_DIRECTORIES)


def tfim_hamiltonian(case: Case) -> np.ndarray:
    """Construct H = J sum Z_i Z_(i+1) + h sum X_i with periodic boundaries."""
    dimension = 2**case.n_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=float)
    for basis_state in range(dimension):
        spins = np.asarray(
            [1.0 if not (basis_state >> site) & 1 else -1.0 for site in range(case.n_qubits)]
        )
        hamiltonian[basis_state, basis_state] = case.J * sum(
            spins[site] * spins[(site + 1) % case.n_qubits]
            for site in range(case.n_qubits)
        )
        for site in range(case.n_qubits):
            flipped_state = basis_state ^ (1 << site)
            hamiltonian[flipped_state, basis_state] += case.h
    return hamiltonian


def exact_ground_state(case: Case) -> tuple[float, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(tfim_hamiltonian(case))
    return float(eigenvalues[0]), eigenvectors[:, 0]


def fidelity_with_vector(states: sp.spmatrix, vector: np.ndarray) -> np.ndarray:
    overlaps = np.asarray(vector.conj() @ states).reshape(-1)
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def active_coefficients(coefficients: np.ndarray) -> np.ndarray:
    return np.count_nonzero(
        np.abs(coefficients) > SUPPORT_TOL,
        axis=tuple(range(1, coefficients.ndim)),
    )


def curve_label(case: Case, kind: str) -> str:
    if kind == "qite":
        return "QITE"
    if kind == "fixed":
        return rf"Fixed ($\lambda={case.fixed_lambda:g}$)"
    return rf"Adaptive ($\epsilon={case.adaptive_epsilon:g}$)"


def plot_curve(axis, x_values, y_values, case: Case, kind: str) -> None:
    axis.plot(
        x_values,
        y_values,
        color=COLORS[kind],
        label=curve_label(case, kind),
        zorder=3 if kind != "qite" else 2,
        **CURVE_STYLES[kind],
    )


def decorate_curve_axis(axis) -> None:
    axis.grid(True, which="major", color="0.82", linewidth=0.45, alpha=0.75)
    axis.tick_params(direction="out")
    axis.set_axisbelow(True)
    axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        borderpad=0.3,
        handlelength=2.0,
        handletextpad=0.45,
        labelspacing=0.25,
    )


def plot_comparison_grid(data_root: Path, output_dir: Path) -> Path:
    """Write the two-row by three-column curve comparison."""
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))

    for row, case in enumerate(CASES):
        runs = load_case(data_root, case)
        ground_energy, ground_vector = exact_ground_state(case)
        for run in runs:
            state_times = np.arange(run.energy.size) * case.dt
            coefficient_times = (
                np.arange(run.coefficients.shape[0]) + 1
            ) * case.dt
            infidelity = np.maximum(
                1.0 - fidelity_with_vector(run.states, ground_vector), 1e-16
            )
            active = active_coefficients(run.coefficients)
            plot_curve(axes[row, 0], state_times, run.energy, case, run.kind)
            plot_curve(axes[row, 1], state_times, infidelity, case, run.kind)
            plot_curve(
                axes[row, 2], coefficient_times, active, case, run.kind
            )

        axes[row, 0].axhline(
            ground_energy,
            color=COLORS["ground"],
            linestyle="--",
            linewidth=0.9,
            label="Exact ground energy",
            zorder=1,
        )
        axes[row, 0].set_ylabel("Energy")
        axes[row, 1].set_ylabel(r"Ground-state infidelity, $1-F$")
        axes[row, 1].set_yscale("log")
        axes[row, 2].set_ylabel("Active coefficients")

        for axis in axes[row]:
            axis.set_xlabel(r"Imaginary time, $\tau$")
            axis.set_xlim(0.0, case.N * case.dt)
            decorate_curve_axis(axis)

    for column, title in enumerate(
        ("Energy", "Ground-state infidelity", "Active coefficients")
    ):
        axes[0, column].set_title(title, pad=4.0)

    for axis, panel in zip(axes.flat, "abcdef"):
        axis.text(
            0.5,
            -0.31,
            f"({panel})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
        )

    figure.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.115,
        top=0.945,
        wspace=0.38,
        hspace=0.55,
    )
    output_path = output_dir / "adaptive_vs_fixed_2x3_revtex.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Wrote {output_path}")
    return output_path


def pauli_word_localities(D: int, expected_words: int) -> np.ndarray:
    """Reconstruct the odd-Y real-QITE Pauli basis and return Pauli weights."""
    localities = []
    for word in product("IXYZ", repeat=D):
        if sum(symbol == "Y" for symbol in word) % 2 == 0:
            continue
        localities.append(sum(symbol != "I" for symbol in word))
    result = np.asarray(localities, dtype=int)
    if result.size != expected_words:
        raise ValueError(
            f"D={D} implies {result.size} odd-Y Pauli words, but the NPZ data "
            f"contain {expected_words} coefficient columns."
        )
    return result


def active_fraction_by_locality(case: Case, run: Run) -> np.ndarray:
    """Compute active fraction among all coefficient slots at each locality."""
    coefficients = run.coefficients
    n_pieces = coefficients.shape[1]
    word_localities = pauli_word_localities(case.D, coefficients.shape[2])
    active = np.abs(coefficients) > SUPPORT_TOL
    fractions = np.zeros((case.D, coefficients.shape[0]), dtype=float)

    for locality in range(1, case.D + 1):
        word_mask = word_localities == locality
        available = n_pieces * int(np.count_nonzero(word_mask))
        if available:
            counts = np.count_nonzero(active[:, :, word_mask], axis=(1, 2))
            fractions[locality - 1] = counts / available
    return np.clip(fractions, 0.0, 1.0)


def heatmap_title(case: Case, kind: str) -> str:
    if kind == "fixed":
        return rf"Fixed Sparse-QITE ($\lambda={case.fixed_lambda:g}$)"
    return rf"Adaptive Sparse-QITE ($\epsilon={case.adaptive_epsilon:g}$)"


def plot_locality_heatmaps(data_root: Path, output_dir: Path) -> Path:
    """Write the two-row by two-column locality-fraction heatmap comparison."""
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("silver")
    image = None

    for row, case in enumerate(CASES):
        runs = {run.kind: run for run in load_case(data_root, case)}
        for column, kind in enumerate(("fixed", "adaptive")):
            axis = axes[row, column]
            fractions = active_fraction_by_locality(case, runs[kind])
            displayed = np.ma.masked_equal(fractions, 0.0)
            n_steps = fractions.shape[1]
            image = axis.imshow(
                displayed,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                vmin=0.0,
                vmax=1.0,
                extent=(
                    0.5 * case.dt,
                    (n_steps + 0.5) * case.dt,
                    0.5,
                    case.D + 0.5,
                ),
                rasterized=True,
            )
            axis.set_xlim(0.0, n_steps * case.dt)
            axis.set_xlabel(r"Imaginary time, $\tau$")
            axis.set_ylabel("Pauli locality")
            axis.set_yticks(np.arange(1, case.D + 1))
            axis.set_xticks(np.arange(0, n_steps * case.dt + 0.1, 1.0))
            axis.set_title(heatmap_title(case, kind), pad=5.0)
            axis.tick_params(direction="out")

    for axis, panel in zip(axes.flat, "abcd"):
        axis.text(
            0.5,
            -0.27,
            f"({panel})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )

    figure.subplots_adjust(
        left=0.075,
        right=0.875,
        bottom=0.13,
        top=0.94,
        wspace=0.26,
        hspace=0.54,
    )
    color_axis = figure.add_axes((0.905, 0.155, 0.018, 0.73))
    color_bar = figure.colorbar(image, cax=color_axis)
    color_bar.set_label("Active fraction at this locality", labelpad=7.0)
    color_bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    output_path = output_dir / "adaptive_vs_fixed_locality_heatmaps_2x2_revtex.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Wrote {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing the two case folders and their NPZ run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to write the two PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_comparison_grid(args.data_root.resolve(), args.output_dir.resolve())
    plot_locality_heatmaps(args.data_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
