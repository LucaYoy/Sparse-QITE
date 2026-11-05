# Sparse QITE

Code and numerical results for **Sparse-QITE: Discovering Compact Generators for Quantum Imaginary-Time Evolution via Regularization Methods**.

The project compares standard quantum imaginary-time evolution (QITE) with sparse variants that use LASSO regularization to reduce the number of Pauli terms needed for the evolution.

The main algorithms are implemented in `modules/`, which provides the backbone of the project. The files in `scripts/` use these modules to run experiments and generate numerical outputs.

## Repository structure

```text
.
├── modules/          Main algorithms: QITE, Hamiltonians, Pauli pools, and circuits
├── notebooks/        Exploratory and explanatory notebooks
├── scripts/          Tools that use modules/ to run experiments and generate outputs
│   └── legacy_analysis/  Earlier analysis scripts
├── outputs/          Paper plotting notebooks, numerical data, and PDF figures
├── final_outputs/    Additional saved QITE runs
└── plots/            Other generated figures
```

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib qiskit jupyter
```

## Running experiments

Edit the parameter block in `scripts/Generate_QITE_configs.py`, then generate configurations:

```bash
python scripts/Generate_QITE_configs.py
```

Run one of the generated TOML files, replacing the placeholders below with its path:

```bash
python scripts/Run_QITE_cluster.py configs/generated/<experiment>/<method>.toml
```

The generator writes configurations to `configs/generated/` and defaults to saving results under `outputs/qite_runs/`. Each run saves its energy, coefficients, and state trajectory.

For adaptive Sparse-QITE, use `scripts/Run_Adaptive_QITE.py`. Circuit reconstruction and transpilation are handled by `scripts/Run_Transpilation_cluster.py`. The `Generate_*_condor_batch.py` scripts support HTCondor batch runs.

## Paper figures

The plotting notebooks and their accompanying data are in `outputs/`. Start Jupyter there so the notebooks can find their inputs:

```bash
cd outputs
jupyter notebook
```

- `plotter_sparseQITE.ipynb`: standard versus sparse QITE.
- `plotter_adaptive.ipynb`: adaptive versus fixed-penalty Sparse-QITE.
- `plotter_freeze.ipynb`: support-freezing results.
- `plotter_sparseQITE_heatmap.ipynb` and `plotter_adaptive_heatmap.ipynb`: coefficient heatmaps.
- `plotter_sparseQITE_pareto.ipynb`: Pareto comparisons.

The corresponding data are grouped under `tfim_ord/`, `tfim_disord/`, `heisenberg/`, and `pareto/`; generated PDF figures are also included in `outputs/`.

## Citation

If you use this code or the numerical data, please cite the associated research paper.
