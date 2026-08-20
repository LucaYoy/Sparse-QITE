# Sparse QITE via LASSO

This repository contains the code and numerical results accompanying the research paper on **Sparse-QITE: Discovering Compact Generators for Quantum Imaginary-Time
Evolution via Regularization Methods**. The project studies how an \(L_1\)-regularized solve can reduce the number of active Pauli terms in QITE while retaining the accuracy of the imaginary-time trajectory and ground-state preparation.

The repository includes implementations of standard QITE and sparse variants, model Hamiltonians, Pauli-basis construction, circuit transpilation, experiment drivers, analysis scripts, exploratory notebooks, and the data and figures used for the paper.

## Repository structure

```text
.
├── modules/          Core QITE, Hamiltonian, Pauli-pool, and transpilation code
├── notebooks/        Exploratory and explanatory Jupyter notebooks
├── scripts/          Experiment, HTCondor, analysis, and plotting scripts
├── final_outputs/    Numerical outputs accompanying the paper
└── final_plots/      Figures and summary tables
```

## Core modules

- `modules/Evolution.py` implements QITE and the coefficient solvers. It includes standard linear-system methods, fixed-penalty LASSO Sparse-QITE, adaptive LASSO selection, and cumulative-support freezing based on Jaccard stability.
- `modules/Hamiltonian.py` constructs the full and Trotter-decomposed Hamiltonians used in the experiments. Implemented models include the transverse-field Ising model, Cluster-Ising model, and Heisenberg chains with open or periodic boundary conditions.
- `modules/PauliStrings.py` constructs local Pauli pools for real or general states and for open or periodic boundary conditions.
- `modules/Transpile.py` builds and transpiles QITE circuits with Qiskit and provides utilities for fixed and adaptive product-formula synthesis.

## Notebooks

The notebooks contain exploratory and explanatory calculations rather than the main reusable implementation:

- `notebooks/QITE_vs_LASSOQITE.ipynb` compares standard QITE with LASSO-based Sparse-QITE.
- `notebooks/LASSOQITE_Gates.ipynb` explores the gate and circuit consequences of sparse QITE coefficients.
- `notebooks/test.ipynb` is a development and experimentation notebook.

For reproducible parameter scans and paper figures, use the scripts rather than the notebooks.

## Scripts and workflow

The main numerical workflow is:

1. Choose the parameter grid in `scripts/Generate_QITE_configs.py` and generate TOML job files.
2. Run each configuration locally with `scripts/Run_QITE_cluster.py`, or generate an HTCondor batch with `scripts/Generate_QITE_condor_batch.py`.
3. Optionally reconstruct and transpile the resulting circuits with `scripts/Run_Transpilation_cluster.py` or the corresponding HTCondor batch generator.
4. Run the scripts in `scripts/plotting/` to analyze trajectories and produce figures and CSV summaries.

The relevant scripts are grouped as follows:

- `Generate_QITE_configs.py` creates portable TOML configurations for Cartesian-product parameter scans.
- `Run_QITE_cluster.py` runs standard QITE, fixed-lambda Sparse-QITE, or cumulative-Jaccard Sparse-QITE from one TOML file. Interrupted runs can be checkpointed and resumed.
- `Generate_QITE_condor_batch.py` generates HTCondor submission files for selected QITE configurations.
- `Run_Transpilation_cluster.py` reconstructs a completed QITE circuit, transpiles it with Qiskit, and records circuit metrics.
- `Generate_Transpilation_condor_batch.py` generates HTCondor jobs for circuit reconstruction and transpilation.
- `Run_Adaptive_QITE.py` runs the adaptive Sparse-QITE cases used in the adaptive comparison.
- `scripts/plotting/` contains the current figure-generation and analysis scripts for individual runs, Jaccard support stability, Pareto fronts, scalability, compiled circuits, and adaptive-versus-fixed comparisons.
- `scripts/legacy_analysis/` contains earlier analysis programs retained for reference.

Most scan and plotting scripts have a clearly marked parameter block in `main()`. Edit that block to select the models, domain sizes, regularization strengths, fidelity thresholds, and input/output locations for a particular experiment.

## Installation

Python 3.11 or newer is recommended because the configuration runner uses the standard-library `tomllib` module. The main dependencies are NumPy, SciPy, Matplotlib, Qiskit, and Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib qiskit jupyter
```

The archived calculations in this repository were developed in an environment containing Python 3.12, NumPy 2.3, SciPy 1.16, Matplotlib 3.10, and Qiskit 2.3. These versions are useful reference points when reproducing the results exactly.

Run commands from the repository root so that the local `modules` package and script imports resolve correctly.

## Running an experiment

First edit the parameter block in `scripts/Generate_QITE_configs.py`, then generate the configurations:

```bash
python scripts/Generate_QITE_configs.py
```

Generated configurations are placed under `configs/generated/`. Run one configuration with:

```bash
python scripts/Run_QITE_cluster.py configs/generated/<experiment>/<method>.toml
```

The configuration selects the Hamiltonian, number of qubits, QITE domain size \(D\), number of time steps, time-step size, initial basis state, solver, LASSO parameters, checkpoint policy, and output directory. Supported Hamiltonian identifiers are `tfim`, `cluster_ising`, `heisenberg_obc`, and `heisenberg_pbc`. The principal evolution modes are:

- `default`: standard QITE;
- `vanilla_lasso`: fixed-penalty Sparse-QITE over the full Pauli pool;
- `Jaccard_lasso`: LASSO followed by cumulative-support freezing once the support is sufficiently stable.

The generated output directory is organized first by physical setup and then by method. For example:

```text
final_outputs/qite_runs/
└── tfim_n8_D4_N100_dt0p05_T2_J0p5_h1/
    ├── qite_default/
    └── sparse_qite_lambda_0p001/
```

Here, `n`, `D`, `N`, `dt`, and `T` denote the number of qubits, QITE domain size, number of steps, imaginary-time step, and Trotter support size, respectively. Decimal points and negative exponents are encoded in directory names to keep paths portable; for example, `0p05` means `0.05` and `1em05` means \(10^{-5}\).

## Output format

Each completed QITE run normally contains:

- `trajectory.npz`, with `Energy` for the energy trajectory and `a` for the QITE Pauli-coefficient tensor;
- `states.npz`, a SciPy sparse matrix whose columns are the state trajectory;
- a `transpilation/` subtree, when circuit reconstruction has been run, containing a Qiskit `circuit.qpy`, `summary.csv`, `metadata.json`, and synthesis-specific diagnostics.

The NPZ files can be loaded with:

```python
import numpy as np
import scipy.sparse as sp

trajectory = np.load("path/to/trajectory.npz")
energies = trajectory["Energy"]
coefficients = trajectory["a"]
states = sp.load_npz("path/to/states.npz")
```

## Reproducing analyses and figures

The scripts in `scripts/plotting/` document their expected inputs and write PDF figures together with CSV summaries. Typical entry points are:

```bash
python scripts/plotting/Plot_QITE_results.py
python scripts/plotting/Plot_QITE_scalability.py
python scripts/plotting/Plot_QITE_pareto.py
python scripts/plotting/Plot_QITE_compiled_scalability.py
python scripts/plotting/Plot_QITE_Jaccard_results.py
python scripts/plotting/Plot_Adaptive_vs_Fixed.py
```

- `final_plots/qite_runs/` contains per-run energy, fidelity, active-coefficient, cumulative-basis, and locality plots.
- `final_plots/scalability/` contains domain-size and compiled-circuit scalability plots and their CSV summaries.
- `final_plots/pareto/` contains fidelity-versus-support-cost Pareto analyses.
- `final_plots/qite_jaccard/` contains cumulative-Jaccard support-stability diagnostics.
- `final_plots/qite_adaptive/` contains adaptive-versus-fixed Sparse-QITE comparisons.

## Computational considerations

The Pauli pool and state-vector representation grow exponentially with the domain size and number of qubits. Larger scans can therefore require substantial memory and runtime. The cluster runners limit BLAS libraries to one thread per process by default to avoid oversubscription on HTCondor workers. Adjust the worker resources and thread environment variables together if using more than one CPU per job.

## Citation

If you use this code or the accompanying numerical data, please cite the associated research paper.
