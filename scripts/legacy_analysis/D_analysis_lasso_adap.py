import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt

from modules import Hamiltonian as ham
from modules import PauliStrings
from modules import Evolution as evol
import pickle

import importlib
importlib.reload(evol)

#Hamiltonian Params Traverse Field Ising model
n_qubits=8
J=0.5
h=1
T=2

#trotter params
H,H_trot=ham.TFIM(J,h,n_qubits,T=T)

#evolution parameters
dt=0.05
NQ=20 #maximum number of steps in QITE

#initial state
psi0np=np.zeros((2**n_qubits,1),dtype=complex)
psi0np[0]=1
psi_0=sp.csc_matrix(psi0np/np.linalg.norm(psi0np))

D_grid = [2, 3, 4, 5, 6, 7, 8]
lasso_lambda = 0.001
results = {}
output_path = Path("results_AdaptiveLASSOQITE_D.pkl")

for D in D_grid:

    print(f"Computing QITE with adaptive lambda and D={D}")
    EQ_lasso_ada, psi_QITE_lasso_ada, a_lasso_ada = evol.QITE(n_qubits, H, H_trot, D, psi_0, NQ, dt,
                                                               method='adaptive_lasso_v1',
                                                               lasso_max_iter=800,
                                                               lasso_tol=1e-9)
    results[D] = {
        "Energy": EQ_lasso_ada,
        "Psi": psi_QITE_lasso_ada,
        "a": a_lasso_ada
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(results, handle)
    temp_path.replace(output_path)
    