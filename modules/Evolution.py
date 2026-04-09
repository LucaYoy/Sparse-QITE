from typing import Tuple, Callable, List, Optional
import numpy as np
import scipy
import cmath 
import scipy.sparse as sp
from . import PauliStrings as pauli_strings
from .Hamiltonian import TrotterHamiltonian

def _soft_threshold_real(x: np.ndarray, threshold: float) -> np.ndarray:
    """Real soft-thresholding used by proximal methods."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def _solve_lasso_qite(A: np.ndarray,
                      b: np.ndarray,
                      lam: float,
                      max_iter: int = 500,
                      tol: float = 1e-8,
                      ridge: float = 1e-10,
                      support_tol: float = 1e-10,
                      x0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solve the QITE LASSO subproblem:
        min_x 0.5 * x^T A x - b^T x + lam * ||x||_1
    for real-valued coefficients x.

    This uses accelerated proximal gradient (FISTA) with a debias step
    on the active support to improve accuracy while preserving sparsity.
    """
    if lam < 0:
        raise ValueError("lasso_lambda must be non-negative.")

    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)

    # QITE coefficients are real in this setup; remove tiny imaginary noise.
    A_real = np.real(A)
    b_real = np.real(b)

    if A_real.shape[0] != b_real.shape[0]:
        raise ValueError("A and b have incompatible shapes for LASSO solve.")

    A_sym = 0.5 * (A_real + A_real.T)
    if ridge > 0:
        A_sym = A_sym + ridge * np.eye(A_sym.shape[0], dtype=float)

    if lam == 0.0:
        return scipy.linalg.solve(A_sym, b_real, assume_a='sym')

    lipschitz = float(np.linalg.norm(A_sym, ord=2))
    if lipschitz == 0.0:
        return np.zeros(A_sym.shape[1], dtype=float)

    step = 1.0 / (lipschitz + 1e-16)

    if x0 is None:
        x = np.zeros(A_sym.shape[1], dtype=float)
    else:
        x = np.asarray(np.real(x0), dtype=float).reshape(-1)
        if x.shape[0] != A_sym.shape[1]:
            raise ValueError("x0 has incompatible shape for LASSO solve.")
    y = x.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = A_sym @ y - b_real
        x_new = _soft_threshold_real(y - step * grad, lam * step)
        if np.linalg.norm(x_new - x) <= tol * (np.linalg.norm(x) + 1e-12):
            x = x_new
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        t = t_new
        x = x_new

    # Debias on active support to recover precision lost to shrinkage.
    active = np.abs(x) > support_tol
    if np.any(active):
        A_active = A_sym[np.ix_(active, active)]
        b_active = b_real[active]
        x_active = scipy.linalg.solve(A_active, b_active, assume_a='sym')
        x_debias = np.zeros_like(x)
        x_debias[active] = x_active
        x = x_debias

    return x

def QITE(n_qubits: int, 
         H: sp.spmatrix, 
         H_trot: TrotterHamiltonian, 
         D: int, 
         psi_0: sp.spmatrix, 
         N: int, 
         dt: float, 
         vervose: bool = False, 
         method: str = 'LU',
         lasso_lambda: float = 1e-5,
         lasso_max_iter: int = 500,
         lasso_tol: float = 1e-8,
         objective_tol:float = 0.001) -> Tuple[np.ndarray, sp.lil_matrix, np.ndarray]:
    """
    Quantum Imaginary Time Evolution (QITE) algorithm for simulating ground state preparation.
    
    This function implements the QITE algorithm, which approximates imaginary time evolution. 
    The algorithm iteratively constructs unitary operators
    that approximate the action of exp(-H*dt) on the quantum state.
    
    Args:
        n_qubits: Number of qubits in the quantum system.
        H: Full Hamiltonian operator as a sparse matrix.
        H_trot: Trotterized Hamiltonian object (TrotterHamiltonian) containing decomposed terms (must have Nk and Hk attributes).
        D: Dimension parameter controlling the Pauli string basis size.
        psi_0: Initial quantum state as a sparse column vector.
        N: Number of time steps for the evolution.
        dt: Time step size for the imaginary time evolution.
        vervose: If True, prints energy at each step. Default is False.
        method: Method for solving linear system ('LU', 'lstsq', 'pinv', or 'lasso'). Default is 'LU'.
        lasso_lambda: L1 penalty weight used when method='lasso'. Default is 1e-5.
        lasso_max_iter: Maximum proximal iterations when method='lasso'. Default is 500.
        lasso_tol: Proximal convergence tolerance when method='lasso'. Default is 1e-8.
        objective_tol: Tolerance for the objective function convergence. Default is 0.001.
    Returns:
        A tuple containing:
            - E_QITE: Array of energies at each time step (length varies if early termination occurs).
            - psi_out: Sparse matrix where each column is the state at a given time step.
            - a: Array of coefficients for Pauli operators at each time step and qubit.
              Shape: (num_steps, num_trotter_terms, num_paulis).
    
    Notes:
        - The method parameter determines how the linear system (S+S^T)*a = b is solved:
          * 'LU': LU decomposition with small regularization
          * 'lstsq': Least squares solution
          * 'pinv': Pseudo-inverse method
          * 'lasso': L1-penalized least squares (sparse coefficients)
    """

    #checking whick method to obtain pauli strings is used
    if np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        #print("Using Real Pauli Strings") 
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
    else:
        #print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits)

    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0.copy()
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    #errors = []
    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #print('Time step',i+1,'/',N-1,'acting on qubit',l)
            #computing matrix S and coeffcient b (steps 1-3 in the algorithm)
            X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
            b=np.zeros((num_paulis),dtype=complex)
            aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
            aux2 = np.real((psi_QITE.getH()@sp.linalg.expm(-H_trot.Hk[l]*dt)@psi_QITE).trace())
            c=cmath.sqrt(aux)
            constant = 2*(1-aux2/c)/dt**2
            expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
            for j in range(num_paulis):
                b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
                X[:,j]=PD[l][j]@psi_QITE
            S=(X.getH()@X).todense()
           
            #obtencio coefficients a
            #invS_ex=np.linalg.pinv(S+S.T)
            #a[i,l]=np.real(invS_ex@b)

            #solution of equation (S+S^T)*a = b (step 4 in algorithm)
            A = np.asarray(S + S.T)
            if method=='lstsq':
                #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
                v,res,rank,s=scipy.linalg.lstsq(A,b,lapack_driver='gelsd')
                a[i,l]=v
                #print(res,rank)
            elif method=='pinv':
                #pseudo-inverse method
                invS_ex=scipy.linalg.pinvh(A)
                #print(np.linalg.norm(invS_ex*(S+S.T)-np.eye(num_paulis)))
                a[i,l]=np.real(invS_ex@b).flatten()
            elif method=='LU':
                ep=1e-10
                a[i,l]=(scipy.linalg.solve(A+ep*np.eye(num_paulis),b,assume_a='her'))
                #errors.append(a[i,l].transpose()@S@a[i,l]-b.transpose()@a[i,l] + constant)
            elif method=='vanilla_lasso':
                x0 = np.real(a[i-1,l]) if i > 0 else None
                a[i,l] = _solve_lasso_qite(
                    A,
                    b,
                    lam=lasso_lambda,
                    max_iter=lasso_max_iter,
                    tol=lasso_tol,
                    ridge=1e-10,
                    support_tol=1e-10,
                    x0=x0,
                )
            elif method=='adaptive_lasso_v1':
                lambda_max = 2*max(np.abs(b))
                lambda_grid = [lambda_max,lambda_max*0.5,lambda_max*0.2,lambda_max*0.1]
                x0 = np.real(a[i-1,l]) if i > 0 else None
                for ld in lambda_grid:
                    # Solve LASSO problem for each lambda in the grid
                    a[i,l] = _solve_lasso_qite(
                        A,
                        b,
                        lam=ld,
                        max_iter=lasso_max_iter,
                        tol=lasso_tol,
                        ridge=1e-10,
                        support_tol=1e-10,
                        x0=x0,
                    )
                    objective = np.real_if_close((a[i,l].transpose()@S@a[i,l]-b.transpose()@a[i,l] + constant).item())
                    if objective < objective_tol:
                        print(f"Step {i}, Lambda {ld}: Objective function is within tolerance.")  # Check if the objective function is within the tolerance
                        break

            else:
                raise ValueError(
                    f"Unknown method '{method}'. Use one of: 'LU', 'lstsq', 'pinv', 'vanilla_lasso', 'adaptive_lasso_v1'."
                )
            
            #construction of the evolution operator (steps 5 and 6 in the algorithm)
            operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*PD[l][j]
            psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
            
            
        #Enegy of the state at time step i
        if np.real((psi_QITE.getH()@H@psi_QITE).trace())>E_QITE[i]:
            print('Energy doubly increased at step',i)
            break
        E_QITE[i+1] = np.real((psi_QITE.getH()@H@psi_QITE).trace())
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
    return E_QITE[0:i+1],psi_out[:,0:i+1],a[0:i+1,:,:]
        
