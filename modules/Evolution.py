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
                      support_tol: float = 1e-14,
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
         lasso_lambda: float = 1e-3,
         lasso_max_iter: int = 500,
         lasso_tol: float = 1e-8,
         jaccard_cumulative_min: float = 0.99,
         jaccard_patience: int = 3,
         support_tol: float = 1e-14,
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
        method: Linear-system method. ``'vanilla_lasso'`` always uses the
            complete pool; ``'Jaccard_lasso'`` freezes cumulative support.
        lasso_lambda: L1 penalty used by the LASSO methods.
        lasso_max_iter: Maximum proximal iterations used by LASSO.
        lasso_tol: Proximal convergence tolerance used by LASSO.
        jaccard_cumulative_min: Minimum cumulative-Jaccard value considered
            stable by ``'Jaccard_lasso'``.
        jaccard_patience: Required number of consecutive stable time steps.
        support_tol: Absolute coefficient threshold defining active support.
        objective_tol: Tolerance for the objective function convergence. Default is 0.001.
    Returns:
        A tuple containing:
            - E_QITE: Array of energies at each time step (length N + 1).
            - psi_out: Sparse matrix where each column is the state at a given time step.
            - a: Array of coefficients for Pauli operators at each time step and qubit.
              Shape: (num_steps, num_trotter_terms, num_paulis).
    
    Notes:
        - The method parameter determines how the linear system (S+S^T)*a = b is solved:
          * 'LU': LU decomposition with small regularization
          * 'lstsq': Least squares solution
          * 'pinv': Pseudo-inverse method
          * 'vanilla_lasso': L1-penalized solve over the complete Pauli pool
          * 'Jaccard_lasso': LASSO followed by cumulative-support freezing
    """

    # Check which method to obtain Pauli strings is used.
    if H_trot.BC == 'OBC':
        if D == n_qubits:
            num_paulis,PD,fail = pauli_strings.OBC_DN(H_trot,n_qubits)
        elif np.isreal(H.data).all() and np.isreal(psi_0.data).all():
            #print("Using Real Pauli Strings")
            num_paulis,PD,fail = pauli_strings.real_OBC(H_trot,D,n_qubits)
        else:
            #print("Using General Pauli Strings")
            num_paulis,PD,fail = pauli_strings.general_OBC(H_trot,D,n_qubits)
    elif np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        #print("Using Real Pauli Strings")
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
    else:
        #print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits)

    if method == 'Jaccard_lasso':
        if not 0.0 <= jaccard_cumulative_min <= 1.0:
            raise ValueError("jaccard_cumulative_min must lie between 0 and 1.")
        if jaccard_patience < 1:
            raise ValueError("jaccard_patience must be at least 1.")
        if support_tol < 0.0:
            raise ValueError("support_tol must be non-negative.")

    # Jaccard_lasso starts with every coefficient slot available. After the
    # cumulative support stabilizes, the mask retains only discovered slots.
    allowed_mask = np.ones((H_trot.Nk, num_paulis), dtype=bool)
    cumulative_support: set[int] = set()
    jaccard_streak = 0
    frozen = False
    freeze_step: Optional[int] = None

    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0.copy()
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    #errors = []
    completed_steps = 0
    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #print('Time step',i+1,'/',N-1,'acting on qubit',l)
            #computing matrix S and coeffcient b (steps 1-3 in the algorithm)
            if method == 'Jaccard_lasso':
                basis_indices = np.flatnonzero(allowed_mask[l])
            else:
                basis_indices = np.arange(num_paulis)
            X=sp.lil_matrix((2**n_qubits,len(basis_indices)),dtype=complex)
            b=np.zeros((len(basis_indices)),dtype=complex)
            aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
            aux2 = np.real((psi_QITE.getH()@sp.linalg.expm(-H_trot.Hk[l]*dt)@psi_QITE).trace())
            c=cmath.sqrt(aux)
            constant = 2*(1-aux2/c)/dt**2
            expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
            for column,j in enumerate(basis_indices):
                b[column] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
                X[:,column]=PD[l][j]@psi_QITE
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
                a[i,l]=np.real(scipy.linalg.solve(A+ep*np.eye(num_paulis),b,assume_a='her'))
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
                    support_tol=support_tol,
                    x0=x0,
                )
            elif method=='Jaccard_lasso':
                if basis_indices.size == 0:
                    a[i,l] = 0.0
                else:
                    x0 = np.real(a[i-1,l,basis_indices]) if i > 0 else None
                    reduced_coefficients = _solve_lasso_qite(
                        A,
                        b,
                        lam=lasso_lambda,
                        max_iter=lasso_max_iter,
                        tol=lasso_tol,
                        ridge=1e-10,
                        support_tol=support_tol,
                        x0=x0,
                    )
                    a[i,l] = 0.0
                    a[i,l,basis_indices] = reduced_coefficients
            elif method=='adaptive_lasso_v1':
                lambda_max = max(np.abs(b))
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
                        support_tol= 1e-14,
                        x0=x0,
                    )
                    objective = np.real_if_close((a[i,l].transpose()@S@a[i,l]-b.transpose()@a[i,l] + constant).item())
                    if objective < objective_tol:
                        print(f"Step {i}, Lambda {ld}: Objective function is within tolerance. with  lambda at index  {lambda_grid.index(ld)}")  # Check if the objective function is within the tolerance
                        break

            else:
                raise ValueError(
                    f"Unknown method '{method}'. Use one of: 'LU', 'lstsq', "
                    "'pinv', 'vanilla_lasso', 'Jaccard_lasso', "
                    "'adaptive_lasso_v1'."
                )
            
            #construction of the evolution operator (steps 5 and 6 in the algorithm)
            operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*PD[l][j]
            psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
            
        if method == 'Jaccard_lasso' and not frozen:
            active_support = set(
                int(index)
                for index in np.flatnonzero(
                    np.abs(np.asarray(a[i]).reshape(-1)) > support_tol
                )
            )
            previous_cumulative_size = len(cumulative_support)
            cumulative_support |= active_support

            if cumulative_support:
                cumulative_jaccard = (
                    previous_cumulative_size / len(cumulative_support)
                )
                stable = cumulative_jaccard >= jaccard_cumulative_min
            else:
                cumulative_jaccard = 1.0
                stable = False

            jaccard_streak = jaccard_streak + 1 if stable else 0
            if jaccard_streak >= jaccard_patience:
                frozen = True
                freeze_step = i
                allowed_mask[:] = False
                for flat_index in cumulative_support:
                    piece, pauli_index = divmod(flat_index, num_paulis)
                    allowed_mask[piece, pauli_index] = True
                if vervose:
                    print(
                        "Cumulative-Jaccard support frozen after step",
                        i + 1,
                        "with",
                        len(cumulative_support),
                        "coefficient slots and Jaccard",
                        cumulative_jaccard,
                    )

        # Energy of the state after time step i.
        new_energy = float(np.real((psi_QITE.getH()@H@psi_QITE).trace()))
        E_QITE[i+1] = new_energy
        psi_out[:,i+1]=psi_QITE.copy()
        completed_steps = i + 1
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
    return E_QITE[0:completed_steps+1],psi_out[:,0:completed_steps+1],a[0:completed_steps,:,:]
        
