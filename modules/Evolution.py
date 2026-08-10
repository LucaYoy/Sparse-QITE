from typing import Any, Tuple, Callable, List, Optional, NamedTuple
import numpy as np
import scipy
import cmath 
import scipy.sparse as sp
from . import PauliStrings as pauli_strings
from .Hamiltonian import TrotterHamiltonian

def _soft_threshold_real(x: np.ndarray, threshold: float) -> np.ndarray:
    """Real soft-thresholding used by proximal methods."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _qite_objective(coefficients: np.ndarray,
                    S: np.ndarray,
                    b: np.ndarray,
                    constant: float) -> float:
    """Evaluate the unpenalized QITE approximation objective."""
    value = (
        coefficients.transpose() @ np.asarray(S) @ coefficients
        - b.transpose() @ coefficients
        + constant
    )
    return float(np.real(value))


def _debias_lasso_support(A: np.ndarray,
                          b: np.ndarray,
                          coefficients: np.ndarray,
                          support_tol: float) -> np.ndarray:
    """Refit coefficients without an L1 penalty on their active support."""
    active = np.abs(coefficients) > support_tol
    if not np.any(active):
        return coefficients

    active_coefficients = scipy.linalg.solve(
        A[np.ix_(active, active)],
        b[active],
        assume_a='sym',
    )
    debiased_coefficients = np.zeros_like(coefficients)
    debiased_coefficients[active] = active_coefficients
    return debiased_coefficients


def _solve_lasso_qite(A: np.ndarray,
                      b: np.ndarray,
                      lam: float,
                      max_iter: int = 500,
                      tol: float = 1e-8,
                      support_tol: float = 1e-14,
                      x0: Optional[np.ndarray] = None,
                      lr: float = 1e-3,
                      debias: bool = True) -> np.ndarray:
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

    if lam == 0.0:
        return scipy.linalg.solve(A, b, assume_a='sym')

    if x0 is None:
        x = np.zeros(A.shape[1], dtype=float)
    else:
        x = np.asarray(np.real(x0), dtype=float).reshape(-1)
        if x.shape[0] != A.shape[1]:
            raise ValueError("x0 has incompatible shape for LASSO solve.")
    y = x.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = A @ y - b
        x_new = _soft_threshold_real(y - lr * grad, lam * lr)
        if np.linalg.norm(x_new - x) <= tol * (np.linalg.norm(x) + 1e-12):
            x = x_new
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        t = t_new
        x = x_new

    if debias:
        x = _debias_lasso_support(A, b, x, support_tol)

    return x


class _AdaptiveLassoResult(NamedTuple):
    coefficients: np.ndarray
    lambda_value: float
    grid_index: int
    objective: float

def _solve_adaptive_lasso_qite(A: np.ndarray,
                               S: np.ndarray,
                               b: np.ndarray,
                               constant: float,
                               objective_tol: float,
                               ridge: float,
                               max_iter: int,
                               tol: float,
                               support_tol: float,
                               x0: Optional[np.ndarray]) -> _AdaptiveLassoResult:
    """Select the largest feasible lambda on a logarithmic grid."""
    A_reg = np.asarray(A) + ridge * np.eye(A.shape[0], dtype=float)
    step_size = 1.0 / (np.linalg.norm(A_reg, ord=2) + 1e-16)

    lambda_max = float(np.max(np.abs(b)))
    if lambda_max == 0.0:
        lambda_grid = np.array([0.0])
    else:
        positive_grid = lambda_max * np.geomspace(1.0, 1e-8, num=25)
        lambda_grid = np.append(positive_grid, 0.0)

    def evaluate_lambda(ld: float,
                        warm_start: Optional[np.ndarray]
                        ) -> tuple[np.ndarray, float]:
        if ld>0:
            coefficients = _solve_lasso_qite(
                A_reg,
                b,
                lam=ld,
                max_iter=max_iter,
                tol=tol,
                support_tol=support_tol,
                x0=warm_start,
                lr=step_size,
                debias=False,
            )
        else:
            ep=1e-10
            coefficients = np.real(scipy.linalg.solve(A+ep*np.eye(A.shape[0]),b,assume_a='her'))
        return coefficients, _qite_objective(
            coefficients,
            S,
            b,
            constant,
        )

    # Lambda zero minimizes the unpenalized objective. Testing it first also
    # identifies tolerances that cannot be met by the current Pauli pool.
    last_index = len(lambda_grid) - 1
    qite_coefficients, qite_objective = evaluate_lambda(0.0, x0)
    tolerance_feasible = qite_objective < objective_tol

    selected_index = last_index
    selected_coefficients = qite_coefficients

    # The solution at lambda_max is exactly zero, so no FISTA solve is
    # needed for the upper endpoint.
    sparsest_coefficients = np.zeros_like(qite_coefficients)
    sparsest_objective = _qite_objective(
        sparsest_coefficients,
        S,
        b,
        constant,
    )

    if sparsest_objective - qite_objective < objective_tol:
        selected_index = 0
        selected_coefficients = sparsest_coefficients
    else:
        # Non-debiased LASSO feasibility is monotone along this descending
        # grid, allowing a binary search for the first feasible index.
        infeasible_index = 0
        feasible_index = last_index
        evaluated = {
            infeasible_index: sparsest_coefficients,
            feasible_index: qite_coefficients,
        }

        while feasible_index - infeasible_index > 1:
            middle_index = (infeasible_index + feasible_index) // 2
            nearest_index = min(
                evaluated,
                key=lambda index: abs(index - middle_index),
            )
            coefficients, objective = evaluate_lambda(
                float(lambda_grid[middle_index]),
                evaluated[nearest_index],
            )
            evaluated[middle_index] = coefficients

            if objective - qite_objective < objective_tol:
                feasible_index = middle_index
            else:
                infeasible_index = middle_index

        selected_index = feasible_index
        selected_coefficients = evaluated[feasible_index]

    selected_lambda = float(lambda_grid[selected_index])
    if selected_lambda > 0.0:
        selected_coefficients = _debias_lasso_support(
            A_reg,
            b,
            selected_coefficients,
            support_tol,
        )

    selected_objective = _qite_objective(
        selected_coefficients,
        S,
        b,
        constant,
    )
    return _AdaptiveLassoResult(
        coefficients=selected_coefficients,
        lambda_value=selected_lambda,
        grid_index=selected_index,
        objective=selected_objective,
    )

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
         resume_state: Optional[dict[str, Any]] = None,
         checkpoint_callback: Optional[Callable[[dict[str, Any]], None]] = None,
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
        resume_state: Previously saved state returned to this routine by the
            cluster runner. Evolution resumes at ``resume_state['next_step']``.
        checkpoint_callback: Optional function called after each complete QITE
            update with all numerical and Jaccard state required for resuming.
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
    ridge = 1e-10
    allowed_mask = np.ones((H_trot.Nk, num_paulis), dtype=bool)
    cumulative_support: set[int] = set()
    jaccard_streak = 0
    frozen = False
    freeze_step: Optional[int] = None

    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    start_step = 0

    if resume_state is None:
        psi_out[:,0]=psi_0.copy()
        psi_QITE=psi_0.copy()
        E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    else:
        start_step = int(resume_state["next_step"])
        if not 0 <= start_step <= N:
            raise ValueError("Checkpoint next_step is outside the requested evolution.")

        saved_energy = np.asarray(resume_state["energy"])
        saved_states = sp.csc_matrix(resume_state["states"])
        saved_coefficients = np.asarray(resume_state["coefficients"])
        expected_coefficient_shape = (start_step, H_trot.Nk, num_paulis)
        if saved_energy.shape != (start_step + 1,):
            raise ValueError("Checkpoint energy length does not match next_step.")
        if saved_states.shape != (2**n_qubits, start_step + 1):
            raise ValueError("Checkpoint state trajectory does not match next_step.")
        if saved_coefficients.shape != expected_coefficient_shape:
            raise ValueError(
                "Checkpoint coefficient shape does not match this Pauli pool: "
                f"{saved_coefficients.shape} != {expected_coefficient_shape}."
            )

        E_QITE[:start_step+1] = saved_energy
        psi_out[:,:start_step+1] = saved_states
        a[:start_step] = saved_coefficients
        saved_current_state = resume_state.get("current_state")
        if saved_current_state is None:
            psi_QITE = saved_states[:,-1].tocsc()
        else:
            psi_QITE = saved_current_state.copy()
            if psi_QITE.shape != (2**n_qubits, 1):
                raise ValueError("Checkpoint current_state has an incompatible shape.")

        if method == 'Jaccard_lasso':
            saved_mask = np.asarray(resume_state["allowed_mask"], dtype=bool)
            if saved_mask.shape != allowed_mask.shape:
                raise ValueError("Checkpoint allowed_mask has an incompatible shape.")
            allowed_mask[:] = saved_mask
            cumulative_support = {
                int(index) for index in resume_state["cumulative_support"]
            }
            jaccard_streak = int(resume_state["jaccard_streak"])
            frozen = bool(resume_state["frozen"])
            saved_freeze_step = resume_state.get("freeze_step")
            freeze_step = (
                None if saved_freeze_step is None else int(saved_freeze_step)
            )

    #errors = []
    completed_steps = start_step
    for i in range(start_step,N):
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
            #remove small imaginary noise
            A = np.real(A)
            b = np.real(b)
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
                A_reg = A + ridge * np.eye(A.shape[0], dtype=float)
                lr = 1.0 / (np.linalg.norm(A_reg, ord=2) + 1e-16)
                a[i,l] = _solve_lasso_qite(
                    A_reg,
                    b,
                    lam=lasso_lambda,
                    max_iter=lasso_max_iter,
                    tol=lasso_tol,
                    support_tol=support_tol,
                    x0=x0,
                    lr=lr
                )
            elif method=='Jaccard_lasso':
                if basis_indices.size == 0:
                    a[i,l] = 0.0
                else:
                    x0 = np.real(a[i-1,l,basis_indices]) if i > 0 else None
                    A_reg = A + ridge * np.eye(A.shape[0], dtype=float)
                    lr = 1.0 / (np.linalg.norm(A_reg, ord=2) + 1e-16)
                    reduced_coefficients = _solve_lasso_qite(
                        A_reg,
                        b,
                        lam=lasso_lambda,
                        max_iter=lasso_max_iter,
                        tol=lasso_tol,
                        support_tol=support_tol,
                        x0=x0,
                        lr=lr
                    )
                    a[i,l] = 0.0
                    a[i,l,basis_indices] = reduced_coefficients
            elif method=='adaptive_lasso_v1':
                x0 = np.real(a[i-1,l]) if i > 0 else None
                adaptive_result = _solve_adaptive_lasso_qite(
                    A=A,
                    S=S,
                    b=b,
                    constant=constant,
                    objective_tol=objective_tol,
                    ridge=ridge,
                    max_iter=lasso_max_iter,
                    tol=lasso_tol,
                    support_tol=support_tol,
                    x0=x0,
                )
                a[i,l] = adaptive_result.coefficients

                print(
                    f"Step {i}, term {l}, Lambda "
                    f"{adaptive_result.lambda_value}: objective is within "
                    "tolerance at grid index "
                    f"{adaptive_result.grid_index}."
                )

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
        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "next_step": completed_steps,
                    "energy": E_QITE[:completed_steps+1],
                    "states": psi_out[:,:completed_steps+1].tocsc(),
                    "current_state": psi_QITE.copy(),
                    "coefficients": a[:completed_steps],
                    "allowed_mask": allowed_mask,
                    "cumulative_support": np.asarray(
                        sorted(cumulative_support), dtype=np.int64
                    ),
                    "jaccard_streak": jaccard_streak,
                    "frozen": frozen,
                    "freeze_step": freeze_step,
                }
            )
    return E_QITE[0:completed_steps+1],psi_out[:,0:completed_steps+1],a[0:completed_steps,:,:]
        
