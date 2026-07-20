from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp,Pauli
from qiskit.circuit.library import PauliEvolutionGate,HamiltonianGate
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2,FakeBelemV2,FakeQuitoV2
from qiskit.quantum_info import Operator
from qiskit.synthesis import qs_decomposition,LieTrotter

def cyclic_range(start: int, stop: int, n: int) -> List[int]:
    """
    Generate a range of indices with periodic boundary conditions.
    
    This function creates a list of consecutive indices from start (inclusive) to stop
    (exclusive), wrapping around at n to handle periodic boundary conditions on a chain
    or ring of n qubits.
    
    Args:
        start: Starting index (inclusive).
        stop: Stopping index (exclusive).
        n: Total number of elements in the cyclic system (wrap-around point).
    
    Returns:
        List of indices in cyclic order. If start < stop, returns [start, start+1, ..., stop-1].
        If start >= stop, returns [start, ..., n-1, 0, 1, ..., stop-1] (wraps around).
    
    Examples:
        cyclic_range(2, 5, 10) -> [2, 3, 4]
        cyclic_range(8, 2, 10) -> [8, 9, 0, 1]
    """
    if start < stop:
        return list(range(start, stop))
    else:
        # wrap around: [start, n) ∪ [0, stop)
        return list(range(start, n)) + list(range(0, stop))

def support_of_Ak(k: int, D: int, T: int, n_qubits: int) -> List[int]:
    """
    Compute the qubit support region for the k-th Hamiltonian piece operator.
    
    This function determines which qubits are involved in the k-th Hamiltonian piece
    when using a domain of size D to approximate a T-local Hamiltonian. The support
    is centered around qubit k with appropriate offset to cover the necessary interaction
    range, respecting periodic boundary conditions.
    
    Args:
        k: Index of the Hamiltonian piece (starting qubit of the T-local term).
        D: Domain size - number of qubits in the Pauli string basis.
        T: Trotter size - locality of the Hamiltonian piece (number of qubits it acts on).
        n_qubits: Total number of qubits in the system.
    
    Returns:
        List of qubit indices forming the support region, in cyclic order.
        The length of this list is D.
    
    Notes:
        - The support is computed with displacement = (D - T) // 2
        - For even (D - T): perfectly centered around the T-local term
        - For odd (D - T): centered with rightward bias (offset = 1)
        - Start index: (k - half) % n_qubits
        - Stop index: (k + T + half + offset) % n_qubits
        - Handles wrap-around for periodic boundary conditions
    
    Examples:
        For n_qubits=10, k=0, T=2, D=4:
            half=1, offset=0 -> support from qubit 9 to qubit 3 (wraps): [9, 0, 1, 2]
    """
    a = D - T
    half = a // 2
    offset = a % 2  # 0 if even, 1 if odd

    start = (k - half) % n_qubits
    stop  = (k + T + half + offset) % n_qubits

    support = cyclic_range(start, stop, n_qubits)
    return support

def transpile_circuit_qite(n_qubits: int, 
                            D: int, 
                            T: int, 
                            PD: List[List[str]], 
                            a: np.ndarray, 
                            dt: float, 
                            steps: int, 
                            backend, 
                            op_level: int, 
                            PEvolution: bool = False) -> Tuple[QuantumCircuit, np.ndarray, float]:
    """
    Transpile a QITE circuit to native gates and compute transpilation error.
    
    This function constructs a quantum circuit implementing the QITE algorithm for a
    specified number of time steps, then transpiles it to the native gate set of a target
    backend. It also computes the exact unitary that should be implemented and measures
    the transpilation error using trace norm.
    
    Args:
        n_qubits: Number of qubits in the quantum system.
        D: Domain size - number of qubits covered by each Pauli string basis.
        T: Trotter size - locality of each Hamiltonian piece.
        PD: Pauli decomposition - list of Pauli string lists for each Hamiltonian piece.
            Structure: PD[piece_index][pauli_index] -> Pauli string (e.g., "IIXYZ").
        a: Coefficients for the Pauli operators.
           Shape: (steps, num_pieces, num_paulis).
        dt: Time step.
        steps: Number of QITE time steps to implement.
        backend: Qiskit backend object defining the target gate set and architecture
        op_level: Optimization level for transpilation (0-3).
                  Higher levels apply more aggressive optimization.
        PEvolution: If True, uses PauliEvolutionGate for circuit construction.
                    If False, uses HamiltonianGate. Default is False.
    
    Returns:
        A tuple containing:
            - tr_qc_v: Transpiled quantum circuit with native gates.
            - v_true: The exact unitary matrix that should be implemented.
            - trace_norm: Frobenius norm of the error matrix (tr_unitary - v_true),
                          quantifying the transpilation error.
    """
    # set up the quantum circuit and backend
    qc_v = QuantumCircuit(n_qubits)
    v_true = np.eye(2**n_qubits)

    # loop through the Pauli operators and construct the circuits
    for step in range(steps):
        for k in range(len(PD)):
            coeffs = a[step,k,:]
            mask = np.abs(coeffs*dt)> 0

            if np.any(mask):
                support = support_of_Ak(k,D,T,n_qubits)
                PD_k_reduced = np.asarray([''.join(string[i] for i in support) for string in PD[k]])
                #print(PD_k_reduced,PD[k],support)

                Ak_reduced = SparsePauliOp(PD_k_reduced[mask],coeffs[mask])

                qiskit_support = [n_qubits-1-i for i in support[::-1]]  #reverse order for qiskit
                if PEvolution:
                    qc_v.append(PauliEvolutionGate(Ak_reduced,dt),qiskit_support) #use support to reduce qubits
                else:
                    qc_v.append(HamiltonianGate(Ak_reduced,dt),qiskit_support) #use support to reduce qubits
                
            Ak = SparsePauliOp(PD[k],a[step,k,:]).to_matrix()
            v_true = sp.linalg.expm(-1j *Ak*dt) @ v_true
            
    # transpile the circuits to the native gates of the backend
    tr_qc_v = transpile(qc_v,backend,seed_transpiler=40,optimization_level=op_level)
    
    # Calculate error using trace norm
    tr_unitary = Operator(tr_qc_v)
    error = (tr_unitary.data - v_true) 
    trace_norm = np.linalg.norm(error)  #trace norm
    #print(f'Error (trace norm) between transpiled circuit unitary and target unitary: {trace_norm:e}')

    return tr_qc_v, v_true, trace_norm