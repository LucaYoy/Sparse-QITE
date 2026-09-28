from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate,HamiltonianGate
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.synthesis import LieTrotter, SuzukiTrotter

def _build_synthesis_metadata(
    *,
    synthesis_formula: str,
    adaptive_repetitions: bool,
    max_coefficient_angle: float,
    max_repetitions: int,
    dt: float,
    repetition_counts: List[int],
    maximum_coefficients: List[float],
    active_term_counts: List[int],
    generator_indices: List[Tuple[int, int]],
) -> dict:
    """Summarize the product-formula choices made for active generators."""
    reps_array = np.asarray(repetition_counts, dtype=int)
    maxima_array = np.asarray(maximum_coefficients, dtype=float)
    terms_array = np.asarray(active_term_counts, dtype=int)
    rotation_multiplier = 1 if synthesis_formula == "lie" else 2

    return {
        "formula": synthesis_formula,
        "adaptive_repetitions": bool(adaptive_repetitions),
        "max_coefficient_angle": float(max_coefficient_angle),
        "max_repetitions": int(max_repetitions),
        "repetition_counts": reps_array.tolist(),
        "generator_records": [
            {
                "step": int(step),
                "piece": int(piece),
                "maximum_coefficient": float(maximum_coefficient),
                "repetitions": int(repetitions),
                "realized_coefficient_angle": float(
                    dt * maximum_coefficient / repetitions
                ),
                "active_terms": int(active_terms),
            }
            for (step, piece), maximum_coefficient, repetitions, active_terms
            in zip(
                generator_indices,
                maxima_array,
                reps_array,
                terms_array,
            )
        ],
        "minimum_repetitions": int(reps_array.min()) if reps_array.size else 0,
        "maximum_repetitions": int(reps_array.max()) if reps_array.size else 0,
        "mean_repetitions": float(reps_array.mean()) if reps_array.size else 0.0,
        "generators_above_one_repetition": int(np.count_nonzero(reps_array > 1)),
        "maximum_generator_coefficient": (
            float(maxima_array.max()) if maxima_array.size else 0.0
        ),
        "maximum_realized_coefficient_angle": (
            float(np.max(dt * maxima_array / reps_array))
            if reps_array.size else 0.0
        ),
        "estimated_pauli_rotations": int(
            np.sum(terms_array * reps_array * rotation_multiplier)
        ),
        "rotation_count_note": (
            "For suzuki2 this is the unmerged upper estimate 2*M*reps; "
            "Qiskit may merge adjacent rotations."
        ),
    }


def transpile_circuit_qite(n_qubits: int, 
                            PD: List[List[str]], 
                            a: np.ndarray, 
                            dt: float, 
                            steps: int, 
                            backend, 
                            op_level: int, 
                            PEvolution: bool = False,
                            compute_unitary_error: bool = True,
                            coefficient_threshold: float = 0.0,
                            synthesis_formula: str = "lie",
                            adaptive_repetitions: bool = False,
                            max_coefficient_angle: float = 0.1,
                            max_repetitions: int = 128,
                            synthesis_metadata: dict | None = None,
                            ) -> Tuple[QuantumCircuit, np.ndarray | None, float | None]:
    """
    Transpile a QITE circuit to native gates and compute transpilation error.
    
    This function constructs a quantum circuit implementing the QITE algorithm for a
    specified number of time steps, then transpiles it to the native gate set of a target
    backend. It also computes the exact unitary that should be implemented and measures
    the transpilation error using trace norm.
    
    Args:
        n_qubits: Number of qubits in the quantum system.
        PD: Pauli decomposition - list of Pauli string lists for each Hamiltonian piece.
            Structure: PD[piece_index][pauli_index] -> Pauli string (e.g., "IIXYZ").
            Each string spans all n_qubits; support is inferred from non-identity positions.
        a: Coefficients for the Pauli operators.
           Shape: (steps, num_pieces, num_paulis).
        dt: Time step.
        steps: Number of QITE time steps to implement.
        backend: Qiskit backend object defining the target gate set and architecture
        op_level: Optimization level for transpilation (0-3).
                  Higher levels apply more aggressive optimization.
        PEvolution: If True, uses PauliEvolutionGate for circuit construction.
                    If False, uses HamiltonianGate. Default is False.
        compute_unitary_error: If True, construct full-system unitaries and
                    calculate their Frobenius-norm difference. Disable this
                    for large systems, where a 2**n by 2**n matrix is costly.
        coefficient_threshold: Exclude coefficients whose absolute value is
                    less than or equal to this threshold. The default 0.0
                    preserves the original exact-nonzero behavior.
        synthesis_formula: Product formula used by PauliEvolutionGate: ``lie``
                    for first-order Lie--Trotter or ``suzuki2`` for the
                    symmetric second-order Suzuki formula.
        adaptive_repetitions: Choose the repetition count independently for
                    each nonempty generator using its largest coefficient.
        max_coefficient_angle: Target upper bound for
                    ``dt * max(abs(coefficient)) / repetitions``. This is a
                    practical step-size heuristic, not a rigorous error bound.
        max_repetitions: Safety cap applied to the adaptive repetition count.
        synthesis_metadata: Optional dictionary populated with the repetition
                    counts and coefficient-angle diagnostics.
    
    Returns:
        A tuple containing:
            - tr_qc_v: Transpiled quantum circuit with native gates.
            - v_true: The exact unitary matrix that should be implemented.
            - trace_norm: Frobenius norm of the error matrix (tr_unitary - v_true),
                          quantifying the transpilation error.
    """
    # set up the quantum circuit and backend
    if coefficient_threshold < 0.0:
        raise ValueError("coefficient_threshold must be non-negative.")
    if synthesis_formula not in {"lie", "suzuki2"}:
        raise ValueError("synthesis_formula must be 'lie' or 'suzuki2'.")
    if max_coefficient_angle <= 0.0:
        raise ValueError("max_coefficient_angle must be positive.")
    if max_repetitions < 1:
        raise ValueError("max_repetitions must be at least 1.")
    qc_v = QuantumCircuit(n_qubits)
    v_true = np.eye(2**n_qubits) if compute_unitary_error else None
    repetition_counts = []
    maximum_coefficients = []
    active_term_counts = []
    generator_indices = []

    # Piece indices need not equal their starting qubits. Infer support from
    # the full-system labels, which also handles open and periodic boundaries.
    # Keep one qubit for identity-only pools to preserve their global phase.
    supports = [
        sorted({i for string in pool for i, pauli in enumerate(string) if pauli != "I"})
        or [0]
        for pool in PD
    ]

    # loop through the Pauli operators and construct the circuits
    for step in range(steps):
        for k in range(len(PD)):
            coeffs = a[step,k,:]
            mask = np.abs(coeffs) > coefficient_threshold
            filtered_coeffs = np.where(mask, coeffs, 0.0)

            if np.any(mask):
                maximum_coefficient = float(np.max(np.abs(coeffs[mask])))
                repetitions = 1
                if adaptive_repetitions:
                    repetitions = int(
                        np.ceil(dt * maximum_coefficient / max_coefficient_angle)
                    )
                    repetitions = min(max(1, repetitions), max_repetitions)
                repetition_counts.append(repetitions)
                maximum_coefficients.append(maximum_coefficient)
                active_term_counts.append(int(np.count_nonzero(mask)))
                generator_indices.append((step, k))

                support = supports[k]
                PD_k_reduced = np.asarray([''.join(string[i] for i in support) for string in PD[k]])

                Ak_reduced = SparsePauliOp(
                    PD_k_reduced[mask], filtered_coeffs[mask]
                )

                qiskit_support = [n_qubits-1-i for i in support[::-1]]  #reverse order for qiskit
                if PEvolution:
                    if synthesis_formula == "lie":
                        synthesis = LieTrotter(reps=repetitions)
                    else:
                        synthesis = SuzukiTrotter(order=2, reps=repetitions)
                    qc_v.append(
                        PauliEvolutionGate(Ak_reduced, dt, synthesis=synthesis),
                        qiskit_support,
                    ) #use support to reduce qubits
                else:
                    qc_v.append(HamiltonianGate(Ak_reduced,dt),qiskit_support) #use support to reduce qubits
                
            if compute_unitary_error:
                Ak = SparsePauliOp(PD[k], filtered_coeffs).to_matrix()
                v_true = sp.linalg.expm(-1j *Ak*dt) @ v_true
            
    # transpile the circuits to the native gates of the backend
    tr_qc_v = transpile(qc_v,backend,seed_transpiler=40,optimization_level=op_level)
    
    # Calculate error using trace norm
    if compute_unitary_error:
        tr_unitary = Operator(tr_qc_v)
        error = (tr_unitary.data - v_true)
        trace_norm = np.linalg.norm(error)
    else:
        trace_norm = None

    if synthesis_metadata is not None:
        synthesis_metadata.update(
            _build_synthesis_metadata(
                synthesis_formula=synthesis_formula,
                adaptive_repetitions=adaptive_repetitions,
                max_coefficient_angle=max_coefficient_angle,
                max_repetitions=max_repetitions,
                dt=dt,
                repetition_counts=repetition_counts,
                maximum_coefficients=maximum_coefficients,
                active_term_counts=active_term_counts,
                generator_indices=generator_indices,
            )
        )
    #print(f'Error (trace norm) between transpiled circuit unitary and target unitary: {trace_norm:e}')

    return tr_qc_v, v_true, trace_norm
