from typing import List, Tuple, Union
from itertools import product
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import Pauli

def general(H_trot, 
            D: int, 
            n_qubits: int, 
            sparse: bool = True, 
            PDstr: bool = False, 
            verbose: bool = False) -> Tuple[int, Union[List[List[sp.csc_matrix]], List[List[str]]], bool]:
    """
    Generate the complete set of Pauli strings for QITE with a general (complex) Hamiltonian.
    
    This function generates all 4^D possible Pauli strings (combinations of I, X, Y, Z) needed
    to construct the unitary evolution operators in the QITE algorithm. This version handles
    general complex Hamiltonians and quantum states (i.e., no assumptions about real-valuedness).
    
    The Pauli strings are centered around each Hamiltonian piece's support region, with
    appropriate displacement to cover the T-local interactions.
    
    Args:
        H_trot: TrotterHamiltonian object containing the Hamiltonian decomposition.
                Must have attributes: T (Trotter size), Nk (number of pieces), and
                indk (starting indices).
        D: Domain size - number of qubits covered by each set of Pauli strings.
           Must satisfy D >= T (Trotter size) to span the Hamiltonian support.
        n_qubits: Total number of qubits in the quantum system.
        sparse: If True, returns Pauli operators as sparse scipy matrices.
                If False, returns dense numpy arrays. Default is True.
        PDstr: If True, returns Pauli string representations instead of matrices.
               Default is False.
        verbose: If True, prints statistics about the generated Pauli strings.
                 Default is False.
    
    Returns:
        A tuple containing:
            - num_paulis: Total number of Pauli strings per Hamiltonian piece (equals 4^D).
            - PD: List of lists containing Pauli operators or strings.
                  Structure: PD[piece_index][pauli_index] -> sparse matrix or string.
                  If PDstr=False: contains sparse/dense Pauli matrices.
                  If PDstr=True: contains Pauli string representations (e.g., "IXYZ").
            - fail: Boolean flag indicating if D < T (invalid configuration).
    
    Notes:
        - The Pauli strings are positioned with displacement = floor((D-T)/2) from each
          Hamiltonian piece's starting qubit, centering the strings over the support.
        - For D = T, no displacement occurs (exact coverage).
        - For D > T with odd difference: center with rightward bias.
        - For D > T with even difference: perfectly centered.
        - Periodic boundary conditions are applied when positioning Pauli strings.
    """
    num_paulis=int(4**D)
    PD = [[0 for i in range(num_paulis)] for j in range(H_trot.Nk)]
    PD_str = [['' for i in range(num_paulis)] for j in range(H_trot.Nk)]
    fail=False
    if H_trot.T>D:
        print("Your domain (D=",D,") is to small for your Trotterization (T=",H_trot.T,')')
        fail=True
    else:
        #displacement of initial qubit in which we compute expectation of Pauli string (if T=D no displacement, if D>T odd: centered with gravity towards right, if D>T even: centered)
        disp=np.floor((D-H_trot.T)/2).astype(int)
        for j in range(H_trot.Nk):
            i=0
            for qi in product('IXYZ',repeat=D):
                s=["I"]*n_qubits
                for k in range(D):
                    ind=(H_trot.indk[j]+k-disp)%n_qubits
                    s[ind]=qi[k]
                pstr=''.join(s)
                PD[j][i]=sp.csc_matrix(Pauli(pstr).to_matrix(sparse=sparse))
                PD_str[j][i]=pstr
                i=i+1
    if sparse:
        PD = [[sp.csc_matrix(val) for val in row] for row in PD]
    
    unique_elements = set(elem for sublist in PD_str for elem in sublist)
    if verbose:
        print("The number of Paulis is ",num_paulis)
        print("The number of Paulis per Hamiltonian piece are",num_paulis)
        print("The number of unique Pauli strings is",len(unique_elements),'from a total of',num_paulis*H_trot.Nk)
    if PDstr:
        return num_paulis,PD_str,fail
    return num_paulis,PD,fail

def real(H_trot, 
         D: int, 
         n_qubits: int, 
         sparse: bool = True, 
         PDstr: bool = False, 
         verbose: bool = False) -> Tuple[int, Union[List[List[sp.csc_matrix]], List[List[str]]], bool]:
    """
    Generate optimized Pauli strings for QITE with real Hamiltonian and real quantum states.
    
    This function generates a reduced set of 2^D*(2^D-1)/2 Pauli strings by exploiting the
    real-valuedness of both the Hamiltonian and quantum states.
    
    This optimization significantly reduces the number of required measurements and
    computational overhead compared to the general case.
    
    Args:
        H_trot: TrotterHamiltonian object containing the Hamiltonian decomposition.
                Must have attributes: T (Trotter size), Nk (number of pieces), and
                indk (starting indices).
        D: Domain size - number of qubits covered by each set of Pauli strings.
           Must satisfy D >= T (Trotter size) to span the Hamiltonian support.
        n_qubits: Total number of qubits in the quantum system.
        sparse: If True, returns Pauli operators as sparse scipy matrices.
                If False, returns dense numpy arrays. Default is True.
        PDstr: If True, returns Pauli string representations instead of matrices.
               Default is False.
        verbose: If True, prints statistics about the generated Pauli strings.
                 Default is False.
    
    Returns:
        A tuple containing:
            - num_paulis: Total number of Pauli strings per piece (equals 2^D*(2^D-1)/2).
            - PD: List of lists containing Pauli operators or strings.
                  Structure: PD[piece_index][pauli_index] -> sparse matrix or string.
                  If PDstr=False: contains sparse/dense Pauli matrices.
                  If PDstr=True: contains Pauli string representations.
            - fail: Boolean flag indicating if D < T (invalid configuration).
    
    Notes:
        - This function assumes BOTH the Hamiltonian and quantum state are real-valued.
        - The displacement logic matches the general() function for consistency.
    """
    num_paulis=int((2**D)*(2**D-1)/2)
    PD = [[0 for i in range(num_paulis)] for j in range(H_trot.Nk)]
    PD_str = [['' for i in range(num_paulis)] for j in range(H_trot.Nk)]
    fail=False
    if H_trot.T>D:
        print("Your domain (D=",D,") is to small for your Trotterization (T=",H_trot.T,')')
        fail=True
    else:
        #displacement of initial qubit in which we compute expectation of Pauli string (if T=D no displacement, if D>T odd centered with gravity towards right, if D>T even centered)
        disp=np.floor((D-H_trot.T)/2).astype(int)
        for j in range(H_trot.Nk):
            i=0
            for qi in product('IXYZ',repeat=D):
                s=["I"]*n_qubits
                for k in range(D):
                    ind=(H_trot.indk[j]+k-disp)%n_qubits
                    s[ind]=qi[k]
                plist=s
                num_Y=0
                for x in plist:
                    if x=="Y":
                        num_Y+=1
                if (num_Y % 2) != 0:
                    pstr=''.join(plist)
                    PD[j][i]=Pauli(pstr).to_matrix(sparse=sparse)
                    PD_str[j][i]=pstr
                    i=i+1
    if sparse:
        PD = [[sp.csc_matrix(val) for val in row] for row in PD]
    unique_elements = set(elem for sublist in PD_str for elem in sublist)
    if verbose:
        print("The number of Paulis is ",num_paulis)
        print("The number of Paulis per Hamiltonian piece are",num_paulis)
        print("The number of unique Pauli strings is",len(unique_elements),'from a total of',num_paulis*H_trot.Nk)
    if PDstr:
        return num_paulis,PD_str,fail
    return num_paulis,PD,fail


def general_OBC(H_trot, 
            D: int, 
            n_qubits: int, 
            sparse: bool = True, 
            PDstr: bool = False, 
            verbose: bool = False) -> Tuple[int, Union[List[List[sp.csc_matrix]], List[List[str]]], bool]:
    """
    Generate the complete set of Pauli strings for QITE with a general (complex) Hamiltonian.
    
    This function generates all 4^D possible Pauli strings (combinations of I, X, Y, Z) needed
    to construct the unitary evolution operators in the QITE algorithm. This version handles
    general complex Hamiltonians and quantum states (i.e., no assumptions about real-valuedness).
    
    The Pauli strings are centered around each Hamiltonian piece's support region, with
    appropriate displacement to cover the T-local interactions.
    
    Args:
        H_trot: TrotterHamiltonian object containing the Hamiltonian decomposition.
                Must have attributes: T (Trotter size), Nk (number of pieces), and
                indk (starting indices).
        D: Domain size - number of qubits covered by each set of Pauli strings.
           Must satisfy D >= T (Trotter size) to span the Hamiltonian support.
        n_qubits: Total number of qubits in the quantum system.
        sparse: If True, returns Pauli operators as sparse scipy matrices.
                If False, returns dense numpy arrays. Default is True.
        PDstr: If True, returns Pauli string representations instead of matrices.
               Default is False.
        verbose: If True, prints statistics about the generated Pauli strings.
                 Default is False.
    
    Returns:
        A tuple containing:
            - num_paulis: Total number of Pauli strings per Hamiltonian piece (equals 4^D).
            - PD: List of lists containing Pauli operators or strings.
                  Structure: PD[piece_index][pauli_index] -> sparse matrix or string.
                  If PDstr=False: contains sparse/dense Pauli matrices.
                  If PDstr=True: contains Pauli string representations (e.g., "IXYZ").
            - fail: Boolean flag indicating if D < T (invalid configuration).
    
    Notes:
        - The Pauli strings are positioned with displacement = floor((D-T)/2) from each
          Hamiltonian piece's starting qubit, centering the strings over the support.
        - For D = T, no displacement occurs (exact coverage).
        - For D > T with odd difference: center with rightward bias.
        - For D > T with even difference: perfectly centered.
        - Periodic boundary conditions are applied when positioning Pauli strings.
    """
    num_paulis=int(4**D)
    PD = [[0 for i in range(num_paulis)] for j in range(H_trot.Nk)]
    PD_str = [['' for i in range(num_paulis)] for j in range(H_trot.Nk)]
    fail=False
    if H_trot.T>D:
        print("Your domain (D=",D,") is to small for your Trotterization (T=",H_trot.T,')')
        fail=True
    else:
        #displacement of initial qubit in which we compute expectation of Pauli string (if T=D no displacement, if D>T odd: centered with gravity towards right, if D>T even: centered)
        disp=np.floor((D-H_trot.T)/2).astype(int)
        for j in range(H_trot.Nk):
            i=0
            for qi in product('IXYZ',repeat=D):
                s=["I"]*n_qubits
                for k in range(D):
                    ind=(H_trot.indk[j]+k-disp) #now we dont wrap around the chain
                    #if j==4:
                    #    print("ind",ind,"D_i=",k,"str=",qi[k])
                    if ind>=0 and ind<(n_qubits):   #we only keep the indices that stay inside the chain
                        s[ind]=qi[k]
                #print(s)
                pstr=''.join(s)
                PD[j][i]=sp.csc_matrix(Pauli(pstr).to_matrix(sparse=sparse))
                PD_str[j][i]=pstr
                i=i+1
    if sparse:
        PD = [[sp.csc_matrix(val) for val in row] for row in PD]
    
    unique_elements = set(elem for sublist in PD_str for elem in sublist)
    if verbose:
        print("The number of Paulis is ",num_paulis)
        print("The number of Paulis per Hamiltonian piece are",num_paulis)
        print("The number of unique Pauli strings is",len(unique_elements),'from a total of',num_paulis*H_trot.Nk)
    if PDstr:
        return num_paulis,PD_str,fail
    return num_paulis,PD,fail

def real_OBC(H_trot, 
         D: int, 
         n_qubits: int, 
         sparse: bool = True, 
         PDstr: bool = False, 
         verbose: bool = False) -> Tuple[int, Union[List[List[sp.csc_matrix]], List[List[str]]], bool]:
    """
    Generate optimized Pauli strings for QITE with real Hamiltonian and real quantum states.
    
    This function generates a reduced set of 2^D*(2^D-1)/2 Pauli strings by exploiting the
    real-valuedness of both the Hamiltonian and quantum states.
    
    This optimization significantly reduces the number of required measurements and
    computational overhead compared to the general case.
    
    Args:
        H_trot: TrotterHamiltonian object containing the Hamiltonian decomposition.
                Must have attributes: T (Trotter size), Nk (number of pieces), and
                indk (starting indices).
        D: Domain size - number of qubits covered by each set of Pauli strings.
           Must satisfy D >= T (Trotter size) to span the Hamiltonian support.
        n_qubits: Total number of qubits in the quantum system.
        sparse: If True, returns Pauli operators as sparse scipy matrices.
                If False, returns dense numpy arrays. Default is True.
        PDstr: If True, returns Pauli string representations instead of matrices.
               Default is False.
        verbose: If True, prints statistics about the generated Pauli strings.
                 Default is False.
    
    Returns:
        A tuple containing:
            - num_paulis: Total number of Pauli strings per piece (equals 2^D*(2^D-1)/2).
            - PD: List of lists containing Pauli operators or strings.
                  Structure: PD[piece_index][pauli_index] -> sparse matrix or string.
                  If PDstr=False: contains sparse/dense Pauli matrices.
                  If PDstr=True: contains Pauli string representations.
            - fail: Boolean flag indicating if D < T (invalid configuration).
    
    Notes:
        - This function assumes BOTH the Hamiltonian and quantum state are real-valued.
        - The displacement logic matches the general() function for consistency.
    """
    num_paulis=int((2**D)*(2**D-1)/2)
    PD = [[0 for i in range(num_paulis)] for j in range(H_trot.Nk)]
    PD_str = [['' for i in range(num_paulis)] for j in range(H_trot.Nk)]
    fail=False
    if H_trot.T>D:
        print("Your domain (D=",D,") is to small for your Trotterization (T=",H_trot.T,')')
        fail=True
    else:
        #displacement of initial qubit in which we compute expectation of Pauli string (if T=D no displacement, if D>T odd centered with gravity towards right, if D>T even centered)
        disp=np.floor((D-H_trot.T)/2).astype(int)
        for j in range(H_trot.Nk):
            i=0
            for qi in product('IXYZ',repeat=D):
                num_Y=0
                for x in qi:
                    if x=="Y":
                        num_Y+=1
                if (num_Y % 2) != 0:
                    s=["I"]*n_qubits
                    for k in range(D):
                        ind=(H_trot.indk[j]+k-disp) #now we dont wrap around the chain
                        if ind>=0 and ind<n_qubits:   #we only keep the indices that stay inside the chain
                            s[ind]=qi[k]
                    plist=s
                    pstr=''.join(plist)
                    PD[j][i]=Pauli(pstr).to_matrix(sparse=sparse)
                    PD_str[j][i]=pstr
                    i=i+1
    if sparse:
        PD = [[sp.csc_matrix(val) for val in row] for row in PD]
    unique_elements = set(elem for sublist in PD_str for elem in sublist)
    if verbose:
        print("The number of Paulis is ",num_paulis)
        print("The number of Paulis per Hamiltonian piece are",num_paulis)
        print("The number of unique Pauli strings is",len(unique_elements),'from a total of',num_paulis*H_trot.Nk)
    if PDstr:
        return num_paulis,PD_str,fail
    return num_paulis,PD,fail


def OBC_DN(H_trot,n_qubits,sparse=True,PDstr=False,verbose=False):
    """
    This function generates the 4^D Pauli strings needed to compute
    the unitary evolution of the ITE step generated by a Hamiltonian
    piece which is T-local
    """
    print("Using Pauli strings on the whole support D=N")
    D=n_qubits
    num_paulis=int(4**D)
    PD = [[0 for i in range(num_paulis)] for j in range(H_trot.Nk)]
    PD_str = [['' for i in range(num_paulis)] for j in range(H_trot.Nk)]
    fail=False
    if H_trot.T>D:
        print("Your domain (D=",D,") is to small for your Trotterization (T=",H_trot.T,')')
        fail=True
    else:
        i=0
        for qi in product('IXYZ',repeat=D):
            s=["I"]*n_qubits
            for k in range(D):
                ind=(H_trot.indk[0]+k)%n_qubits
                s[ind]=qi[k]
            pstr=''.join(s)
            PD[0][i]=sp.csc_matrix(Pauli(pstr).to_matrix(sparse=sparse))
            PD_str=pstr
            i=i+1

    unique_elements = set(elem for sublist in PD_str for elem in sublist)
    if verbose:
        print("The number of Paulis is ",num_paulis)
        print("The number of Paulis per Hamiltonian piece are",num_paulis)
        print("The number of unique Pauli strings is",len(unique_elements),'from a total of',num_paulis*H_trot.Nk)
    if PDstr:
        return num_paulis,PD_str,fail
    return num_paulis,PD,fail
