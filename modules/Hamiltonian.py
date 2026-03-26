from typing import List, Tuple, Union
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import Pauli,SparsePauliOp



class TrotterHamiltonian:
    """
    Container class for a Trotterized Hamiltonian decomposition.
    
    This class stores the decomposition of a Hamiltonian into smaller pieces suitable
    for Trotterization, including metadata about the decomposition structure and how
    operators are distributed across qubits.
    
    Attributes:
        T: Trotter size (number of qubits covered by each Hamiltonian piece).
        Nk: Number of Hamiltonian pieces in the decomposition.
        Rq: Number of times each single qubit operator appears.
        Hk: List of sparse matrices representing each Hamiltonian term/piece.
        indk: Array of starting qubit indices for each Hamiltonian piece.
    """
    def __init__(self, T: int, Nk: int, Rq: int, Hk: List[sp.spmatrix], indk: np.ndarray):
        self.T = T          #Trotter size
        self.Nk = Nk        #number of pieces
        self.Rq = Rq        #number of times each single qubit operator acts
        self.Hk = Hk        #list of all the Hammiltonian terms
        self.indk = indk    #first qbit on which each piece acts on

# %%
def TFIM(J: float, 
         h: float, 
         n_qubits: int, 
         T: int = 2, 
         sparse: bool = True, 
         verbose: bool = False) -> Tuple[Union[np.ndarray, sp.csc_matrix], TrotterHamiltonian]:
    """
    Generate the Transverse Field Ising Model (TFIM) Hamiltonian and its Trotterization.
    
    The TFIM Hamiltonian is defined as:
    H = J * ∑ᵢ Zᵢ Zᵢ₊₁ + h * ∑ᵢ Xᵢ
    
    This function constructs both the full Hamiltonian and a Trotterized decomposition
    into Nk pieces, suitable for efficient time evolution simulation. Periodic boundary
    conditions are applied (chain is a ring).
    
    Args:
        J: Coupling strength for the ZZ interaction terms.
        h: Transverse field strength for the X terms.
        n_qubits: Number of qubits in the chain.
        T: Trotter size (number of qubits per Hamiltonian piece). Must be >= 2 and <= n_qubits.
           Default is 2.
        sparse: If True, returns sparse matrices (scipy.sparse.csc_matrix).
                If False, returns dense numpy arrays. Default is True.
        verbose: If True, prints Trotterization details. Default is False.
    
    Returns:
        A tuple containing:
            - H: The full TFIM Hamiltonian (sparse or dense based on 'sparse' parameter).
            - H_trot: TrotterHamiltonian object containing the decomposed Hamiltonian pieces.

    """
    X=[]
    ZZ=[]
    M=[]
    for i in range(n_qubits):
        # Hamiltonian
        sx=["I"]*n_qubits
        szz=["I"]*n_qubits
        i1= (i+1)%n_qubits #periodic index
        sx[i]="X"
        szz[i]="Z"
        szz[i1]="Z"
        X.append(Pauli(''.join(sx)).to_matrix(sparse=sparse))
        ZZ.append(Pauli(''.join(szz)).to_matrix(sparse=sparse))
        #Magnetization
        mag=["I"]*n_qubits
        mag[i]="Z"
        M.append(Pauli(''.join(mag)).to_matrix(sparse=sparse))
        
    H = J*np.sum(ZZ,axis=0) + h*np.sum(X,axis=0)
    if sparse:
        H = sp.csc_matrix(H)
    #############################################################
    H_T=[]
    if T>n_qubits:
        print("Your Trotterization size is bigger than you system")
    if T<2:
        print("Your Trotterization size is too small, min value is T=2")
    N_k=Fraction(n_qubits,T).numerator
    R=Fraction(n_qubits,T).denominator
    if R==1:
        N_k=N_k*2
        R=R*2
    index=np.floor(n_qubits/N_k*np.arange(N_k)).astype(int).tolist()
    for i in range(N_k):
        ind=index[i]
        h_k=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
        #Two body terms
        for j in range(T-1):
            indT=(ind+j)%n_qubits
            ocr=index.count((indT+1)%n_qubits)
            h_k=h_k+J*ZZ[indT]/(R-ocr)
            #print('YY in {',indT,indT+1,'} appears',R-ocr,'times')
        #Single body terms
        for j in range(T):
            indT=(ind+j)%n_qubits
            h_k=h_k+h*X[indT]/R
        
        #for j in range(T):
        #    indT=(ind+j)%n_qubits
        #    if indT==(index[(i+1)%N_k]-1)%n_qubits:
        #        h_k=h_k+J*(ZZ[indT])/(R-1)+h*X[indT]/R
        #        #print(i,indT,"eliminat central")
        #    elif j == T-1:
        #        h_k=h_k+h*X[indT]/R
        #        #print(i,indT,"eliminat final")
        #    else:
        #        h_k=h_k+J*(ZZ[indT])/R+h*X[indT]/R
        #        #print(i,indT,"terme normal")
        if sparse:
            H_T.append(sp.csc_matrix(h_k))
        else:
            H_T.append(h_k)
    
    ###################################################
    #This is a check to see if the trotterization is the same as the original Hamiltonian
    if sparse:
        difH=sp.linalg.norm((H-sum(H_T)),ord=2)
    else:
        difH=np.linalg.norm((H-sum(H_T)),ord=2)
    if difH<1e-14:
        if verbose:
            print('Succesfull Troterization')
            print("The Trotterization consists of",N_k,"terms with the starting qubit of each piece at",index)
            print("Each single qubit term appears",R,"times")
    else:
        print('Failed Trotterization, you can still use the generated full Hamiltonian')

    H_trot=TrotterHamiltonian(T,N_k,R,H_T,index)

    return H,H_trot


def ClusterIsing(Lambda: float, 
                 n_qubits: int, 
                 T: int = 3, 
                 sparse: bool = True, 
                 verbose: bool = False) -> Tuple[Union[np.ndarray, sp.csc_matrix], TrotterHamiltonian]:
    """
    Generate the Cluster-Ising Hamiltonian and its Trotterization.
    
    The Cluster-Ising Hamiltonian is defined as:
    H = -∑ᵢ Zᵢ Xᵢ₊₁ Zᵢ₊₂ + λ * ∑ᵢ Yᵢ Yᵢ₊₁
    
    This Hamiltonian contains both two-body (YY) and three-body (ZXZ) interaction terms.
    Periodic boundary conditions are applied. The function constructs both the full
    Hamiltonian and a Trotterized decomposition suitable for efficient simulation.
    
    Args:
        Lambda: Coupling strength for the YY interaction terms.
        n_qubits: Number of qubits in the chain.
        T: Trotter size (number of qubits per Hamiltonian piece). Must be >= 3 due to
           three-body terms. Default is 3.
        sparse: If True, returns sparse matrices (scipy.sparse.csc_matrix).
                If False, returns dense numpy arrays. Default is True.
        verbose: If True, prints Trotterization details. Default is False.
    
    Returns:
        A tuple containing:
            - H: The full Cluster-Ising Hamiltonian (sparse or dense based on 'sparse' parameter).
            - H_trot: TrotterHamiltonian object containing the decomposed Hamiltonian pieces.
    """
    ZXZ=[]
    YY=[]
    for i in range(n_qubits):
        szxz=[]
        syy=[]
        for k in range(n_qubits):
            szxz.append("I")
            syy.append("I")
        i1= (i+1)%n_qubits
        i2= (i+2)%n_qubits
        
        szxz[i]="Z"
        szxz[i1]="X"
        szxz[i2]="Z"
        
        syy[i]="Y"
        syy[i1]="Y"
        zxzstr=''.join(szxz)
        yystr=''.join(syy)
        ZXZ.append(Pauli(zxzstr).to_matrix(sparse=sparse))
        YY.append(Pauli(yystr).to_matrix(sparse=sparse))
    H = -np.sum(ZXZ,axis=0) + Lambda*np.sum(YY,axis=0)
    if sparse:
        H = sp.csc_matrix(H)

    ###############################################################################################################################
    Tstr=3
    #Tstr is the Trotterization strategy, when your chain can be divided into whole pieces (n_qubits/T is a whole number)
    #to include the interaction terms between pieces you have to add orther pieces, Tstr=2 add one of them, Tstr=3 adds two
    #For this model that contains two and three body terms we default it to Tstr=2 for simplicity
    
    H_T=[]
    if T>n_qubits:
        print("Your Trotterization size is bigger than you system")
    if T<3:
        print("Your Trotterization size is too small, min value is T=3")
    
    N_k=Fraction(n_qubits,T).numerator
    R=Fraction(n_qubits,T).denominator
    if R==1:
        N_k=N_k*Tstr
        R=R*Tstr
    index=np.floor(n_qubits/N_k*np.arange(N_k)).astype(int).tolist()
    for i in range(N_k):
        ind=index[i]
        if sparse:
            h_k=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
        else:
            h_k=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)

        #Two body terms
        for j in range(T-1):
            indT=(ind+j)%n_qubits
            ocr=index.count((indT+1)%n_qubits)
            h_k=h_k+Lambda*YY[indT]/(R-ocr)
            #print('YY in {',indT,indT+1,'} appears',R-ocr,'times')
        #Three body terms
        for j in range(T-2):
            indT=(ind+j)%n_qubits
            ocr=index.count((indT+1)%n_qubits)+index.count((indT+2)%n_qubits)
            h_k=h_k-ZXZ[indT]/(R-ocr)
            #print('ZXZ in {',indT,indT+1,indT+2,'} appears',R-ocr,'times')
        
        H_T.append(h_k)
    
    ###################################################
    #This is a check to see if the trotterization is the same as the original Hamiltonian
    if sparse:
        difH=sp.linalg.norm((H-sum(H_T)),ord=2)
    else:
        difH=np.linalg.norm((H-sum(H_T)),ord=2)
    if difH<1e-14:
        if verbose:
            print('Succesfull Troterization')
            print("The Trotterization consists of",N_k,"terms with the starting qubit of each piece at",index)
            print("Each 2-qubit term appears",R-1,"times. And each 3-qubit term appears",R-2,"times")
    else:
        print('Failed Trotterization, you can still use the generated full Hamiltonian')

    H_trot=TrotterHamiltonian(T,N_k,R,H_T,index)
    
    return H,H_trot


def Heisenberg(J: float, 
               n_qubits: int, 
               sparse: bool = True) -> Tuple[Union[np.ndarray, sp.csc_matrix], TrotterHamiltonian]:
    """
    Generate the Heisenberg model Hamiltonian and its Trotterization.
    
    The Heisenberg Hamiltonian is defined as:
    H = ∑ᵢ (Xᵢ Xᵢ₊₁ + Yᵢ Yᵢ₊₁ + J Zᵢ Zᵢ₊₁)
    
    Uses periodic boundary conditions
    
    Args:
        J: Coupling strength for the ZZ interaction terms.
        n_qubits: Number of qubits in the chain.
        sparse: If True, returns sparse matrices. Default is True.
    
    Returns:
        Tuple of (H, H_trot) where H is the full Hamiltonian and H_trot 
        is the TrotterHamiltonian decomposition with T=2.
    """
    XX=[]
    YY=[]
    ZZ=[]
    for i in range(n_qubits-1): #open boundary
        # Hamiltonian
        sxx=["I"]*n_qubits
        syy=["I"]*n_qubits
        szz=["I"]*n_qubits
        i1= (i+1)%n_qubits #periodic index
        sxx[i]="X"
        sxx[i1]="X"
        syy[i]="Y"
        syy[i1]="Y"
        szz[i]="Z"
        szz[i1]="Z"
        XX.append(Pauli(''.join(sxx)).to_matrix(sparse=sparse))
        YY.append(Pauli(''.join(syy)).to_matrix(sparse=sparse))
        ZZ.append(Pauli(''.join(szz)).to_matrix(sparse=sparse))
        
    H = np.sum(XX,axis=0) + np.sum(YY,axis=0) + J*np.sum(ZZ,axis=0)
    if sparse:
        H = sp.csc_matrix(H)
    #############################################################
    H_T=[]
    T=2
    N_k=Fraction(n_qubits,T).numerator
    R=Fraction(n_qubits,T).denominator
    if R==1:
        N_k=N_k*2
        R=R*2
    index=np.floor(n_qubits/N_k*np.arange(N_k)).astype(int).tolist()
    for i in range(N_k-1):
        ind=index[i]
        h_k=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
        
        #Two body terms
        for j in range(T-1):
            indT=(ind+j)%n_qubits
            ocr=index.count((indT+1)%n_qubits)
            h_k=h_k+(XX[indT]+YY[indT]+J*ZZ[indT])/(R-ocr)
            #print('YY in {',indT,indT+1,'} appears',R-ocr,'times')
        
        if sparse:
            H_T.append(sp.csc_matrix(h_k))
        else:
            H_T.append(h_k)
    N_k=N_k-1 #open boundary, remove last term
    ###################################################
    #This is a check to see if the trotterization is the same as the original Hamiltonian
    if sparse:
        difH=sp.linalg.norm((H-sum(H_T)),ord=2)
    else:
        difH=np.linalg.norm((H-sum(H_T)),ord=2)
    print('Heisenberg model with OBC and Hamiltonian pieces of locality T=2')
    if difH<1e-14:
        print('Succesfull Troterization')
        print("The Trotterization consists of",N_k,"terms with the starting qubit of each piece at",index[0:-1])
    else:
        print('Failed Trotterization, you can still use the generated full Hamiltonian')

    H_trot=TrotterHamiltonian(T,N_k,R,H_T,index)

    return H,H_trot