from analytic_gradient import optimization_routine
from optimizer import U_from_theta
import cirq
import numpy as np
from typing import List, Tuple

def matrix_to_sycamore_operations(target_qubits: List[cirq.GridQubit], matrix: np.ndarray) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """ A method to convert a unitary matrix to a list of Sycamore operations. 
    
    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla 
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`. 
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`. 

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list 
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns: 
        A tuple of operations and ancilla qubits allocated. 
            Operations: In case the matrix is supported, a list of operations `ops` is returned. 
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up 
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to 
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise 
                an empty list.
        .   
    """
    #Identitiy Matrices for all qubits
    for n in range(1, 9):
        if len(target_qubits) == n and np.allclose(np.identity(2**n), matrix):
            return [], []
        
    #1 QUBIT
    if len(target_qubits) == 1:
        return cirq.PhasedXZGate.from_matrix(matrix)(*target_qubits)

    #2 QUBITS
    elif len(target_qubits) == 2:
        syc_mat = cirq.unitary(cirq.google.SycamoreGate()(*target_qubits))
        iswapsqrt_mat = cirq.unitary((cirq.ISWAP(target_qubits[0],target_qubits[1])**0.5))
        if np.allclose(syc_mat, matrix):
            return cirq.google.SycamoreGate()(*target_qubits), []
        elif np.allclose(iswapsqrt_mat, matrix):
            return (cirq.ISWAP(a,b)**0.5)(*target_qubits), []
        else :
            l,theta = optimization_routine(matrix)
            #print(cirq.unitary(U_from_theta(theta[0:-1], len(target_qubits), l)))
            return U_from_theta(theta[0:-1], len(target_qubits), l).all_operations(), []
        
    #3+ QUBITS
    elif 2 < len(target_qubits) < 5:
        #else :
        l,theta = optimization_routine(matrix)
        return U_from_theta(theta[0:-1], len(target_qubits), l).all_operations(), []
    
    else:
        return [], []