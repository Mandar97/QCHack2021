from typing import List, Tuple
import numpy as np
import cirq
from cirq.protocols import trace_distance_from_angle_list
from scipy.optimize import minimize

global params_per_rot
params_per_rot = 3





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
    return NotImplemented, []

def cost_fn(theta,l,expected_unitary):
	n = int(np.log2(len(expected_unitary)))
	response_unitary = U_from_theta(theta,n,l)
	u = response_unitary @ expected_unitary.conj().T
	return trace_distance_from_angle_list(np.angle(np.linalg.eigvals(u)))

def cost_fn2(theta,l,expected_unitary):
	n = int(np.log2(len(expected_unitary)))
	response_unitary = U_from_theta(theta,n,l)
	return -np.real(np.trace(response_unitary @ expected_unitary.conj().T))
	

def optimize(l,expected_unitary,method='Nelder-Mead'):
	n = int(np.log2(len(expected_unitary)))
	theta0 = 2*np.pi*np.random.rand(num_parameters(n,l))
	return minimize(cost_fn2,theta0,args=(l,expected_unitary),method=method)
	
def optimization_routine(expected_unitary):
	l = 0
	finished = False
	while not finished:
		print('starting l = '+str(l)+' optimization...')
		opt = optimize(l,expected_unitary)
		finished = opt.fun < 1e-4
		if not finished:
			l += 1
	print('suceeded at l = '+str(l)+' layers with trace distance of '+str(opt.fun))
	return [l,opt.x]


def U_from_theta(theta, N_qubits,l):

	if N_qubits < 4:
		qubits = cirq.GridQubit.rect(1, N_qubits, 3, 3)
	elif int(np.sqrt(N_qubits)) ** 2 == N_qubits:
		qubits = cirq.GridQubit.square(int(np.sqrt(N_qubits)), 3, 3)
	elif N_qubits % 2 == 0:
		qubits = cirq.GridQubit.rect(2, int(N_qubits / 2), 3, 3)
	else:
		qubits = cirq.GridQubit.rect(2, int((N_qubits + 1) / 2), 3, 3)[:-1]

	#Helper funcitons
	def Sycamore_Gate(control_bit_index, bit_index, qbts):
		return cirq.google.SYC(qbts[control_bit_index], qbts[bit_index])

	def ISWAP(control_bit_index, bit_index, qbts):
		return cirq.ISWAP(qbts[control_bit_index], qbts[bit_index])**0.5

	def Phased_XZ(thetas, qubit_index, qbts):
		return cirq.ops.PhasedXZGate(x_exponent = thetas[0], z_exponent = thetas[1], \
									 axis_phase_exponent = thetas[2])(qbts[qubit_index])

	def Two_Qubit_Layer(qbts, crct, var_params, mode, var_param_ind, N_qubits = 2):
		if (mode == "Sycamore"):
			crct.append(Sycamore_Gate(0, 1, qbts))
		elif (mode == "ISWAP"):
			crct.append(ISWAP(0, 1, qbts))
		crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts) \
								 for i in range(N_qubits)]))
		return crct  

	def Three_Qubit_Layer1(qbts, crct, var_params, mode, var_param_ind, N_qubits = 3):
		if (mode == "Sycamore"):
			crct.append(Sycamore_Gate(0, 1, qbts))
		elif (mode == "ISWAP"):
			crct.append(ISWAP(0, 1, qbts))
		crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts) \
								 for i in range(2)]))
		return crct

	def Three_Qubit_Layer2(qbts, crct, var_params, mode, var_param_ind, N_qubits = 3):
		if (mode == "Sycamore"):
			crct.append(Sycamore_Gate(1, 2, qbts))
		elif (mode == "ISWAP"):
			crct.append(ISWAP(1, 2, qbts))
		crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i + 1, qbts) \
								 for i in range(2)]))
		return crct


	def Layer0(qbts, crct, N_qubits, var_params):
		crct.append(cirq.Moment([Phased_XZ(var_params[3*i:3*i+3], i, qbts) for i in range(N_qubits)]))
		return crct

	def LayerV(qbts, crct, N_qubits, var_params, mode, var_param_ind : int): #var_param_ind is the reference index
		n_vertical = int(N_qubits/2)
		if (mode == "Sycamore"):
			if (N_qubits%2 == 1 and N_qubits > 3):
				crct.append(cirq.Moment([Sycamore_Gate(i, i + n_vertical + 1, qbts) for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + var_param_ind + 3], i, qbts) \
										 for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind + 3*n_vertical:3*i + var_param_ind + \
															  3*n_vertical + 3], i + n_vertical + 1, qbts) for i in range(n_vertical)]))
			elif (N_qubits%2 == 0 and N_qubits > 3):
				crct.append(cirq.Moment([Sycamore_Gate(i, i + n_vertical, qbts) for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i+3 + var_param_ind], i, qbts) \
										 for i in range(N_qubits)]))
		if (mode == "ISWAP"):
			if (N_qubits%2 == 1 and N_qubits > 3):
				crct.append(cirq.Moment([ISWAP(i, i + n_vertical + 1, qbts) for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + var_param_ind + 3], i, qbts) \
										 for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind + 3*n_vertical:3*i + var_param_ind + \
															  3*n_vertical + 3], i + n_vertical + 1, qbts) for i in range(n_vertical)]))
			elif (N_qubits%2 == 0 and N_qubits > 3):
				crct.append(cirq.Moment([ISWAP(i, i + n_vertical, qbts) for i in range(n_vertical)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i+3 + var_param_ind], i, qbts) \
										 for i in range(N_qubits)]))
		return crct

	def LayerH1(qbts, crct, N_qubits, var_params, mode, var_param_ind : int):
		if (mode == "Sycamore"):
			if (N_qubits == 4):
				crct.append(cirq.Moment([Sycamore_Gate(2*i, 2*i + 1, qbts) for i in range (2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits)]))
			if (4 < N_qubits <= 6):
				crct.append(cirq.Moment([Sycamore_Gate(3*i, 3*i + 1, qbts) for i in range (2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind + 6:3*i + 3 + var_param_ind + 6], i + 3, qbts)\
										 for i in range(2)]))
			if (N_qubits == 7):
				crct.append(cirq.Moment([Sycamore_Gate(2*i, 2*i + 1, qbts) for i in range (3)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits - 1)]))
			if (N_qubits == 8):
				crct.append(cirq.Moment([Sycamore_Gate(2*i, 2*i + 1, qbts) for i in range (4)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits)]))
		if (mode == "ISWAP"):
			if (N_qubits == 4):
				crct.append(cirq.Moment([ISWAP(2*i, 2*i + 1, qbts) for i in range (2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits)]))
			if (4 < N_qubits <= 6):
				crct.append(cirq.Moment([ISWAP(3*i, 3*i + 1, qbts) for i in range (2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(2)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind + 6:3*i + 3 + var_param_ind + 6], i + 3, qbts)\
										 for i in range(2)]))
			if (N_qubits == 7):
				crct.append(cirq.Moment([ISWAP(2*i, 2*i + 1, qbts) for i in range (3)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits - 1)]))
			if (N_qubits == 8):
				crct.append(cirq.Moment([ISWAP(2*i, 2*i + 1, qbts) for i in range (4)]))
				crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind:3*i + 3 + var_param_ind], i, qbts)\
										 for i in range(N_qubits)]))
		return crct

	def LayerH2(qbts, crct, N_qubits, var_params, mode,var_param_ind:int):
		n_vertical = int(np.ceil(N_qubits/2))
		if mode =='Sycamore':
			two_q_gate = Sycamore_Gate
		else:
			two_q_gate = Phased_XZ
		moment = []
		acted_list = []
		for i in range(n_vertical):
			if i%2 ==1:
				if (i%n_vertical)+1<n_vertical:
					moment.append(Sycamore_Gate(i, i+1, qbts))
					acted_list.append(i)
					acted_list.append(i+1)
					if n_vertical + i+1 < N_qubits:
						moment.append(two_q_gate(n_vertical+i, n_vertical+i+1, qbts))
						acted_list.append(i+n_vertical)
						acted_list.append(i+1+n_vertical)
		moment = cirq.Moment(moment)
		crct.append(moment)
		acted_list.sort()
		crct.append(cirq.Moment([Phased_XZ(var_params[3*i + var_param_ind: 3*i + var_param_ind + 3], acted_list[i], qbts) \
									 for i in range(len(acted_list))]))
		return crct

	crct = cirq.Circuit()
	var_param_ind = 0
	crct = Layer0(qubits, crct, N_qubits, theta)
	var_param_ind += 3*N_qubits
	if N_qubits == 2:
		for li in range(l):
			crct = Two_Qubit_Layer(qubits, crct, theta, 'Sycamore', var_param_ind)
			var_param_ind += 2*3
	elif N_qubits == 3:
		for li in range(l):
			if li%2 == 0:
				crct = Three_Qubit_Layer1(qubits, crct, theta, 'Sycamore', var_param_ind)
				var_param_ind += 2*3
			elif li%2 == 1:
				crct = Three_Qubit_Layer2(qubits, crct, theta, 'Sycamore', var_param_ind)
				var_param_ind += 2*3
	elif N_qubits==4:			
		for li in range(l):
			if li%2 == 0:
				crct = LayerV(qubits, crct, N_qubits, theta, 'Sycamore', var_param_ind)
				var_param_ind += 4*params_per_rot
				
			elif li%2 == 1:
				crct = LayerH1(qubits, crct, N_qubits, theta, 'Sycamore', var_param_ind)
				var_param_ind += 4*params_per_rot
	
	elif N_qubits>4 and N_qubits<=8:
		for li in range(l):
			if li%3 == 0:
				U_tot = LayerV(qubits, crct, N_qubits, theta, 'Sycamore', var_param_ind)
				var_param_ind = num_parameters(N_qubits,li)
			
			elif li%3 == 1:
				U_tot = LayerH1(qubits, crct, N_qubits, theta, 'Sycamore', var_param_ind)
				var_param_ind = num_parameters(N_qubits,li)
			
			elif li%3 == 2:
				U_tot= LayerH2(qubits, crct, N_qubits, theta, 'Sycamore', var_param_ind)
				var_param_ind = num_parameters(N_qubits,li)

	
	
	return cirq.unitary(crct)

def num_parameters(n,l):
	global params_per_rot
	if n == 1:
		return params_per_rot
	elif n == 2:
		return 2*params_per_rot*(l+1)
	elif n == 3:
		return 3*params_per_rot + 2*l*params_per_rot
	elif n == 4:
		return 4*params_per_rot*(l+1)
	elif n == 5:
		if l%3 == 0:
			return 5*params_per_rot + 10*params_per_rot*l//3
		elif l%3 == 1:
			return 5*params_per_rot + 10*params_per_rot*(l-1)//3 + 4*params_per_rot
		elif l%3 == 2:
			return 5*params_per_rot + 10*params_per_rot*(l-2)//3 + 8*params_per_rot
	elif n == 6:
		if l%3 == 0:
			return 6*params_per_rot + 14*params_per_rot*l//3
		elif l%3 == 1:
			return 6*params_per_rot + 14*params_per_rot*(l-1)//3 + 6*params_per_rot
		elif l%3 == 2:
			return 6*params_per_rot + 14*params_per_rot*(l-2)//3 + 10*params_per_rot
	elif n == 7:
		if l%3 == 0:
			return 7*params_per_rot + 16*params_per_rot*l//3
		elif l%3 == 1:
			return 7*params_per_rot + 16*params_per_rot*(l-1)//3 + 6*params_per_rot
		elif l%3 == 2:
			return 7*params_per_rot + 16*params_per_rot*(l-2)//3 + 12*params_per_rot
	elif n == 8:
		if l%3 == 0:
			return 8*params_per_rot + 20*params_per_rot*l//3
		elif l%3 == 1:
			return 8*params_per_rot + 20*params_per_rot*(l-1)//3 + 8*params_per_rot
		elif l%3 == 2:
			return 8*params_per_rot + 20*params_per_rot*(l-2)//3 + 16*params_per_rot