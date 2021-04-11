import numpy as np
from scipy.optimize import minimize

global params_per_rot
params_per_rot = 3


def U_gate(theta,output_gradient=True):
	# implements U from https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html
	U = np.array([[np.cos(theta[0]/2), -np.exp(1j*theta[2])*np.sin(theta[0]/2)], [ np.exp(1j*theta[1])*np.sin(theta[0]/2), np.exp(1j*(theta[1]+theta[2]))*np.cos(theta[0]/2)]])
	
	if output_gradient:
		dU = np.empty((3,2,2),dtype=np.cdouble)
		dU[0,:,:] = np.array([[-np.sin(theta[0]/2)/2, -np.exp(1j*theta[2])*np.cos(theta[0]/2)/2],[ np.exp(1j*theta[1])*np.cos(theta[0]/2)/2, -np.exp(1j*(theta[1]+theta[2]))*np.sin(theta[0]/2)/2]])
		dU[1,:,:] = np.array([[0, 0],[ 1j*np.exp(1j*theta[1])*np.sin(theta[0]/2), 1j*np.exp(1j*(theta[1]+theta[2]))*np.cos(theta[0]/2)]])
		dU[2,:,:] = np.array([[0, -1j*np.exp(1j*theta[2])*np.sin(theta[0]/2)],[ 0, 1j*np.exp(1j*(theta[1]+theta[2]))*np.cos(theta[0]/2)]])
	else:
		dU = []
	return [U,dU]



def one_qubit_gate(theta,output_gradient=True):
	x = theta[0]
	z = theta[1]
	a = theta[2]
	U = np.array([[np.exp(1j*np.pi*x*0.5)*np.cos(np.pi*x*0.5), \
                       -1j*np.exp(-1j*np.pi*a)*np.exp(1j*np.pi*x*0.5)*np.sin(np.pi*x*0.5)],\
                      [-1j*np.exp(1j*np.pi*z)*np.exp(1j*np.pi*a)*np.exp(1j*np.pi*x*0.5)*np.sin(np.pi*x*0.5),\
                       np.exp(1j*np.pi*z)*np.exp(1j*np.pi*x*0.5)*np.cos(np.pi*x*0.5)]])
	dU = [np.pi/2*np.exp(1j*np.pi*x)*np.array([[1j, -1j*np.exp(-1j*np.pi*a)],[-1j*np.exp(1j*np.pi*(a+z)), 1j*np.exp(1j*np.pi*z)]]),  np.array([[0., 0.], \
                     [np.pi*np.exp(1j*np.pi*z)*np.exp(1j*np.pi*a)*np.exp(1j*np.pi*x*0.5)*np.sin(np.pi*x*0.5),\
                      1j*np.pi*np.exp(1j*np.pi*z)*np.exp(1j*np.pi*x*0.5)*np.cos(np.pi*x*0.5)]]), np.array([[0., \
                     -1.*np.pi*np.exp(-1j*np.pi*a)*np.exp(1j*np.pi*x*0.5)*np.sin(np.pi*x*0.5)],\
                     [np.pi*np.exp(1j*np.pi*z)*np.exp(1j*np.pi*a)*np.exp(1j*np.pi*x*0.5)*np.sin(np.pi*x*0.5), \
                      0.]])]
	
	return [U,dU]


def two_qubit_gate(type="Sycamore"):
    Sycamore =  [[1, 0, 0, 0],
                 [0, 0, -1j, 0],
                 [0, -1j, 0, 0],
                 [0, 0, 0, np.exp(- 1j * np.pi/6)]]
    
    Iswapsqroot = [[1.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.70710678+0.j        ,
        0.        +0.70710678j, 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.70710678j,
        0.70710678+0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.        +0.j        ,
        0.        +0.j        , 1.        +0.j        ]]
    
    if type == "Sycamore":
        return Sycamore
    elif type == "Iswap":
        return Iswapsqroot
    else:
        return NotImplemented
    
def Layer0(U_tot,N_qubits,theta,output_gradient=True):
	current_theta = 0 
	for q in range(N_qubits):
		[U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
		current_theta += params_per_rot
		if output_gradient:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
		else:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
	return U_tot

def Two_Qubit_Layer(U_tot, N_qubits, theta,current_theta,mode,output_gradient=True):
	U_tot = multiply_state(two_qubit_gate(type =mode),U_tot,[0,1],output_gradient)
	for q in range(N_qubits):
		[U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
		current_theta += params_per_rot
		if output_gradient:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
		else:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
	return U_tot
        
def Three_Qubit_Layer1(U_tot, N_qubits, theta,current_theta,mode,output_gradient=True):
    U_tot = multiply_state(two_qubit_gate(type =mode),U_tot,[0,1],output_gradient)
    for q in [0,1]:
        [U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
        current_theta += params_per_rot
        if output_gradient:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
        else:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
    return U_tot

def Three_Qubit_Layer2(U_tot, N_qubits, theta,current_theta,mode,output_gradient=True):
	U_tot = multiply_state(two_qubit_gate(type =mode),U_tot,[1,2],output_gradient)
	for q in [1,2]:
		[U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
		current_theta += params_per_rot
		if output_gradient:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
		else:
			U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
	return U_tot
	


def LayerV(U_tot, N_qubits,theta,current_theta,mode,output_gradient=True):
    n_vertical = int(np.ceil(N_qubits/2))
    #Keeps track of on which qubits the entangling gates act
    acted_list = [] 
    for i in range(n_vertical):
            if (i+n_vertical)<N_qubits:
                U_tot = multiply_state(two_qubit_gate(mode),U_tot,[i,i+n_vertical],output_gradient)
                acted_list.append(i)
                acted_list.append(i+n_vertical)
    acted_list.sort()
    for q in acted_list:
        [U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
        current_theta += params_per_rot
        if output_gradient:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
        else:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
    #print(acted_list)
    return U_tot

def LayerH1(U_tot, N_qubits,theta,current_theta,mode,output_gradient=True):
    n_vertical = int(np.ceil(N_qubits/2))
    #Keeps track of on which qubits the entangling gates act
    acted_list = [] 
    for i in range(n_vertical):
        if i%2 ==0:
            if (i%n_vertical)+1<n_vertical:
                #print(i,i+1)
                U_tot = multiply_state(two_qubit_gate(mode),U_tot,[i,i+1],output_gradient)
                acted_list.append(i)
                acted_list.append(i+1)
                if n_vertical + i+1 < N_qubits:
                    #print(i+n_vertical,i+1+n_vertical)
                    U_tot = multiply_state(two_qubit_gate(mode),U_tot,[i+n_vertical,i+n_vertical+1],output_gradient)
                    acted_list.append(i+n_vertical)
                    acted_list.append(i+1+n_vertical)
    acted_list.sort()
    for q in acted_list:
        [U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
        current_theta += params_per_rot
        if output_gradient:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
        else:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
    #print(acted_list)
    return U_tot

def LayerH2(U_tot, N_qubits,theta,current_theta,mode,output_gradient=True):
    n_vertical = int(np.ceil(N_qubits/2))
    #Keeps track of on which qubits the entangling gates act
    acted_list = [] 
    for i in range(n_vertical):
        if i%2 ==1:
            if (i%n_vertical)+1<n_vertical:
                U_tot = multiply_state(two_qubit_gate(mode),U_tot,[i,i+1],output_gradient)
                acted_list.append(i)
                acted_list.append(i+1)
                if n_vertical + i +1< N_qubits:
                    U_tot = multiply_state(two_qubit_gate(mode),U_tot,[i+n_vertical,i+n_vertical+1],output_gradient)
                    acted_list.append(i+n_vertical)
                    acted_list.append(i+n_vertical+1)
    acted_list.sort()
    for q in acted_list:
        [U,dU] = one_qubit_gate(theta[current_theta:(current_theta+params_per_rot)])
        current_theta += params_per_rot
        if output_gradient:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits, dU )
        else:
            U_tot = apply_one_qubit_matrix(U,U_tot, q, N_qubits )
    #print(acted_list)
    return U_tot

	
def ansatz_U(N_qubits,N_layers,theta,mode="Sycamore",output_gradient=True,return_as_matrix=False):
    U_tot = np.eye(2**N_qubits)
    if N_qubits == 1:
        U_tot = Layer0(U_tot,N_qubits,theta,output_gradient=output_gradient)
    
    # 2 qubit case, only one kind of layer
    if N_qubits==2:
        U_tot = Layer0(U_tot,N_qubits,theta,output_gradient=output_gradient)
        current_theta = 2*params_per_rot
        for l in range(N_layers):
            U_tot = Two_Qubit_Layer(U_tot, 2, theta,current_theta,mode,output_gradient=output_gradient)
            current_theta += 2*params_per_rot

    if N_qubits==3:
        U_tot = Layer0(U_tot,N_qubits,theta,output_gradient=output_gradient)
        current_theta = 3*params_per_rot
        for l in range(N_layers):
            if l%2 == 0:
                U_tot = Three_Qubit_Layer1(U_tot, 3, theta,current_theta,mode,output_gradient=output_gradient)
                current_theta += 2*params_per_rot
            elif l%2 == 1:
                U_tot = Three_Qubit_Layer2(U_tot, 3, theta,current_theta,mode,output_gradient=output_gradient)
                current_theta += 2*params_per_rot
                
    if N_qubits==4:
        U_tot = Layer0(U_tot,N_qubits,theta,output_gradient=output_gradient)
        current_theta = 4*params_per_rot
        for l in range(N_layers):
            if l%2 == 0:
                U_tot = LayerV(U_tot, 4, theta,current_theta,mode,output_gradient=output_gradient)
                current_theta += 4*params_per_rot
                
            elif l%2 == 1:
                U_tot = LayerH1(U_tot, 4, theta,current_theta,mode,output_gradient=output_gradient)
                current_theta += 4*params_per_rot
        
    if N_qubits>4 and N_qubits<=8:
        U_tot = Layer0(U_tot,N_qubits,theta,output_gradient=output_gradient)
        current_theta = N_qubits*params_per_rot
        for l in range(N_layers):
            if l%3 == 0:
                U_tot = LayerV(U_tot, N_qubits,theta,current_theta,mode,output_gradient=output_gradient)
                current_theta = num_parameters(N_qubits,l)
            
            elif l%3 == 1:
                U_tot = LayerH1(U_tot, N_qubits,theta,current_theta,mode,output_gradient=output_gradient)
                current_theta = num_parameters(N_qubits,l)
            
            elif l%3 == 2:
                U_tot=LayerH2(U_tot, N_qubits,theta,current_theta,mode,output_gradient=output_gradient)
                current_theta = num_parameters(N_qubits,l)
            
    if return_as_matrix and not output_gradient:
        U_tot = np.reshape(U_tot,(2**N_qubits,2**N_qubits),order='C')
    return U_tot



def multiply_state(U,U_tot,qubits,leave_first_index=False,array_of_Us=False,return_as_matrix=False):

	# multiples the U_tot on the left by the matrix U. qubits indicates the indices to contract. If leave_first_index = True, then U_tot is treated as an array of unitaries, where the first index indices the different unitaries. All of the unitaries in that array are multiplied by U.
	

	if not hasattr(qubits,'__iter__'):
		qubits = [qubits]
	
	n_to_contract = len(qubits)
	qubits = np.array(qubits)
	
	sh_U = np.shape(U)
	if len(sh_U) != 2*n_to_contract and not array_of_Us:
		U = np.reshape(U,2*np.ones(2*n_to_contract,dtype=np.int), order='C')

	sh = np.shape(U_tot)
	if not leave_first_index:
		n = int(np.log2(np.size(U_tot)))//2
	elif leave_first_index:
		n = (len(sh) - 1)//2
	if (not leave_first_index) and len(sh) != 2*n:
		U_tot = np.reshape(U_tot.T,2*np.ones(2*n,dtype=np.int),order='F')
	
	U_axes = np.flip(np.arange(n_to_contract,2*n_to_contract)) + array_of_Us
	U_tot_axes = qubits + leave_first_index
	extra_indices = leave_first_index + array_of_Us
	if not array_of_Us:
		axesInv = list(np.concatenate( (np.flip(qubits)+ extra_indices, np.delete(np.arange(n+extra_indices) ,qubits+extra_indices), np.arange(n,2*n) + extra_indices )))
	elif array_of_Us:
		axesInv = list(np.concatenate(  ([0], np.flip(qubits)+ extra_indices, np.delete(np.arange(1,n+extra_indices) ,qubits+extra_indices-1), np.arange(n,2*n) + extra_indices )))
	axes = [axesInv.index(i) for i in range(2*n+extra_indices)]
	U_tot = np.tensordot(U,U_tot,axes=(U_axes,U_tot_axes))
	U_tot = np.transpose(U_tot,axes=axes)
	
	
	if (not leave_first_index) and len(sh) != 2*n and return_as_matrix:
		U_tot = np.reshape(U_tot,sh,order='F').T
	
	return U_tot



	
def apply_one_qubit_matrix(U,U_tot, qubit, n, dU = []):
	if len(np.shape(U_tot)) == 2*n and np.size(U_tot) == 4**n:
		first_gate = True
	elif len(np.shape(U_tot)) == 2*n+1:
		first_gate = False
	elif np.size(U_tot) == 4**n:
		first_gate = True
		U_tot = np.reshape(U_tot.T,2*np.ones(2*n,dtype=np.int),order='F')
	output_gradient = np.size(dU) > 0

	array_of_Us = output_gradient and len(np.shape(dU)) == 3
	
	if output_gradient and first_gate:
		dU_tot = multiply_state(dU,U_tot,qubit,array_of_Us=array_of_Us)
	elif output_gradient:
		dU_tot = multiply_state(dU,U_tot[0],qubit,array_of_Us=array_of_Us)
	U_tot = multiply_state(U,U_tot,qubit,leave_first_index = not first_gate)
	if output_gradient and first_gate:
		U_tot = [U_tot]
	if output_gradient and array_of_Us:
		U_tot = np.concatenate((U_tot,dU_tot),axis=0)
	elif output_gradient and not array_of_Us:
		U_tot = np.append(U_tot,[dU_tot],axis=0)
		
	return U_tot



def cost_fn(theta,l,expected_unitary,output_gradient=True,free_global_phase=True):
	n = int(np.log2(len(expected_unitary)))
	U_tot = ansatz_U(n,l,theta,output_gradient=output_gradient)
	# first reshape expected_unitary
	expected_unitary = np.reshape(expected_unitary,2*np.ones(2*n,dtype=np.int),order='C')
	# now compute cost function.
	if free_global_phase:
		phase = theta[-1]
	else:
		phase = 0
	C = -np.real( np.exp(-1j*phase)* np.tensordot( U_tot, expected_unitary.conj(), axes=(np.arange(2*n)+output_gradient,np.arange(2*n))))
	if free_global_phase and output_gradient:
		C = np.append(C,  -np.real(-1j * np.exp(-1j*phase)* np.tensordot( U_tot[0,:], expected_unitary.conj(), axes=(np.arange(2*n),np.arange(2*n)))) )
	if output_gradient:
		return (C[0], C[1:])
	else:
		return C
		


def cost_fn_google(theta,l,expected_unitary):
	from cirq.protocols import trace_distance_from_angle_list
	n = int(np.log2(len(expected_unitary)))
	response_unitary = ansatz_U(n,l,theta,output_gradient=False,return_as_matrix=True)
	u = response_unitary @ expected_unitary.conj().T
	return trace_distance_from_angle_list(np.angle(np.linalg.eigvals(u)))

def halt_google(xk, state):
	return state.fun < 1e-4

def optimize(l,expected_unitary,method='BFGS',jac=True,free_global_phase=True):
	n = int(np.log2(len(expected_unitary)))
	theta0 = np.random.rand(num_parameters(n,l)+free_global_phase)
	return minimize(cost_fn,theta0,args=(l,expected_unitary),method=method,jac=jac)



def optimization_routine(expected_unitary,method='BFGS'):
    import time
    t_start = time.time()
    n = int(np.log2(len(expected_unitary)))
    if n == 1:
        from cirq.ops import PhasedXZGate
        G = PhasedXZGate.from_matrix(expected_unitary)
        theta = [G.x_exponent, G.z_exponent, G.axis_phase_exponent]
        return [0,theta]
    elif n == 2:
        l = 3
    elif n == 3:
        l = 14
    elif n == 4:
        l = 31
    elif n > 4:
        l = 100
    finished = False
    while not finished:
        print('starting l = '+str(l)+' optimization...')
        opt = optimize(l,expected_unitary,method=method)
        print('our cost function: '+str(opt.fun))
        trace_distance = cost_fn_google( opt.x, l, expected_unitary )
        print('Google cost function: '+str( trace_distance ) )
        finished = trace_distance < 1e-4
        
        if not finished and trace_distance < 1.06e-4:
            # try optimizing using Google's cost function:
            print('Starting from this point using Google cost function')
            opt = minimize(cost_fn_google,opt.x,args=(l,expected_unitary),method='Nelder-Mead',callback=halt_google)
            trace_distance = opt.fun
            print('Google cost function: '+str( trace_distance ) )
            finished = trace_distance < 1e-4
        
        if not finished:
            l += 1
    t_end = time.time()
    print('succeeded at l = '+str(l)+' layers with trace distance of '+str(trace_distance))
    print('time elapsed = '+str(t_end - t_start)+' seconds')
    return [l,opt.x]




def num_parameters(n,l):
	params_per_rot = 3
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