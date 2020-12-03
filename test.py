from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
import numpy as np
which = 37 # IEEE 37-node or IEEE 906-node

if which == 37:
    from Network37 import *
else:
    from Network906 import *    

# [V, P_line, Q_line, e_max, k] =LinDistFlowBackwardForwardSweep(P_Load,Q_Load,which)

[V_mag,V_ang,Voltage,S_line,I_line,I_load,e_max,k] =BackwardForwardSweep(P_Load,Q_Load,which)
V = V_mag

# when running full network
P_line = {key:val.real for key, val in S_line.items()} # resistance of every line
Q_line = {key:val.imag for key, val in S_line.items()} # reactancce of every line


# below is state estimation

# get jacobain matrix

# ground truth
gt_P_load = list(P_Load.values())
gt_Q_load = list(Q_Load.values())
gt_V = V[0]
x = np.asarray(gt_P_load + gt_Q_load) # ground truth for states
x = np.insert(x, len(x), gt_V) # ground truth for states

# states = ['P_Load', 'Q_Load', 'V0']
# estimate_states(states)



# measurement vector
meas_P_line = list(P_line.values())
meas_Q_line = list(Q_line.values())
meas_P_load = list(P_Load.values())
meas_Q_load = list(Q_Load.values())
meas_V = list(V.values())
z = np.asarray(meas_P_line + meas_Q_line + meas_P_load + meas_Q_load + meas_V) # ground truth for meas
weight_array = np.ones((len(z)))*0.04
# weight_array = np.insert(weight_array, len(weight_array), 0.00001)
W = np.diag(weight_array) # Weight mat
W = np.linalg.inv(W)

# initialze state vars
# remove p0 = 0 and the rest have values equally divided from p_ij
p_distributed = P_line[(0,1)]/(len(P_Load) - 1)
p_states = np.zeros((len(P_Load)-1)) + p_distributed
p_states = np.insert(p_states, 0, 0) # insert 0 at loc 0

q_distributed = Q_line[(0,1)]/(len(Q_Load) - 1)
q_states = np.zeros((len(Q_Load)-1)) + q_distributed
q_states = np.insert(q_states, 0, 0) # insert 0 at loc 0

v0 = 1 # slack bus

x_est = np.concatenate((p_states, q_states))
x_est = np.insert(x_est, len(x_est), v0) # initialized state vars

# some preprocessing for
G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
Ginv = np.linalg.inv(G)

count = 0
delta_mat = np.zeros((len(x), 100))
tol = 10e-12
results = x_est
emax = 100 # chosen higher than the tol

while emax > tol:

    # distflow backward sweep for calculating measurements

    # distflow forward sweep for calculating measurements

    # calculate h(x)    
    hx = np.matmul(jacobian_matrix, x_est)
    
    # calculate measurement residuals
    residuals = z - hx
    
    # calculate deltax
    deltax = np.matmul(np.matmul(np.matmul(Ginv, jacobian_matrix.T), W), residuals)
    delta_mat[:,count] = deltax
    
    # get tolerance
    emax = np.max(deltax)
    
    # update values of state vars
    x_est = x_est + deltax
    results = np.vstack((results, x_est))
    count+=1
    print(count-1)
    
# calculate error between state vectors
error = x - x_est
max_error = np.max(abs(error))
# sum of residuals
sum_residuals = np.sum(abs(residuals))
results = results.T
