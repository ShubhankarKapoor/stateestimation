from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian
from path_to_nodes import path_to_nodes

which = 37 # IEEE 37-node or IEEE 906-node

if which == 37:
    from Network37 import *
else:
    from Network906 import *    

[V, Vmag, P_line, Q_line, e_max, k] = LinDistFlowBackwardForwardSweep(P_Load, Q_Load, which)

# [V_mag,V_ang,Voltage,S_line,I_line,I_load,e_max,k] = BackwardForwardSweep(P_Load, Q_Load,which)
# Vsq =  {key:val**2 for key, val in V_mag.items()} # square of V_mag
# V = Vsq

# when running full network
# P_line = {key:val.real for key, val in S_line.items()} # resistance of every line
# Q_line = {key:val.imag for key, val in S_line.items()} # reactancce of every line

# below is state estimation

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
z_true = np.asarray(meas_P_line + meas_Q_line + meas_P_load + meas_Q_load + meas_V) # ground truth for meas

var = 0.04
weight_array = np.ones((len(z_true)))*var
# weight_array = np.insert(weight_array, len(weight_array), 0.00001)
W = np.diag(weight_array) # Weight mat
W = np.linalg.inv(W)

# add noise to measurement set
mu, sigma = 0, 0.01 # mean and standatd deviation
noise = np.random.normal(mu, sigma, len(z_true))
# noise = 0
z = z_true + noise # noisy data

# initialze state vars

# consider the state vars only for non ZIBs
P_Load_state = {}
zib_index, non_zib_index = [], [] # index of zibs and non zibs
for k,v in P_Load.items():
    if v != 0:
        P_Load_state[k] = v
        non_zib_index.append(k)
    else:
        zib_index.append(k)
        
# remove p0 = 0 and the rest have values equally divided from p_ij
p_distributed = P_line[(0,1)]/(len(P_Load_state))
p_states = np.zeros((len(P_Load_state))) + p_distributed
# p_states = np.insert(p_states, 0, 0) # insert 0 at loc 0

q_distributed = Q_line[(0,1)]/(len(P_Load_state))
q_states = np.zeros((len(P_Load_state))) + q_distributed
# q_states = np.insert(q_states, 0, 0) # insert 0 at loc 0

v0 = 1 # slack bus

x_est = np.concatenate((p_states, q_states))
x_est = np.insert(x_est, len(x_est), v0) # initialized state vars

# get paths from slack bus to all nodes
path_to_all_nodes = path_to_nodes(which)

# get jacobain matrix
# jacobian_matrix = create_jacobian(P_line, P_Load_state, P_Load_meas, path_to_all_nodes,
#                                   V, R_line, X_line, len(x_est), len(z))


# we arent using the values of P_line, P_Load_state or P_Load in jacobian_calc
# only their keys
jacobian_matrix = create_jacobian(P_line, P_Load_state, P_Load, path_to_all_nodes,
                                  V, R_line, X_line, len(x_est), len(z))

# some preprocessing for time saving during iterative newton method
G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
Ginv = np.linalg.inv(G)

count = 0
delta_mat = np.zeros((len(x_est), 100000))
tol = 10e-16
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

# get the full vector for xest
full_x_est = np.zeros((len(x)))
full_x_est[non_zib_index] = x_est[0:len(non_zib_index)] # insert p vals
full_x_est[len(P_Load)+np.asarray(non_zib_index)] = x_est[len(non_zib_index):2*len(non_zib_index)] # insert q vals
full_x_est[-1] = x_est[-1]

# calculate error between state vectors
error = x - full_x_est
max_error = np.max(abs(error))

# sum of residuals
sum_residuals = np.sum(abs(residuals))
results = results.T

# Regenerated measurements
# format it for lindistflow func
keys = list(range(len(P_Load)))
array = full_x_est[0:len(P_Load)]
P_Load_est = dict(zip(keys, array))
Q_Load_est = dict(zip(keys, full_x_est[len(P_Load):len(P_Load)*2]))
[V_con, _ ,P_line_con, Q_line_con, e_max_con, k_con] = LinDistFlowBackwardForwardSweep(
    P_Load_est, Q_Load_est, which, full_x_est[-1])

# # error calc between 
# meas_P_line = list(P_line_con.values())
# meas_Q_line = list(Q_line_con.values())
# meas_P_load = list(P_Load.values())
# meas_Q_load = list(Q_Load.values())
# meas_V = list(V_con.values())
# z_true = np.asarray(meas_P_line + meas_Q_line + meas_P_load + meas_Q_load + meas_V) # ground truth for meas

# fix/ come up with error calc
