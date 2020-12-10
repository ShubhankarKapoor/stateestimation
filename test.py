from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian
from path_to_nodes import path_to_nodes
import pandas as pd
from some_funcs import error_calc, create_mes_set

which = 37 # IEEE 37-node or IEEE 906-node

if which == 37:
    from Network37 import *
else:
    from Network906 import *

data_lin = 1
data_full_ac = 0
est_lin = 1
est_full_ac = 0
comparison = 0

if data_lin == 1:
    [V, V_mag, P_line, Q_line, S_line, e_max, k] = LinDistFlowBackwardForwardSweep(P_Load, Q_Load, which)

if data_full_ac == 1:
    [V_mag, V_ang, _, S_line, I_line, I_load, e_max, k] = BackwardForwardSweep(
                                                                P_Load, Q_Load, which)
    Vsq =  {key:val**2 for key, val in V_mag.items()} # square of V_mag
    V = Vsq
    
    # when running full network
    P_line = {key:val.real for key, val in S_line.items()} # resistance of every line
    Q_line = {key:val.imag for key, val in S_line.items()} # reactancce of every line

# below is state estimation

# ground truth
gt_P_load = list(P_Load.values())
gt_Q_load = list(Q_Load.values())
gt_V = V[0]
x = np.asarray(gt_P_load + gt_Q_load) # ground truth for states
x = np.insert(x, len(x), gt_V) # ground truth for states

# states = ['P_Load', 'Q_Load', 'V0']
# estimate_states(states)

# true measurement vector
z_true = np.asarray(list(P_line.values()) + list(Q_line.values()) + 
                    list(P_Load.values()) + list(Q_Load.values()) + list(V.values())) # ground truth for meas

sd = 0.04 # 0.01: 1% error
# add noise to measurement set
mu, sigma = 0, 0.04 # mean and standatd deviation
noise = np.random.normal(mu, sigma, len(z_true))
# noise = 0
z = z_true + noise # noisy data

meas_P_line, meas_Q_line, meas_P_load, meas_Q_load, meas_V  = P_line, Q_line, P_Load, Q_Load, V
'''
# reading measurement sets from csv files
filename_bus = 'data/r1_bus_meas.csv'
filename_branch = 'data/r1_branch_meas.csv'
meas_P_line, meas_Q_line, meas_P_load, meas_Q_load, meas_V = create_mes_set(
    filename_bus, filename_branch)

z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
               list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # ground truth for meas
'''

'''
# get measurement vectors from csv files
f1 = pd.read_csv('data/mm_branch_pq_noisy.csv')
meas_P_line = f1['p_from_mw']*10
meas_Q_line = f1['q_from_mvar']*10

f2 = pd.read_csv('data/mm_bus_pq_noisy.csv')
meas_P_load = f2['p_mw']*10
meas_Q_load = f2['q_mvar']*10

f3 = pd.read_csv('data/mm_bus_v_noisy.csv')
meas_V = f3['vm_pu']**2
z = np.concatenate((meas_P_line, meas_Q_line, meas_P_load, meas_Q_load, meas_V)) # ground truth for meas
'''
# call the function for reading measurements from csv
w1, w2, w3 = 0.04, 0.03, 0.01
# w1, w2, w3 = 0.01, 0.01, 0.01
w1, w2, w3 = 0.005, 0.0001, 0.001
w1, w2, w3 = 0.05, 0.01, 0.001
# lower the sd here more the trust in the measurement
weight_array1 = np.ones((len(meas_P_line)*2))*w1
weight_array2 = np.ones((len(meas_P_load)*2))*w2
weight_array3 = np.ones((len(meas_V)))*w3
weight_array = np.concatenate((weight_array1, weight_array2,weight_array3))
# check below
# weight_array[-1]=0.01 # 
# weight_array = np.insert(weight_array, len(weight_array), 0.00001)
W = np.diag(weight_array) # Weight mat
W = np.linalg.inv(W)

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

q_distributed = Q_line[(0,1)]/(len(P_Load_state))
q_states = np.zeros((len(P_Load_state))) + q_distributed

v0 = 1 # slack bus

x_est = np.concatenate((p_states, q_states))
x_est = np.insert(x_est, len(x_est), v0) # initialized state vars

# get paths from slack bus to all nodes
path_to_all_nodes = path_to_nodes(which)

# get jacobain matrix
# we arent using the values of P_line, P_Load_state or P_Load in jacobian_calc
# only their keys
jacobian_matrix = create_jacobian(P_line, P_Load_state, P_Load, path_to_all_nodes,
                                  V, R_line, X_line, len(x_est), len(z))

# jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
#                                   meas_V, R_line, X_line, len(x_est), len(z))

# some preprocessing for time saving during iterative newton method
G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
Ginv = np.linalg.inv(G)

count = 0
delta_mat = np.zeros((len(x_est), 1000))
residuals_mat = np.zeros((len(z), 1000))
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
    residuals_mat[:,count] = residuals
    
    # calculate deltax
    deltax = np.matmul(np.matmul(np.matmul(Ginv, jacobian_matrix.T), W), residuals)
    delta_mat[:,count] = deltax
    
    # get tolerance
    emax = np.max(deltax)
    
    # update values of state vars
    x_est = x_est + deltax
    results = np.vstack((results, x_est))
    count+=1
    print(count)

# get the full vector for xest
full_x_est = np.zeros((len(x)))
full_x_est[non_zib_index] = x_est[0:len(non_zib_index)] # insert p vals
full_x_est[len(P_Load)+np.asarray(non_zib_index)] = x_est[len(non_zib_index):2*len(non_zib_index)] # insert q vals
full_x_est[-1] = x_est[-1] # slack bus square voltage

# calculate error between state vectors
error = x - full_x_est
max_error = np.max(abs(error))
st_err_p, mean_error_st_p, max_error_st_p, max_error_st_abs_p = error_calc(x[0:len(P_Load)], full_x_est[0:len(P_Load)])
st_err_q, mean_error_st_q, max_error_st_q, max_error_st_abs_q = error_calc(x[len(P_Load):2*len(P_Load)], full_x_est[len(P_Load):2*len(P_Load)])

# print some results
print(w1, w2, w3)
print(mean_error_st_p, max_error_st_p, mean_error_st_q, max_error_st_q)
# sum of residuals
sum_residuals = np.sum(abs(residuals))
results = results.T

###############################################################################
###############################################################################

# Regenerated measurements using the estimated states
keys = list(range(len(P_Load)))
array = full_x_est[0:len(P_Load)]
P_Load_est = dict(zip(keys, array))
Q_Load_est = dict(zip(keys, full_x_est[len(P_Load):len(P_Load)*2]))

if est_lin == 1:
    [V_con, V_mag_con ,P_line_con, Q_line_con, _, e_max_con, k_con] = LinDistFlowBackwardForwardSweep(
        P_Load_est, Q_Load_est, which, full_x_est[-1]) # using lindistflow

# using Full AC Network
if est_full_ac == 1:
    [V_mag_con,_,_,S_line_con,_,_,e_max,k] = BackwardForwardSweep(P_Load_est,
            Q_Load_est, which, full_x_est[-1])
    Vsq_con =  {key:val**2 for key, val in V_mag_con.items()} # square of V_mag
    V_con = Vsq_con
    
    # when running full network
    P_line_con = {key:val.real for key, val in S_line_con.items()} # resistance of every line
    Q_line_con = {key:val.imag for key, val in S_line_con.items()} # reactancce of every line

# error calc between measurements
# V_mag^2 and V_mag error
_, mean_vsq_err, max_vsq_err, max_abs_vsq_err = error_calc(np.array(list(V.values())), np.array(list(V_con.values())))
_, mean_vmag_err, max_vmag_err, max_abs_vmag_err = error_calc(np.array(list(V_mag.values())), np.array(list(V_mag_con.values())))

# pflow and qflow error
_, mean_pflow_err, max_pflow_err, max_abs_pflow_err = error_calc(np.array(list(P_line.values())), np.array(list(P_line_con.values())))
_, mean_qflow_err, max_qflow_err, max_abs_qflow_err = error_calc(np.array(list(Q_line.values())), np.array(list(Q_line_con.values())))

# error calc between lindistflow and full AC
if comparison == 1:
    p_err, p_mean_err, p_max_err, p_max_err_abs = error_calc(np.array(list(P_line2.values())), np.array(list(P_line.values())))
    q_err, q_mean_err, q_max_err, q_max_err_abs = error_calc(np.array(list(Q_line2.values())), np.array(list(Q_line.values())))
    
# some manipulation for excel sheet
aa=[]
aa = [a for a in Q_line_con.values()]