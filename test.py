from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian
from solvers import se_wls, se_ols, se_wrr, se_rr, batch_gradient_descent, \
    stochastic_gradient_descent, stochastic_gradient_descent2, \
    WLeastSquaresRegressorTorch, cost
from path_to_nodes import path_to_nodes
import pandas as pd
from itertools import combinations
import seaborn, time
from some_funcs import error_calc, create_mes_set, subset_of_measurements, \
                       weight_vals, noise_addition, bus_measurements_equal_distribution, \
                       error_calc_refactor    
import torch
import matplotlib.pyplot as plt
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

# ground truth
gt_P_load = list(P_Load.values())
gt_Q_load = list(Q_Load.values())
gt_V = V[0]
x = np.asarray(gt_P_load + gt_Q_load) # ground truth for states
x = np.insert(x, len(x), gt_V) # ground truth for states

# ground truth for measurements
z_true = np.asarray(list(P_line.values()) + list(Q_line.values()) + 
                    list(P_Load.values()) + list(Q_Load.values()) + list(V.values())) # ground truth for meas

##############################################################################
##############################################################################

# initialze state vars
# consider the state vars only for non ZIBs
P_Load_state = {}
zib_index, non_zib_index = [], [] # index of zibs and non zibs
for k,v in P_Load.items():
    if v != 0:
        P_Load_state[k] = v
        non_zib_index.append(k)
    else:
        # P_Load_state[k] = v # included to consider all nodes
        # non_zib_index.append(k) # included to consider all nodes
        zib_index.append(k)

non_zib_index_array = np.asarray(non_zib_index)
# remove p0 = 0 and the rest have values equally divided from p_ij
p_distributed = P_line[(0,1)]/(len(P_Load_state))
p_states = np.zeros((len(P_Load_state))) + p_distributed

q_distributed = Q_line[(0,1)]/(len(P_Load_state))
q_states = np.zeros((len(P_Load_state))) + q_distributed
v0 = 1 # slack bus

x_est = np.concatenate((p_states, q_states))
x_est = np.insert(x_est, len(x_est), v0) # initialized state vars

# random initialization of state vars instead of above
torch.manual_seed(0)
x_est = torch.rand(len(x_est)) # so that the initial condn is same as pytorch
x_est =  x_est.detach().cpu().numpy()

x_true = np.concatenate((x[non_zib_index], x[non_zib_index_array + len(gt_P_load)]))
x_true = np.insert(x_true, len(x_true), gt_V) # ground truth for states
##############################################################################
##############################################################################

# if any time you want to consider states other than nonzib states
# modify non_zib_index
# I think should work: needs testing
##############################################################################
##############################################################################

# get subset of lineflow measurement set
num_plow_meas = 1
num_voltage_meas = 1
# chose lineflows
meas_P_line, meas_Q_line = subset_of_measurements(
    num_plow_meas, arcs, P_line, Q_line, V)

# different combinations of known nodes
i = 9
arr = np.arange(len(non_zib_index)) # used for combinations
combs = list(combinations(arr,i))
# chosing bus powers
# indices = np.array(np.arange(5))
indices = np.asarray(combs[7])

# 37
# [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
# [0,   1,  2,  3,  4,  5,  6,  7,  8,  9]

# indices = np.asarray(()) # [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]

# 906
# [ 34, 70, 73, 74, 225, 289, 349, 387, 388, 502, 562, 563, 611, 629, 817, 860, 861, 896, 898, 900, 906]
# [ 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20]
  # [     1,      3,  4,   5,        7,   8,   9,             12,  13,  14,       16,            19,  20]

# indices = np.asarray((0,     4,  5,  6,  7,  )) # [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
# indices = np.asarray((0,  1,  2,  3,  4,   5,   6,   7,   8,   9,   10,  11,  
#                       12,  13,  14,  15,  16, 17, 18, 19, 20))

# see the known meas
if len(indices) !=0:
    corresponding_nodes = non_zib_index_array[indices]
else:
    corresponding_nodes = np.asarray(())
# unknown buses    
not_considered = np.setdiff1d(non_zib_index_array, corresponding_nodes)
not_considered_indices = np.setdiff1d(arr, indices)

P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas, meas_V =  bus_measurements_equal_distribution(
    P_Load, Q_Load, V, P_line[(0,1)], Q_line[(0,1)], 
    non_zib_index, zib_index, num_known_meas=len(indices), indices = indices)    

meas_P_load = {**P_known_meas, **P_pseudo_meas}
meas_P_load = dict(sorted(meas_P_load.items()))
meas_Q_load = {**Q_known_meas, **Q_pseudo_meas}
meas_Q_load = dict(sorted(meas_Q_load.items()))
# meas_V = {key: V[key] for key in not_considered}
# # del meas_V[34]
# meas_V[0]=1
# meas_V = dict(sorted(meas_V.items()))

z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
               list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set

# noise addition
sd = 0 # 0.01: 1% error
z = noise_addition(z, sd)

# static weights but different for pseudo and known measurements
w1 = 1 # weight value for pflow, qflow
w21 = 1 # known measurements for p,q at buses
w22 = 1000000 #100000000000 # pseudo measurements for p,q at buses
w3 = 0.1 # weight for voltage value; use 0.1 for grad descent & 0.0001 for WLS
print(w1, w21, w22, w3)

weight_array1 = np.ones((len(meas_P_line)*2))*w1 # for pline and qline
weight_array2 = np.ones((len(meas_P_load)))
weight_array2[list(P_known_meas.keys())] = weight_array2[list(P_known_meas.keys())]*w21 # for known p meas
weight_array2[list(P_pseudo_meas.keys())] = weight_array2[list(P_pseudo_meas.keys())]*w22# for unknown p meas
weight_array2 = np.concatenate((weight_array2, weight_array2)) # for p and q bus
weight_array3 = np.ones((len(meas_V)))*w3 # for v mag
weight_array = np.concatenate((weight_array1, weight_array2,weight_array3)) # entire weight vector

W = np.diag(weight_array) # Weight mat
W = np.linalg.inv(W)

##############################################################################
##############################################################################
# state estimation
# get paths from slack bus to all nodes
path_to_all_nodes = path_to_nodes(which)

# get jacobain matrix
# we arent using the values of P_line, P_Load_state or P_Load in jacobian_calc
# only their keys

jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
                                  meas_V, R_line, X_line, len(x_est), len(z))

# run WLS/OLS SE
k_range = np.arange(1,1.6,0.1)
# for coeff in k_range:

# weight matrix on estimates from RR
W_rr = np.ones((len(x_est))) * w21 # weights on know p, q bus meas
W_rr[not_considered_indices] = w22 # weights on unknown p_buses
W_rr[not_considered_indices + len(non_zib_index)] = w22 # weights on unknown q_buses
W_rr[-1] = w3

x_estn, emax, count, residuals_mat, delta_mat, results = se_wls(
    x_est, z, jacobian_matrix, W)
costsn = cost(x_estn, jacobian_matrix, z, W)
##############################################################################
sum_residuals = np.sum(abs(residuals_mat[:,count-1]))
results = results.T

##############################################################################
##############################################################################
# Running the gradient Algorithm
lr, iterations = 0.1, 30000 # Learning Rate and Number of iterations

# x_est=x_estb
# Batch Gradient Descent
# print('Running BGD')
x_estb, thetasb, costsb, countsb, emaxb = batch_gradient_descent(
    jacobian_matrix, z, x_est, W, lr, iterations)

# Running Stochastic Gradient Descent
# start_time = time.time()
# x_estt, thetas, costs, counts = stochastic_gradient_descent(
#     jacobian_matrix, z, x_est, W, lr, iterations)
# print('------First Function Run Time------', time.time() - start_time)

# start_time = time.time()
# x_est2, thetas2, costs2, counts2, emaxs = stochastic_gradient_descent2(
#     jacobian_matrix, z, x_est, W, lr, iterations)
# print('------Second Function Run Time------', time.time() - start_time)
# print('Final Cost/MSE(L2 Loss) Value: {:0.3f}'.format(costs2[-1]))

# test pytorch implementation
# can tune the lr below, dependent on your weights
# can try different n_iters & batch size
print('Running Pytorch Implementation')
# n_iter=countsb: to immitate the resuls from BGD
regr = WLeastSquaresRegressorTorch(n_iter=countsb, eta=lr, batch_size=len(z))
xx, emaxp = regr.fit(jacobian_matrix, z, W, x_est)
plt.figure()
plt.plot(regr.history, '.-') # plot the cost function
plt.plot(costsb, '.-')
print('Final Cost', costsn, costsb[-1], regr.history[-1])
##############################################################################
##############################################################################
# Error Calculations

# the following function is used when the states are non zib buses
error_calc_refactor(x, x_estn, non_zib_index, P_Load, est_lin, est_full_ac, 
                        which, V, V_mag) # for WLS
error_calc_refactor(x, x_estb, non_zib_index, P_Load, est_lin, est_full_ac, 
                        which, V, V_mag) # for self GD
error_calc_refactor(x, xx.detach().numpy(), non_zib_index, P_Load, est_lin, est_full_ac, 
                        which, V, V_mag) # for pytorch GD
##############################################################################
##############################################################################
# error calc between lindistflow and full AC
if comparison == 1:
    p_err, p_mean_err, p_max_err, p_max_err_abs, _ = error_calc(np.array(list(P_line2.values())), np.array(list(P_line.values())))
    q_err, q_mean_err, q_max_err, q_max_err_abs, _ = error_calc(np.array(list(Q_line2.values())), np.array(list(Q_line.values())))

# call the function for reading measurements from csv
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
##############################################################################
##############################################################################

# static weights
# lower the sd here more the trust in the measurement
# w1, w2, w3 = 0.04, 0.03, 0.01
# # w1, w2, w3 = 0.01, 0.01, 0.01
# w1, w2, w3 = 0.005, 0.0001, 0.001
# w1, w2, w3 = 0.05, 0.01, 0.001
# print(w1, w2, w3)
# weight_array1 = np.ones((len(meas_P_line)*2))*w1
# weight_array2 = np.ones((len(meas_P_load)*2))*w2
# weight_array3 = np.ones((len(meas_V)))*w3
# weight_array = np.concatenate((weight_array1, weight_array2,weight_array3))

# # dynamic weights
# # these weights are computed from greedy placement algo
# w1 = weight_vals(meas_P_line, c = 0.005, abs_error = 0.01)
# w2 = weight_vals(meas_Q_line, c = 0.005, abs_error = 0.01)
# w3 = weight_vals(meas_P_load, c = 0.2, abs_error = 0.01)
# w4 = weight_vals(meas_Q_load, c = 0.2, abs_error = 0.01)
# w5 = weight_vals(meas_V, c = 0.001, abs_error = 0.01)
# print(w1, w2, w3, w4, w5)
# weight_array1 = np.ones((len(meas_P_line)))*w1
# weight_array2 = np.ones((len(meas_P_line)))*w2
# weight_array3 = np.ones((len(meas_P_load)))*w3
# weight_array4 = np.ones((len(meas_Q_load)))*w4
# weight_array5 = np.ones((len(meas_V)))*w5
# weight_array = np.concatenate((weight_array1, weight_array2, weight_array3,
#                                 weight_array4, weight_array5))

# some manipulation to copy results in excel sheet
# aa=[]
# aa = [a for a in Q_line_con.values()]
