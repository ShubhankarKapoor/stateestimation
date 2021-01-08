from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian, se_wls, se_ols, se_wrr, se_rr
from path_to_nodes import path_to_nodes
import pandas as pd
from itertools import combinations
import seaborn
from some_funcs import error_calc, create_mes_set, subset_of_measurements, \
                       weight_vals, noise_addition, bus_measurements_equal_distribution

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

# states = ['P_Load', 'Q_Load', 'V0']
# estimate_states(states)

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
indices = np.asarray(combs[9])

# [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
# [0,   1,  2,  3,  4,  5,  6,  7,  8,  9]
# indices = np.asarray((0,  1,  2,  3,  4,  5,  6,  7,  8,  9)) # [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
if len(indices) !=0:
    corresponding_nodes = non_zib_index_array[indices]
else:
    corresponding_nodes = np.asarray(())
    
not_considered = np.setdiff1d(non_zib_index_array, corresponding_nodes)
not_considered_indices = np.setdiff1d(arr, indices)

P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas, meas_V =  bus_measurements_equal_distribution(
    P_Load, Q_Load, V, P_line[(0,1)], Q_line[(0,1)], 
    non_zib_index, zib_index, num_known_meas=len(indices), indices = indices)    

meas_P_load = {**P_known_meas, **P_pseudo_meas}
meas_P_load = dict(sorted(meas_P_load.items()))
meas_Q_load = {**Q_known_meas, **Q_pseudo_meas}
meas_Q_load = dict(sorted(meas_Q_load.items()))

z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
               list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set

# noise addition
sd = 0 # 0.01: 1% error
z = noise_addition(z, sd)

'''
##############################################################################
##############################################################################
# run different combinations of pseudo measurements with equally distrbuted 
# load among pseudo measurements

# saving different errors
list_of_errors_p, list_of_errors_q, list_of_errors_v = [], [], []
list_of_all_errors_p, list_of_all_errors_q = [], []
list_all_error_known_p, list_all_error_known_q = [], [] 
list_all_error_unknown_p, list_all_error_unknown_q = [], [] 

# save all estimate results and combinations of different meas
store_estimates, list_of_all_combs = [], []
list_max_error_index_p = []
arr = np.arange(len(non_zib_index)) # used for combinations
# use the below arr2 for considering all 10 meas as well for i
# arr2 = np.arange(len(non_zib_index)+1) # used for combinations
for i in arr: # i are number of known measurements
    err_for_diff_known_meas_p, err_for_diff_known_meas_q, err_for_diff_known_meas_v = [], [], []
    all_err_for_diff_known_meas_p, all_err_for_diff_known_meas_q = [], []
    itermediate_results, max_abs_error_index_p = [], []
    error_for_known_p, error_for_unknown_p = [], []
    error_for_known_q, error_for_unknown_q = [], []
    print('known meas implementation:', i)
    combs = list(combinations(arr,i)) # combinations for i
    list_of_all_combs.append(combs)

    for indices in combs:
        P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas, meas_V =  bus_measurements_equal_distribution(
                P_Load, Q_Load, V, P_line[(0,1)], Q_line[(0,1)], 
                non_zib_index, zib_index, indices = np.asarray(indices))
            
        meas_P_load = {**P_known_meas, **P_pseudo_meas}
        meas_P_load = dict(sorted(meas_P_load.items()))
        meas_Q_load = {**Q_known_meas, **Q_pseudo_meas}
        meas_Q_load = dict(sorted(meas_Q_load.items()))

        z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
               list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set
    

        w1 = 1 # weight value for pflow, qflow
        w21 = 1 # known measurements for p,q at buses
        w22 = 1000000 # pseudo measurements for p,q at buses
        w3 = 0.0001 # weight for voltage value
        # print1(w1, w21, w22, w3)
        
        weight_array1 = np.ones((len(meas_P_line)*2))*w1
        weight_array2 = np.ones((len(meas_P_load)))
        weight_array2[list(P_known_meas.keys())] = weight_array2[list(P_known_meas.keys())]*w21
        weight_array2[list(P_pseudo_meas.keys())] = weight_array2[list(P_pseudo_meas.keys())]*w22
        weight_array2 = np.concatenate((weight_array2, weight_array2))
        weight_array3 = np.ones((len(meas_V)))*w3
        weight_array = np.concatenate((weight_array1, weight_array2,weight_array3))

        W = np.diag(weight_array) # Weight mat
        W = np.linalg.inv(W)
        
        # get paths from slack bus to all nodes
        path_to_all_nodes = path_to_nodes(which)
        
        # get jacobain matrix
        jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
                                          meas_V, R_line, X_line, len(x_est), len(z))
        
        # run WLS SE
        x_est, emax, count, residuals_mat, delta_mat, results = se_wls(
            x_est, z, jacobian_matrix, W, tol = None)        

        # get the full vector for xest
        full_x_est = np.zeros((len(x)))
        full_x_est[non_zib_index] = x_est[0:len(non_zib_index)] # insert p vals
        full_x_est[len(P_Load)+np.asarray(non_zib_index)] = x_est[len(non_zib_index):2*len(non_zib_index)] # insert q vals
        full_x_est[-1] = x_est[-1] # slack bus square voltage

        # Regenerated measurements using the estimated states
        keys = list(range(len(P_Load)))
        array = full_x_est[0:len(P_Load)]
        P_Load_est = dict(zip(keys, array))
        Q_Load_est = dict(zip(keys, full_x_est[len(P_Load):len(P_Load)*2]))
        [V_con, V_mag_con ,P_line_con, Q_line_con, _, e_max_con, k_con] = LinDistFlowBackwardForwardSweep(
        P_Load_est, Q_Load_est, which, full_x_est[-1]) # using lindistflow

        # calculate error between state vectors
        st_err_p, mean_error_st_p, max_error_st_p, max_error_st_abs_p, max_index_p = error_calc(x[0:len(P_Load)], full_x_est[0:len(P_Load)])
        st_err_q, mean_error_st_q, max_error_st_q, max_error_st_abs_q, _ = error_calc(x[len(P_Load):2*len(P_Load)], full_x_est[len(P_Load):2*len(P_Load)])
        # error for voltage meas
        _, mean_vmag_err, max_vmag_err, max_abs_vmag_err, _ = error_calc(np.array(list(V_mag.values())), np.array(list(V_mag_con.values())))
        # append the absolute error
        err_for_diff_known_meas_p.append(max_error_st_abs_p)
        err_for_diff_known_meas_q.append(max_error_st_abs_q)
        # err_for_diff_known_meas_v.append(abs(full_x_est[-1] - x[-1]))
        err_for_diff_known_meas_v.append(max_abs_vmag_err)
        itermediate_results.append(full_x_est)
        max_abs_error_index_p.append(max_index_p)

        # plot all the errors as well
        all_err_for_diff_known_meas_p.extend(st_err_p)
        all_err_for_diff_known_meas_q.extend(st_err_q)

        # get known and unknown errors separately
        known_indices = np.asarray(list(P_known_meas.keys()))
        unknown_indices = np.asarray(list(P_pseudo_meas.keys ()))
        error_for_known_p.extend(st_err_p[known_indices]) # known p erorrs
        error_for_known_q.extend(st_err_q[known_indices]) # known q erorrs
        error_for_unknown_p.extend(st_err_p[unknown_indices]) # unknown p errors
        error_for_unknown_q.extend(st_err_q[unknown_indices]) # unknown q errors

    # max abs error
    list_of_errors_p.append(err_for_diff_known_meas_p)
    list_of_errors_q.append(err_for_diff_known_meas_q)
    list_of_errors_v.append(err_for_diff_known_meas_v)
    # all errors
    list_of_all_errors_p.append(all_err_for_diff_known_meas_p)
    list_of_all_errors_q.append(all_err_for_diff_known_meas_q)
    store_estimates.append(itermediate_results)
    list_max_error_index_p.append(max_abs_error_index_p)
    # known and unknown errors
    list_all_error_known_p.append(error_for_known_p)
    list_all_error_known_q.append(error_for_known_q)
    list_all_error_unknown_p.append(error_for_unknown_p)
    list_all_error_unknown_q.append(error_for_unknown_q)

# plot the max error graph
plt.figure()
seaborn.boxplot(data=list_of_errors_p)
seaborn.swarmplot(data=list_of_errors_p, color=".25")
plt.xlabel('Known number of measurements')
plt.ylabel('Max absolute error in pu')
plt.title('Max Absolute Error Corresponding to known number of Measurements')

# plot all error graph
plt.figure()
seaborn.boxplot(data=list_of_all_errors_p)
# seaborn.swarmplot(data=list_of_all_errors_p, color=".25") 
plt.xlabel('Known number of measurements')
plt.ylabel(' absolute error in pu')
plt.title('All Absolute Errors Corresponding to known number of Measurements')

# plot known errors
plt.figure()
seaborn.boxplot(data=list_all_error_known_p)
seaborn.swarmplot(data=list_of_errors_p, color=".25")
plt.xlabel('Known number of measurements')
plt.ylabel('absolute error in pu')
plt.title('Absolute Error Corresponding to known Measurements Buses')

# plot known errors
plt.figure()
seaborn.boxplot(data=list_all_error_unknown_p)
seaborn.swarmplot(data=list_all_error_unknown_p, color=".25")
plt.xlabel('Known number of measurements')
plt.ylabel('absolute error in pu')
plt.title('Absolute Error Corresponding to unknown Measurement Buses')
# check if max error is at pseudo buses
# count = 0 
print('checking if error nodes are known measurement nodes or not')
for i,val in enumerate(list_max_error_index_p):
    count+=1
    for j, val2 in enumerate(val):
        # print(i,j)
        if i !=0: # for avoiding empty set error
            # have to do the following becasue list_of_all_combs contains
            # indices for chosing the non zibs values but not the actual
            # non zib values
            known_meas_idx = non_zib_index_array[np.asarray(list_of_all_combs[i][j])]
            if (np.any(np.in1d(val2, known_meas_idx))): # if any val2 is in known meas
                # if the error is known bus meas then something is wrong    
                print('Something wrong')
            
##############################################################################
##############################################################################
'''
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

# static weights but different for pseudo and known measurements
w1 = 1 # weight value for pflow, qflow
w21 = 1 # known measurements for p,q at buses
w22 = 1000000 # pseudo measurements for p,q at buses
w3 = 0.0001 # weight for voltage value
print(w1, w21, w22, w3)

weight_array1 = np.ones((len(meas_P_line)*2))*w1
weight_array2 = np.ones((len(meas_P_load)))
weight_array2[list(P_known_meas.keys())] = weight_array2[list(P_known_meas.keys())]*w21
weight_array2[list(P_pseudo_meas.keys())] = weight_array2[list(P_pseudo_meas.keys())]*w22
weight_array2 = np.concatenate((weight_array2, weight_array2))
weight_array3 = np.ones((len(meas_V)))*w3
weight_array = np.concatenate((weight_array1, weight_array2,weight_array3))

# check below
# weight_array[-1]=0.01 # 
# weight_array = np.insert(weight_array, len(weight_array), 0.00001)
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

p_distributed = P_line[(0,1)]/(len(P_Load_state))
p_states = np.zeros((len(P_Load_state))) + p_distributed

q_distributed = Q_line[(0,1)]/(len(P_Load_state))
q_states = np.zeros((len(P_Load_state))) + q_distributed

v0 = 1 # slack bus

# weight matrix on estimates from RR
W_rr = np.ones((len(x_est))) * w21 # weights on know p, q bus meas
W_rr[not_considered_indices] = w22 # weights on unknown p_buses
W_rr[not_considered_indices + len(non_zib_index)] = w22 # weights on unknown q_buses
W_rr[-1] = w3

x_est = np.concatenate((p_states, q_states))
x_est = np.insert(x_est, len(x_est), v0) # initialized state vars
x_est, emax, count, residuals_mat, delta_mat, results = se_wrr(
    x_est, z, jacobian_matrix, W, k=1)

##############################################################################
##############################################################################
# Error Calculations

# get the full vector for xest
full_x_est = np.zeros((len(x)))
full_x_est[non_zib_index] = x_est[0:len(non_zib_index)] # insert p vals
full_x_est[len(P_Load)+np.asarray(non_zib_index)] = x_est[len(non_zib_index):2*len(non_zib_index)] # insert q vals
full_x_est[-1] = x_est[-1] # slack bus square voltage

print(x[not_considered], full_x_est[not_considered])
print(x[not_considered+37], full_x_est[not_considered+37])

# calculate error between state vectors
st_err_p, mean_error_st_p, max_error_st_p, max_error_st_abs_p, _ = error_calc(x[0:len(P_Load)], full_x_est[0:len(P_Load)])
st_err_q, mean_error_st_q, max_error_st_q, max_error_st_abs_q, _ = error_calc(x[len(P_Load):2*len(P_Load)], full_x_est[len(P_Load):2*len(P_Load)])

# print some results
print(mean_error_st_p, max_error_st_p, max_error_st_abs_p) 
print(mean_error_st_q, max_error_st_q, max_error_st_abs_q)
# sum of residuals
sum_residuals = np.sum(abs(residuals_mat[:,count-1]))
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
_, mean_vsq_err, max_vsq_err, max_abs_vsq_err, _ = error_calc(np.array(list(V.values())), np.array(list(V_con.values())))
_, mean_vmag_err, max_vmag_err, max_abs_vmag_err, _ = error_calc(np.array(list(V_mag.values())), np.array(list(V_mag_con.values())))
print(mean_vmag_err, max_vmag_err, max_abs_vmag_err)

# pflow and qflow error
_, mean_pflow_err, max_pflow_err, max_abs_pflow_err, _ = error_calc(np.array(list(P_line.values())), np.array(list(P_line_con.values())))
_, mean_qflow_err, max_qflow_err, max_abs_qflow_err, _ = error_calc(np.array(list(Q_line.values())), np.array(list(Q_line_con.values())))

# error calc between lindistflow and full AC
if comparison == 1:
    p_err, p_mean_err, p_max_err, p_max_err_abs, _ = error_calc(np.array(list(P_line2.values())), np.array(list(P_line.values())))
    q_err, q_mean_err, q_max_err, q_max_err_abs, _ = error_calc(np.array(list(Q_line2.values())), np.array(list(Q_line.values())))
    
# some manipulation for excel sheet
aa=[]
aa = [a for a in Q_line_con.values()]
