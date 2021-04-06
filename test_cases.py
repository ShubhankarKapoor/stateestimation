#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:01:17 2021

@author: shub
"""
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian
from solvers import se_wls, se_ols, se_wrr, se_rr, batch_gradient_descent, \
    stochastic_gradient_descent, stochastic_gradient_descent2, \
    WLeastSquaresRegressorTorch, cost
from solvers_with_loss import se_wls_nonlin, se_wls_nonlin_ass, se_wls_la_bgd
from path_to_nodes import path_to_nodes
import pandas as pd
from itertools import combinations
import seaborn, time
from some_funcs import error_calc, create_mes_set, subset_of_measurements, \
                       weight_vals, noise_addition, bus_measurements_equal_distribution, \
                       error_calc_refactor, countour_plot
import torch
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

which = 37 # IEEE 37-node or IEEE 906-node

if which == 37:
    from Network37 import *
else:
    from Network906 import *

# model for measurement set
data_lin = 0
data_full_ac = 1
# model used for reconstruction set
est_lin = 1 # lindisflow or distflow depending on a few more params
est_full_ac = 0
comparison = 0

# masurement set
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

# ground truth for all measurements
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
x_est = torch.rand(len(x_est)).double() # so that the initial condn is same as pytorch
# x_est = torch.ones(len(x_est)).double() # so that the initial condn is same as pytorch
x_est =  x_est.detach().cpu().numpy()

x_true = np.concatenate((x[non_zib_index], x[non_zib_index_array + len(gt_P_load)]))
x_true = np.insert(x_true, len(x_true), gt_V) # ground truth for states
###############################################################################
###############################################################################

# get subset of lineflow measurement set
num_plow_meas = 1
num_voltage_meas = 1
# chose lineflows
meas_P_line, meas_Q_line = subset_of_measurements(
    num_plow_meas, arcs, P_line, Q_line, V)

# get paths from slack bus to all nodes
path_to_all_nodes, path_to_all_nodes_list = path_to_nodes(which)
num_known = [8, 5, 3] # known number of measurements
# number of known measurements
# i = 8
# arr = np.arange(len(non_zib_index)) # used for combinations
# combs = list(combinations(arr,i))

# holds errors for different number of known measurements
ll_no_feed_perc_v, ll_no_feed_perc_p, ll_no_feed_abs_v, ll_no_feed_abs_p = [], [], [], []
ll_v_feed_perc_v, ll_v_feed_perc_p, ll_v_feed_abs_v, ll_v_feed_abs_p = [], [], [], []
ll_p_feed_perc_v, ll_p_feed_perc_p, ll_p_feed_abs_v, ll_p_feed_abs_p = [], [], [], []
ll_both_feed_perc_v, ll_both_feed_perc_p, ll_both_feed_abs_v, ll_both_feed_abs_p = [], [], [], []
ll_la_perc_v, ll_la_perc_p, ll_la_abs_v, ll_la_abs_p = [], [], [], []

for i in num_known:
    arr = np.arange(len(non_zib_index)) # used for combinations
    combs = list(combinations(arr,i))

    # to hold all eroors for all combinations for a fixed num of missing meass 
    l_no_feed_perc_v, l_no_feed_perc_p, l_no_feed_abs_v, l_no_feed_abs_p = [], [], [], []
    l_v_feed_perc_v, l_v_feed_perc_p, l_v_feed_abs_v, l_v_feed_abs_p = [], [], [], []
    l_p_feed_perc_v, l_p_feed_perc_p, l_p_feed_abs_v, l_p_feed_abs_p = [], [], [], []
    l_both_feed_perc_v, l_both_feed_perc_p, l_both_feed_abs_v, l_both_feed_abs_p = [], [], [], []
    l_la_perc_v, l_la_perc_p, l_la_abs_v, l_la_abs_p = [], [], [], []
    
    # different combs for known number of meas
    for indices in combs:
        indices = np.asarray(indices)
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
        
        z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
                       list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set
        
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
        
        jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
                                          meas_V, R_line, X_line, len(x_est), len(z))
        
        # GN-WLS
        lossy_volt_est = {'tot_states':len(x), 'non_zib_index':non_zib_index, 
                          'num_buses':len(P_Load), 'which':which, 'volt_buses': meas_V.keys(),
                          'plines':meas_P_line.keys()}
        
        # to include non linear voltage feedback and pflow/qflow
        loss, pflow = 0, 0
        # LinDist
        x_estn, emax, countsn, residuals_mat, delta_mat, results, costsn = se_wls(
            x_est, z, jacobian_matrix, W, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # costsn = cost(x_estn, jacobian_matrix, z, W)
        print('GN-WLS based on linear jacobian with no feedback/ feedback') 
        perc_v_nofeed, perc_p_nofeed, abs_v_nofeed, abs_p_nofeed = error_calc_refactor(x, x_estn, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = loss, pflow = pflow) # for WLS
        l_no_feed_perc_v.extend(perc_v_nofeed), l_no_feed_perc_p.extend(perc_p_nofeed), 
        l_no_feed_abs_v.extend(abs_v_nofeed), l_no_feed_abs_p.extend(abs_p_nofeed)
        ###############################################################################
        loss, pflow = 1, 0
        # LinDist + Voltage Feedback
        x_estn, emax, countsn, residuals_mat, delta_mat, results, costsn = se_wls(
            x_est, z, jacobian_matrix, W, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # costsn = cost(x_estn, jacobian_matrix, z, W)
        print('GN-WLS based on linear jacobian with no feedback/ feedback')
        perc_v_vfeed, perc_p_vfeed, abs_v_vfeed, abs_p_vfeed = error_calc_refactor(x, x_estn, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = loss, pflow = pflow) # for WLS
        l_v_feed_perc_v.extend(perc_v_vfeed), l_v_feed_perc_p.extend(perc_p_vfeed), 
        l_v_feed_abs_v.extend(abs_v_vfeed), l_v_feed_abs_p.extend(abs_p_vfeed)
        ###############################################################################
        loss, pflow = 0, 1
        # LinDist + Pflow Feedback
        x_estn, emax, countsn, residuals_mat, delta_mat, results, costsn = se_wls(
            x_est, z, jacobian_matrix, W, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # costsn = cost(x_estn, jacobian_matrix, z, W)
        print('GN-WLS based on linear jacobian with no feedback/ feedback')
        perc_v_pfeed, perc_p_pfeed, abs_v_pfeed, abs_p_pfeed = error_calc_refactor(x, x_estn, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = loss, pflow = pflow) # for WLS
        l_p_feed_perc_v.extend(perc_v_pfeed), l_p_feed_perc_p.extend(perc_p_pfeed), 
        l_p_feed_abs_v.extend(abs_v_pfeed), l_p_feed_abs_p.extend(abs_p_pfeed)
        ###############################################################################
        loss, pflow = 1, 1
        # LinDist + Voltage & Pflow Feedback
        x_estn, emax, countsn, residuals_mat, delta_mat, results, costsn = se_wls(
            x_est, z, jacobian_matrix, W, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # costsn = cost(x_estn, jacobian_matrix, z, W)   
        ###############################################################################
        print('Implementing loss based with a few assumptions')
        x_est_la, emax_la, count_la, residuals_mat_la, delta_mat_la, results_la, jacobian_matrix_la = se_wls_nonlin_ass(
            x_est, z, W, meas_P_line, meas_Q_line, P_Load_state, meas_P_load, 
            path_to_all_nodes_list, path_to_all_nodes, non_zib_index, meas_V, R_line, 
            X_line, LineData_Z_pu, len(x_est), len(z), len(x), which)
        
        ###############################################################################
        ###############################################################################
        
        print('GN-WLS based on linear jacobian with no feedback/ feedback')
        perc_v_n, perc_p_n, abs_v_n, abs_p_n = error_calc_refactor(x, x_estn, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = loss, pflow = pflow) # for WLS
        l_both_feed_perc_v.extend(perc_v_n), l_both_feed_perc_p.extend(perc_p_n), 
        l_both_feed_abs_v.extend(abs_v_n), l_both_feed_abs_p.extend(abs_p_n)
        print('GN-WLS based on non-linear with ass')
        perc_v_la, perc_p_la, abs_v_la, abs_p_la = error_calc_refactor(x, x_est_la, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = 1, pflow = 1) # non linear GN with assumption
        l_la_perc_v.extend(perc_v_la), l_la_perc_p.extend(perc_p_la), 
        l_la_abs_v.extend(abs_v_la), l_la_abs_p.extend(abs_p_la)
        
        # break
    ll_no_feed_perc_v.append(l_no_feed_perc_v), ll_no_feed_perc_p.append(l_no_feed_perc_p), 
    ll_no_feed_abs_v.append(l_no_feed_abs_v), ll_no_feed_abs_p.append(l_no_feed_abs_p)
    ll_v_feed_perc_v.append(l_v_feed_perc_v), ll_v_feed_perc_p.append(l_v_feed_perc_p), 
    ll_v_feed_abs_v.append(l_v_feed_abs_v), ll_v_feed_abs_p.append(l_v_feed_abs_p)
    ll_p_feed_perc_v.append(l_p_feed_perc_v), ll_p_feed_perc_p.append(l_p_feed_perc_p), 
    ll_p_feed_abs_v.append(l_p_feed_abs_v), ll_p_feed_abs_p.append(l_p_feed_abs_p)
    ll_both_feed_perc_v.append(l_both_feed_perc_v), ll_both_feed_perc_p.append(l_both_feed_perc_p), 
    ll_both_feed_abs_v.append(l_both_feed_abs_v), ll_both_feed_abs_p.append(l_both_feed_abs_p)
    ll_la_perc_v.append(l_la_perc_v), ll_la_perc_p.append(l_la_perc_p), 
    ll_la_abs_v.append(l_la_abs_v), ll_la_abs_p.append(l_la_abs_p)
# plot histogram

# plot histogram for different methods but same number of missing nodes on 1 graph
# plt.figure()
# bins = np.linspace(0, 2, 500)
# plt.hist(ll_no_feed_perc_v[1], bins, alpha=0.5, label='n')
# plt.hist(ll_v_feed_perc_v[1], bins, alpha=0.5, label='v')
# plt.hist(ll_p_feed_perc_v[1], bins, alpha=0.5, label='p')
# plt.hist(ll_both_feed_perc_v[1], bins, alpha=0.5, label='b')
# plt.hist(ll_la_perc_v[1], bins, alpha=0.5, label='l')
# plt.legend(loc='upper right')
# plt.show()
# plot histogram for same method but different number of missing nodes on 1 graph

# plots for manuscripts
# box and whisker plot for same method with different number of meas(s)
plt.figure()
data = [ll_no_feed_perc_v[0], ll_v_feed_perc_v[0], ll_p_feed_perc_v[0], 
        ll_both_feed_perc_v[0], ll_la_perc_v[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage V Error')
plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_perc_p[0], ll_v_feed_perc_p[0], ll_p_feed_perc_p[0], 
        ll_both_feed_perc_p[0], ll_la_perc_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage P Error')
plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_v[0], ll_v_feed_abs_v[0], ll_p_feed_abs_v[0], 
        ll_both_feed_abs_v[0], ll_la_abs_v[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute V Error')
plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_p[0], ll_v_feed_abs_p[0], ll_p_feed_abs_p[0], 
        ll_both_feed_abs_p[0], ll_la_abs_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute P Error')
plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

###############################################################################
plt.figure()
data = [ll_no_feed_perc_v[1], ll_v_feed_perc_v[1], ll_p_feed_perc_v[1], 
        ll_both_feed_perc_v[1], ll_la_perc_v[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage V Error')
plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_perc_p[1], ll_v_feed_perc_p[1], ll_p_feed_perc_p[1], 
        ll_both_feed_perc_p[1], ll_la_perc_p[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage P Error')
plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_v[1], ll_v_feed_abs_v[1], ll_p_feed_abs_v[1], 
        ll_both_feed_abs_v[1], ll_la_abs_v[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute V Error')
plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_p[1], ll_v_feed_abs_p[1], ll_p_feed_abs_p[1], 
        ll_both_feed_abs_p[1], ll_la_abs_p[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute P Error')
plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

###############################################################################
plt.figure()
data = [ll_no_feed_perc_v[2], ll_v_feed_perc_v[2], ll_p_feed_perc_v[2], 
        ll_both_feed_perc_v[2], ll_la_perc_v[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage V Error')
plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_perc_p[2], ll_v_feed_perc_p[2], ll_p_feed_perc_p[2], 
        ll_both_feed_perc_p[2], ll_la_perc_p[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Percentage P Error')
plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_v[2], ll_v_feed_abs_v[2], ll_p_feed_abs_v[2], 
        ll_both_feed_abs_v[2], ll_la_abs_v[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute V Error')
plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.figure()
data = [ll_no_feed_abs_p[2], ll_v_feed_abs_p[2], ll_p_feed_abs_p[2], 
        ll_both_feed_abs_p[2], ll_la_abs_p[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
plt.ylabel('Different SE Models')
plt.xlabel('Absolute P Error')
plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

###############################################################################
# SUBPLOT FOR 3 CASES #
###############################################################################

plt.figure()
data = [ll_no_feed_perc_v[0], ll_v_feed_perc_v[0], ll_p_feed_perc_v[0], 
        ll_both_feed_perc_v[0], ll_la_perc_v[0]]
plt.subplot(3,4,1)
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(a) Percentage V Error')
# plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,2)
data = [ll_no_feed_perc_p[0], ll_v_feed_perc_p[0], ll_p_feed_perc_p[0], 
        ll_both_feed_perc_p[0], ll_la_perc_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(b) Percentage P Error')
# plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,3)
data = [ll_no_feed_abs_v[0], ll_v_feed_abs_v[0], ll_p_feed_abs_v[0], 
        ll_both_feed_abs_v[0], ll_la_abs_v[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(c) Absolute V Error')
# plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,4)
data = [ll_no_feed_abs_p[0], ll_v_feed_abs_p[0], ll_p_feed_abs_p[0], 
        ll_both_feed_abs_p[0], ll_la_abs_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(d) Absolute P Error')
# plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

###############################################################################
plt.subplot(3,4,5)
data = [ll_no_feed_perc_v[1], ll_v_feed_perc_v[1], ll_p_feed_perc_v[1], 
        ll_both_feed_perc_v[1], ll_la_perc_v[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(e) Percentage V Error')
# plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,6)
data = [ll_no_feed_perc_p[1], ll_v_feed_perc_p[1], ll_p_feed_perc_p[1], 
        ll_both_feed_perc_p[1], ll_la_perc_p[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(f) Percentage P Error')
# plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,7)
data = [ll_no_feed_abs_v[1], ll_v_feed_abs_v[1], ll_p_feed_abs_v[1], 
        ll_both_feed_abs_v[1], ll_la_abs_v[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(g) Absolute V Error')
# plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,8)
data = [ll_no_feed_abs_p[1], ll_v_feed_abs_p[1], ll_p_feed_abs_p[1], 
        ll_both_feed_abs_p[1], ll_la_abs_p[1]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(h) Absolute P Error')
# plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

###############################################################################
plt.subplot(3,4,9)
data = [ll_no_feed_perc_v[2], ll_v_feed_perc_v[2], ll_p_feed_perc_v[2], 
        ll_both_feed_perc_v[2], ll_la_perc_v[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(i) Percentage V Error')
# plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,10)
data = [ll_no_feed_perc_p[2], ll_v_feed_perc_p[2], ll_p_feed_perc_p[2], 
        ll_both_feed_perc_p[2], ll_la_perc_p[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(j) Percentage P Error')
# plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,11)
data = [ll_no_feed_abs_v[2], ll_v_feed_abs_v[2], ll_p_feed_abs_v[2], 
        ll_both_feed_abs_v[2], ll_la_abs_v[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(k) Absolute V Error')
# plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])

plt.subplot(3,4,12)
data = [ll_no_feed_abs_p[2], ll_v_feed_abs_p[2], ll_p_feed_abs_p[2], 
        ll_both_feed_abs_p[2], ll_la_abs_p[2]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(l) Absolute P Error')
# plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2, 3, 4], ['N', 'LV', 'LP', 'LB', 'LA'])
plt.tight_layout()

###############################################################################
# Comparing Results with and without pflow
###############################################################################

plt.figure(100)
# data = [ll_no_feed_perc_v[0],
#         ll_both_feed_perc_v[0], ll_la_perc_v[0]]
# plt.subplot(2,4,1)
# seaborn.boxplot(data=data, orient="h")
# # seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# # plt.ylabel('Different SE Models')
# plt.xlabel('(a) Percentage V Error')
# # plt.title('Percentage V Error for Different SE Models')
# plt.yticks([0, 1, 2,], ['N', 'LB','LA'])

# plt.subplot(2,4,2)
# data = [ll_no_feed_perc_p[0],
#         ll_both_feed_perc_p[0], ll_la_perc_p[0]]
# seaborn.boxplot(data=data, orient="h")
# # seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# # plt.ylabel('Different SE Models')
# plt.xlabel('(b) Percentage P Error')
# # plt.title('Percentage P Error for Different SE Models')
# plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])

# plt.subplot(2,4,3)
# data = [ll_no_feed_abs_v[0],
#         ll_both_feed_abs_v[0], ll_la_abs_v[0]]
# seaborn.boxplot(data=data, orient="h")
# # seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# # plt.ylabel('Different SE Models')
# plt.xlabel('(c) Absolute V Error')
# # plt.title('Absolute V Error for Different SE Models')
# plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])

# plt.subplot(2,4,4)
# data = [ll_no_feed_abs_p[0],
#         ll_both_feed_abs_p[0], ll_la_abs_p[0]]
# seaborn.boxplot(data=data, orient="h")
# # seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# # plt.ylabel('Different SE Models')
# plt.xlabel('(d) Absolute P Error')
# # plt.title('Absolute P Error for Different SE Models')
# plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])

# plt.figure(100)
data = [ll_no_feed_perc_v[0],
        ll_both_feed_perc_v[0], ll_la_perc_v[0]]
plt.subplot(2,4,5)
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(e) Percentage V Error')
# plt.title('Percentage V Error for Different SE Models')
plt.yticks([0, 1, 2,], ['N', 'LB','LA'])

plt.subplot(2,4,6)
data = [ll_no_feed_perc_p[0],
        ll_both_feed_perc_p[0], ll_la_perc_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(f) Percentage P Error')
# plt.title('Percentage P Error for Different SE Models')
plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])

plt.subplot(2,4,7)
data = [ll_no_feed_abs_v[0],
        ll_both_feed_abs_v[0], ll_la_abs_v[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(g) Absolute V Error')
# plt.title('Absolute V Error for Different SE Models')
plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])

plt.subplot(2,4,8)
data = [ll_no_feed_abs_p[0],
        ll_both_feed_abs_p[0], ll_la_abs_p[0]]
seaborn.boxplot(data=data, orient="h")
# seaborn.swarmplot(data=ll_v_feed_perc_v, color=".25")
# plt.ylabel('Different SE Models')
plt.xlabel('(h) Absolute P Error')
# plt.title('Absolute P Error for Different SE Models')
plt.yticks([0, 1, 2,], ['N', 'LB', 'LA'])
'''
# saving different errors
list_of_errors_p, list_of_errors_q, list_of_errors_v = [], [], [] # max abs error
list_of_all_errors_p, list_of_all_errors_q = [], [] # all abs error
list_all_error_known_p, list_all_error_known_q = [], []  # error for known buses
list_all_error_unknown_p, list_all_error_unknown_q = [], [] # error for unknown buses

# save all estimate results and combinations of different meas
store_estimates, list_of_all_combs = [], []
list_max_error_index_p = []
arr = np.arange(len(non_zib_index)-8) # used for combinations
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
    list_of_all_combs.append(combs) # stores all combinations

    # can put the following loop in a function
    for indices in combs:
        P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas, meas_V =  bus_measurements_equal_distribution(
                P_Load, Q_Load, V, P_line[(0,1)], Q_line[(0,1)], 
                non_zib_index, zib_index, num_known_meas=len(indices), indices = np.asarray(indices))

        meas_P_load = {**P_known_meas, **P_pseudo_meas}
        meas_P_load = dict(sorted(meas_P_load.items()))
        meas_Q_load = {**Q_known_meas, **Q_pseudo_meas}
        meas_Q_load = dict(sorted(meas_Q_load.items()))

        z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
               list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set
    
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
        
        # get jacobain matrix
        jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
                                          meas_V, R_line, X_line, len(x_est), len(z))
        
        # run WLS SE
        x_estn, emax, count, residuals_mat, delta_mat, results = se_wls(
            x_est, z, jacobian_matrix, W)
        
        # Running the gradient Algorithm
        lr, iterations = 0.1, 30000 # Learning Rate and Number of iterations
         
        # x_est=x_estb
        # Batch Gradient Descent
        # print('Running BGD')
        # x_estb, thetasb, costsb, countsb = batch_gradient_descent(
        #     jacobian_matrix, z, x_est, W, lr, iterations)
        
        # print('Running Pytorch Implementation')
        # regr = WLeastSquaresRegressorTorch(n_iter=30000, eta=0.01, batch_size=len(z))
        # xx = regr.fit(jacobian_matrix, z, W)

        # Error Calculations
        # the following function is used when the states are non zib buses
        errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
        errperc_vectorp, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \
        errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err = \
        error_calc_refactor(x, x_estn, non_zib_index, P_Load, est_lin, est_full_ac, 
                                which, V, V_mag) # for WLS
        # error_calc_refactor(x, x_estb, non_zib_index, P_Load, est_lin, est_full_ac, 
        #                         which, V, V_mag) # for self GD
        # error_calc_refactor(x, xx.detach().numpy(), non_zib_index, P_Load, est_lin, est_full_ac, 
        #                         which, V, V_mag) # for pytorch GD

        # append the max absolute error
        err_for_diff_known_meas_p.append(max_error_st_abs_p)
        err_for_diff_known_meas_q.append(max_error_st_abs_q)
        # err_for_diff_known_meas_v.append(abs(full_x_est[-1] - x[-1]))
        err_for_diff_known_meas_v.append(max_abs_vmag_err)
        # itermediate_results.append(full_x_est)
        max_abs_error_index_p.append(max_index_p)

        # plot all the errors as well
        all_err_for_diff_known_meas_p.extend(st_err_p)
        all_err_for_diff_known_meas_q.extend(st_err_q)

        # get known and unknown errors separately
        known_indices = np.asarray(list(P_known_meas.keys()))
        unknown_indices = np.asarray(list(P_pseudo_meas.keys ()))
        error_for_known_p.extend(st_err_p[known_indices]) # erorrs of all known p
        error_for_known_q.extend(st_err_q[known_indices]) # erorrs of all known q
        error_for_unknown_p.extend(st_err_p[unknown_indices]) # erorrs of all unknown p
        error_for_unknown_q.extend(st_err_q[unknown_indices]) # erorrs of all unknown q

    # max abs error
    list_of_errors_p.append(err_for_diff_known_meas_p)
    list_of_errors_q.append(err_for_diff_known_meas_q)
    list_of_errors_v.append(err_for_diff_known_meas_v)
    # all errors
    list_of_all_errors_p.append(all_err_for_diff_known_meas_p)
    list_of_all_errors_q.append(all_err_for_diff_known_meas_q)
    # store_estimates.append(itermediate_results)
    list_max_error_index_p.append(max_abs_error_index_p)
    # known and unknown errors
    list_all_error_known_p.append(error_for_known_p)
    list_all_error_known_q.append(error_for_known_q)
    list_all_error_unknown_p.append(error_for_unknown_p)
    list_all_error_unknown_q.append(error_for_unknown_q)

# plot the max error graph
# plt.subplot(3,4,12)
# seaborn.boxplot(data=list_of_errors_p)
# seaborn.swarmplot(data=list_of_errors_p, color=".25")
# plt.xlabel('Known number of measurements')
# plt.ylabel('Max absolute error in pu')
# plt.title('Max Absolute Error Corresponding to known number of Measurements')

# # plot all error graph
# plt.figure()
# seaborn.boxplot(data=list_of_all_errors_p)
# # seaborn.swarmplot(data=list_of_all_errors_p, color=".25") 
# plt.xlabel('Known number of measurements')
# plt.ylabel(' absolute error in pu')
# plt.title('All Absolute Errors Corresponding to known number of Measurements')

# # plot known errors
# plt.figure()
# seaborn.boxplot(data=list_all_error_known_p)
# seaborn.swarmplot(data=list_of_errors_p, color=".25")
# plt.xlabel('Known number of measurements')
# plt.ylabel('absolute error in pu')
# plt.title('Absolute Error Corresponding to known Measurements Buses')

# # plot known errors
# plt.figure()
# seaborn.boxplot(data=list_all_error_unknown_p)
# seaborn.swarmplot(data=list_all_error_unknown_p, color=".25")
# plt.xlabel('Known number of measurements')
# plt.ylabel('absolute error in pu')
# plt.title('Absolute Error Corresponding to unknown Measurement Buses')

# check if max error is at pseudo buses
# count = 0 
print('checking if error nodes are known measurement nodes or not')
for i,val in enumerate(list_max_error_index_p):
    count+=1
    for j, val2 in enumerate(val):
        # print(i,j)
        if i !=0: # for avoiding empty set error
            # have to do the following becasue list_of_all_combs contains
            # indices for chosing the non zibs values but not the actual non zib values
            known_meas_idx = non_zib_index_array[np.asarray(list_of_all_combs[i][j])] # gives the known index
            if (np.any(np.in1d(val2, known_meas_idx))): # if any val2 is in known meas
                # if the error is known bus meas then something is wrong    
                print('Something wrong')
            
##############################################################################
##############################################################################
arr = np.arange(len(non_zib_index)) # used for combinations
i = 2 # number of known measurements
combs = list(combinations(arr,i)) # all for combination of buses with known number of i

# run one scenario for different solvers
vmag_perc_error = [] # abs vmag error
p_perc_error, q_perc_error = [], [] # perc p q bus error
p_abs_error, q_abs_error = [], [] # abs p q bus error
for indices in combs:
    # [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
    # [0,   1,  2,  3,  4,  5,  6,  7,  8,  9]
    indices = np.asarray(( 2,  3,  4,  5,  6,  7,  8,  9)) # select manually
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
            non_zib_index, zib_index, num_known_meas=len(indices), indices = np.asarray(indices))

    meas_P_load = {**P_known_meas, **P_pseudo_meas}
    meas_P_load = dict(sorted(meas_P_load.items()))
    meas_Q_load = {**Q_known_meas, **Q_pseudo_meas}
    meas_Q_load = dict(sorted(meas_Q_load.items()))

    z = np.asarray(list(meas_P_line.values()) + list(meas_Q_line.values()) + 
            list(meas_P_load.values()) + list(meas_Q_load.values()) + list(meas_V.values())) # meas set

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
    
    # get jacobain matrix
    jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
                                      meas_V, R_line, X_line, len(x_est), len(z))

    # run WLS SE
    x_estn, emax, count, residuals_mat, delta_mat, results = se_wls(
        x_est, z, jacobian_matrix, W)

    # Running the gradient Algorithm
    lr, iterations = 0.1, 30000 # Learning Rate and Number of iterations

    # x_est=x_estb
    # Batch Gradient Descent
    print('Running BGD')
    x_estb, thetasb, costsb, countsb = batch_gradient_descent(
        jacobian_matrix, z, x_est, W, lr, iterations)

    print('Running Pytorch Implementation')
    regr = WLeastSquaresRegressorTorch(n_iter=30000, eta=0.01, batch_size=len(z))
    xx = regr.fit(jacobian_matrix, z, W)

    # Error Calculations
    # the following function is used when the states are non zib buses
    errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
    errperc_vectorq, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \
    errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err = \
    error_calc_refactor(x, x_estn, non_zib_index, P_Load, est_lin, est_full_ac, 
                            which, V, V_mag) # for WLS
    # append the results
    vmag_perc_error.append(errperc_vector_vmag)
    p_perc_error.append(errperc_vectorp), q_perc_error.append(errperc_vectorq)
    p_abs_error.append(st_err_p), q_abs_error.append(st_err_q)

    errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
    errperc_vectorq, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \
    errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err = \
    error_calc_refactor(x, x_estb, non_zib_index, P_Load, est_lin, est_full_ac, 
                            which, V, V_mag) # for self GD
    # append the results
    vmag_perc_error.append(errperc_vector_vmag)
    p_perc_error.append(errperc_vectorp), q_perc_error.append(errperc_vectorq)
    p_abs_error.append(st_err_p), q_abs_error.append(st_err_q)

    errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
    errperc_vectorq, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \
    errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err = \
    error_calc_refactor(x, xx.detach().numpy(), non_zib_index, P_Load, est_lin, est_full_ac, 
                            which, V, V_mag) # for pytorch GD
    # append the results
    vmag_perc_error.append(errperc_vector_vmag)
    p_perc_error.append(errperc_vectorp), q_perc_error.append(errperc_vectorq)
    p_abs_error.append(st_err_p), q_abs_error.append(st_err_q)
    break

# plot V mag percentage error
plt.figure()
seaborn.boxplot(data=vmag_perc_error, orient='h')
seaborn.swarmplot(data=vmag_perc_error, color=".25", orient='h')
plt.ylabel('Different Solvers')
plt.xlabel('Percentage Voltage Error')
plt.title('Percentage Voltage Error for Different Solvers')
plt.yticks([0, 1, 2], ['WLS', 'GD', 'Pytorch GD'])

# plot P mag percentage error
plt.figure()
seaborn.boxplot(data=p_perc_error, orient='h')
# seaborn.swarmplot(data=p_perc_error, color=".25")
plt.ylabel('Different Solvers')
plt.xlabel('Percentage P Error')
plt.title('Percentage P Error for Different Solvers')
plt.yticks([0, 1, 2], ['WLS', 'GD', 'Pytorch GD'])

# plot P mag percentage error
plt.figure()
seaborn.boxplot(data=p_abs_error)
# seaborn.swarmplot(data=p_abs_error, color=".25")
plt.xlabel('Different Solvers')
plt.ylabel('Absolute P Error')
plt.title('Absolute P Error for Different Solvers')
plt.xticks([0, 1, 2], ['WLS', 'GD', 'Pytorch GD'])
'''