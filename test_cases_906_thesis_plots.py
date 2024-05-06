#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:01:17 2021
@author: shub
"""
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
import numpy as np
from jacobian_calc import create_jacobian, vnode_with_v0_pre_calculated_terms, \
    combination_of_loads, get_r_x_z_mat, pline_with_p_pre_calculated_terms, \
    pline_with_vnode_calculated_terms, vnode_with_v0_pre_calc_terms_fast
from solvers import se_wls, se_ols, se_wrr, se_rr, batch_gradient_descent, \
    stochastic_gradient_descent, stochastic_gradient_descent2, \
    WLeastSquaresRegressorTorch, cost
from solvers_with_loss import se_wls_nonlin, se_wls_nonlin_ass, se_wls_la_bgd
from path_to_nodes import path_to_nodes
import pandas as pd
from itertools import combinations
import seaborn as sns
from some_funcs import error_calc, create_mes_set, subset_of_measurements, \
                       weight_vals, noise_addition, bus_measurements_equal_distribution, \
                       error_calc_refactor, countour_plot, inc_avg, get_nodes_downstream_of_each_branch
from power_flow_modelling.networks import Network
import time
import torch
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

which = 906 # IEEE 37-node or IEEE 906-node

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

all_index_array = np.asarray(list(P_Load.keys()))
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
# x_est = torch.rand(len(x_est)).double() # so that the initial condn is same as pytorch
x_est = torch.ones(len(x_est)).double() # so that the initial condn is same as pytorch
x_est =  x_est.detach().cpu().numpy()

x_true = np.concatenate((x[non_zib_index], x[non_zib_index_array + len(gt_P_load)]))
x_true = np.insert(x_true, len(x_true), gt_V) # ground truth for states
###############################################################################
###############################################################################
# load the network object for sped up distflow
network906 = Network('network906', sparse=False)
###############################################################################
###############################################################################

# get paths from slack bus to all nodes
path_to_all_nodes, path_to_all_nodes_list = path_to_nodes(which)

# get subset of lineflow measurement set
num_plow_meas = 1
# chose lineflows
meas_P_line, meas_Q_line = subset_of_measurements(
    num_plow_meas, arcs, P_line, Q_line, V)
if meas_P_line:
    downstream_matrix = get_nodes_downstream_of_each_branch(meas_P_line, P_Load_state, path_to_all_nodes)
else:
    downstream_matrix = 0
# num_known = [8, 5, 3] # known number of measurements
# num_known = [9,] # known number of measurements
num_known = np.arange(len(non_zib_index))[::-1]

num_known = [20, 17, 14, 11] # known number of measurements
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

# function used to get max vals for each node
def max_val(A, current_calc_error, non_zib_index):
    # A prev_max_error for non zib buses
    B = current_calc_error[non_zib_index]
    # B = current_calc_error
    A[B>A] = B[B>A]
    return A

def max_val_for_index(A, current_calc_error, non_zib_index, node):
    ''' returns a flag when a max value is changed for a specific node 
        doing it for power only
    '''
    # node is the index number and not the node number
    # [ 2,  8, 10, 11, 21, 22, 23, 26, 35, 36]
    # [0,   1,  2,  3,  4,  5,  6,  7,  8,  9]

    # A prev_max_error for non zib buses
    flag = 0 # flag for getting max error in node 26
    node26 = A[node] # error val for node 26   
    B = current_calc_error[non_zib_index]
    A[B>A] = B[B>A]
    
    if node26 != A[node]:
        flag = 1
    return A, flag

def subplot_heatmap(array_2d, indices, vmin, vmax, cbar = None, cbar_ax = None, ax = None):
    cbar = 0 if cbar is None else cbar
    cbar_ax = 0 if cbar_ax is None else cbar_ax
    ax = 0 if ax is None else ax
    # plt.figure()
    ax = sns.heatmap(array_2d, vmin, vmax, cbar = cbar, cbar_ax = cbar_ax)
    # plt.xlabel('Node Number')
    # plt.ylabel('Number of Missing Measurements')
    plt.yticks(np.arange(len(num_known))+0.5, num_known, rotation = 360) # num known meas
    plt.xticks(np.arange(len(indices))+0.5, indices, rotation = 360) # node number

def plot_heatmap(array_2d):
    # plt.figure()
    ax = sns.heatmap(array_2d)
    # plt.xlabel('Node Number')
    # plt.ylabel('Number of Missing Measurements')
    plt.yticks(np.arange(len(num_known))+0.5, num_known)

# heatmap array
# num missing meas * num nodes
heatmap_volt_abs_no_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_abs_no_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_volt_perc_no_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_perc_no_feed = np.zeros((len(num_known), len(non_zib_index)))
###############################################################################
heatmap_volt_abs_v_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_abs_v_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_volt_perc_v_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_perc_v_feed = np.zeros((len(num_known), len(non_zib_index)))
###############################################################################
heatmap_volt_abs_p_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_abs_p_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_volt_perc_p_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_perc_p_feed = np.zeros((len(num_known), len(non_zib_index)))
###############################################################################
heatmap_volt_abs_both_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_abs_both_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_volt_perc_both_feed = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_perc_both_feed = np.zeros((len(num_known), len(non_zib_index)))
###############################################################################
heatmap_volt_abs_la = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_abs_la = np.zeros((len(num_known), len(non_zib_index)))
heatmap_volt_perc_la = np.zeros((len(num_known), len(non_zib_index)))
heatmap_p_perc_la = np.zeros((len(num_known), len(non_zib_index)))
###############################################################################

# all average variables
avg_perc_v_nofeed, avg_perc_p_nofeed, avg_abs_p_nofeed, avg_abs_v_nofeed = 0, 0, 0, 0
avg_perc_v_vfeed, avg_perc_p_vfeed, avg_abs_p_vfeed, avg_abs_v_vfeed = 0, 0, 0, 0
avg_perc_v_pfeed, avg_perc_p_pfeed, avg_abs_p_pfeed, avg_abs_v_pfeed = 0, 0, 0, 0
avg_perc_v_bothfeed, avg_perc_p_bothfeed, avg_abs_p_bothfeed, avg_abs_v_bothfeed = 0, 0, 0, 0
avg_perc_v_la, avg_perc_p_la, avg_abs_p_la, avg_abs_v_la = 0, 0, 0, 0
total_counts_v, total_counts_p = 0, 0 # total number of vars for average

###############################################################################
# get pre calculated info beforehand that can be used to calc jacobians for LA
pre_calculated_info = {}

# add node 0 in non zibs if it doesnt exist for the precalculated values for v meas
# as we always have slack bus voltage in the meas set
meas_V_nodes = np.insert(non_zib_index_array, 0, 0) if 0 not in non_zib_index_array else non_zib_index_array # consist all possible locs of V meas
meas_V_nodes_index = np.arange((len(meas_V_nodes))) # index corresponding to all v nodes

# used for vnode with p
R_mat, X_mat, Z_mat, additional_mat_r, additional_mat_x = get_r_x_z_mat(
    meas_V_nodes, P_Load_state, path_to_all_nodes, R_line, X_line, LineData_Z_pu)

# combination of elems of non-zib nodes
elems_comb = combination_of_loads(P_Load_state)

# used for vnode with V0
v_node_RX_comb, z_common_path = vnode_with_v0_pre_calculated_terms(meas_V_nodes, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, LineData_Z_pu)

# used for vode with V0 fast
df_vnode_with_v0, v_RX_Z_comb = vnode_with_v0_pre_calc_terms_fast(meas_V_nodes, elems_comb, path_to_all_nodes, 
                                      R_line, X_line, LineData_Z_pu, non_zib_index_array)

# used for pline with p
r_hat, x_hat = pline_with_p_pre_calculated_terms(meas_P_line, P_Load_state, path_to_all_nodes, 
                            R_line, X_line)

# used for pline with vnode
df_pline_with_v0, mat_r, mat_x =  pline_with_vnode_calculated_terms(meas_P_line, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, elems_comb, non_zib_index_array)
###############################################################################

node_26_error_for_diff_known_meas = [] # to store known indices for max error
count = 0 # total number of iters, should be sum of all combs at the end
iters_n0, iters_n1, iters_n2, iters_n, iters_la = 0, 0, 0, 0, 0
time_n0, time_n1, time_n2, time_nn, time_la = 0, 0, 0, 0, 0
tot_counts = 0
for row, i in enumerate(num_known):
    arr = np.arange(len(non_zib_index)) # used for combinations
    combs = list(combinations(arr,i))
    if len(combs) > 1000: # sampling
        np.random.seed(40) # 40 for latex, 28 alternative
        idx = np.random.choice(len(combs), 1000, replace=False)
        combs = np.array(combs)
        combs=combs[idx]

    # stores the max abs voltage error for each node
    volt_max_abs_nofeed, p_max_abs_nofeed = np.zeros((len(non_zib_index))),  np.zeros((len(non_zib_index)))
    volt_max_perc_nofeed = np.zeros((len(non_zib_index)))
    p_max_perc_nofeed = np.zeros((len(non_zib_index)))

    volt_max_perc_vfeed, p_max_perc_vfeed, volt_max_abs_vfeed, p_max_abs_vfeed = np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index)))

    volt_max_perc_pfeed, p_max_perc_pfeed, volt_max_abs_pfeed, p_max_abs_pfeed = np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index)))

    volt_max_perc_bothfeed, p_max_perc_bothfeed, volt_max_abs_bothfeed, p_max_abs_bothfeed = np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index)))
    
    volt_max_perc_la, p_max_perc_la, volt_max_abs_la, p_max_abs_la = np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index))), np.zeros((len(non_zib_index)))
    # to hold all eroors for all combinations for a fixed num of missing meass 
    l_no_feed_perc_v, l_no_feed_perc_p, l_no_feed_abs_v, l_no_feed_abs_p = [], [], [], []
    l_v_feed_perc_v, l_v_feed_perc_p, l_v_feed_abs_v, l_v_feed_abs_p = [], [], [], []
    l_p_feed_perc_v, l_p_feed_perc_p, l_p_feed_abs_v, l_p_feed_abs_p = [], [], [], []
    l_both_feed_perc_v, l_both_feed_perc_p, l_both_feed_abs_v, l_both_feed_abs_p = [], [], [], []
    l_la_perc_v, l_la_perc_p, l_la_abs_v, l_la_abs_p = [], [], [], []

    # different combs for known number of meas
    for indices in combs:
        count+=1
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
        w22 = 100000 #1000000 # pseudo measurements for p,q at buses
        w3 = 0.1 # weight for voltage value; use 0.1 for grad descent & 0.0001 for WLS
        # print(w1, w21, w22, w3)

        weight_array1 = np.ones((len(meas_P_line)*2))*w1 # for pline and qline
        weight_array2 = np.ones((len(meas_P_load)))
        weight_array2[list(P_known_meas.keys())] = weight_array2[list(P_known_meas.keys())]*w21 # for known p meas
        weight_array2[list(P_pseudo_meas.keys())] = weight_array2[list(P_pseudo_meas.keys())]*w22# for unknown p meas
        # modified weight for reactive power
        # weight_array2 = np.concatenate((weight_array2, np.ones((len(meas_P_load)))*0.1)) # for p and q bus
        weight_array2 = np.concatenate((weight_array2, weight_array2))
        weight_array3 = np.ones((len(meas_V)))*w3 # for v mag
        weight_array = np.concatenate((weight_array1, weight_array2,weight_array3)) # entire weight vector

        W = np.diag(weight_array) # Weight mat
        W = np.linalg.inv(W)

        # jacobian_matrix = create_jacobian(meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
        #                                   meas_V, R_line, X_line, len(x_est), len(z))

        # GN-WLS
        lossy_volt_est = {'tot_states':len(x), 'non_zib_index':non_zib_index, 
                          'num_buses':len(P_Load), 'which':which, 'volt_buses': meas_V.keys(),
                          'plines':meas_P_line.keys()}

        # to include non linear voltage feedback and pflow/qflow
        # to include non linear voltage feedback and pflow/qflow
        print('GN-WLS based on linear jacobian with no feedback') 
        loss, pflow = 0, 0
        # LinDist
        start = time.time()
        x_estn0, emax, countsn0, residuals_mat, delta_mat, results, costsn, jacobian_matrix = se_wls(
            x_est, z, W, meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
            meas_V, R_line, X_line, network906, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        end = time.time()
        tot_time = end-start
        time_n0+= tot_time
        # costsn = cost(x_estn, jacobian_matrix, z, W)
        iters_n0+=countsn0      # average of all elements
        perc_v_nofeed, perc_p_nofeed, abs_v_nofeed, abs_p_nofeed = error_calc_refactor(x, x_estn0, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = 1, pflow = 1) # for WLS
        # average of all elements        
        avg_perc_v_nofeed = inc_avg(avg_perc_v_nofeed, total_counts_v, perc_v_nofeed[non_zib_index])
        avg_abs_v_nofeed = inc_avg(avg_abs_v_nofeed, total_counts_v, abs_v_nofeed[non_zib_index])
        avg_perc_p_nofeed = inc_avg(avg_perc_p_nofeed, total_counts_p, perc_p_nofeed[non_zib_index])
        avg_abs_p_nofeed = inc_avg(avg_abs_p_nofeed, total_counts_p, abs_p_nofeed[non_zib_index])
        # uncomment below to store all errors
        l_no_feed_perc_v.extend(perc_v_nofeed), l_no_feed_perc_p.extend(perc_p_nofeed), 
        l_no_feed_abs_v.extend(abs_v_nofeed), l_no_feed_abs_p.extend(abs_p_nofeed[non_zib_index])
        ################### HEATMAP ##########################################
        volt_max_perc_nofeed = max_val(volt_max_perc_nofeed, perc_v_nofeed, non_zib_index)
        # _, flag = max_val_for_index(p_max_perc_nofeed, perc_p_nofeed, non_zib_index, 7)
        # print('Flaaaaaaaaaaaag', flag)
        # if flag == 1:
        #     max_node_26_error = corresponding_nodes        
        p_max_perc_nofeed = max_val(p_max_perc_nofeed, perc_p_nofeed, non_zib_index)    
        volt_max_abs_nofeed = max_val(volt_max_abs_nofeed, abs_v_nofeed, non_zib_index)
        p_max_abs_nofeed = max_val(p_max_abs_nofeed, abs_p_nofeed, non_zib_index)
        #######################################################################
        # print('GN-WLS based on linear jacobian with V feedback')
        # loss, pflow = 1, 0
        # # LinDist + Voltage Feedback
        # start = time.time()
        # x_estn1, emax, countsn1, residuals_mat, delta_mat, results, costsn, _ = se_wls(
        #     x_est, z, W, meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
        #     meas_V, R_line, X_line, network906, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # end = time.time()
        # tot_time = end-start
        # time_n1+= tot_time
        # # costsn = cost(x_estn, jacobian_matrix, z, W)
        # iters_n1+=countsn1
        # perc_v_vfeed, perc_p_vfeed, abs_v_vfeed, abs_p_vfeed = error_calc_refactor(x, x_estn1, non_zib_index, len(P_Load), est_lin, est_full_ac, 
        #                         which, V, V_mag, loss = 1, pflow = 1) # for WLS
        # # average of all elements
        # avg_perc_v_vfeed = inc_avg(avg_perc_v_vfeed, total_counts_v, perc_v_vfeed[non_zib_index])
        # avg_abs_v_vfeed = inc_avg(avg_abs_v_vfeed, total_counts_v, abs_v_vfeed[non_zib_index])
        # avg_perc_p_vfeed = inc_avg(avg_perc_p_vfeed, total_counts_p, perc_p_vfeed[non_zib_index])
        # avg_abs_p_vfeed = inc_avg(avg_abs_p_vfeed, total_counts_p, abs_p_vfeed[non_zib_index])        
        # # uncomment below to store all errors        
        # l_v_feed_perc_v.extend(perc_v_vfeed), l_v_feed_perc_p.extend(perc_p_vfeed), 
        # l_v_feed_abs_v.extend(abs_v_vfeed), l_v_feed_abs_p.extend(abs_p_vfeed[non_zib_index])
        # ################### HEATMAP ##########################################
        # volt_max_perc_vfeed = max_val(volt_max_perc_vfeed, perc_v_vfeed, non_zib_index)
        # p_max_perc_vfeed = max_val(p_max_perc_vfeed, perc_p_vfeed, non_zib_index)       
        # volt_max_abs_vfeed = max_val(volt_max_abs_vfeed, abs_v_vfeed, non_zib_index)
        # p_max_abs_vfeed = max_val(p_max_abs_vfeed, abs_p_vfeed, non_zib_index)
        #######################################################################
        # print('GN-WLS based on linear jacobian with P feedback')
        # loss, pflow = 0, 1
        # # LinDist + Pflow Feedback
        # start = time.time()
        # x_estn2, emax, countsn2, residuals_mat, delta_mat, results, costsn, _ = se_wls(
        #     x_est, z, W, meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
        #     meas_V, R_line, X_line, network906, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        # end = time.time()
        # tot_time = end-start
        # time_n2+= tot_time        
        # # costsn = cost(x_estn, jacobian_matrix, z, W)
        # iters_n2+=countsn2
        # perc_v_pfeed, perc_p_pfeed, abs_v_pfeed, abs_p_pfeed = error_calc_refactor(x, x_estn2, non_zib_index, len(P_Load), est_lin, est_full_ac, 
        #                         which, V, V_mag, loss = 1, pflow = 1) # for WLS
        # # average of all elements
        # avg_perc_v_pfeed = inc_avg(avg_perc_v_pfeed, total_counts_v, perc_v_pfeed[non_zib_index])
        # avg_abs_v_pfeed = inc_avg(avg_abs_v_pfeed, total_counts_v, abs_v_pfeed[non_zib_index])
        # avg_perc_p_pfeed = inc_avg(avg_perc_p_pfeed, total_counts_p, perc_p_pfeed[non_zib_index])
        # avg_abs_p_pfeed = inc_avg(avg_abs_p_pfeed, total_counts_p, abs_p_pfeed[non_zib_index])           
        # # uncomment below to store all errors        
        # l_p_feed_perc_v.extend(perc_v_pfeed), l_p_feed_perc_p.extend(perc_p_pfeed), 
        # l_p_feed_abs_v.extend(abs_v_pfeed), l_p_feed_abs_p.extend(abs_p_pfeed[non_zib_index])
        # ################### HEATMAP ##########################################
        # volt_max_perc_pfeed = max_val(volt_max_perc_pfeed, perc_v_pfeed, non_zib_index)
        # p_max_perc_pfeed = max_val(p_max_perc_pfeed, perc_p_pfeed, non_zib_index)       
        # volt_max_abs_pfeed = max_val(volt_max_abs_pfeed, abs_v_pfeed, non_zib_index)
        # p_max_abs_pfeed = max_val(p_max_abs_pfeed, abs_p_pfeed, non_zib_index)
        #######################################################################
        print('GN-WLS based on linear jacobian with both feedback')
        loss, pflow = 1, 1
        # LinDist + Voltage & Pflow Feedback
        start = time.time()
        x_estn, emax, countsn, residuals_mat, delta_mat, results, costsn, _ = se_wls(
            x_est, z, W, meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes,
            meas_V, R_line, X_line, network906, loss = loss, pflow = pflow, lossy_volt_est = lossy_volt_est)
        end = time.time()
        tot_time = end-start
        time_nn+= tot_time                
        # costsn = cost(x_estn, jacobian_matrix, z, W)
        iters_n+=countsn
        perc_v_n, perc_p_n, abs_v_n, abs_p_n = error_calc_refactor(x, x_estn, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = 1, pflow = 1) # for WLS
        # average of all elements
        avg_perc_v_bothfeed = inc_avg(avg_perc_v_bothfeed, total_counts_v, perc_v_n[non_zib_index])
        avg_abs_v_bothfeed = inc_avg(avg_abs_v_bothfeed, total_counts_v, abs_v_n[non_zib_index])
        avg_perc_p_bothfeed = inc_avg(avg_perc_p_bothfeed, total_counts_p, perc_p_n[non_zib_index])
        avg_abs_p_bothfeed = inc_avg(avg_abs_p_bothfeed, total_counts_p, abs_p_n[non_zib_index])   
        # uncomment below to store all errors        
        l_both_feed_perc_v.extend(perc_v_n), l_both_feed_perc_p.extend(perc_p_n), 
        l_both_feed_abs_v.extend(abs_v_n), l_both_feed_abs_p.extend(abs_p_n[non_zib_index])
        ################### HEATMAP ##########################################
        volt_max_perc_bothfeed = max_val(volt_max_perc_bothfeed, perc_v_n, non_zib_index)
        p_max_perc_bothfeed = max_val(p_max_perc_bothfeed, perc_p_n, non_zib_index)       
        volt_max_abs_bothfeed = max_val(volt_max_abs_bothfeed, abs_v_n, non_zib_index)
        p_max_abs_bothfeed = max_val(p_max_abs_bothfeed, abs_p_n, non_zib_index)

        #######################################################################
        # get characteristics beforehand that can be used to calc jacobians for LA
        pre_calculated_info = {}
        # V meas indices considered
        meas_V_keys = np.array(list(meas_V.keys()))
        meas_V_idx = np.nonzero(np.in1d(meas_V_nodes,meas_V_keys))[0]
        R_mat_req = R_mat[meas_V_idx, :]
        X_mat_req = X_mat[meas_V_idx, :]
        v_RX_Z_comb_req = v_RX_Z_comb[meas_V_idx, :]

        Z_mm = np.concatenate([Z_mat[i*len(P_Load_state):i*len(P_Load_state) + len(P_Load_state)] for i in meas_V_idx])
        addn_rr = np.concatenate([additional_mat_r[i*len(P_Load_state):i*len(P_Load_state) + len(P_Load_state)] for i in meas_V_idx])
        addn_xx = np.concatenate([additional_mat_x[i*len(P_Load_state):i*len(P_Load_state) + len(P_Load_state)] for i in meas_V_idx])

        pre_calculated_info['v_node_RX_comb'] = v_node_RX_comb
        pre_calculated_info['z_common_path'] = z_common_path
        pre_calculated_info['elems_comb'] = elems_comb
        pre_calculated_info['R_mat'] = R_mat_req
        pre_calculated_info['X_mat'] = X_mat_req
        pre_calculated_info['Z_mat'] = Z_mm
        pre_calculated_info['additional_mat_r'] = addn_rr
        pre_calculated_info['additional_mat_x'] = addn_xx
        pre_calculated_info['r_hat'] = r_hat
        pre_calculated_info['x_hat'] = x_hat
        pre_calculated_info['mat_r'] = mat_r
        pre_calculated_info['mat_x'] = mat_x
        pre_calculated_info['downstream_matrix'] = downstream_matrix        
        if num_plow_meas!=0:
            pre_calculated_info['comb_idx1'] = np.array(df_pline_with_v0.idx1)
            pre_calculated_info['comb_idx2'] = np.array(df_pline_with_v0.idx2)
        else:
            pre_calculated_info['comb_idx1'] = np.array(df_vnode_with_v0.idx1)
            pre_calculated_info['comb_idx2'] = np.array(df_vnode_with_v0.idx2)   
        pre_calculated_info['sum_r'] = np.array(df_pline_with_v0.sum_r)
        pre_calculated_info['sum_x'] = np.array(df_pline_with_v0.sum_x)      
        pre_calculated_info['v_RX_Z_comb_req'] = v_RX_Z_comb_req
        # same comb_idx can be used for vnode_wit_V0 as above?
        #######################################################################

        #######################################################################
        # print('Implementing loss based with a few assumptions')
        print('GN-WLS based on non-linear with ass')
        start = time.time()        
        x_est_la, emax_la, counts_la, residuals_mat_la, delta_mat_la, results_la, jacobian_matrix_la = se_wls_nonlin_ass(
            x_est, z, W, meas_P_line, P_Load_state, meas_P_load, path_to_all_nodes, 
            non_zib_index, meas_V, R_line, X_line, LineData_Z_pu, pre_calculated_info,
            len(x_est), len(z), len(x), which)
        end = time.time()
        tot_time = end-start
        time_la+= tot_time                
        iters_la+=counts_la
        #######################################################################
        perc_v_la, perc_p_la, abs_v_la, abs_p_la = error_calc_refactor(x, x_est_la, non_zib_index, len(P_Load), est_lin, est_full_ac, 
                                which, V, V_mag, loss = 1, pflow = 1) # non linear GN with assumption
        # average of all elements
        avg_perc_v_la = inc_avg(avg_perc_v_la, total_counts_v, perc_v_la[non_zib_index])
        avg_abs_v_la = inc_avg(avg_abs_v_la, total_counts_v, abs_v_la[non_zib_index])
        avg_perc_p_la = inc_avg(avg_perc_p_la, total_counts_p, perc_p_la[non_zib_index])
        avg_abs_p_la = inc_avg(avg_abs_p_la, total_counts_p, abs_p_la[non_zib_index]) 
        # uncomment below to store all errors
        l_la_perc_v.extend(perc_v_la), l_la_perc_p.extend(perc_p_la), 
        l_la_abs_v.extend(abs_v_la), l_la_abs_p.extend(abs_p_la[non_zib_index])
        ################### HEATMAP ##########################################
        volt_max_perc_la = max_val(volt_max_perc_la, perc_v_la, non_zib_index)
        p_max_perc_la = max_val(p_max_perc_la, perc_p_la, non_zib_index)       
        volt_max_abs_la = max_val(volt_max_abs_la, abs_v_la, non_zib_index)
        p_max_abs_la = max_val(p_max_abs_la, abs_p_la, non_zib_index)
        # break

        # update total counts
        total_counts_v+= len(perc_v_nofeed[non_zib_index]) # count * len(perc_v_nofeed)
        total_counts_p+= len(perc_p_nofeed[non_zib_index])
    # node_26_error_for_diff_known_meas.append(max_node_26_error) # append indices
    tot_counts+= len(combs)
    # insert the values for heatmap
    heatmap_volt_abs_no_feed[row,:] = volt_max_abs_nofeed * Vbase
    heatmap_p_abs_no_feed[row,:] = p_max_abs_nofeed * Sbase
    heatmap_volt_perc_no_feed[row,:] = volt_max_perc_nofeed
    heatmap_p_perc_no_feed[row,:] = p_max_perc_nofeed
    ###########################################################################
    heatmap_volt_abs_v_feed[row,:] = volt_max_abs_vfeed * Vbase
    heatmap_p_abs_v_feed[row,:] = p_max_abs_vfeed * Sbase
    heatmap_volt_perc_v_feed[row,:] = volt_max_perc_vfeed
    heatmap_p_perc_v_feed[row,:] = p_max_perc_vfeed
    ###########################################################################
    heatmap_volt_abs_p_feed[row,:] = volt_max_abs_pfeed * Vbase
    heatmap_p_abs_p_feed[row,:] = p_max_abs_pfeed * Sbase
    heatmap_volt_perc_p_feed[row,:] = volt_max_perc_pfeed
    heatmap_p_perc_p_feed[row,:] = p_max_perc_pfeed
    ###########################################################################
    heatmap_volt_abs_both_feed[row,:] = volt_max_abs_bothfeed * Vbase
    heatmap_p_abs_both_feed[row,:] = p_max_abs_bothfeed * Sbase
    heatmap_volt_perc_both_feed[row,:] = volt_max_perc_bothfeed
    heatmap_p_perc_both_feed[row,:] = p_max_perc_bothfeed
    ###########################################################################
    heatmap_volt_abs_la[row,:] = volt_max_abs_la * Vbase
    heatmap_p_abs_la[row,:] = p_max_abs_la * Sbase
    heatmap_volt_perc_la[row,:] = volt_max_perc_la
    heatmap_p_perc_la[row,:] = p_max_perc_la
    ###########################################################################
    
    # values for all errors
    # uncomment below to store all errors for each case
    # ll_no_feed_perc_v.append(l_no_feed_perc_v), ll_no_feed_perc_p.append(l_no_feed_perc_p), 
    # ll_no_feed_abs_v.append(l_no_feed_abs_v), ll_no_feed_abs_p.append(l_no_feed_abs_p)
    # ll_v_feed_perc_v.append(l_v_feed_perc_v), ll_v_feed_perc_p.append(l_v_feed_perc_p), 
    # ll_v_feed_abs_v.append(l_v_feed_abs_v), ll_v_feed_abs_p.append(l_v_feed_abs_p)
    # ll_p_feed_perc_v.append(l_p_feed_perc_v), ll_p_feed_perc_p.append(l_p_feed_perc_p), 
    # ll_p_feed_abs_v.append(l_p_feed_abs_v), ll_p_feed_abs_p.append(l_p_feed_abs_p)
    # ll_both_feed_perc_v.append(l_both_feed_perc_v), ll_both_feed_perc_p.append(l_both_feed_perc_p), 
    # ll_both_feed_abs_v.append(l_both_feed_abs_v), ll_both_feed_abs_p.append(l_both_feed_abs_p)
    # ll_la_perc_v.append(l_la_perc_v), ll_la_perc_p.append(l_la_perc_p), 
    # ll_la_abs_v.append(l_la_abs_v), ll_la_abs_p.append(l_la_abs_p)

    # uncomment below to store all errors together
    ll_no_feed_perc_v.extend(l_no_feed_perc_v), ll_no_feed_perc_p.extend(l_no_feed_perc_p), 
    ll_no_feed_abs_v.extend(l_no_feed_abs_v), ll_no_feed_abs_p.extend(l_no_feed_abs_p)
    ll_v_feed_perc_v.extend(l_v_feed_perc_v), ll_v_feed_perc_p.extend(l_v_feed_perc_p), 
    ll_v_feed_abs_v.extend(l_v_feed_abs_v), ll_v_feed_abs_p.extend(l_v_feed_abs_p)
    ll_p_feed_perc_v.extend(l_p_feed_perc_v), ll_p_feed_perc_p.extend(l_p_feed_perc_p), 
    ll_p_feed_abs_v.extend(l_p_feed_abs_v), ll_p_feed_abs_p.extend(l_p_feed_abs_p)
    ll_both_feed_perc_v.extend(l_both_feed_perc_v), ll_both_feed_perc_p.extend(l_both_feed_perc_p), 
    ll_both_feed_abs_v.extend(l_both_feed_abs_v), ll_both_feed_abs_p.extend(l_both_feed_abs_p)
    ll_la_perc_v.extend(l_la_perc_v), ll_la_perc_p.extend(l_la_perc_p), 
    ll_la_abs_v.extend(l_la_abs_v), ll_la_abs_p.extend(l_la_abs_p)    

###############################################################################
# new plots for thesis
# voltage errors
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
all_lists_abs_V = ll_no_feed_abs_v + ll_both_feed_abs_v + ll_la_abs_v

# Create the method labels
method_labels = ['LN'] * len(ll_no_feed_abs_v) + ['LB'] * len(ll_both_feed_abs_v) + ['LA'] * len(ll_la_abs_v)

# Create a DataFrame
dfAbsV = pd.DataFrame({
    'method': method_labels,
    'error_abs_V': all_lists_abs_V
})

pal = sns.cubehelix_palette(len(dfAbsV["method"].unique()), start=1.4, rot=-.25, light=.7, dark=.4)
g = sns.FacetGrid(dfAbsV, row="method", hue="method", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps 
# for BSP
g.map(sns.kdeplot, "error_abs_V",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "error_abs_V", clip_on=False, color="w", lw=2, bw_adjust=.5)
# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "error_abs_V")
# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.5)
# Remove axes details that don't play well with overlap
g.set_titles("")
# g.set(yticks=[], xlabel="", ylabel="", xlim=(None, 680), title="")
g.set(yticks=[], ylabel="", xlabel="ABSOLUTE VOLTAGE ERROR",title="", xlim=(None, 0.02))
g.despine(bottom=True, left=True)

# Add a common y-axis label
g.fig.text(0.06, 0.4, "DISTRIBUTION OF DIFFERENT METHODS", va='center', rotation='vertical', fontsize=BIGGER_SIZE)
print("Max voltage for LN: {}, LB: {}, LA: {}".format(max(ll_no_feed_abs_v), max(ll_both_feed_abs_v), max(ll_la_abs_v)))

######## for NO BSP ########
g = sns.FacetGrid(dfAbsV, row="method", hue="method", aspect=15, height=.5, palette=pal)
g.map(sns.kdeplot, "error_abs_V",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "error_abs_V", clip_on=False, color="w", lw=2, bw_adjust=.5)
# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "error_abs_V")
# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.2)
# Remove axes details that don't play well with overlap
g.set_titles("")
# g.set(yticks=[], xlabel="", ylabel="", xlim=(None, 680), title="")
g.set(yticks=[], ylabel="", xlabel="ABSOLUTE VOLTAGE ERROR",title="", xlim=(None, 0.012))
g.despine(bottom=True, left=True)

# Add a common y-axis label
g.fig.text(0.02, 0.5, "DISTRIBUTION OF DIFFERENT METHODS", va='center', rotation='vertical', fontsize=BIGGER_SIZE)

print("Max voltage for LN: {}, LB: {}, LA: {}".format(max(ll_no_feed_abs_v), max(ll_both_feed_abs_v), max(ll_la_abs_v)))

#############################################################

# power errors
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
all_lists_abs_P = ll_no_feed_abs_p + ll_both_feed_abs_p + ll_la_abs_p

# Create the method labels
method_labelsP = ['LN'] * len(ll_no_feed_abs_p) + ['LB'] * len(ll_both_feed_abs_p) + ['LA'] * len(ll_la_abs_p)

# Create a DataFrame
dfAbsP = pd.DataFrame({
    'method': method_labelsP,
    'error_abs_P': all_lists_abs_P
})

palP = sns.cubehelix_palette(len(dfAbsP["method"].unique()), start=1.4, rot=-.25, light=.7, dark=.4)
gP = sns.FacetGrid(dfAbsP, row="method", hue="method", aspect=15, height=.5, palette=palP)

# Draw the densities in a few steps
gP.map(sns.kdeplot, "error_abs_P",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
gP.map(sns.kdeplot, "error_abs_P", clip_on=False, color="w", lw=2, bw_adjust=.5)
# passing color=None to refline() uses the hue mapping
gP.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

gP.map(label, "error_abs_P")
# Set the subplots to overlap
gP.figure.subplots_adjust(hspace=-.5)
# Remove axes details that don't play well with overlap
gP.set_titles("")
# g.set(yticks=[], xlabel="", ylabel="", xlim=(None, 680), title="")
gP.set(yticks=[], ylabel="", xlabel="ABSOLUTE POWER ERROR",title="", xlim=(None,None))
gP.despine(bottom=True, left=True)

# Add a common y-axis label
# gP.fig.text(0.06, 0.4, "DISTRIBUTION OF DIFFERENT METHODS", va='center', rotation='vertical', fontsize=BIGGER_SIZE)
print("Max P Error for LN: {}, LB: {}, LA: {}".format(max(ll_no_feed_abs_p), max(ll_both_feed_abs_p), max(ll_la_abs_p)))
print("Mean P Error for LN: {}, LB: {}, LA: {}".format(avg_abs_p_nofeed, avg_abs_p_bothfeed, avg_abs_p_la))

########################## computational time plot #######################
# creating the dataset
approaches = ['LN', 'LB', 'LA']
comp_time = [time_n0/tot_counts*1e3, time_nn/tot_counts*1e3, time_la/tot_counts*1e3] # in ms

fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(approaches, comp_time, color ='maroon',
        width = 0.4)
plt.xlabel("Model")
plt.ylabel("Computational Time (ms)")
plt.title("Total Computational Times for Different Models")
plt.show()
