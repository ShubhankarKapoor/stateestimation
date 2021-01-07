#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:44:59 2020

@author: shub
"""
import numpy as np

def create_jacobian(P_line_mes, P_Load_state, P_Load_meas, path_to_all_nodes,
                    Vsq_mes, R_line, X_line, num_states, num_meas):
    ''' LinDistflow based Jacobian'''
    
    # V is square of voltage mag
    # initial case jacobian matrix
    # num_meas = 36*2 + 37*3 # 2 for pij and qij 3 for pj, qj, vj^2
    # num_states = 2 * 37 + 1 # 2 for pj, qj 1 for v0^2
    jacobian_matrix = np.zeros((num_meas, num_states))
    
    # call jacobian for pline_plflow
    grad_array, index_of_lines = grad_pline_with_p(P_line_mes, P_Load_state, path_to_all_nodes)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix[0:meas_rows, 0:state_cols] = grad_array
    last_row_inserted = meas_rows
    
    # might have to modify if qline is missing where there is pline or vice versa
    # grad array would be different in that case
    # similarly for other grad arrays as well
    # call jacobian for qline_qflow, should be exactly same as above
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly
    
    # call jacobian for ppseudo_p
    grad_array = grad_pseudo_with_p(P_Load_meas, P_Load_state)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
    last_row_inserted = last_row_inserted + meas_rows
    
    # call jacobian for qpseudo_q, should be same as above
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    last_row_inserted += meas_rows
    
    # call jacobian for vnode^2 with p
    grad_array = grad_vnode_with_p(Vsq_mes, P_Load_state, path_to_all_nodes, R_line)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
    
    # call jacobian for vnode^2 with q
    grad_array = grad_vnode_with_p(Vsq_mes, P_Load_state, path_to_all_nodes, X_line)
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    
    # call jacobian with v0^2
    grad_array = grad_vnode_with_v0(Vsq_mes)
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array
    
    return jacobian_matrix

def grad_pline_with_p(S_line_meas, p_states, path_to_all_nodes):
    grad_array = np.zeros((len(S_line_meas), len(p_states))) # meas*states

    index_of_lines = {}
    for i , (k,v) in enumerate(S_line_meas.items()): # iterate over measurements line
        # print(i,k)
        index_of_lines[i] = k
        # print(i, k, v)
        for j, node in enumerate(p_states.keys()): # iterate over states node
            # print(j, node)
            if k in path_to_all_nodes[node]:
                # print(i, k, node)
                # below we use j instead of node becasue not all nodes in the distribution are considered as state
                # all the zib are known hence not a part of state
                grad_array[i,j] = 1 # it will be a matrix of 0s and 1s
    return grad_array, index_of_lines
    
def grad_pseudo_with_p(p_pseudo, p_states):
    grad_array = np.zeros((len(p_pseudo), len(p_states)))
    for i, node_i in enumerate(p_pseudo.keys()): # meas node
        for j, node_j in enumerate(p_states.keys()): # state node
            if node_i == node_j:
                # print(node_i, node_j, j)
                grad_array[i][j] = 1
                break # if match found, break the search
    return grad_array

def grad_vnode_with_p(v_meas, p_states, path_to_all_nodes, R_line):
    grad_array = np.zeros((len(v_meas), len(p_states)))
    for i, node_i in enumerate(v_meas.keys()): # meas node
        for j, node_j in enumerate(p_states.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            grad_array[i][j] = -(sum(R_line[item] for item in common_lines)) * 2
    return grad_array
            
def grad_vnode_with_v0(v_meas):
    grad_array = np.ones((len(v_meas),1))
    return grad_array

def se_wls(x_est, z, jacobian_matrix, W, tol = None):
    ''' Weighted Least Square Estimate'''

    # some preprocessing for time saving during iterative newton method
    G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
    # G = np.matmul(jacobian_matrix.T, jacobian_matrix) # OLS
    Ginv = np.linalg.inv(G)
    
    count = 0
    delta_mat = np.zeros((jacobian_matrix.shape[1], 1000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 1000)) # meas residuals
    tol = tol if tol is not None else 10e-12
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
        # deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals) # OLS
        delta_mat[:,count] = deltax

        # get tolerance
        emax = np.max(deltax)

        # update values of state vars
        x_est = x_est + deltax
        results = np.vstack((results, x_est))
        count+=1
        # print(count)
    
    return x_est, emax, count, residuals_mat, delta_mat, results

def se_rr(x_est, z, jacobian_matrix, W, k = None, tol = None):
    ''' Ridge regression'''

    k = k if k is not None else 0 # 0 makes it ols
    # some preprocessing for time saving during iterative newton method
    G = np.matmul(jacobian_matrix.T, jacobian_matrix) + k * np.diag(np.ones((len(x_est))))
    # G = np.matmul(jacobian_matrix.T, jacobian_matrix) # OLS
    Ginv = np.linalg.inv(G)

    count = 0
    delta_mat = np.zeros((jacobian_matrix.shape[1], 1000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 1000)) # meas residuals
    tol = tol if tol is not None else 10e-12
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol and count < 1:

        # distflow backward sweep for calculating measurements

        # distflow forward sweep for calculating measurements

        # calculate h(x)    
        hx = np.matmul(jacobian_matrix, x_est)

        # calculate measurement residuals
        residuals = z - hx
        residuals_mat[:,count] = residuals

        # calculate deltax
        deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals)
        # deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals) # OLS
        delta_mat[:,count] = deltax

        # get tolerance
        emax = np.max(deltax)

        # update values of state vars
        x_est = x_est + deltax
        results = np.vstack((results, x_est))
        count+=1
    
    return x_est, emax, count, residuals_mat, delta_mat, results

def se_ols(x_est, z, jacobian_matrix, W, tol = None):
    ''' Ordinary Least Square Estimate'''

    # some preprocessing for time saving during iterative newton method
    G = np.matmul(jacobian_matrix.T, jacobian_matrix)
    # G = np.matmul(jacobian_matrix.T, jacobian_matrix) # OLS
    try:
        pseudo_inv = 0
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            pseudo_inv = 1
            print('pseudo')
            Ginv = np.linalg.pinv(jacobian_matrix)

    count = 0
    delta_mat = np.zeros((jacobian_matrix.shape[1], 1000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 1000)) # meas residuals
    tol = tol if tol is not None else 10e-12
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
        if pseudo_inv == 0:
            deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals)
        else: # cannot remember why I'm doing this
            deltax = np.matmul(Ginv, residuals)
        delta_mat[:,count] = deltax

        # get tolerance
        emax = np.max(deltax)

        # update values of state vars
        x_est = x_est + deltax
        results = np.vstack((results, x_est))
        count+=1
        # print(count)
    
    return x_est, emax, count, residuals_mat, delta_mat, results