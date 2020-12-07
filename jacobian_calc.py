#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:44:59 2020

@author: shub
"""
import numpy as np
def create_jacobian(P_line, P_Load_state, P_Load_meas, path_to_all_nodes,
                    V, R_line, X_line, num_states, num_meas):
    
    num_states, num_meas
    
    # initial case jacobian matrix
    # num_meas = 36*2 + 37*3 # 2 for pij and qij 3 for pj, qj, vj^2
    # num_states = 2 * 37 + 1 # 2 for pj, qj 1 for v0^2
    jacobian_matrix = np.zeros((num_meas, num_states))
    
    # call jacobian for pline_plflow
    grad_array, index_of_lines = grad_pline_with_p(P_line, P_Load_state, path_to_all_nodes)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix[0:meas_rows, 0:state_cols] = grad_array
    last_row_inserted = meas_rows
    
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
    grad_array = grad_vnode_with_p(V, P_Load_state, path_to_all_nodes, R_line)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
    
    # call jacobian for vnode^2 with q
    grad_array = grad_vnode_with_p(V, P_Load_state, path_to_all_nodes, X_line)
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    
    # call jacobian with v0^2
    grad_array = grad_vnode_with_v0(V)
    jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array
    
    return jacobian_matrix

def calc_residuals():
    residuals = 0
    return residuals

def estimate_states():
    x = 0
    return x

def grad_pline_with_p(S_line_meas, P_Load_states, path_to_all_nodes):
    grad_array = np.zeros((len(S_line_meas), len(P_Load_states))) # meas*states

    index_of_lines = {}
    for i , (k,v) in enumerate(S_line_meas.items()): # iterate over measurements line
        # print(i,k)
        index_of_lines[i] = k
        # print(i, k, v)
        for j, node in enumerate(P_Load_states.keys()): # iterate over states node
            # print(j, node)
            if k in path_to_all_nodes[node]:
                # print(i, k, node)
                # below we use j instead of node becasue not all nodes in the distribution are considered as state
                # all the zib are known hence not a part of state
                grad_array[i,j] = 1 # it will be a matrix of 0s and 1s
    return grad_array, index_of_lines
    
def grad_pseudo_with_p(p_pseudo, p_states):
    grad_array = np.zeros((len(p_pseudo), len(p_states)))
    for node_i in p_pseudo.keys(): # meas node
        for j, node_j in enumerate(p_states.keys()): # state node
            if node_i == node_j:
                # print(node_i, node_j, j)
                grad_array[node_i][j] = 1
                break # if match found, break the search
    return grad_array

def grad_vnode_with_p(v_meas, p_states, path_to_all_nodes, R_line):
    grad_array = np.zeros((len(v_meas), len(p_states)))
    for node_i in v_meas.keys(): # meas node
        for j, node_j in enumerate(p_states.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            grad_array[node_i][j] = -(sum(R_line[item] for item in common_lines)) * 2
    return grad_array
            
def grad_vnode_with_v0(v_meas):
    grad_array = np.ones((len(v_meas),1))
    return grad_array

'''
# make jacobian without the function
# initial case jacobian matrix
num_meas = 36*2 + 37*3 # 2 for pij and qij 3 for pj, qj, vj^2
num_states = 2 * 37 + 1 # 2 for pj, qj 1 for v0^2
jacobian_matrix = np.zeros((num_meas, num_states))

# call jacobian for pline_plflow
grad_array, index_of_lines = grad_pline_with_p(P_line, P_Load, path_to_all_nodes)
meas_rows = grad_array.shape[0]
state_cols = grad_array.shape[1]
jacobian_matrix[0:meas_rows, 0:state_cols] = grad_array
last_row_inserted = meas_rows

# call jacobian for qline_qflow, should be exactly same as above
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly

# call jacobian for ppseudo_p
grad_array = grad_pseudo_with_p(P_Load, P_Load)
meas_rows = grad_array.shape[0]
state_cols = grad_array.shape[1]
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
last_row_inserted = last_row_inserted + meas_rows

# call jacobian for qpseudo_q, should be same as above
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
last_row_inserted += meas_rows

# call jacobian for vnode^2 with p
grad_array = grad_vnode_with_p(V, P_Load, path_to_all_nodes, R_line)
meas_rows = grad_array.shape[0]
state_cols = grad_array.shape[1]
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array

# call jacobian for vnode^2 with q
grad_array = grad_vnode_with_p(V, P_Load, path_to_all_nodes, X_line)
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array

# call jacobian with v0^2
grad_array = grad_vnode_with_v0(V)
jacobian_matrix[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array
'''
