#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:44:59 2020

@author: shub
"""
import numpy as np
import pandas as pd
def create_loss_jacobian(P_Load_state, P_line_meas, Q_line_meas, P_Load_meas, 
                         Vsq_mes, path_to_all_nodes_list, path_to_all_nodes, R_line, X_line, LineData_Z_pu, V_est, 
                         Pline_est, Qline_est, num_states, num_meas):

    ''' Loss based Jacobian wo assumptions
        Potentially incorrect
    '''

    jacobian_loss_matrix = np.zeros((num_meas, num_states))

    # jacobian for pline wrt p
    grad_array, grad_array2, dict_for_pline_with_p_derivatives = grad_pline_with_p_loss(P_line_meas, 
                    P_Load_state, path_to_all_nodes_list, R_line, Pline_est, V_est)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_loss_matrix[0:meas_rows, 0:state_cols] = grad_array
    last_row_inserted = meas_rows

    # jacobian for qline wrt q
    grad_array, grad_array2, dict_for_qline_with_q_derivatives = grad_pline_with_p_loss(Q_line_meas, P_Load_state, path_to_all_nodes_list, 
                       X_line, Qline_est, V_est)
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly

    # jacobian for pbus wrt p
    grad_array = grad_pseudo_with_p(P_Load_meas, P_Load_state)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
    last_row_inserted = last_row_inserted + meas_rows

    # jacobian for qbus wrt q, should be same as above
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    last_row_inserted += meas_rows

    # jacobain of pbus/qbus with v0^2 is 0

    # jacobain of v^2 with v0^2
    grad_array, dict_for_v_derivatives = grad_vsq_with_v0sq_loss(Vsq_mes, Pline_est, Qline_est, 
                                         LineData_Z_pu, V_est, path_to_all_nodes_list)
    meas_rows = grad_array.shape[0]
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array

    # jacobain of pflow with v0^2
    grad_array, dict_for_v_derivatives = grad_pline_with_v0sq_loss(P_line_meas, 
                            Pline_est, Qline_est, R_line, LineData_Z_pu, V_est,  
                              path_to_all_nodes_list, dict_for_v_derivatives, grad_var = 'p')
    # grad_array = np.zeros((grad_array.shape))
    meas_rows = grad_array.shape[0]
    jacobian_loss_matrix[0:meas_rows, 2*state_cols:] = grad_array
    last_row_inserted = meas_rows

    # jacobain of qflow with v0^2
    grad_array, dict_for_v_derivatives = grad_pline_with_v0sq_loss(Q_line_meas, 
                            Pline_est, Qline_est, X_line, LineData_Z_pu, V_est,  
                              path_to_all_nodes_list, dict_for_v_derivatives, grad_var = 'q')
    # grad_array = np.zeros((grad_array.shape))
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly
    # below will help us to insert above calculated vals
    last_row_inserted = last_row_inserted + 2*len(V_est) # derivative of pbus and qbus is 0 wrt v0

    # jacobian for v^2 with p
    # grad_array = grad_vnode_with_p(Vsq_mes, P_Load_state, path_to_all_nodes, R_line)

    grad_array = grad_vnode_with_p_loss(Vsq_mes, P_Load_state, path_to_all_nodes_list, 
                                    Pline_est, R_line, LineData_Z_pu, V_est, dict_for_pline_with_p_derivatives)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array

    # jacobian for v^2 with q
    # grad_array = grad_vnode_with_p(Vsq_mes, P_Load_state, path_to_all_nodes, X_line)
    grad_array = grad_vnode_with_p_loss(Vsq_mes, P_Load_state, path_to_all_nodes_list, 
                                    Qline_est, X_line, LineData_Z_pu, V_est, dict_for_qline_with_q_derivatives)

    jacobian_loss_matrix[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array

    return jacobian_loss_matrix

def grad_pline_with_succeeding_p(lineflow, lineres, v1):
    '''
    Calculates the gradient of pline with the immediate succession node

    lineflow : pflow/ qflow value
    lineres : resistance/ reactance of the line
    v1 : voltage sq of the preceeding node

    '''
    return 1/(1-(2*lineres*lineflow/v1))

def grad_pline_with_p_loss(P_line_meas, P_Load_state, path_to_all_nodes_list, 
                           R_line, Pline_est, V_est):

    grad_array = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states)
    grad_array2 = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states)
    dict_for_pline_with_p_derivatives = {}
    for i , (k,v) in enumerate(P_line_meas.items()): # iterate over measurements line
        # break
        # print(i,k)

        for j, node in enumerate(P_Load_state.keys()): # iterate over states node
            # print('Node:', node)
            if k in path_to_all_nodes_list[node]: # calculate the grad
                grad = 1
                # iterate over set of lines in repeat order till you hit the meas line
                for line in reversed(path_to_all_nodes_list[node]):
                    if line == k: # for gradient calculation
                        # print('yeehaw', line)
                        grad = grad * grad_pline_with_succeeding_p(Pline_est[line], R_line[line], V_est[line[0]])
                        break    
                    else:
                        # calculate gradient for each line
                        grad = grad * grad_pline_with_succeeding_p(Pline_est[line], R_line[line], V_est[line[0]])
                        # print(line)
                # print(i, k, node)
                grad_array2[i,j] = 1 # it will be a matrix of 0s and 1s
                grad_array[i,j] = grad
                dict_for_pline_with_p_derivatives[k, node] = grad
    return grad_array, grad_array2, dict_for_pline_with_p_derivatives

def grad_v2_with_preceeding_v1(pflow, qflow, lineimp, v1):
    '''
    Calculates the gradient of v^2 with the immediate previous node

    pflow/qflow : pflow/ qflow value
    lineimp : impedance of the line
    v1 : voltage sq of the preceeding node

    '''
    return 1 + ((abs(lineimp)**2) * (pflow**2 + qflow**2) * (1/(v1**2)))

def grad_vsq_with_v0sq_loss(Vsq_mes, Pline_est, Qline_est, 
                                         LineData_Z_pu, V_est, path_to_all_nodes_list):

    dict_for_v_derivatives = {} # so that don't have to recalculate
    grad_array = np.zeros((len(Vsq_mes),1))
    for i, (node,v) in enumerate(Vsq_mes.items()):
        grad = 1
        for line in path_to_all_nodes_list[node]:
            try: # see if the voltage derivative val is already calculated
                grad = grad * dict_for_v_derivatives[line]
            except KeyError: # when the voltage derivative value isn't in dict
                grad_val = grad_v2_with_preceeding_v1(Pline_est[line], Qline_est[line], 
                                                 LineData_Z_pu[line], V_est[line[0]])
                grad = grad * grad_val
                dict_for_v_derivatives[line] = grad_val
        grad_array[i] = grad
        # print(i,node,v, grad)   
            # print(line)

    return grad_array, dict_for_v_derivatives

def grad_pline_with_preceeding_v(pflow, qflow, lineres, v1):
    '''
    Calculates the gradient of v^2 with the immediate previous node
    pflow/qflow : pflow/ qflow value
    lineres : resistance/ reactance of the line
    v1 : voltage of the preceeding node
    '''

    # double check this

    # val = -lineres * (pflow**2) * (1/(v1**2)) * grad_pline_with_succeeding_p(pflow, lineres, v1)

    # return val
    return (lineres) * (pflow**2 + qflow**2) * (1/(v1**2))

def grad_pline_with_v0sq_loss(P_line_meas, Pline_est, Qline_est, R_line, LineData_Z_pu, V_est,  
                              path_to_all_nodes_list, dict_for_v_derivatives, grad_var):


    grad_array = np.zeros((len(P_line_meas),1))

    for i , (k,v) in enumerate(P_line_meas.items()): # iterate over measurements line
        # print(i,k)
        grad = 1
        pline_with_preceeding_v_flag = 0
        # print(k, pline_with_preceeding_v_flag)

        for line in reversed(path_to_all_nodes_list[k[-1]]):

            if pline_with_preceeding_v_flag == 0:
                # print('only once',line)
                pline_with_preceeding_v_flag = 1 # once the first grad is calculated differently
                if grad_var == 'p':
                    grad = grad *  grad_pline_with_preceeding_v(Pline_est[line], Qline_est[line],
                                 R_line[line], V_est[line[0]])
                if grad_var == 'q':
                    grad = grad *  grad_pline_with_preceeding_v(Pline_est[line], Qline_est[line],
                                 R_line[line], V_est[line[0]])
            else:
                try: # see if the voltage derivative val is already calculated
                    grad = grad * dict_for_v_derivatives[line]
                except KeyError: # when the voltage derivative value isn't in dict
                    grad_val = grad_v2_with_preceeding_v1(Pline_est[line], Qline_est[line], 
                                                 LineData_Z_pu[line], V_est[line[0]])
                    grad = grad * grad_val
                    dict_for_v_derivatives[line] = grad_val
        grad_array[i] = grad

    return grad_array, dict_for_v_derivatives

def grad_vnode_with_preceeding_pline(pflow, lineres, lineimp, v1):
    '''
    Calculates the gradient of v^2 with the immediate previous pline
    eg: V_j wrt pij
    pflow : pflow/ qflow value
    lineres : resistance/ reactance of the line
    lineimp : impedance of the line
    v1 : voltage sq of the preceeding node (node i)

    '''
    # grad vsq_j with pline_ij/ qline_ij
    return -2*lineres + 2 * (abs(lineimp)**2) * pflow * (1/v1)

def grad_pline_with_preceeding_pline(pflow, lineres, v1):
    '''
    Calculates the gradient of pline with the immediate previous pline
    gradient of line j(j+1) wrt ij

    pflow : pflow/qflow of ij (previous line)
    lineres : resistance/ reactance of the line
    v1 : voltage sq of the preceeding node (node i)

    '''

    return 1 - (2*pflow*lineres)* (1/(v1**2))

def grad_vnode_with_p_loss(Vsq_mes, P_Load_state, path_to_all_nodes_list,
                            Pline_est, R_line, LineData_Z_pu, V_est, dict_for_pline_with_p_derivatives):

    dict_for_v_with_preceeding_pline, dict_for_pline_with_preceeding_pline = {}, {} # so that don't have to recalculate
    grad_array = np.zeros((len(Vsq_mes), len(P_Load_state)))
    for i, node_i in enumerate(Vsq_mes.keys()): # meas node
        reversed_ordered_path = list(reversed(path_to_all_nodes_list[node_i]))
        for j, node_j in enumerate(P_Load_state.keys()): # state node
            # print(node_i, node_j)
            if len(reversed_ordered_path) == 0:
                grad = 0 # it is 0 if for volt at slack bus as there is no path
            else: 
                grad = 1

            v_with_preceeding_pline_flag = 0
            for line_num, line in enumerate(reversed_ordered_path):
                # break
                if v_with_preceeding_pline_flag == 0: # gradient of v once with prev line flow
                    # print('v with pl')
                    v_with_preceeding_pline_flag = 1 # once the first grad is calculated differently
                    try: # see if the voltage derivative with pline is already calculated
                        grad = grad * dict_for_v_with_preceeding_pline[node_i, line]
                    except KeyError:
                        grad_val = grad_vnode_with_preceeding_pline(Pline_est[line], 
                                    R_line[line], LineData_Z_pu[line], V_est[line[0]])
                        grad = grad * grad_val
                        dict_for_v_with_preceeding_pline[(node_i, line)] = grad_val                          

                if len(path_to_all_nodes_list[node_i]) - 1 == line_num: # gradient of the last line with the state bus
                    # print('pl with p')
                    # grad pline with pbus, use the previous values calculated
                    # all the keys should exist
                    # all the keys will only exist if you have pflow in your meas else not
                    grad = grad * dict_for_pline_with_p_derivatives[(line),node_j]
                else: # gradient of pline with preceeding line
                    # print('pl with pl')    

                    try: # see if the pline derivative with prev pline is already calculated
                        preceding_line = reversed_ordered_path[line_num + 1]
                        grad = grad * dict_for_pline_with_preceeding_pline[line, preceding_line]
                    except KeyError: # when the voltage derivative value isn't in dict
                        grad_val = grad_pline_with_preceeding_pline(Pline_est[preceding_line], 
                                                R_line[preceding_line], V_est[preceding_line[0]])
                        grad = grad * grad_val
                        dict_for_pline_with_preceeding_pline[line, preceding_line] = grad_val
            # print(node_i, node_j, grad)
            grad_array[i][j] = grad

    return grad_array

###############################################################################
###############################################################################

def create_jacobian(P_line_meas, P_Load_state, P_Load_meas, path_to_all_nodes,
                    Vsq_mes, R_line, X_line, num_states, num_meas):
    ''' LinDistflow based Jacobian'''

    # V is square of voltage mag
    # initial case jacobian matrix
    # num_meas = 36*2 + 37*3 # 2 for pij and qij 3 for pj, qj, vj^2
    # num_states = 2 * 37 + 1 # 2 for pj, qj 1 for v0^2
    jacobian_matrix = np.zeros((num_meas, num_states))
    
    # call jacobian for pline_plflow
    grad_array, index_of_lines = grad_pline_with_p(P_line_meas, P_Load_state, path_to_all_nodes)
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

def grad_pline_with_p(P_line_meas, P_Load_state, path_to_all_nodes):
    grad_array = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states

    index_of_lines = {}
    for i , (k,v) in enumerate(P_line_meas.items()): # iterate over measurements line
        # print(i,k)
        index_of_lines[i] = k
        # print(i, k, v)
        for j, node in enumerate(P_Load_state.keys()): # iterate over states node
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

###############################################################################
###############################################################################

def create_loss_jacobian_ass(meas_P_line, P_Load_state, P_Load_meas, P_Load_est, Q_Load_est, path_to_all_nodes,
                    Vsq_mes, R_line, X_line, LineData_Z_pu, num_states, num_meas, iter_num,
                    jacobian_matrix_la, pre_calculated_info, x_est):
    ''' creates jacobian while considering losses and some assumptions'''

    # jacobian_matrix_la = np.zeros((num_meas, num_states)) # jacobian with loss assumption
    
    # grad for pline_p, pline_q, qline_p, qline_q
    # this changes every iteration
    # grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q = grad_pline_with_p_loss_ass(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, x_est[-1], P_Load_est, Q_Load_est)
    # grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q = grad_pline_with_p_loss_ass_updated(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, x_est[-1], P_Load_est, Q_Load_est, x_est)
    grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q = grad_pline_with_p_loss_ass_updated_fast(
        meas_P_line, pre_calculated_info['r_hat'], pre_calculated_info['x_hat'], x_est, P_Load_state, x_est[-1])
    # grad_array_pline_p2, grad_array_pline_q2, grad_array_qline_p2, grad_array_qline_q2 = grad_pline_with_p_loss_ass_updated_new(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, x_est[-1], P_Load_est, Q_Load_est)
    meas_rows = grad_array_pline_p.shape[0]
    state_cols = grad_array_pline_p.shape[1]
    # for pflow
    jacobian_matrix_la[0:meas_rows, 0:state_cols] = grad_array_pline_p # wrt p
    jacobian_matrix_la[0:meas_rows, state_cols:2*state_cols] = grad_array_pline_q # wrt q
    last_row_inserted = meas_rows

    # for qflow
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array_qline_p
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array_qline_q
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly

    # call jacobian for ppseudo_p
    # this is constant
    if iter_num == 0:
        grad_array = grad_pseudo_with_p(P_Load_meas, P_Load_state)
        meas_rows = grad_array.shape[0]
        state_cols = grad_array.shape[1]
        jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
        last_row_inserted = last_row_inserted + meas_rows

        # call jacobian for qpseudo_q, should be same as above
        # this is constant
        jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
        last_row_inserted += meas_rows

    else:
        last_row_inserted+= len(P_Load_meas)*2
    # gradient for vnode^2 with p and q
    # this changes every iteration
    # grad_array_v_p, grad_array_v_q = grad_vnode_with_p_loss_ass(Vsq_mes, P_Load_state, path_to_all_nodes, 
    #                                 R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    # grad_array_v_p, grad_array_v_q = grad_vnode_with_p_loss_ass_new(Vsq_mes, P_Load_state, path_to_all_nodes, 
    #                                 R_mat, X_mat, Z_mat, x_est, P_Load_est, Q_Load_est, Vsq_mes[0])
    grad_array_v_p, grad_array_v_q = grad_vnode_with_p_loss_ass_updated(Vsq_mes, P_Load_state, path_to_all_nodes, 
                        pre_calculated_info['R_mat'], pre_calculated_info['X_mat'], 
                        pre_calculated_info['Z_mat'], pre_calculated_info['additional_mat_r'], 
                        pre_calculated_info['additional_mat_x'], x_est, P_Load_est, Q_Load_est, Vsq_mes[0])
    # grad_array_v_p, grad_array_v_q = grad_vnode_with_p_loss_ass_updated_new(Vsq_mes, P_Load_state, path_to_all_nodes, 
    #                    R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    meas_rows = grad_array_v_p.shape[0]
    state_cols = grad_array_v_p.shape[1]
    # for vsq with p
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array_v_p
    # for vsq with q
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array_v_q

    # gradient for pflow/ qflow with v0^2
    # this changes every iteration
    # grad_array_pline_vnode, grad_array_qline_vnode = grad_pline_with_vnode_loss_ass(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, P_Load_est, Q_Load_est, x_est[-1])
    # grad_array_pline_vnode, grad_array_qline_vnode = grad_pline_with_vnode_loss_ass_updated(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, P_Load_est, Q_Load_est, x_est[-1])
    grad_array_pline_vnode, grad_array_qline_vnode = grad_pline_with_vnode_loss_ass_updated_fast(
        meas_P_line, pre_calculated_info['comb_idx1'], pre_calculated_info['comb_idx2'], 
        pre_calculated_info['sum_r'], pre_calculated_info['sum_x'], P_Load_state, x_est, x_est[-1])
    # grad_array_pline_vnode2, grad_array_qline_vnode2 = grad_pline_with_vnode_loss_ass_updated_new(
    #     meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, P_Load_est, Q_Load_est, x_est[-1])
    meas_rows = grad_array_pline_vnode.shape[0]

    # for pflow
    jacobian_matrix_la[0:meas_rows, 2*state_cols] = grad_array_pline_vnode
    last_row_inserted = meas_rows

    # for qflow
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols] = grad_array_qline_vnode   
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly

    # gradient for p/q with v0^2 is 0
    # this is constant, zero vector
    last_row_inserted = last_row_inserted + 2* len(P_Load_meas)
    # gradient for v^2 with v0^2
    # this changes every iteration
    # grad_array_vnode_v = grad_vnode_with_v0_loss_ass(Vsq_mes, P_Load_state, 
    #                                                   path_to_all_nodes, R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    # grad_array_vnode_v = grad_vnode_with_v0_loss_ass_new(Vsq_mes, P_Load_state, 
    #                                                   path_to_all_nodes, R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    # grad_array_vnode_v = grad_vnode_with_v0_loss_ass_updated(Vsq_mes, P_Load_state, path_to_all_nodes, 
    #                         R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    # grad_array_vnode_v = grad_vnode_with_v0_loss_ass_updated_new(Vsq_mes, P_Load_state, path_to_all_nodes, 
    #                         R_line, X_line, LineData_Z_pu, pre_calculated_info, 
    #                         P_Load_est, Q_Load_est, Vsq_mes[0])
    grad_array_vnode_v = grad_vnode_with_v0_loss_ass_updated_fast(pre_calculated_info['comb_idx1'], 
                pre_calculated_info['comb_idx2'], pre_calculated_info['v_RX_Z_comb_req'], 
                    P_Load_state, x_est, Vsq_mes[0])
    meas_rows = grad_array_vnode_v.shape[0]
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols] = grad_array_vnode_v

    return jacobian_matrix_la

def grad_pline_with_p_loss_ass(meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, V0, P_Load_est, Q_Load_est):

    grad_array_pline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_pline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states

    for i , (k,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,k)
        sum_p, sum_q = 0, 0 # sum of p and q downstream to that line
        idx = np.asarray(()) # indices where the nodes contribute to pflow/ qflow
        for j, node in enumerate(P_Load_state.keys()): # iterate over states node
            # print(j, node)
            if k in path_to_all_nodes[node]:
                sum_p+=P_Load_est[node]
                sum_q+=Q_Load_est[node]
                idx = np.append(idx, j)
                # print(i, k, node)
        grad_pline_with_p = 1 + 2 * R_line[k]*sum_p/V0
        grad_pline_with_q = 2 * R_line[k]*sum_q/V0
        grad_qline_with_p = 2 * X_line[k]*sum_p/V0        
        grad_qline_with_q = 1 + 2 * X_line[k]*sum_q/V0

        grad_array_pline_p[i, idx.astype(int)] = grad_pline_with_p
        grad_array_pline_q[i, idx.astype(int)] = grad_pline_with_q
        grad_array_qline_p[i, idx.astype(int)] = grad_qline_with_p
        grad_array_qline_q[i, idx.astype(int)] = grad_qline_with_q

    return grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q

def grad_pline_with_p_loss_ass_updated(meas_P_line, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, V0, P_Load_est, Q_Load_est, x_est):

    grad_array_pline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_pline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states

    for i , (k,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,k)

        idx = np.asarray(()) # indices where the nodes contribute to pflow/ qflow
        for j, node_a in enumerate(P_Load_state.keys()): # iterate over states node
            sum_pline_with_p, sum_pline_with_q, sum_qline_with_p, sum_qline_with_q = 0, 0, 0, 0 # sum of p and q downstream to that line
            # print(j, node_a)
            if k in path_to_all_nodes[node_a]: # if node is downstream
                # print(node_a,'Downstream')
                for _, node_k in enumerate(P_Load_state.keys()): # iterate over states node
                    if k in path_to_all_nodes[node_k]: # if node is downstream
                        common_path = path_to_all_nodes[node_a].intersection(path_to_all_nodes[node_k])
                        common_path = common_path-path_to_all_nodes[k[0]]
                        sum_R = sum(R_line[item] for item in common_path)
                        sum_X = sum(X_line[item] for item in common_path)
                        # print(node_a, node_k, common_path)
                        # pre calculate sum_R and sum_X
                        temp_sum_pline_p = P_Load_est[node_k] * sum_R
                        temp_sum_pline_q = Q_Load_est[node_k] * sum_R
                        temp_sum_qline_p = P_Load_est[node_k] * sum_X
                        temp_sum_qline_q = Q_Load_est[node_k] * sum_X
                        sum_pline_with_p+=temp_sum_pline_p
                        sum_pline_with_q+=temp_sum_pline_q
                        sum_qline_with_p+=temp_sum_qline_p
                        sum_qline_with_q+=temp_sum_qline_q                        

                grad_array_pline_p[i, j] = 1 + 2 * sum_pline_with_p/V0
                grad_array_pline_q[i, j] = 2 * sum_pline_with_q/V0
                grad_array_qline_p[i, j] = 2 * sum_qline_with_p/V0
                grad_array_qline_q[i, j] = 1 + 2 * sum_qline_with_q/V0
                
    return grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q

def pline_with_p_pre_calculated_terms(meas_P_line, P_Load_state, path_to_all_nodes, 
                            R_line, X_line):
    r_hat = np.zeros((len(P_Load_state), len(P_Load_state)))
    x_hat = np.zeros((len(P_Load_state), len(P_Load_state)))
    for i , (line,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,line)

        for j, node_a in enumerate(P_Load_state.keys()): # iterate over states node
            sum_pline_with_p, sum_pline_with_q, sum_qline_with_p, sum_qline_with_q = 0, 0, 0, 0 # sum of p and q downstream to that line
            # print(j, node_a)
            if line in path_to_all_nodes[node_a]: # if node is downstream
                # print(node_a,'Downstream')
                for k, node_k in enumerate(P_Load_state.keys()): # iterate over states node
                    if line in path_to_all_nodes[node_k]: # if node is downstream
                        common_path = path_to_all_nodes[node_a].intersection(path_to_all_nodes[node_k])
                        common_path = common_path-path_to_all_nodes[line[0]]
                        sum_R = sum(R_line[item] for item in common_path)
                        sum_X = sum(X_line[item] for item in common_path)
                    # else: # error above when considering all nodes, so trying to fix. Not sure if entirely correcct
                    #     sum_R = 0
                    #     sum_X = 0
                    r_hat[j][k] = sum_R
                    x_hat[j][k] = sum_X
    return r_hat, x_hat
                    
def grad_pline_with_p_loss_ass_updated_fast(meas_P_line, r_hat, x_hat, x_est, P_Load_state, V0):
    # would only wprk for (0,1) flow
    # could be modified to include more line flows
    if meas_P_line:
        grad_array_pline_p = (1 + 2 * np.matmul(r_hat,x_est[0:len(P_Load_state)])/ V0).reshape(1,-1)
        grad_array_pline_q = (    2 * np.matmul(r_hat,x_est[len(P_Load_state):2*len(P_Load_state)])/V0).reshape(1,-1)
        grad_array_qline_p = (    2 * np.matmul(x_hat,x_est[0:len(P_Load_state)])/V0).reshape(1,-1)
        grad_array_qline_q = (1 + 2 * np.matmul(x_hat,x_est[len(P_Load_state):2*len(P_Load_state)])/V0 ).reshape(1,-1)
    else:
        grad_array_pline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
        grad_array_pline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
        grad_array_qline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
        grad_array_qline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    return grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q
                       
def grad_pline_with_p_loss_ass_updated_new(meas_P_line, P_Load_state, path_to_all_nodes, R_line, X_line, V0, P_Load_est, Q_Load_est):

    grad_array_pline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_pline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_p = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states
    grad_array_qline_q = np.zeros((len(meas_P_line), len(P_Load_state))) # meas*states

    for i , (line,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,line)
        for j, (node_a, load_a) in enumerate(P_Load_est.items()): # state vars
            # print(j, node_a, load_a)
            temp_pline_with_p, temp_pline_with_q, temp_qline_with_p, temp_qline_with_q = 0, 0, 0, 0
            if line in path_to_all_nodes[node_a]: # see if node_a is downstream of line
                for node_k, load_k in P_Load_est.items(): # for every node in the DN
                    # print(node_k, load_k)
                    if line in path_to_all_nodes[node_k]: # if node_k is downstream of line
                        common_path = path_to_all_nodes[node_a].intersection(path_to_all_nodes[node_k]) # coomon path bw 2 nodes
                        common_path = common_path - path_to_all_nodes[line[0]] # consider common path downstream of line[0] as per formulation
                        R_hat = sum(R_line[item] for item in common_path)
                        X_hat = sum(X_line[item] for item in common_path)
                        # print(node_a, node_k, load_k, R_hat, X_hat)
                        temp_pline_with_p+=R_hat*load_k
                        temp_pline_with_q+=R_hat*Q_Load_est[node_k]
                        temp_qline_with_p+=X_hat*load_k
                        temp_qline_with_q+=X_hat*Q_Load_est[node_k]
                grad_array_pline_p[i][j] = 1 + 2/V0 * temp_pline_with_p
                grad_array_pline_q[i][j] =     2/V0 * temp_pline_with_q 
                grad_array_qline_p[i][j] =     2/V0 * temp_qline_with_p
                grad_array_qline_q[i][j] = 1 + 2/V0 * temp_qline_with_q
            
    return grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q

def get_r_x_z_mat(meas_V_nodes, P_Load_state, path_to_all_nodes, R_line, X_line, LineData_Z_pu):
    ''' Returns the constant matrices used for grad_vnode_with_p_loss_ass'''
    # somehow define it in the beginning
    # and extract a part of it depending on the problem
    # sensitivity matrices
    R_mat, X_mat, Z_mat = np.zeros((len(meas_V_nodes), len(P_Load_state))), np.zeros((len(meas_V_nodes), len(P_Load_state))), \
        np.zeros((len(meas_V_nodes)*len(P_Load_state), len(P_Load_state)))
    a = 0
    for i, node_i in enumerate(meas_V_nodes): # meas node
        path_to_node_i = path_to_all_nodes[node_i]
        for j, node_j in enumerate(P_Load_state.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            R_mat[i][j] = sum(R_line[item] for item in common_lines)
            X_mat[i][j] = sum(X_line[item] for item in common_lines)
            for k, node_k in enumerate(P_Load_state): # for sum of sq of impedance with each power term, see formula
                # print(a, k)
                # break
                # print(node_i, node_j, node_k)
                common_path = common_lines.intersection(path_to_all_nodes[node_k])
                # break
            #     sumzsq_p = sum(((abs(LineData_Z_pu[item])**2)*P_Load_est[node_k]) for item in common_path)
                Z_mat[a, k] = sum((abs(LineData_Z_pu[item])**2) for item in common_path)
            a+=1

    a = 0
    additional_mat_r = np.zeros((len(meas_V_nodes)*len(P_Load_state), len(P_Load_state)))
    additional_mat_x = np.zeros((len(meas_V_nodes)*len(P_Load_state), len(P_Load_state)))
    for i, node_i in enumerate(meas_V_nodes): # meas node
        path_to_node_i = path_to_all_nodes[node_i]
        # break
        for path in path_to_node_i: # each line param to node_i, r_ij
            for j, node_j in enumerate(P_Load_state.keys()): # state node, p_a
                # print(path)
                a=i*len(P_Load_state)+j
                for k, node_k in enumerate(P_Load_state): # p_n', for sum of resistance and reactance with each power term, see formula
                    # if node is downstream of ij and not node j
                    if path in path_to_all_nodes[node_k] and node_k!=path[1]:
                        common_path = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                        # only consider the nodes downstream of ij
                        common_path = common_path-path_to_all_nodes[path[1]]
                        
                        R_hat = sum(R_line[item] for item in common_path)
                        X_hat = sum(X_line[item] for item in common_path)
                        temp_term_r = R_line[path] * R_hat
                        temp_term_x = X_line[path] * X_hat
                        additional_mat_r[a][k] += temp_term_r
                        additional_mat_x[a][k] += temp_term_x
                    # print(a,k)
                # if first_path_run == 0:
                # a=i*len(P_Load_state)+j
                # print(i, j, a)
        # break

    return R_mat, X_mat, Z_mat, additional_mat_r, additional_mat_x

def grad_vnode_with_p_loss_ass(meas_V, P_Load_state, path_to_all_nodes, R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, V0):
    grad_array_v_p = np.zeros((len(meas_V), len(P_Load_state))) # vnode_with_p
    grad_array_v_q = np.zeros((len(meas_V), len(P_Load_state))) # vnode_with_q
    f = 0
    for i, node_i in enumerate(meas_V.keys()): # meas node
        for j, node_j in enumerate(P_Load_state.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            grad_array_v_p[i][j] = -(sum(R_line[item] for item in common_lines)) * 2
            grad_array_v_q[i][j] = -(sum(X_line[item] for item in common_lines)) * 2
            sumzsq_p, sumzsq_q = 0, 0
            f+=1
            # vv = []
            for k, node_k in enumerate(P_Load_state): # for sum of sq of impedance with each power term, see formula
                # print(node_i, node_j, node_k)
                common_path = common_lines.intersection(path_to_all_nodes[node_k])
                ##
                # sum_of_com_path = sum(abs(LineData_Z_pu[item])**2 for item in common_path)
                # sumzsq_p_temp = sum_of_com_path*P_Load_est[node_k]
                # sumzsq_p+=sumzsq_p_temp
                ## more compact form below
                sumzsq_p += sum(((abs(LineData_Z_pu[item])**2)*P_Load_est[node_k]) for item in common_path)
                sumzsq_q += sum(((abs(LineData_Z_pu[item])**2)*Q_Load_est[node_k]) for item in common_path)
                # vv.append(sumzsq_p)
                # sumzsq_p = sum(((abs(LineData_Z_pu[item])**2)*P_Load_est[node_k]) for item in common_path)
                # sumzsq_q = sum(((abs(LineData_Z_pu[item])**2)*Q_Load_est[node_k]) for item in common_path)
                # sumzsq_p = 2*sumzsq_p/V0
                # sumzsq_q = 2*sumzsq_q/V0
                ######
                # temp_sumsq = sum((abs(LineData_Z_pu[item])**2) for item in common_path)
                # temp_sumsq= temp_sumsq * (P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
                # sumzsq_pcomb+= temp_sumsq                
                # if k in path_to_all_nodes[node_a]: # if node is downstream
            # print(sumzsq_q)
            sumzsq_p = -2*sumzsq_p/V0
            sumzsq_q = -2*sumzsq_q/V0
            grad_array_v_p[i][j] += sumzsq_p
            grad_array_v_q[i][j] += sumzsq_q
        #     if f == 20:
        #         break
        # if f == 20:
        #     break
    return grad_array_v_p, grad_array_v_q

def grad_vnode_with_p_loss_ass_new(meas_V, P_Load_state, path_to_all_nodes, 
                                   R_mat, X_mat, Z_mat, x_est, P_Load_est, Q_Load_est, V0):
    Z_hat_p = np.matmul(Z_mat,x_est[0:len(P_Load_state)])
    # Z_hat_p = np.matmul(Z_mat,np.asarray(list(P_Load_est.values())))
    Z_hat_p = Z_hat_p.reshape(len(meas_V), len(P_Load_state))
    Z_hat_q = np.matmul(Z_mat,x_est[len(P_Load_state):2*len(P_Load_state)])
    # Z_hat_q = np.matmul(Z_mat,np.asarray(list(Q_Load_est.values())))
    Z_hat_q = Z_hat_q.reshape(len(meas_V), len(P_Load_state))
    # use the above ones to get the grads
    grad_array_v_p = -2*R_mat - 2/V0*(Z_hat_p) # vnode_with_p
    grad_array_v_q = -2*X_mat - 2/V0*(Z_hat_q) # vnode_with_q
    return grad_array_v_p, grad_array_v_q

def grad_vnode_with_p_loss_ass_updated(meas_V, P_Load_state, path_to_all_nodes, 
                       R_mat, X_mat, Z_mat, additional_mat_r, additional_mat_x, 
                           x_est, P_Load_est, Q_Load_est, V0):
    Z_hat_p = np.matmul(Z_mat,x_est[0:len(P_Load_state)])
    # Z_hat_p = np.matmul(Z_mat,np.asarray(list(P_Load_est.values())))
    Z_hat_p = Z_hat_p.reshape(len(meas_V), len(P_Load_state))
    Z_hat_q = np.matmul(Z_mat,x_est[len(P_Load_state):2*len(P_Load_state)])
    # Z_hat_q = np.matmul(Z_mat,np.asarray(list(Q_Load_est.values())))
    Z_hat_q = Z_hat_q.reshape(len(meas_V), len(P_Load_state))
    # for the additional term
    r_p_term = np.matmul(additional_mat_r,x_est[0:len(P_Load_state)])
    x_p_term = np.matmul(additional_mat_x,x_est[0:len(P_Load_state)])
    r_q_term = np.matmul(additional_mat_r,x_est[len(P_Load_state):2*len(P_Load_state)])
    x_q_term = np.matmul(additional_mat_x,x_est[len(P_Load_state):2*len(P_Load_state)])
    # reshape them
    r_p_term = r_p_term.reshape(len(meas_V), len(P_Load_state))
    x_p_term = x_p_term.reshape(len(meas_V), len(P_Load_state))
    r_q_term = r_q_term.reshape(len(meas_V), len(P_Load_state))
    x_q_term = x_q_term.reshape(len(meas_V), len(P_Load_state))
    # use the above ones to get the grads
    grad_array_v_p = -2*R_mat - 2/V0*(Z_hat_p) - 4/V0*(r_p_term + x_p_term) # vnode_with_p
    grad_array_v_q = -2*X_mat - 2/V0*(Z_hat_q) - 4/V0*(r_q_term + x_q_term) # vnode_with_q
    return grad_array_v_p, grad_array_v_q

def grad_vnode_with_p_loss_ass_updated_new(meas_V, P_Load_state, path_to_all_nodes, 
                       R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, V0):
    grad_array_v_p = np.zeros((len(meas_V), len(P_Load_state))) # vnode_with_p
    grad_array_v_q = np.zeros((len(meas_V), len(P_Load_state))) # vnode_with_q
    f = 0
    elems_comb = combination_of_loads(P_Load_state) # combination of node elems
    # common path of node combs and their respective line combs
    common_path_of_combs, lines_comb = combination_of_lines_to_nodes(elems_comb, path_to_all_nodes)
    for i, node_i in enumerate(meas_V.keys()): # meas node
        path_to_node_i = path_to_all_nodes[node_i]
        for j, node_j in enumerate(P_Load_state.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_node_i.intersection(path_to_all_nodes[node_j])
            grad_array_v_p[i][j] = -(sum(R_line[item] for item in common_lines)) * 2
            grad_array_v_q[i][j] = -(sum(X_line[item] for item in common_lines)) * 2
            sumzsq_p, sumzsq_q = 0, 0
            addn_term_p, addn_term_q = 0, 0
            f+=1
            # vv = []
            for k, node_k in enumerate(P_Load_state): # for sum of sq of impedance with each power term, see formula
                # print(node_i, node_j, node_k)
                common_path = common_lines.intersection(path_to_all_nodes[node_k])
                sumzsq_p += sum(((abs(LineData_Z_pu[item])**2)*P_Load_est[node_k]) for item in common_path)
                sumzsq_q += sum(((abs(LineData_Z_pu[item])**2)*Q_Load_est[node_k]) for item in common_path)
                R_hat, X_hat = 0, 0
                # comb of node_j and node_k
                try:
                    comb_line = lines_comb[(node_j, node_k)]
                except KeyError:
                    comb_line = lines_comb[(node_k, node_j)]
                _, _, sum_RX_comb = sum_comb_of_lines(comb_line, path_to_all_nodes, node_i, R_line, X_line)
                addn_term_p+= sum_RX_comb * P_Load_est[node_k]
                addn_term_q+= sum_RX_comb * Q_Load_est[node_k]
            sumzsq_p = -2*sumzsq_p/V0
            sumzsq_q = -2*sumzsq_q/V0
            addn_term_p, addn_term_q = -4*addn_term_p/ V0, -4*addn_term_q/ V0
            grad_array_v_p[i][j] += sumzsq_p + addn_term_p
            grad_array_v_q[i][j] += sumzsq_q + addn_term_q
    return grad_array_v_p, grad_array_v_q

def grad_pline_with_vnode_loss_ass(meas_P_line, P_Load_state, path_to_all_nodes, 
                                   R_line, X_line, P_Load_est, Q_Load_est, V0):
    ''' V0 is sq of voltage at slack bus '''
    grad_array_pline_vnode = np.zeros((len(meas_P_line))) # meas*states
    grad_array_qline_vnode = np.zeros((len(meas_P_line))) # meas*states

    for i , (k,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,k)
        sum_p, sum_q = 0, 0 # sum of p and q contributing to pflow/ qflow
        idx = np.asarray(()) # indices where the nodes contribute to pflow/ qflow
        for j, node in enumerate(P_Load_state.keys()): # iterate over states node
            # print(j, node)
            if k in path_to_all_nodes[node]:
                sum_p+=P_Load_est[node] # sum of nodes downstream of branch k
                sum_q+=Q_Load_est[node]
                # print(i, k, node)
        # print(sum_p, sum_q)
        grad_array_pline_vnode[i] = -R_line[k]/(V0**2) * (sum_p**2 + sum_q**2)
        grad_array_qline_vnode[i] = -X_line[k]/(V0**2) * (sum_p**2 + sum_q**2)

    return grad_array_pline_vnode, grad_array_qline_vnode

def grad_pline_with_vnode_loss_ass_updated(meas_P_line, P_Load_state, path_to_all_nodes, 
                                   R_line, X_line, P_Load_est, Q_Load_est, V0):
    ''' V0 is sq of voltage at slack bus '''
    # use grad_vnode_with_v0_loss_ass_new to generate combs of nodes
    grad_array_pline_vnode = np.zeros((len(meas_P_line))) # meas*states
    grad_array_qline_vnode = np.zeros((len(meas_P_line))) # meas*states

    # get combs of elements
    # make sure fix it for nodes downstream for line ij
    # at this stage it would work because everything is downstream of ij
    elems_comb = combination_of_loads(P_Load_state)

    for i , (k,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,k,v)
        sq_pline_term, sq_qline_term, double_pline_term, double_qline_term = 0, 0, 0, 0
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            if k in path_to_all_nodes[node_j] and k in path_to_all_nodes[node_k]:
                if node_j == node_k:
                    common_lines = path_to_all_nodes[node_j]
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    sum_R = sum(R_line[item] for item in common_lines)
                    sum_X = sum(X_line[item] for item in common_lines)
                    sq_term_temp = (P_Load_est[node_j]**2 + Q_Load_est[node_j]**2)
                    pline_sq_term_temp = sq_term_temp * sum_R
                    qline_sq_term_temp = sq_term_temp * sum_X
                    sq_pline_term+=pline_sq_term_temp
                    sq_qline_term+=qline_sq_term_temp
                else:
                    common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    sum_R = sum(R_line[item] for item in common_lines)
                    sum_X = sum(X_line[item] for item in common_lines)
                    double_term_temp = 2*(P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
                
                    pline_double_term_temp = double_term_temp * sum_R
                    qline_double_term_temp = double_term_temp * sum_X
                    double_pline_term+=pline_double_term_temp
                    double_qline_term+=qline_double_term_temp

        grad_array_pline_vnode[i] = - (1/(V0**2)) * (sq_pline_term + double_pline_term)
        grad_array_qline_vnode[i] = - (1/(V0**2)) * (sq_qline_term + double_qline_term)
        
    return grad_array_pline_vnode, grad_array_qline_vnode

def pline_with_vnode_calculated_terms(meas_P_line, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, elems_comb, non_zib_index_array):
# maybe create a df for pline with vnode and see if it works
# elems_comb[1] elems_comb[2] indices_for_elems_comb R_hat X_hat calc_PQ(could use indices) Final_result()
    
    # the 2 terms below can be directly used for any pline/qline calculation
    # these are the 2 big matrices multiplied with combination of loads in pline/qline calculation
    # should hold true for grads of any line with vnode: needs verification
    mat_r = np.zeros((len(meas_P_line), len(elems_comb))) # matrix used for multiplication with combs
    mat_x = np.zeros((len(meas_P_line), len(elems_comb))) # matrix used for multiplication with combs

    data = []
    meas_P_nodes_index = np.arange((len(P_Load_state)))
    for i , (k,v) in enumerate(meas_P_line.items()): # iterate over measurements line
        # print(i,k,v)
        r_list, x_list = [], []
        sq_pline_term, sq_qline_term, double_pline_term, double_qline_term = 0, 0, 0, 0
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            if k in path_to_all_nodes[node_j] and k in path_to_all_nodes[node_k]:
                if node_j == node_k:
                    common_lines = path_to_all_nodes[node_j]
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    sum_R = sum(R_line[item] for item in common_lines)
                    sum_X = sum(X_line[item] for item in common_lines)
                    idx1 = np.where(node_j == non_zib_index_array)[0][0] # index of node_j in non_zib_index_array
                    data.append([node_j, node_k, idx1, idx1, sum_R, sum_X])
                    r_list.append(sum_R) # for big mat_r
                    x_list.append(sum_X) # for big mat_x
                else:
                    common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    sum_R = sum(R_line[item] for item in common_lines) * 2 # 2 comes from 2P1P2 terms
                    sum_X = sum(X_line[item] for item in common_lines) * 2 
                    idx1 = np.where(node_j == non_zib_index_array)[0][0] # index of node_j in non_zib_index_array
                    idx2 = np.where(node_k == non_zib_index_array)[0][0] # index of node_j in non_zib_index_array
                    data.append([node_j, node_k, idx1, idx2, sum_R, sum_X])
                    r_list.append(sum_R) # for big mat_r
                    x_list.append(sum_X) # for big mat_x
            else: # these terms will be 0
                sum_R, sum_X = 0, 0
                idx1 = np.where(node_j == non_zib_index_array)[0][0] # index of node_j in non_zib_index_array
                idx2 = np.where(node_k == non_zib_index_array)[0][0] # index of node_j in non_zib_index_array
                data.append([node_j, node_k, idx1, idx2, sum_R, sum_X])
                r_list.append(sum_R) # for big mat_r
                x_list.append(sum_X) # for big mat_x
        # make a matrix of sum_r and sum_x for each matrix
        mat_r[i:] = r_list # vector of size elems comb
        mat_x[i:] = x_list # vector of size elems comb
    df = pd.DataFrame(data, columns=['elem1', 'elem2', 'idx1', 'idx2', 'sum_r', 'sum_x'])
    # return mat_r, mat_x for generic calcualtion of pline/ qline in distlow assumption pf
    return df, mat_r, mat_x

def grad_pline_with_vnode_loss_ass_updated_fast(meas_P_line, comb_idx1, comb_idx2, sum_r, sum_x, P_Load_state, x_est, V0):
    # had to separate from df as the pd wrapper around np is expensive
    # would only work for 0,1 lineflow
    if meas_P_line:
        p_term = x_est[comb_idx1]*x_est[comb_idx2]
        q_term = x_est[comb_idx1 + len(P_Load_state)]*x_est[comb_idx2 + len(P_Load_state)]
        pq_term = p_term + q_term
        grad_array_pline_vnode = np.array([(-1/V0**2)*(sum(pq_term * sum_r))])
        grad_array_qline_vnode = np.array([(-1/V0**2)*(sum(pq_term * sum_x))])
    else:
        grad_array_pline_vnode = np.zeros((len(meas_P_line))) # meas*states
        grad_array_qline_vnode = np.zeros((len(meas_P_line))) # meas*states
    return grad_array_pline_vnode, grad_array_qline_vnode

def grad_pline_with_vnode_loss_ass_updated_new(meas_P_line, P_Load_state, path_to_all_nodes, 
                                   R_line, X_line, P_Load_est, Q_Load_est, V0):

    grad_array_pline_vnode = np.zeros((len(meas_P_line))) # meas*states
    grad_array_qline_vnode = np.zeros((len(meas_P_line))) # meas*states

    # get combs of elements
    elems_comb = combination_of_loads(P_Load_state)
    pqs = []                
    for i, (line,v) in enumerate(meas_P_line.items()):
        # print(i, line, v)
        grad_line_p, grad_line_q = 0, 0 
        for node_j, node_k in elems_comb:
            if line in path_to_all_nodes[node_j] and line in path_to_all_nodes[node_k]: # if both nodes are downstream of the line
                # print(node_j, node_k)
                if node_j == node_k:
                    common_path = path_to_all_nodes[node_j]
                    pq_sum = P_Load_est[node_j]**2 + Q_Load_est[node_j]**2
                else: 
                    common_path = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    pq_sum = 2*(P_Load_est[node_j] * P_Load_est[node_k] + Q_Load_est[node_j] * Q_Load_est[node_k])

                pqs.append(pq_sum)
                common_path = common_path - path_to_all_nodes[line[0]] # only consider lines downstream of line[0] as per the formulation
                R_hat = sum(R_line[item] for item in common_path)
                X_hat = sum(X_line[item] for item in common_path)
                grad_line_p+= R_hat * pq_sum
                grad_line_q+= X_hat * pq_sum
                # print(node_j, node_k, R_hat, X_hat, grad_line_p, grad_line_q)
        grad_array_pline_vnode[i] = (-1/V0**2)*grad_line_p
        grad_array_qline_vnode[i] = (-1/V0**2)*grad_line_q
        pqs = np.array(pqs)
    return grad_array_pline_vnode, grad_array_qline_vnode

def grad_vnode_with_v0_loss_ass(meas_V, P_Load_state, path_to_all_nodes, R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, V0):
    grad_array_vnode_v = np.zeros((len(meas_V))) # vnode_with_p
    for i, node_i in enumerate(meas_V.keys()): # meas node
        sumzsq_p, sumzsq_pcomb, temp_sumzsq_p = 0, 0, 0

        for j, node_j in enumerate(P_Load_state.keys()): # for impedance sq bw p/q and vnode
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            temp_sumzsq_p = sum((abs(LineData_Z_pu[item])**2) for item in common_lines)
            temp_sumzsq_p= temp_sumzsq_p * (P_Load_est[node_j]**2 + Q_Load_est[node_j]**2)
            sumzsq_p+=temp_sumzsq_p

            temp_sumsq = 0
            for k, node_k in enumerate(P_Load_state): # for impedance sq bw i and combination of p/q nodes
                # below works on the assumption the nodes are ordered
                if node_k > node_j: # to avoid repitive combinations of nodes
                    # print(node_i, node_j, node_k)
                    common_path = common_lines.intersection(path_to_all_nodes[node_k])
                    temp_sumsq = sum((abs(LineData_Z_pu[item])**2) for item in common_path)
                    temp_sumsq= temp_sumsq * (P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
                    sumzsq_pcomb+= temp_sumsq
                else:
                    pass

        grad_array_vnode_v[i] = 1 + (1/(V0**2) * (sumzsq_p)) + (2/(V0**2) * sumzsq_pcomb)

    return grad_array_vnode_v

def grad_vnode_with_v0_loss_ass_new(meas_V, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, V0):
    grad_array_vnode_v = np.zeros((len(meas_V))) # vnode_with_p

    # get combs of elements
    elems_comb = combination_of_loads(P_Load_state)

    for i, node_i in enumerate(meas_V.keys()): # meas node

        sq_term, double_term = 0, 0
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            if node_j == node_k:
                common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
                com_path_sens = sum((abs(LineData_Z_pu[item])**2) for item in common_lines)
                sq_term_temp = (P_Load_est[node_j]**2 + Q_Load_est[node_j]**2)
                sq_term_temp = sq_term_temp * com_path_sens
                sq_term+=sq_term_temp
            else:
                common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j]).intersection(path_to_all_nodes[node_k])
                com_path_sens = sum((abs(LineData_Z_pu[item])**2) for item in common_lines)
                double_term_temp = 2*(P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
                double_term_temp = double_term_temp * com_path_sens
                double_term+=double_term_temp

        grad_array_vnode_v[i] = 1 +  (1/(V0**2)) * (sq_term + double_term)

    return grad_array_vnode_v

def grad_vnode_with_v0_loss_ass_updated(meas_V, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, V0):
    grad_array_vnode_v = np.zeros((len(meas_V))) # vnode_with_p

    # get combs of elements
    elems_comb = combination_of_loads(P_Load_state)

    for i, node_i in enumerate(meas_V.keys()): # meas node
        path_to_node_i = path_to_all_nodes[node_i]
        sq_term, double_term = 0, 0
        addn_sq_term, addn_double_term = 0, 0
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            if node_j == node_k:
                common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                common_lines = path_to_all_nodes[node_i].intersection(common_lines_power_nodes)
                com_path_sens = sum((abs(LineData_Z_pu[item])**2) for item in common_lines)
                sq_term_load = (P_Load_est[node_j]**2 + Q_Load_est[node_j]**2)
                sq_term_temp = sq_term_load * com_path_sens
                sq_term+=sq_term_temp
                # shouldn't be consdiered for first node on the line from slack node
                # it wont affect here as that node is ZIB
                # could be fixed in future to fit the case for a non-ZIB
                # sum of lines in common_path that are downstream of every line
                R_hat, X_hat = 0, 0
                for line in path_to_node_i:
                   for item in common_lines_power_nodes:
                       if line in path_to_all_nodes[item[0]]:
                           # print(line, item)
                           R_hat+= R_line[line]*R_line[item]
                           X_hat+= X_line[line]*X_line[item]
                RX_hat = R_hat + X_hat
                
                addn_sq_term_temp = sq_term_load * RX_hat
                addn_sq_term+=addn_sq_term_temp
                # additional_loss+= sum_RX_comb * power_term
            else:
                common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                common_lines = path_to_all_nodes[node_i].intersection(common_lines_power_nodes)
                com_path_sens = sum((abs(LineData_Z_pu[item])**2) for item in common_lines)
                double_term_load = 2*(P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
                double_term_temp = double_term_load * com_path_sens
                double_term+=double_term_temp
                R_hat, X_hat = 0, 0
                for line in path_to_node_i:
                   for item in common_lines_power_nodes:
                       if line in path_to_all_nodes[item[0]]:
                           # print(line, item)
                           R_hat+= R_line[line]*R_line[item]
                           X_hat+= X_line[line]*X_line[item]
                RX_hat = R_hat + X_hat
                addn_double_term_temp = double_term_load * RX_hat
                addn_double_term+=addn_double_term_temp

        grad_array_vnode_v[i] = 1 +  (1/(V0**2)) * (sq_term + double_term) +  (2/(V0**2)) * (addn_sq_term + addn_double_term)
        
    # for i, node_i in enumerate(meas_V.keys()): # meas node
    #     path_to_node_i = path_to_all_nodes[node_i]
    #     # break
    #     sq_term, double_term = 0, 0
    #     for path in path_to_node_i: # each line param to node_i, r_ij
    #         for (node_j, node_k) in elems_comb:
    #             if node_j == node_k: # sq terms
    #                 # if node is downstream of ij and not node j
    #                 if path in path_to_all_nodes[node_k] and node_k!=path[1]:
    #                     common_path = path_to_all_nodes[node_j]
    #                     common_path = common_path-path_to_all_nodes[path[1]]
    #                     R_hat = sum(R_line[item] for item in common_path)
    #                     X_hat = sum(X_line[item] for item in common_path)
    #                     sq_temp_term = (P_Load_est[node_j]**2 + Q_Load_est[node_j]**2)
    #                     sq_temp_term1 = R_line[path] * R_hat * sq_temp_term
    #                     sq_temp_term2 = X_line[path] * X_hat * sq_temp_term
    #                     sq_term = sq_temp_term1 + sq_temp_term2
    #             else:
    #                 # if node is downstream of ij and not node j
    #                 if path in path_to_all_nodes[node_j] and path in path_to_all_nodes[node_k] and node_j!=path[1] and node_k!=path[1]:
    #                     common_path = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
    #                     common_path = common_path-path_to_all_nodes[path[1]]
    #                     R_hat = sum(R_line[item] for item in common_path)
    #                     X_hat = sum(X_line[item] for item in common_path)
    #                     double_temp_term = 2*(P_Load_est[node_j] * P_Load_est[node_k]  + Q_Load_est[node_j] * Q_Load_est[node_k])
    #                     double_temp_term1 = R_line[path] * R_hat * double_temp_term
    #                     double_temp_term2 = X_line[path] * X_hat * double_temp_term
    #                     double_term = double_temp_term1 + double_temp_term2
    #                     pass
    #     grad_array_vnode_v[i] += (2/(V0**2)) * (sq_term + double_term)

    return grad_array_vnode_v
    
def grad_vnode_with_v0_loss_ass_updated_new(meas_V, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, LineData_Z_pu, pre_calculated_info, 
                            P_Load_est, Q_Load_est, V0):
    grad_array_vnode_v = np.zeros((len(meas_V))) # vnode_with_v
    
    # get combs of elements
    elems_comb = combination_of_loads(P_Load_state)
    # common_path_of_combs, lines_comb = combination_of_lines_to_nodes(elems_comb, path_to_all_nodes)

    for idxv, (node_v, val) in enumerate(meas_V.items()):

        # print (idxv, node_v, val)
        original_loss, additional_loss = 0, 0
        # break
        
        # all downstream nodes
        # common impedance bw subscripts of power terms and voltage node
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            # print(node_j, node_k, sum_RX_comb[(node_j, node_k)])
            if node_j == node_k: # square terms
                # common_path = path_to_all_nodes[node_v].intersection(path_to_all_nodes[node_j]) # for original loss
                power_term = P_Load_est[node_j]**2 + Q_Load_est[node_j]**2
                # no additional term for first node on the line from slack bus
                # in this node 1, wont affect becasue it is ZIB
                # could be fixed for the above case in future
                # the line comb terms should always have a line that is in path of vnode
                # else shouldn't be considered
            else: # other coupled terms
                # common_path = path_to_all_nodes[node_v].intersection(path_to_all_nodes[node_j]).intersection(path_to_all_nodes[node_k])
                power_term = 2*(P_Load_est[node_j]*P_Load_est[node_k] + Q_Load_est[node_j]*Q_Load_est[node_k])
                # no additional term for first node on the power line from slack bus
                # in this node 1, wont affect becasue it is 0
            # print(common_path)    
            # Z_hat = sum((abs(LineData_Z_pu[item]))**2 for item in common_path)
            # original_loss+= Z_hat * power_term
            original_loss+= pre_calculated_info['z_common_path'][node_v, (node_j,node_k)] * power_term
            # if z_common_path[(node_j, node_k)]!= Z_hat:
            #     print(node_j, node_k, z_common_path[(node_j, node_k)], Z_hat)
            #     break
            # for additional loss
            # common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k]) # for additional loss
            # _, _, sum_RX_comb = sum_comb_of_lines(lines_comb[(node_j, node_k)], path_to_all_nodes, node_v, R_line, X_line)
            # _, _, sum_RX_comb = sum_comb_of_lines2(node_v, R_line, X_line, common_lines_power_nodes, path_to_all_nodes)
            # ff = 0
            # print(sum_RX_comb, sum_RX_comb2)
            # if sum_RX_comb!= sum_RX_comb2:
            #     print(node_v, node_j, node_k, sum_RX_comb, sum_RX_comb2)
            #     # break
            #     ff = 1
        # if ff == 1:
        #     break
            # additional_loss+= sum_RX_comb * power_term
            additional_loss+= pre_calculated_info['v_node_RX_comb'][node_v, (node_j,node_k)] * power_term
            # get lines combs for node
        
        grad_array_vnode_v[idxv] = 1 + 1/V0*original_loss + 2/(V0**2)*additional_loss
    return grad_array_vnode_v

def vnode_with_v0_pre_calculated_terms(meas_V_nodes, P_Load_state, path_to_all_nodes, 
                            R_line, X_line, LineData_Z_pu):
    '''
    meas_V_nodes : all nodes where meas can be found, should be same as P_Load_state
    P_Load_state : all nodes that can have loads
    Returns pre calculated values required for v_node with v0 grad
    '''
    # get combs of elements
    elems_comb = combination_of_loads(P_Load_state)

    common_path_of_combs, lines_comb = combination_of_lines_to_nodes(elems_comb, path_to_all_nodes)

    v_node_RX_comb = {}
    z_common_path = {}

    for node_v in meas_V_nodes:
        # print (node_v)
        # all downstream nodes
        # common impedance bw subscripts of power terms and voltage node
        for (node_j, node_k) in elems_comb:
            # print(node_j, node_k)
            # print(node_j, node_k, sum_RX_comb[(node_j, node_k)])
            if node_j == node_k: # square terms
                common_path = path_to_all_nodes[node_v].intersection(path_to_all_nodes[node_j]) # for original loss
            else: # other coupled terms
                common_path = path_to_all_nodes[node_v].intersection(path_to_all_nodes[node_j]).intersection(path_to_all_nodes[node_k])
            # sum of square impedance of common lines
            Z_hat = sum((abs(LineData_Z_pu[item]))**2 for item in common_path)
            z_common_path[node_v, (node_j,node_k)] = Z_hat
            # for additional loss
            common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k]) # for additional loss
            # _, _, sum_RX_comb = sum_comb_of_lines(lines_comb[(node_j, node_k)], path_to_all_nodes, node_v, R_line, X_line)
            _, _, sum_RX_comb = sum_comb_of_lines2(node_v, R_line, X_line, common_lines_power_nodes, path_to_all_nodes)
            v_node_RX_comb[node_v, (node_j,node_k)] = sum_RX_comb
            
    return v_node_RX_comb, z_common_path

def vnode_with_v0_pre_calc_terms_fast(meas_V_nodes, elems_comb, path_to_all_nodes, 
                                      R_line, X_line, LineData_Z_pu, non_zib_index_array):
    data = []
    v_node_RX_comb = np.zeros((len(meas_V_nodes), len(elems_comb)))
    z_common_path = np.zeros((len(meas_V_nodes), len(elems_comb)))

    for idxv, node_v in enumerate(meas_V_nodes):
        # all downstream nodes
        # common impedance bw subscripts of power terms and voltage node
        for idx_elem, (node_j, node_k) in enumerate(elems_comb):
            # print(idx_elem, node_j, node_k)
            if node_j == node_k: # square terms
                common_lines_power_nodes = path_to_all_nodes[node_j] # for additional loss
                common_path = path_to_all_nodes[node_v].intersection(common_lines_power_nodes) # for original loss
                _, _, sum_RX_comb = sum_comb_of_lines2(node_v, R_line, X_line, common_lines_power_nodes, path_to_all_nodes)
                Z_hat = sum((abs(LineData_Z_pu[item]))**2 for item in common_path)                
                v_node_RX_comb[idxv][idx_elem] = sum_RX_comb
                z_common_path[idxv][idx_elem] = Z_hat
                if idxv == 0: # needs to be done only once
                    idx1 = np.where(node_j == non_zib_index_array)[0][0]
                    data.append([node_j, node_k, idx1, idx1])
            else: # other coupled terms
                common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k]) # for additional loss
                common_path = path_to_all_nodes[node_v].intersection(common_lines_power_nodes)
                _, _, sum_RX_comb = sum_comb_of_lines2(node_v, R_line, X_line, common_lines_power_nodes, path_to_all_nodes)
                Z_hat = sum((abs(LineData_Z_pu[item]))**2 for item in common_path)
                v_node_RX_comb[idxv][idx_elem] = sum_RX_comb * 2 # 2 is for coupled terms
                z_common_path[idxv][idx_elem] = Z_hat * 2
                if idxv == 0: # needs to be done only once
                    idx1 = np.where(node_j == non_zib_index_array)[0][0]
                    idx2 = np.where(node_k == non_zib_index_array)[0][0]
                    data.append([node_j, node_k, idx1, idx2])

    v_RX_Z_comb = z_common_path + 2 * v_node_RX_comb # 2 is here because it is in the formulation
    df = pd.DataFrame(data, columns=['elem1', 'elem2', 'idx1', 'idx2'])
    return df, v_RX_Z_comb

def grad_vnode_with_v0_loss_ass_updated_fast(comb_idx1, comb_idx2, v_RX_Z_comb_req, 
                                             P_Load_state, x_est, V0):
    # had to separate from df as the pd wrapper around np is expensive
    # would only work for 0,1 lineflow
    # comb_idx1 = np.array(df.idx1)
    # comb_idx2 = np.array(df.idx2)
    p_term = x_est[comb_idx1]*x_est[comb_idx2]
    q_term = x_est[comb_idx1 + len(P_Load_state)]*x_est[comb_idx2 + len(P_Load_state)]
    pq_term = p_term + q_term

    grad_array_vnode_v = 1 + 1/V0*(np.matmul(v_RX_Z_comb_req, pq_term))
    return grad_array_vnode_v


def sum_comb_of_lines(lines_comb_for_nodes, path_to_all_nodes, node_v, R_line, X_line):
    ''' returns sum of combs of lines for the node wrt voltage meas'''
    # only need to be run once
    path_to_node_v = path_to_all_nodes[node_v]

    R_hat = sum(R_line[item[0]]*R_line[item[1]] for item in lines_comb_for_nodes if path_to_node_v.intersection(item))
    X_hat = sum(X_line[item[0]]*X_line[item[1]] for item in lines_comb_for_nodes if path_to_node_v.intersection(item))
    sum_RX_hat = R_hat + X_hat
    # you can get a dictionary for vnode key and line key
    return R_hat, X_hat, sum_RX_hat

def sum_comb_of_lines2(node_v, R_line, X_line, common_lines_power_nodes, path_to_all_nodes):
    ''' returns sum of combs of lines for the node wrt voltage meas'''
    path_to_node_i =  path_to_all_nodes[node_v]
    R_hat, X_hat = 0, 0
    for line in path_to_node_i:
       for item in common_lines_power_nodes:
           if line in path_to_all_nodes[item[0]]:
               # print(line, item)
               R_hat+= R_line[line]*R_line[item]
               X_hat+= X_line[line]*X_line[item]
    RX_hat = R_hat + X_hat
    
    return R_hat, X_hat, RX_hat

def combination_of_loads(P_Load_state):
    ''' return combination of load elements '''
    # get combs of elements
    elems = list(P_Load_state.keys())
    elems_comb = [] # all elements in square of p and q
    for i,elema  in enumerate(elems):
        for j, elemb in enumerate(elems):
            if j>=i:
                elems_comb.append((elema,elemb))
    return elems_comb

def combination_of_lines_to_nodes(elems_comb, path_to_all_nodes):
    ''' returns common path of nodes and the combinations of common path '''
    # common path to coupled and squared node terms
    common_path_of_combs = {}
    for (node_j, node_k) in elems_comb:
        # print(node_j, node_k)
        # could use this straight away rather than calculating at every iteration for grad calc
        common_path_of_combs[(node_j, node_k)] = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])

    # get lines combs for each node for the additional loss term
    # only needs to be done once
    lines_comb = {}
    for key,val in common_path_of_combs.items():
        # print(key,val)
        # if key == (1,2):
        #     break
        lines_comb_entries = []
        for i,elema  in enumerate(val):
            for j, elemb in enumerate(val):
                if j>i:
                    lines_comb_entries.append((elema,elemb))
        lines_comb[key] = lines_comb_entries
    
    return common_path_of_combs, lines_comb
