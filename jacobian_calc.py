#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:44:59 2020

@author: shub
"""
import numpy as np

def create_loss_jacobian(P_Load_state, P_line_meas, Q_line_meas, P_Load_meas, 
                         Vsq_mes, path_to_all_nodes_list, path_to_all_nodes, R_line, X_line, LineData_Z_pu, V_est, 
                         Pline_est, Qline_est, num_states, num_meas):

    ''' Loss based Jacobian wo assumptions
        Potentially incorrect
    '''

    jacobian_loss_matrix = np.zeros((num_meas, num_states))

    # jacobian for pline wrt p
    grad_array, grad_array2, dict_for_pline_with_p_derivatives = grad_pline_with_p_loss(P_line_meas, P_Load_state, path_to_all_nodes_list, 
                           R_line, Pline_est, V_est)
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
                              path_to_all_nodes_list, dict_for_v_derivatives, grad_var = 'p')
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

    return (lineres) * (pflow**2 + qflow**2) * (1/(v1**4))

def grad_pline_with_v0sq_loss(P_line_meas, Pline_est, Qline_est, R_line, LineData_Z_pu, V_est,  
                              path_to_all_nodes_list, dict_for_v_derivatives):

    # val = -lineres * (pflow**2) * (1/(v1**2)) * grad_pline_with_succeeding_p(pflow, lineres, v1)

    # return val
    return (lineres) * (pflow**2 + qflow**2) * (1/(v1**4))

def grad_pline_with_v0sq_loss(P_line_mes, Pline_est, Qline_est, R_line, LineData_Z_pu, V_est,  
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

    return 1 - (2*pflow*lineres)* (1/v1**2)

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

def create_loss_jacobian_ass(P_line_meas, P_Load_state, P_Load_meas, P_Load_est, Q_Load_est, path_to_all_nodes,
                    Vsq_mes, R_line, X_line, LineData_Z_pu, num_states, num_meas):
    ''' creates jacobian while considering losses and some assumptions'''

    jacobian_matrix_la = np.zeros((num_meas, num_states)) # jacobian with loss assumption
    
    # grad for pline_p, pline_q, qline_p, qline_q
    grad_array_pline_p, grad_array_pline_q, grad_array_qline_p, grad_array_qline_q = grad_pline_with_p_loss_ass(
        P_line_meas, P_Load_state, path_to_all_nodes, R_line, X_line, Vsq_mes[0], P_Load_est, Q_Load_est)
    meas_rows = grad_array_pline_p.shape[0]
    state_cols = grad_array_pline_p.shape[1]
    # for pflow
    jacobian_matrix_la[0:meas_rows, 0:state_cols] = grad_array_pline_p
    jacobian_matrix_la[0:meas_rows, state_cols:2*state_cols] = grad_array_pline_q
    last_row_inserted = meas_rows

    # for qflow
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array_qline_p
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array_qline_q
    last_row_inserted = 2*meas_rows # didn't do -1 because then this can be used directly
    
    # call jacobian for ppseudo_p
    grad_array = grad_pseudo_with_p(P_Load_meas, P_Load_state)
    meas_rows = grad_array.shape[0]
    state_cols = grad_array.shape[1]
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array
    last_row_inserted = last_row_inserted + meas_rows
    
    # call jacobian for qpseudo_q, should be same as above
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array
    last_row_inserted += meas_rows

    # call jacobian for vnode^2 with p and q
    grad_array_v_p, grad_array_v_q = grad_vnode_with_p_loss_ass(Vsq_mes, P_Load_state, path_to_all_nodes, 
                                   R_line, X_line, LineData_Z_pu, P_Load_est, Q_Load_est, Vsq_mes[0])
    meas_rows = grad_array_v_p.shape[0]
    state_cols = grad_array_v_p.shape[1]
    # for vsq with p
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 0:state_cols] = grad_array_v_p
    # for vsq with q
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, state_cols:2*state_cols] = grad_array_v_q
    # done till here

    # call jacobian with v0^2
    grad_array = grad_vnode_with_v0(Vsq_mes)
    jacobian_matrix_la[last_row_inserted:last_row_inserted + meas_rows, 2*state_cols:] = grad_array

    return jacobian_matrix_la

def grad_pline_with_p_loss_ass(P_line_meas, P_Load_state, path_to_all_nodes, R_line, X_line, V0, P_Load_est, Q_Load_est):

    grad_array_pline_p = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states
    grad_array_pline_q = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states
    grad_array_qline_p = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states
    grad_array_qline_q = np.zeros((len(P_line_meas), len(P_Load_state))) # meas*states

    for i , (k,v) in enumerate(P_line_meas.items()): # iterate over measurements line
        # print(i,k)
        sum_p, sum_q = 0, 0 # sum of p and q contributing to pflow/ qflow
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

def grad_vnode_with_p_loss_ass(v_meas, P_Load_state, path_to_all_nodes, R_line, X_line, Z_line, P_Load_est, Q_Load_est, V0):
    grad_array_v_p = np.zeros((len(v_meas), len(P_Load_state))) # vnode_with_p
    grad_array_v_q = np.zeros((len(v_meas), len(P_Load_state))) # vnode_with_q
    for i, node_i in enumerate(v_meas.keys()): # meas node
        for j, node_j in enumerate(P_Load_state.keys()): # state node
            # print(node_i, node_j)
            common_lines = path_to_all_nodes[node_i].intersection(path_to_all_nodes[node_j])
            grad_array_v_p[i][j] = -(sum(R_line[item] for item in common_lines)) * 2
            grad_array_v_q[i][j] = -(sum(X_line[item] for item in common_lines)) * 2
            sumzsq_p, sumzsq_q = 0, 0
            for k, node_k in enumerate(P_Load_state): # for sum of sq of impedance with each power term, see formula
                # print(node_i, node_j, node_k)    
                common_path = common_lines.intersection(path_to_all_nodes[node_k])
                sumzsq_p = sum(((abs(Z_line[item])**2)*P_Load_est[node_k]) for item in common_path)
                sumzsq_p = 2*sumzsq_p/V0
                sumzsq_q = sum(((abs(Z_line[item])**2)*Q_Load_est[node_k]) for item in common_path)
                sumzsq_q = 2*sumzsq_p/V0
            grad_array_v_p[i][j] += sumzsq_p
            grad_array_v_q[i][j] += sumzsq_q
    return grad_array_v_p, grad_array_v_q