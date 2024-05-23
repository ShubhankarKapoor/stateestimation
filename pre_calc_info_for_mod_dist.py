#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:57:59 2022

@author: shub
"""
# copied from impedance estimation on 20/05/2024
import sys
import os
sys.path.append("..")
# print(sys.path)
import numpy as np

parent_path = os.path.abspath("..")
sys.path.append(parent_path+"/stateestimation")
sys.path.append("/home/shub/Documents/phd/stateestimation/") # to deal with when running code from impedanceestimation


# from some_funcs import get_nodes_downstream_of_branch_with_req_nodes
from jacobian_calc import combination_of_loads_with_indices

def get_pre_calc_info(lines_key, non_zib_index_array, num_buses, path_to_all_nodes):

    pre_calculated_info = {}
    elems_comb, comb_idx1, comb_idx2 = combination_of_loads_with_indices(non_zib_index_array)
    pre_calculated_info['comb_idx1'] = comb_idx1
    pre_calculated_info['comb_idx2'] = comb_idx2
    pre_calculated_info['elems_comb'] = elems_comb

    # nodes interested in for V meas -- here all nodes
    V_node_idx = np.arange(num_buses)

    # for linear term in lineflow calc
    downstream_matrix = get_nodes_downstream_of_branch_with_req_nodes(lines_key, 
                                        V_node_idx, path_to_all_nodes)
    pre_calculated_info['downstream_matrix'] = downstream_matrix

    # using 2d coeff matrix
    # for loss term in lineflow calc
    big_r_mat_coeff = get_coeffs_for_big_r_mat_2d(elems_comb, lines_key, path_to_all_nodes)
    pre_calculated_info['big_r_mat_coeff'] = big_r_mat_coeff

    # test
    big_r_mat_coeff_test = get_coeffs_for_big_r_mat_2d_test(elems_comb, lines_key, path_to_all_nodes,num_buses)
    pre_calculated_info['big_r_mat_coeff_test'] = big_r_mat_coeff_test
    # coeff for sens matrix used in lin term of volt calc
    sens_mat_r_coeff = get_coeffs_for_r_in_sens_mat(V_node_idx, non_zib_index_array, 
                                                    lines_key, path_to_all_nodes)
    pre_calculated_info['sens_mat_r_coeff'] = sens_mat_r_coeff
    
    # z coeff for volt non linear term
    z_mat_coeff = get_z_coeff_for_non_lin_volt_term(V_node_idx, elems_comb, 
                                                lines_key, path_to_all_nodes)
    pre_calculated_info['z_mat_coeff'] = z_mat_coeff

    # inner rmat coeff for non linear term
    rhat_inner_coeff = get_inner_r_mat_coeff(elems_comb, lines_key, path_to_all_nodes)
    pre_calculated_info['rhat_inner_coeff'] = rhat_inner_coeff

    return  pre_calculated_info

def get_coeffs_for_big_r_mat_2d(elems_comb, lines_key, path_to_all_nodes):
    # this func is motivated from pline_with_vnode_calculated_terms

    big_r_mat_coeff = np.zeros((len(lines_key), len(elems_comb) * len(lines_key)))
    for i , k in enumerate(lines_key): # iterate over measurements line
        # print(i,k,v)
        for idx_comb, (node_j, node_k) in enumerate(elems_comb):
            # print(i*len(elems_comb)+idx_comb)
            # print(idx_comb, node_j, node_k)
            if k in path_to_all_nodes[node_j] and k in path_to_all_nodes[node_k]:
                if node_j == node_k:
                    common_lines = path_to_all_nodes[node_j]
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    big_r_mat_coeff[N, i*len(elems_comb)+idx_comb] = 1
                else:
                    common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    big_r_mat_coeff[N, i*len(elems_comb)+idx_comb] = 2
    return big_r_mat_coeff

def get_coeffs_for_big_r_mat_2d_test(elems_comb, lines_key, path_to_all_nodes, num_buses):
    # this func is motivated from pline_with_vnode_calculated_terms
    # i think this is correct, just incorrect implementation
    # you need to align the keys with r and x vector which i think is not happening
    # double check
    big_r_mat_coeff_test = np.zeros((len(lines_key), len(elems_comb)))
    for i , k in enumerate(lines_key): # iterate over measurements line
        # print(i,k,v)
        for idx_comb, (node_j, node_k) in enumerate(elems_comb):
            # print(i*len(elems_comb)+idx_comb)
            # print(idx_comb, node_j, node_k)
            if k in path_to_all_nodes[node_j] and k in path_to_all_nodes[node_k]:
                if node_j == node_k:
                    common_lines = path_to_all_nodes[node_j]
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    # big_r_mat_coeff_test[N, i*len(elems_comb)+idx_comb] = 1
                    big_r_mat_coeff_test[N, idx_comb] += 1
                else:
                    common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    # big_r_mat_coeff_test[N, i*len(elems_comb)+idx_comb] = 2
                    big_r_mat_coeff_test[N, idx_comb] += 2
    return big_r_mat_coeff_test

def get_coeffs_for_big_r_mat_3d(elems_comb, lines_key, path_to_all_nodes):
    # this func is motivated from pline_with_vnode_calculated_terms

    big_r_mat_coeff = np.zeros((len(lines_key), len(elems_comb), len(lines_key))) # as a 3d matrix
    for i , k in enumerate(lines_key): # iterate over measurements line
        # print(i,k,v)
        for idx_comb, (node_j, node_k) in enumerate(elems_comb):
            # print(idx_comb, node_j, node_k)
            if k in path_to_all_nodes[node_j] and k in path_to_all_nodes[node_k]:
                if node_j == node_k:
                    common_lines = path_to_all_nodes[node_j]
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    big_r_mat_coeff[N, idx_comb, i] = 1
                else:
                    common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k])
                    common_lines = common_lines - path_to_all_nodes[k[0]]
                    N = get_idx_of_common_line_in_keys(common_lines, lines_key)
                    big_r_mat_coeff[N, idx_comb, i] = 2
    return big_r_mat_coeff

def big_r_mat_from_3d(big_r_mat_coeff_3d, r, x, elems_comb_len):
    big_r_mat = np.zeros((len(r), elems_comb_len))
    big_x_mat = np.zeros((len(x), elems_comb_len))
    for i in range(big_r_mat_coeff_3d.shape[2]):
        val_r = np.matmul(r, big_r_mat_coeff_3d[:,:,i])
        val_x = np.matmul(x, big_r_mat_coeff_3d[:,:,i])
        big_r_mat[i,:] = val_r
        big_x_mat[i,:] = val_x
    return big_r_mat, big_x_mat

def get_coeffs_for_r_in_sens_mat(V_node_idx, non_zib_index_array, lines_key, path_to_all_nodes):
    sens_mat_r_coeff = np.zeros((len(lines_key), len(V_node_idx) * len(non_zib_index_array)))
    for i, node_i in enumerate(V_node_idx): # meas node
        path_to_node_i = path_to_all_nodes[node_i]
        for j, node_j in enumerate(non_zib_index_array): # state node
            # print(node_i, node_j)
            common_lines = path_to_node_i.intersection(path_to_all_nodes[node_j])
            N = get_idx_of_common_line_in_keys(common_lines, lines_key)
            sens_mat_r_coeff[N, i*len(non_zib_index_array)+j] = 1
    return sens_mat_r_coeff

def get_z_coeff_for_non_lin_volt_term(V_node_idx, elems_comb, lines_key, path_to_all_nodes):
    # for zmat coeff of non-linear volt term
    # motivated from vnode_with_v0_pre_calc_terms_fast

    z_mat_coeff = np.zeros((len(lines_key), len(V_node_idx) * len(elems_comb)))
    for idxv, node_v in enumerate(V_node_idx):
        # all downstream nodes
        for idx_elem, (node_j, node_k) in enumerate(elems_comb):
            # print(idx_elem, node_j, node_k)
            if node_j == node_k: # square terms
                common_lines_power_nodes = path_to_all_nodes[node_j] # for additional loss
                common_path = path_to_all_nodes[node_v].intersection(common_lines_power_nodes) # for original loss
                N = get_idx_of_common_line_in_keys(common_path, lines_key)
                z_mat_coeff[N, idxv*len(elems_comb)+idx_elem] = 1
            else: # other coupled terms
                common_lines_power_nodes = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k]) # for additional loss
                common_path = path_to_all_nodes[node_v].intersection(common_lines_power_nodes)
                N = get_idx_of_common_line_in_keys(common_path, lines_key)
                z_mat_coeff[N, idxv*len(elems_comb)+idx_elem] = 2
    return z_mat_coeff

def get_inner_r_mat_coeff(elems_comb, lines_key, path_to_all_nodes):
    # for r_hat coeff -- used to calculate the inner term of the coeff
    # motivated from vnode_with_v0_pre_calc_terms_fast

    rhat_inner_coeff = np.zeros((len(lines_key), len(elems_comb) * len(lines_key)))
    for idx_r, key in enumerate(lines_key):
        # print(idx_r, key)
        for idx_elem, (node_j, node_k) in enumerate(elems_comb):
            # print(idx_elem, node_j, node_k)
            if node_j == node_k: # square terms
                common_lines = path_to_all_nodes[node_j] # for additional loss
                # get idx of branches in path of node_j and node_k and downstream of key
                N = idx_of_branches_downstream_of_ij(key, common_lines, lines_key, path_to_all_nodes)
                rhat_inner_coeff[N, idx_r*len(elems_comb)+idx_elem] = 1

            else: # other coupled terms
                common_lines = path_to_all_nodes[node_j].intersection(path_to_all_nodes[node_k]) # for additional loss
                # get idx of branches in path of node_j and node_k and downstream of key
                N = idx_of_branches_downstream_of_ij(key, common_lines, lines_key, path_to_all_nodes)
                rhat_inner_coeff[N, idx_r*len(elems_comb)+idx_elem] = 2 # 2

    return rhat_inner_coeff

def idx_of_branches_downstream_of_ij(branch_ij, common_lines, lines_key, path_to_all_nodes):
    ''' returns index of branches downstream of branch ij in common lines '''
    downstream_path_of_ij = []
    for item in common_lines:
        if branch_ij in path_to_all_nodes[item[0]]: # branches downstream of ij, item refers to fg
            # print(branch_ij, item)
            downstream_path_of_ij.append(item)
    N = get_idx_of_common_line_in_keys(downstream_path_of_ij, lines_key)
    return N

def get_idx_of_common_line_in_keys(common_lines, lines_key):
    N = []
    if type(common_lines)!=list:
        common_lines = list(common_lines)
    for i in range(len(lines_key)):
        if lines_key[i] in common_lines:
            N.append(i)
    return N

def get_nodes_downstream_of_branch_with_req_nodes(lines_key, nodes, path_to_all_nodes):
    '''
    same as get_nodes_downstream_of_each_branch with only line keys and node num required 
    lines_key: lines you want to consider
    nodes: nodes you want to consider
    '''
    downstream_matrix = np.zeros((len(lines_key), len(nodes)))
    for i , k in enumerate(lines_key): # iterate over measurements line
        for j, node_a in enumerate(nodes): # iterate over states node
            if k in path_to_all_nodes[node_a]: # if node is downstream
                downstream_matrix[i][j] = 1
    return downstream_matrix
