#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:28:38 2022

@author: shub
"""
import numpy as np
import scipy as sp
import scipy.optimize
from networks import Network

def func_calc_lineflow(x, P_load, Q_load, node_a, node_b, downstream_branch, line_z_pu, V,
              V0=None, loss=None, pflow = None):

    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term

    # you can already have the matrices pre computed

    # x is the vector of P_ijs and Q_ijs

    # define matrix for multiplication with p and q vector
    # dont consider slack node in p, q vector
    # diag(node b) can be used for that
    load_vec = np.concatenate((P_load[1:], Q_load[1:])) # don't consider slack node
    p_load_mat = np.diag(np.ones(len(node_b)))
    load_mat = np.zeros((2*len(node_a), 2*len(node_a)))
    load_mat[0:len(node_a),0:len(node_a)] = p_load_mat
    load_mat[len(node_a):2*len(node_a),len(node_a):2*len(node_a)] = p_load_mat
    # mat*p and q vector
    f1 = np.matmul(load_mat, load_vec)

    # alternatively you could just use load values straight away
    # f1 = P_load[node_b], Q_load[node_b]

    # define matrix for multiplication with p_01/ p_ij, q_01/ q_ij terms
    line_flow_mat = np.zeros((2*len(node_a), 2*len(node_a)))
    line_flow_mat[0:len(node_a),0:len(node_a)] = downstream_branch
    line_flow_mat[len(node_a):2*len(node_a),len(node_a):2*len(node_a)] = downstream_branch

    # use the downstream matrix * x
    f2 = np.matmul(line_flow_mat, x)

    if pflow == 1:
        line_z_pu = np.expand_dims(line_z_pu, axis=1)
        # define matrix for multiplication with p_01^2/ p_ij^2, q_01^2/ q_ij^2 terms
        # same matrix as p and q multiplied with resistance() / volt(node_a) to obtain a new matrix -->> double check
        # pflow_loss_mat = np.zeros((len(node_a), len(node_a)))
        # qflow_loss_mat = np.zeros((len(node_a), len(node_a)))
        line_flow_loss_mat = np.zeros((2*len(node_a), 2*len(node_a)))
        # separate voltage term from the loop
        # then you have to run the loop only once
        # for i in range(pflow_loss_mat.shape[0]):
        #     pflow_loss_mat[i,i] = line_z_pu[i].real/V[node_a[i]]
        #     qflow_loss_mat[i,i] = line_z_pu[i].imag/V[node_a[i]]
        # replace for-loop
        pflow_loss_mat = np.diag(np.divide(line_z_pu.real,V[node_a]).reshape(len(node_a),))
        qflow_loss_mat = np.diag(np.divide(line_z_pu.imag,V[node_a]).reshape(len(node_a),))
        line_flow_loss_mat[0:len(node_a),0:len(node_a)] = pflow_loss_mat
        line_flow_loss_mat[0:len(node_a),len(node_a):2*len(node_a)] = pflow_loss_mat
        line_flow_loss_mat[len(node_a):2*len(node_a),0:len(node_a)] = qflow_loss_mat
        line_flow_loss_mat[len(node_a):2*len(node_a),len(node_a):2*len(node_a)] = qflow_loss_mat

        # martix * x^2
        f3 = np.matmul(line_flow_loss_mat, x**2)
    else:
        f3 = 0
    # add these up and return the vector of p_ij and q_ij
    x = f1 + f2 + f3 # check signs

    return x

def func_calc_volt(V, x, node_a, node_b, downstream_branch, line_z_pu,
              loss=None, pflow = None):

    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term

    # matrix mutiplication for v node
    # here we have considered all voltage values for multiplication
    # row elements refer to the voltages we are calculating and columns represent the preceding node
    # potentially last column should consist of 0s, this will hold true for all leaf nodes as well
    volt_conn = downstream_branch.T
    # f1 = np.matmul(volt_conn, V)
    f1 = V[node_a]

    # const matrix mul of res./ reac with line flow
    # pflow_mat = np.zeros((len(node_b), len(node_a)))
    # qflow_mat = np.zeros((len(node_b), len(node_a)))
    # # line_flow_mat = np.zeros((len(node_b), 2*len(node_a)))
    # for i in range(pflow_mat.shape[0]):
    #     pflow_mat[i,i] = line_z_pu[i].real
    #     qflow_mat[i,i] = line_z_pu[i].imag
    # replace for loop
    pflow_mat = np.diag(line_z_pu.real)
    qflow_mat = np.diag(line_z_pu.imag)
    # probably could replace by taking diagonal

    line_flow_mat = np.concatenate((pflow_mat, qflow_mat), axis =1)
    f2 = 2 * np.matmul(line_flow_mat, x)
    # mul of impedance and line flow with inverse of volt

    f3 = 0
    if loss == 1: # voltage loss term
        # define matrix for multiplication with p_01^2/ p_ij^2, q_01^2/ q_ij^2 terms
        # same matrix as p and q multiplied with resistance() / volt(node_a) to obtain a new matrix -->> double check
        # pflow_loss_mat = np.zeros((len(node_a), len(node_a)))
        # qflow_loss_mat = np.zeros((len(node_a), len(node_a)))
        # line_flow_loss_mat = np.zeros((2*len(node_a), 2*len(node_a)))
        # probs just use downstream.T matrix
        imp_term = np.expand_dims((np.abs(line_z_pu)**2),axis=1)
        line_flow_term = x[0:len(node_b)]**2 + x[len(node_b):2*len(node_b)]**2
        volt_invert_terms= imp_term * line_flow_term
        f3 = volt_invert_terms * 1/V[node_a]

    # add them up
    V = f1 - f2 + f3

    return V

def jacob_calc(x):
    pass

def newton(network, tol_x = None, max_iter= None):
    busNo = network.busNo
    V_slack = network.V_slack
    node_a = network.node_a
    node_b = network.node_b
    sparse = network.sparse
    # node_b = network.node_b
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    downstream_branch = network.downstream_branch
    load_powers = network.load_powers
    P_load = load_powers.real
    Q_load = load_powers.imag

    # initialize p_ij, q_ij
    np.random.seed(0)
    x = np.expand_dims(np.random.rand(len(node_a)*2),1)
    # initialize V
    V = np.expand_dims(np.random.rand((len(node_a)+1)),1)
    V[0][0] = 1. # make sure slack bus voltage is 1 pu

    # define matrices for p_ij/q_ij calc
    # pflow_loss_mat = np.zeros((len(node_a), len(node_a)))
    # qflow_loss_mat = np.zeros((len(node_a), len(node_a)))
    # line_flow_loss_mat = np.zeros((2*len(node_a), 2*len(node_a)))
    # for i in range(pflow_loss_mat.shape[0]):
    #     pflow_loss_mat[i,i] = line_z_pu[i].real/V[node_a[i]]
    #     qflow_loss_mat[i,i] = line_z_pu[i].imag/V[node_a[i]]
    
    #     line_flow_loss_mat[0:len(node_a),0:len(node_a)] = pflow_loss_mat
    #     line_flow_loss_mat[0:len(node_a),len(node_a):2*len(node_a)] = pflow_loss_mat
    #     line_flow_loss_mat[len(node_a):2*len(node_a),0:len(node_a)] = qflow_loss_mat
    #     line_flow_loss_mat[len(node_a):2*len(node_a),len(node_a):2*len(node_a)] = qflow_loss_mat

    results = x
    results_V = V
    # giving an insanely high number below so it converges with tol when max_iters are missing
    max_iter = 10e12 if max_iter is None else max_iter # max iterations without considering tolerance  
    tol_x = 0.000000000001 if tol_x is None else tol_x
    for i in range(int(max_iter)):
        # print(i, max_iter)
        # solve for lineflows, forward sweep
        x_new = func_calc_lineflow(x, P_load, Q_load, node_a, node_b, downstream_branch[1:, :], 
            line_z_pu, V, V0=None, loss=None, pflow = 1)

        # solve for node voltages, backward sweep
        V_new = func_calc_volt(V, x, node_a, node_b, downstream_branch, line_z_pu,
              loss=1, pflow = None)
        V_new = np.vstack((np.atleast_2d(V[0]), V_new))
        # J = jacob_calc(x)
        # dx = sp.sparse.linalg.spsolve(J, -F)
        # normF = np.linalg.norm(F)
        # normdx = np.linalg.norm(dx)
        # x = x + dx
        # if normF < tolF and normdx < toldx:         
        #     print ('Converged to x= {} in {} iterations'.format(x, i) )
        #     return x

        # if iter == max_iter:    
        #     print ('Non-Convergence after {} iterations!!!'.format(i))
        #     return x
        diff_x = x_new - x
        diff_V = V_new - V
        x = 1. * x_new
        V = 1. * V_new
        results = np.hstack((results, x))
        results_V = np.hstack((results_V, V))
        if np.max(np.abs(diff_x)) < tol_x and np.max(np.abs(diff_V)) < tol_x:
            print ('Converged to x= {} in {} iterations'.format(x, i) )
            return x, V, results, results_V

        if i == max_iter-1: # iter == max_iter:    
            print(x)
            print ('Non-Convergence after {} iterations!!!'.format(i))

            return x, V, results, results_V

network37 = Network('network37', sparse=False)

x, V, results, results_V = newton(network37)
