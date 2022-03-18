#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:28:38 2022

@author: shub
"""
import numpy as np
import scipy as sp
import scipy.optimize
from scipy.sparse import csc_matrix
import time
# from networks import Network

def func_calc_lineflow(x, P_load, Q_load, node_a, node_b, num_lines, line_z_pu,
                V, load_mat, line_flow_mat, pflow_mat, qflow_mat, loss, pflow):
    '''
    x is the vector of P_ijs and Q_ijs
    load_mat: matrix is for multiplication with p and q vector
    line_flow_mat: for multiplication with p_01/ p_ij, q_01/ q_ij terms
    '''

    # dont consider slack node in p, q vector
    load_vec = np.concatenate((P_load[1:], Q_load[1:])) # don't consider slack node
    # mat*p and q vector
    f1 = np.matmul(load_mat, load_vec)
    # alternatively you could just use load values straight away
    # f1 = P_load[node_b], Q_load[node_b]

    # second term
    f2 = np.matmul(line_flow_mat, x)

    if pflow == 1:
        # define matrix for multiplication with p_01^2/ p_ij^2, q_01^2/ q_ij^2 terms
        line_flow_loss_mat = np.zeros((2*num_lines, 2*num_lines))
        # separate voltage term from the loop
        # then you have to run the loop only once
        # for i in range(pflow_loss_mat.shape[0]):
        #     pflow_loss_mat[i,i] = line_z_pu[i].real/V[node_a[i]]
        #     qflow_loss_mat[i,i] = line_z_pu[i].imag/V[node_a[i]]
        # replace for-loop
        # this matrix has to be recomputed every time as V changes each iter
        pflow_loss_mat = pflow_mat/V[node_a]
        qflow_loss_mat = qflow_mat/V[node_a]
        line_flow_loss_mat[0:num_lines,0:num_lines] = pflow_loss_mat
        line_flow_loss_mat[0:num_lines,num_lines:2*num_lines] = pflow_loss_mat
        line_flow_loss_mat[num_lines:2*num_lines,0:num_lines] = qflow_loss_mat
        line_flow_loss_mat[num_lines:2*num_lines,num_lines:2*num_lines] = qflow_loss_mat

        # martix * x^2
        f3 = np.matmul(line_flow_loss_mat, (x**2))
        # jacob = 2*(line_flow_loss_mat*x) + line_flow_mat
        # jacob = jacob + np.diag(np.ones(2*num_lines))
        jacob = 1
    else:
        f3 = 0
        jacob = 1
    # add these up and return the vector of p_ij and q_ij
    x = f1 + f2 + f3 # check signs
    return x, jacob

def func_calc_volt(V, x, node_a, node_b, num_lines, downstream_branch, line_z_pu, 
                  line_flow_mat_V, loss, pflow):

    # matrix mutiplication for v node
    # here we have considered all voltage values for multiplication
    # row elements refer to the voltages we are calculating (1 to N) and columns represent the preceding node
    # potentially last row should consist of 0s, this will hold true for all leaf nodes as well
    # coz they are never a preceeding node
    volt_conn = downstream_branch.T
    # f11 = np.matmul(volt_conn, V)
    f1 = V[node_a]

    # const matrix mul of res./ reac with line flow
    # pflow_mat = np.zeros((num_lines, num_lines))
    # qflow_mat = np.zeros((num_lines, num_lines))
    # # line_flow_mat = np.zeros((num_lines, 2*num_lines))
    # for i in range(pflow_mat.shape[0]):
    #     pflow_mat[i,i] = line_z_pu[i].real
    #     qflow_mat[i,i] = line_z_pu[i].imag
    # replace for loop
    f2 = 2 * np.matmul(line_flow_mat_V, x)
    # mul of impedance and line flow with inverse of volt

    f3 = 0
    if loss == 1: # voltage loss term
        # define matrix for multiplication with p_01^2/ p_ij^2, q_01^2/ q_ij^2 terms
        # same matrix as p and q multiplied with resistance() / volt(node_a) to obtain a new matrix -->> double check
        # pflow_loss_mat = np.zeros((num_lines, num_lines))
        # qflow_loss_mat = np.zeros((num_lines, num_lines))
        # line_flow_loss_mat = np.zeros((2*num_lines, 2*num_lines))
        # probs just use downstream.T matrix

        # this matrix has to be calculated every iteration
        imp_term = np.abs(line_z_pu)**2
        line_flow_term = x[0:num_lines]**2 + x[num_lines:2*num_lines]**2
        volt_invert_terms= imp_term * line_flow_term
        f3 = volt_invert_terms * 1/V[node_a]
    V_new  = 1. * V
    # add them up
    V_new[1:] = f1 - f2 + f3 # we arent solving for slack bus

    # jacob_term2 = volt_invert_terms * (1/(V[node_a]**2))
    # jacob_term2 = volt_conn * jacob_term2
    # jacob_V = volt_conn - jacob_term2

    # jacob_V = np.insert(jacob_V, 0, 0, 0) # adding how V0 changes for other terms
    # jacob_V = jacob_V + np.diag(np.ones(len(V))) # gradient of voltage of each node wrt itself
    # volt_conn[0][0] = 1 # how V0 changes with V0

    jacob_V = 1
    # jacob_V = volt_conn + jacob_term2

    return V_new, jacob_V

def jacob_calc(x):
    pass

def newton_no_jacob(network, P_load, Q_load, V0=None, loss=None, pflow = None, 
           tol_x = None, max_iter= None):
    ''' numpy based distflow '''
    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term

    busNo = network.busNo
    # V_slack = network.V_slack
    node_a = network.node_a
    node_b = network.node_b
    num_lines = network.num_lines
    sparse = network.sparse
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    downstream_branch = network.downstream_branch
    load_mat = network.load_mat
    line_flow_mat = network.line_flow_mat
    line_flow_mat_V = network.line_flow_mat_V
    pflow_mat = network.pflow_mat
    qflow_mat = network.qflow_mat

    # initialize p_ij, q_ij
    x = np.zeros((num_lines*2,1))
    # initialize V
    V = np.ones((num_lines+1,1))
    V[0][0] = 1 if V0 is None else V0 # replace 0 with slack node
    # print(V[0][0])

    # results = x
    # results_V = V
    # giving an insanely high number below so it converges with tol when max_iters are missing
    max_iter = 10e2 if max_iter is None else max_iter # max iterations without considering tolerance  
    tol_x = 0.000000000001 if tol_x is None else tol_x

    for i in range(int(max_iter)):

        # solve for lineflows, backward sweep
        x_new, _ = func_calc_lineflow(x, P_load, Q_load, node_a, node_b, 
            num_lines, line_z_pu, V, load_mat, line_flow_mat, pflow_mat, 
            qflow_mat, loss=loss, pflow = pflow) # get the func value and jacobian
        # solve for node voltages, forward sweep
        V_new, _ = func_calc_volt(V, x_new, node_a, node_b, num_lines, 
            downstream_branch, line_z_pu, line_flow_mat_V, loss=loss, pflow = pflow)

        # V_new = np.insert(V_new, 0, 1, 0)
        # V_new = np.vstack((np.atleast_2d(V[0]), V_new)) # write this in an efficient way
        diff_V = V_new - V
        x = 1. * x_new
        V = 1. * V_new
        # results = np.hstack((results, x))
        # results_V = np.hstack((results_V, V))
        if np.max(np.abs(diff_V)) < tol_x:
            # print ('Converged to  in {} iterations'.format( i) )
            # return x, V, results, results_V
            return x, V
        if i == max_iter-1: # iter == max_iter:    
            # print(x)
            print ('Non-Convergence after {} iterations!!!'.format(i))
            # return x, V, results, results_V
            return x, V
# network37 = Network('network37', sparse=False)
# load_powers = network37.load_powers
# P_load = load_powers.real
# Q_load = load_powers.imag
# loss, pflow = 1, 1
# # x2, V2, results, results_V = newton_no_jacob(network37, P_load, Q_load, loss=1, pflow = 1)
# x2, V2 = newton_no_jacob(network37, P_load, Q_load, loss=1, pflow = 1)

###############################################################################
###############################################################################

def func_jacob_calc(x, P_load, Q_load, node_a, node_b, num_lines, line_z_pu,
            V, load_mat, line_flow_mat_V, pflow_mat, qflow_mat, line_flow_mat_all, 
            voltage_graph, downstream_branch, pline_val, volt_val, loss, pflow):
    '''
    x is the vector of P_ijs and Q_ijs
    load_mat: matrix is for multiplication with p and q vector
    '''
    # calculate function values based on lineflows
    # p, q bus injection doesnt represent the vars we are solving for in this case
    if pline_val == 1:
        load_vec = np.concatenate((P_load[1:], Q_load[1:])) # don't consider slack node
        # mat*p and q vector
        f1 = np.matmul(load_mat, load_vec)
    
        # mat * p_ijs/ q_ijs
        f2 =  np.matmul(line_flow_mat_all, x)
        if pflow == 1:
            # matrix multiplication with p_01^2/ p_ij^2, q_01^2/ q_ij^2 terms
            line_flow_loss_mat = np.zeros((2*num_lines, 2*num_lines))
            # this matrix has to be recomputed every time as V changes each iter
            pflow_loss_mat = pflow_mat/V[node_a]
            qflow_loss_mat = qflow_mat/V[node_a]
            line_flow_loss_mat[0:num_lines,0:num_lines] = pflow_loss_mat
            line_flow_loss_mat[0:num_lines,num_lines:2*num_lines] = pflow_loss_mat
            line_flow_loss_mat[num_lines:2*num_lines,0:num_lines] = qflow_loss_mat
            line_flow_loss_mat[num_lines:2*num_lines,num_lines:2*num_lines] = qflow_loss_mat
    
            # martix * x^2
            f3 = np.matmul(line_flow_loss_mat, (x**2))
            # jacobian of lineflow with lineflow
            jacob =  line_flow_mat_all - 2*(line_flow_loss_mat*x)
        else:
            f3 = 0
            jacob = line_flow_mat_all
        # add these up and return the vector of p_ij and q_ij
        jacob = csc_matrix(jacob)
        f_x = -f1 + f2 -f3 # func value
        return f_x, jacob

    # for voltages
    if volt_val == 1:
        volt_conn = downstream_branch.T
        # matrix for multiplication with voltage vals
        # volt_mat = np.zeros((len(V),len(V))) # pre calc it
        # volt_mat[1:,1:] = voltage_graph
        # volt_mat[0][0] = 1  # how V0 changes with V0
        # volt_mat[1][0] = -1 # how V1 changes with V0
        # modifying it so not considering V0
        # volt_mat = voltage_graph
        # matrix * voltages
        f1 = np.matmul(voltage_graph, V[1:]) # looks correct
        f1[0] = f1[0] - V[0] # updating value becasue V0 wasnt considered
        # matrix multiplication with lineflow terms
        f2 = 2 * np.matmul(line_flow_mat_V, x)
        # mul of impedance and line flow with inverse of volt
    
        f3 = 0
        if loss == 1: # voltage loss term

            # this matrix has to be calculated every iteration
            imp_term = np.abs(line_z_pu)**2
            line_flow_term = x[0:num_lines]**2 + x[num_lines:2*num_lines]**2
            volt_invert_terms= imp_term * line_flow_term
            f3 = volt_invert_terms * 1/V[node_a]
            # calculate jacobian value for voltages
            # this term consist of how every voltage apart from V0 changes with other voltages
            jacob_term2 = volt_invert_terms * (1/(V[node_a]**2))
            jacob_term2 = volt_conn[:,1:] * jacob_term2
            # jacob_V = volt_mat[1:,:] + jacob_term2
            jacob_V = voltage_graph + jacob_term2

            # adding how V0 changes for other terms
            # jacob_V = np.insert(jacob_V, 0, 0, 0)
            # jacob_V[0][0] = 1 # how V0 changes with V0 
        else:
            jacob_V = voltage_graph
        # add them up
        f_V = f1 + f2 - f3 # we arent solving for slack bus
        jacob_V = csc_matrix(jacob_V)

        return f_V, jacob_V

def newton_with_jacob(network, P_load, Q_load, V0=None, loss=None, pflow = None, 
           tol_x = None, max_iter= None):
    ''' newton based distflow '''
    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term

    busNo = network.busNo
    # V_slack = network.V_slack
    node_a = network.node_a
    node_b = network.node_b
    num_lines = network.num_lines
    sparse = network.sparse
    # node_b = network.node_b
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    downstream_branch = network.downstream_branch
    load_mat = network.load_mat
    line_flow_mat = network.line_flow_mat
    line_flow_mat_V = network.line_flow_mat_V
    pflow_mat = network.pflow_mat
    qflow_mat = network.qflow_mat
    line_flow_mat_all = network.line_flow_mat_all

    # initialize p_ij, q_ij
    np.random.seed(0)
    # x = np.expand_dims(np.random.rand(num_lines*2),1)
    x = np.zeros((num_lines*2,1))
    # initialize V
    # V = np.expand_dims(np.random.rand((num_lines+1)),1)
    V = np.ones((num_lines+1,1))   
    V[0][0] = 1 if V0 is None else V0 # replace 0 with slack node

    # results = x
    # results_V = V
    # giving an insanely high number below so it converges with tol when max_iters are missing
    max_iter = 10e2 if max_iter is None else max_iter # max iterations without considering tolerance  
    tol_x = 0.000000000001 if tol_x is None else tol_x
    h = 1
    for i in range(int(max_iter)):

        # in powerflow term there isn't any need to include bus injections 
        # as it isn't a variable here
        # the variables are lineflows and bus voltages

        # solve for lineflows and their grads
        f_x, J = func_jacob_calc(x, P_load, Q_load, 
            node_a, node_b, num_lines, line_z_pu, V, load_mat, line_flow_mat_V, 
            pflow_mat, qflow_mat, line_flow_mat_all, voltage_graph, 
            downstream_branch, pline_val=1, volt_val=0, loss = loss, pflow = pflow)
        # GN
        dx = np.expand_dims((sp.sparse.linalg.spsolve(h*J, -f_x)), 1)
        x_new = x + dx # update the value

        # solve for voltage and their grads
        f_V, jacob_V = func_jacob_calc(x_new, P_load, Q_load, 
            node_a, node_b, num_lines, line_z_pu, V, load_mat, line_flow_mat_V, 
            pflow_mat, qflow_mat, line_flow_mat_all, voltage_graph, 
            downstream_branch, pline_val=0, volt_val=1, loss = loss, pflow = pflow)
        # GN
        dV = np.expand_dims((sp.sparse.linalg.spsolve(h*jacob_V, -f_V)),1)
        V_new  = 1. * V
        V_new[1:] = V[1:] + dV # update the value

        diff_x = x_new - x
        diff_V = V_new - V
        x = 1. * x_new
        V = 1. * V_new
        # results = np.hstack((results, x))
        # results_V = np.hstack((results_V, V))
        if np.max(np.abs(diff_V)) < tol_x:
            # print ('Converged to  in {} iterations'.format( i) )
            return x, V

        if i == max_iter-1: # iter == max_iter:    
            # print(x)
            print ('Non-Convergence after {} iterations!!!'.format(i))

            return x, V # x is a vec of p_ijs and q_ijs. V is sq of voltage mag

# x3, V3 = newton_with_jacob(network37, P_load, Q_load, loss=1, pflow = 1)
# x3, V3, results, results_V = newton_with_jacob(network37, P_load, Q_load, loss=1, pflow = 1)
