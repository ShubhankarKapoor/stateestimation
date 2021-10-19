import numpy as np
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from some_funcs import refactor_estimates, measurements_estimated_from_states
from jacobian_calc import create_loss_jacobian, create_loss_jacobian_ass
from BackwardForwardSweep import BackwardForwardSweep

def se_wls_nonlin_ass(x_est, z, W, meas_P_line, P_Load_state, meas_P_load,
           path_to_all_nodes, non_zib_index, meas_V, R_line, X_line, LineData_Z_pu, 
           num_states, num_meas, tot_state_vars, which, tol = None, iters= None):
    ''' Weighted Least Square Estimate with assumptions on losses
        num_states: state vars being estimated
        tot_state_vars: state vars being estimated + zib buses
    '''

    tol = tol if tol is not None else 10e-12 #10e-14
    iters = iters if iters is not None else 10e12 #10e-14

    count = 0
    delta_mat = np.zeros((num_states, 1000)) # delta in states
    residuals_mat = np.zeros((num_meas, 1000)) # meas residuals
    results = x_est
    emax = 100 # chosen higher than the tol
    while emax > tol:

        if count == 0:
            jacobian_matrix = np.zeros((num_meas, num_states)) # initialize jacobian with zeros
            R_mat, X_mat, Z_mat = np.zeros((len(meas_V), len(P_Load_state))), np.zeros((len(meas_V), len(P_Load_state))), \
                np.zeros((len(meas_V)*len(P_Load_state), len(P_Load_state))) # initialize with zeros
            additional_mat_r, additional_mat_x = np.zeros((len(meas_V)*len(P_Load_state), len(P_Load_state))), np.zeros((len(meas_V)*len(P_Load_state), len(P_Load_state)))
        # i think you are getting the below vals from PF: might be incorrect
        # should be just getting the vals from jacobian * xest
        # you are right
        # hx, _, _, _, _, _, _ = measurements_estimated_from_states(x_est, meas_P_line, 
        #         meas_V, which, non_zib_index, len(meas_P_load), tot_state_vars)

        # used in jacobian calc
        P_Load_est = dict(zip(P_Load_state.keys(), x_est[0:len(P_Load_state)]))
        Q_Load_est = dict(zip(P_Load_state.keys(), x_est[len(P_Load_state):2*len(P_Load_state)]))

        # recalculate jacobian: changes every iter here
        # R_mat, X_mat, Z_mat remains consistent: only calculated once
        jacobian_matrix, R_mat, X_mat, Z_mat, additional_mat_r, additional_mat_x = create_loss_jacobian_ass(
            meas_P_line, P_Load_state, meas_P_load, P_Load_est, Q_Load_est, path_to_all_nodes,
            meas_V, R_line, X_line, LineData_Z_pu, num_states, num_meas, count,
            jacobian_matrix, R_mat, X_mat, Z_mat, additional_mat_r, additional_mat_x, x_est)

        # calculate h(x)
        hx = np.matmul(jacobian_matrix, x_est)

        # calculate measurement residuals
        residuals = z - hx
        # residuals_mat[:,count] = residuals

        # changes every iter unlike lindist based SE
        G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
        Ginv = np.linalg.inv(G)

        # calculate deltax
        deltax = np.matmul(np.matmul(np.matmul(Ginv, jacobian_matrix.T), W), residuals)
        # deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals) # OLS
        # delta_mat[:,count] = deltax

        # update values of state vars
        x_est = x_est + deltax
        results = np.vstack((results, x_est))

        # get tolerance
        emax = np.max(np.abs(deltax))
        count+=1
        # print(count, emax)
    return x_est, emax, count, residuals_mat, delta_mat, results, jacobian_matrix

def se_wls_nonlin(x_est, z, W, P_line_meas, Q_line_meas, P_Load_state, P_Load_meas, 
        path_to_all_nodes_list, path_to_all_nodes, non_zib_index, Vsq_meas, R_line, 
        X_line, LineData_Z_pu, num_states, num_meas, which, tol = None, 
        lossy_volt_est = None):
    ''' Weighted Least Square Estimate'''

    tol = tol if tol is not None else 10e-15
    # loss = loss if loss is not None else 0
    # pflow = pflow if pflow is not None else 0
    lossy_volt_est = lossy_volt_est if lossy_volt_est is not None else {}

    count = 0
    delta_mat = np.zeros((num_states, 1000)) # delta in states
    residuals_mat = np.zeros((num_meas, 1000)) # meas residuals
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol:

        # distflow backward sweep for calculating measurements
        hx, full_x_est, P_Load_est, Q_Load_est, V_est, Pline_est, Qline_est = measurements_estimated_from_states(
            x_est, P_line_meas, Vsq_meas, which, non_zib_index, len(P_Load_meas), 
            lossy_volt_est['tot_states'])
        # # distflow forward sweep for calculating measurements

        # voltage/ pflow loss feedback
        # full_x_est, P_Load_est, Q_Load_est = refactor_estimates(lossy_volt_est['tot_states'], 
        #                                                         x_est, lossy_volt_est['non_zib_index'], lossy_volt_est['num_buses'])
        # V_est, _, Pline_est, Qline_est, _, _, k = LinDistFlowBackwardForwardSweep(
        #         P_Load_est, Q_Load_est, lossy_volt_est['which'], full_x_est[-1], loss = 1, pflow = 1)
        # print(V_est[0])
        jacobian_matrix = create_loss_jacobian(P_Load_state, P_line_meas, 
                    Q_line_meas, P_Load_meas, Vsq_meas, path_to_all_nodes_list, path_to_all_nodes,
                    R_line, X_line, LineData_Z_pu, V_est, Pline_est, Qline_est, num_states, num_meas)
        # print(jacobian_matrix)
        G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
        Ginv = np.linalg.inv(G)

        # calculate h(x)
        # some of these values should match fro, the above functions
        # good way to check it
        # hx = np.matmul(jacobian_matrix, x_est)

        # calculate measurement residuals
        residuals = z - hx
        residuals_mat[:,count] = residuals

        # calculate deltax
        deltax = np.matmul(np.matmul(np.matmul(Ginv, jacobian_matrix.T), W), residuals)
        # deltax = np.matmul(np.matmul(Ginv, jacobian_matrix.T), residuals) # OLS
        # delta_mat[:,count] = deltax

        # get tolerance
        emax = np.max(np.abs(deltax))
        # print(emax)
        # update values of state vars
        x_est = x_est + deltax
        results = np.vstack((results, x_est))
        count+=1
        # print(count)
        # break
    return x_est, emax, count, residuals_mat, delta_mat, results, jacobian_matrix

def cost(estimates, y, W):
    '''
    (): shows the vars used in our code
    Calculates cost of the function.
    H & y(z) have their usual meaning.
    theta(xest) - vector of coefficients.
    '''
    m = len(y)
    # m = 1 # for SGD
    # Calculating Cost
    # Loss
    Error = estimates-y
    c = (1/(2*m)) * np.sum(np.square(Error) * W)
    # c = np.sum(np.square(Error) * np.diag(W))/ (m)
    return c

def se_wls_la_bgd(x_est, z, W, lr, iterations, P_line_meas, Q_line_meas, P_Load_state, 
        P_Load_meas, path_to_all_nodes_list, path_to_all_nodes, non_zib_index, 
        Vsq_meas, R_line, X_line, LineData_Z_pu, num_states, num_meas, 
        tot_state_vars, which, tol = None):
    
    '''
    returns array of thetas, cost of every iteration
    H - H matrix
    y - target variable matrix
    theta - matrix of regression coefficients
    W - diagonal matrix for weights
    lr - learning rate
    iteration - number of iteration to be run
    '''
    #Getting number of observations.
    m = len(z)
    tol = tol if tol is not None else 10e-14
    # Initializing cost and theta's arrays with zeroes.



    # thetas = theta # to store result every iter
    costs = []
    count = 0
    emax = 100 # chosen higher than the tol

    while emax > tol and count < iterations :
        # print(count)
        x_ests = x_est # to see emax without storing results

        # these are your estimates here
        hx, _, _, _, _, _, _ = measurements_estimated_from_states(x_est, P_line_meas, 
                Vsq_meas, which, non_zib_index, len(P_Load_meas), tot_state_vars)
        
        # used in jacobian calc
        P_Load_est = dict(zip(P_Load_state.keys(), x_est[0:len(P_Load_state)]))
        Q_Load_est = dict(zip(P_Load_state.keys(), x_est[len(P_Load_state):2*len(P_Load_state)]))
        
        jacobian_matrix = create_loss_jacobian_ass(P_line_meas, P_Load_state, P_Load_meas, P_Load_est, Q_Load_est, path_to_all_nodes,
                    Vsq_meas, R_line, X_line, LineData_Z_pu, num_states, num_meas)
        cur_cost = cost(hx, z, W)
        costs.append(cur_cost)
        residuals = hx -z
        w_residuals = np.dot(W, residuals) # weighted residuals
        gradient = 1/m*(np.dot(jacobian_matrix.T, w_residuals)) # this is correct

        x_est = x_est - lr * gradient # new weights/ thetas
        # thetas = np.vstack((thetas, theta)) # store the result in a matrix
        # emax = np.max(np.abs(thetas[count+1,:]-thetas[count,:])) # when you store every result
        emax = np.max(np.abs(x_ests-x_est)) # without storing every result
        # grads[:,count] = gradient
        count+=1
        if count % 30000==0:
            print(count, emax, cur_cost)
    return x_est, x_ests, costs, count, emax, jacobian_matrix