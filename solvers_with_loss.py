import numpy as np
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from some_funcs import refactor_estimates, measurements_estimated_from_states
from jacobian_calc import create_loss_jacobian, create_loss_jacobian_ass
from BackwardForwardSweep import BackwardForwardSweep

def se_wls_nonlin(x_est, z, W, P_line_meas, Q_line_meas, P_Load_state, P_Load_meas, path_to_all_nodes_list,
           path_to_all_nodes, Vsq_mes, R_line, X_line, LineData_Z_pu, num_states, num_meas, tol = None,
           loss = None, pflow = None, lossy_volt_est = None):
    ''' Weighted Least Square Estimate'''

    tol = tol if tol is not None else 10e-12
    loss = loss if loss is not None else 0
    pflow = pflow if pflow is not None else 0
    lossy_volt_est = lossy_volt_est if lossy_volt_est is not None else {}

    count = 0
    delta_mat = np.zeros((num_states, 1000)) # delta in states
    residuals_mat = np.zeros((num_meas, 1000)) # meas residuals
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol:

        # distflow backward sweep for calculating measurements

        # distflow forward sweep for calculating measurements

        # voltage/ pflow loss feedback
        full_x_est, P_Load_est, Q_Load_est = refactor_estimates(lossy_volt_est['tot_states'], 
                                                                x_est, lossy_volt_est['non_zib_index'], lossy_volt_est['num_buses'])
        V_est, _, Pline_est, Qline_est, _, _, k = LinDistFlowBackwardForwardSweep(
                P_Load_est, Q_Load_est, lossy_volt_est['which'], full_x_est[-1], loss, pflow, max_iter=1)
        # print(V_est[0])
        jacobian_matrix = create_loss_jacobian(P_Load_state, P_line_meas, 
                    Q_line_meas, P_Load_meas, Vsq_mes, path_to_all_nodes_list, path_to_all_nodes,
                    R_line, X_line, LineData_Z_pu, V_est, Pline_est, Qline_est, num_states, num_meas)
        # print(jacobian_matrix)
        G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
        Ginv = np.linalg.inv(G)

        # calculate h(x)
        # some of these values should match fro, the above functions
        # good way to check it
        hx = np.matmul(jacobian_matrix, x_est)

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

def se_wls_nonlin_ass(x_est, z, W, P_line_meas, Q_line_meas, P_Load_state, P_Load_meas, path_to_all_nodes_list,
           path_to_all_nodes, non_zib_index, Vsq_meas, R_line, X_line, LineData_Z_pu, 
           num_states, num_meas, tot_state_vars, which, tol = None):
    ''' Weighted Least Square Estimate with assumptions on losses
        num_states: state vars being estimated
        tot_state_vars: state vars being estimated + zib buses
    '''

    tol = tol if tol is not None else 10e-15

    count = 0
    delta_mat = np.zeros((num_states, 1000)) # delta in states
    residuals_mat = np.zeros((num_meas, 1000)) # meas residuals
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol:
        
        P_Load_est = dict(zip(P_Load_state.keys(), x_est[0:len(P_Load_state)]))
        Q_Load_est = dict(zip(P_Load_state.keys(), x_est[len(P_Load_state):2*len(P_Load_state)]))

        # distflow backward sweep for calculating measurements
        hx = measurements_estimated_from_states(x_est, P_line_meas, Vsq_meas, 
                                which, non_zib_index, len(P_Load_meas), tot_state_vars, max_iter=1)
        # distflow forward sweep for calculating measurements

        jacobian_matrix = create_loss_jacobian_ass(P_line_meas, P_Load_state, P_Load_meas, P_Load_est, Q_Load_est, path_to_all_nodes,
                    Vsq_meas, R_line, X_line, LineData_Z_pu, num_states, num_meas)

        G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix)
        Ginv = np.linalg.inv(G)

        # calculate h(x)
        # some of these values should match fro, the above functions
        # good way to check it
        # hx = np.matmul(jacobian_matrix, x_est)

        # calculate measurement residuals
        residuals = z - hx
        # residuals_mat[:,count] = residuals

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

    return x_est, emax, count, residuals_mat, delta_mat, results, jacobian_matrix