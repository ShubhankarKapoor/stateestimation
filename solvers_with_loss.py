import numpy as np
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from some_funcs import refactor_estimates
from jacobian_calc import create_loss_jacobian

def se_wls_nonlin(x_est, z, W, P_line_meas, Q_line_mes, P_Load_state, P_Load_meas, path_to_all_nodes_list,
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
                    Q_line_mes, P_Load_meas, Vsq_mes, path_to_all_nodes_list, path_to_all_nodes,
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

# def cost(theta, H, y, W):
#     '''
#     (): shows the vars used in our code
#     Calculates cost of the function.
#     H & y(z) have their usual meaning.
#     theta(xest) - vector of coefficients.
#     '''
#     m = len(y)
#     # m = 1 # for SGD
#     # Calculating Cost
#                     # Loss
#     Error = H.dot(theta)-y
#     c = (1/(2*m)) * np.sum(np.square(Error) * W)
#     # c = np.sum(np.square(Error) * np.diag(W))/ (m)
#     return c

# def batch_gradient_descent(H, y, theta, W, lr, iterations, tol = None,
#                            loss = None, pflow = None, lossy_volt_est = None):
#     '''
#     returns array of thetas, cost of every iteration
#     H - H matrix
#     y - target variable matrix
#     theta - matrix of regression coefficients
#     W - diagonal matrix for weights
#     lr - learning rate
#     iteration - number of iteration to be run
#     '''
#     #Getting number of observations.
#     m = len(y)
#     tol = tol if tol is not None else 10e-12
#     loss = loss if loss is not None else 0
#     pflow = pflow if pflow is not None else 0
#     lossy_volt_est = lossy_volt_est if lossy_volt_est is not None else {}
#     # Initializing cost and theta's arrays with zeroes.
    
#     # thetas = theta # to store result every iter
#     costs = []
#     count = 0
#     emax = 100 # chosen higher than the tol
#     # grads = np.zeros((len(theta), iterations))
#     # Calculating theta for every iteration.
#     # for i in range(iterations):
#     while emax > tol and count < iterations :
#         # print(count)
#         thetas = theta # to see emax without storing results
#         cur_cost = cost(theta, H, y, W)
#         costs.append(cur_cost)
#         estimates = H.dot(theta)
#         if loss == 1 or pflow == 1: # voltage/pflow feedback using non linear estimates
#             if len(lossy_volt_est) == 6:    
#                 full_x_est, P_Load_est, Q_Load_est = refactor_estimates(lossy_volt_est['tot_states'], 
#                                                                         theta, lossy_volt_est['non_zib_index'], lossy_volt_est['num_buses'])
#                 V_est, _, Pline_est, Qline_est, _, _, k = LinDistFlowBackwardForwardSweep(
#                         P_Load_est, Q_Load_est, lossy_volt_est['which'], full_x_est[-1], loss, pflow, max_iter=1)
    
#                 # update the pline/qline
#                 Pline_known_meas = {k:Pline_est[k] for k in lossy_volt_est['plines']} # get pflow vals for known measurements
#                 Qline_known_meas = {k:Qline_est[k] for k in lossy_volt_est['plines']} # get qflow vals for known measurements
#                 estimates[0:len(Pline_known_meas)]=list(Pline_known_meas.values()) # update values
#                 estimates[len(Pline_known_meas):2*len(Pline_known_meas)]=list(Qline_known_meas.values()) # update values
                
#                 # update the voltage value for buses with measurements
#                 V_known_meas = {k:V_est[k] for k in lossy_volt_est['volt_buses']} # get voltage vals for known measurements            
#                 # print('max volt diff', max(abs(np.asarray(estimates[-len(V_known_meas):]) - np.asarray(list(V_known_meas.values())))))
#                 estimates[-len(V_known_meas):]=list(V_known_meas.values()) # update values
#             else:
#                 raise ValueError('Length of lossy_volt_est should be 6')
                
#         residuals = estimates -y
#         w_residuals = np.dot(W, residuals) # weighted residuals
#         gradient = 1/m*(np.dot(H.T, w_residuals)) # this is correct

#         theta = theta - lr * gradient # new weights/ thetas
#         # thetas = np.vstack((thetas, theta)) # store the result in a matrix
#         # emax = np.max(np.abs(thetas[count+1,:]-thetas[count,:])) # when you store every result
#         emax = np.max(np.abs(thetas-theta)) # without storing every result
#         # grads[:,count] = gradient
#         count+=1
#         if count % 30000==0:
#             print(count, emax, cur_cost)
#     return theta, thetas, costs, count, emax