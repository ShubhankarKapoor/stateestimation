import numpy as np

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

def se_ols(x_est, z, jacobian_matrix, W, tol = None):
    ''' Ordinary Least Square Estimate
        Left the W here because it is easier to change the func in test.py
    '''

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

def se_wrr(x_est, z, jacobian_matrix, W, k, tol = None):
    ''' Weighted Least Square Estimate with L2 regularisation OR
        Weighted Ridge Regression
    '''
    k = k if k is not None else 0 # 0 makes it wls
    
    # some preprocessing for time saving during iterative newton method
    G = np.matmul(np.matmul(jacobian_matrix.T, W), jacobian_matrix) + k * np.diag(np.ones((len(x_est))))
    # G = np.matmul(jacobian_matrix.T, jacobian_matrix) # OLS
    Ginv = np.linalg.inv(G)
    
    count = 0
    delta_mat = np.zeros((jacobian_matrix.shape[1], 1000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 1000)) # meas residuals
    tol = tol if tol is not None else 10e-12
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol and count<2:

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

def se_rr(x_est, z, jacobian_matrix, W_rr, k = None, tol = None):
    ''' Ridge regression
        W_rr is the weight on the estimates here not measurements
    '''

    W_rr = W_rr if W_rr is not None else np.ones((len(x_est)))
    k = k if k is not None else 0 # 0 makes it ols
    # some preprocessing for time saving during iterative newton method
    G = np.matmul(jacobian_matrix.T, jacobian_matrix) + k * np.diag(W_rr)
    # G = np.matmul(jacobian_matrix.T, jacobian_matrix) # OLS
    Ginv = np.linalg.inv(G)

    count = 0
    delta_mat = np.zeros((jacobian_matrix.shape[1], 1000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 1000)) # meas residuals
    tol = tol if tol is not None else 10e-12
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol and count < 1000: # to run it only once

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

def cost(theta,X,y):
    '''
    (): shows the vars used in our code
    Calculates cost of the function.
    X(H) & y(z) have their usual meaning.
    theta(xest) - vector of coefficients.
    '''
    # m = len(y)
    # Calculating Cost
    c = np.sum(np.square((X.dot(theta))-y))  
    return c

def batch_gradient_descent(X,y,theta,W,alpha,iterations, tol = None):
    '''
    returns array of thetas, cost of every iteration
    X - X matrix
    y - target variable matrix
    theta - matrix of regression coefficients
    W - diagonal matrix for weights
    alpha - learning rate
    iteration - number of iteration to be run
    '''
    #Getting number of observations.
    m = len(y)
    tol = tol if tol is not None else 10e-9
    # Initializing cost and theta's arrays with zeroes.
    
    thetas = theta
    costs = []
    count = 0
    emax = 100 # chosen higher than the tol
    # Calculating theta for every iteration.
    # for i in range(iterations):
    while emax > tol:
        # print(count)
        residuals = X.dot(theta)-y
        w_residuals = np.dot(W,residuals) # weighted residuals
        gradient = X.T.dot(w_residuals)/m
        theta = theta - alpha * gradient # new weights/ thetas
        thetas = np.vstack((thetas, theta)) # store the result in a matrix
        costs.append(cost(theta,X,y))
        emax = np.max(np.abs(thetas[count+1,:]-thetas[count,:]))
        count+=1
        if count % 30000==0:
            print(count)
        # if count == iterations*25:
        #     break
        # if i!=0 and : # tol
        #     break
    return theta,thetas,costs, count

def stochastic_gradient_descent(X,y,theta,W,alpha,iterations, tol = None):
    ''' implements SGD '''
    tol = tol if tol is not None else 10e-12
    # Initializing cost and theta's arrays with zeroes.

    thetas = theta
    costs = []
    # count = 0
    emax = 100 # chosen higher than the tol
    # parameters are updated with every new measurement
    for j in range(iterations):
        if j%300 == 0:
            print(j)
        for i in range(X.shape[0]):
            # yhat = np.matmul(, x2)
            residuals = X[i,:].dot(theta)-y[i] # scalar in SGD
            w_residuals = np.dot(W[i,i],residuals) # weighted residuals
            gradient = X[i,:].T.dot(w_residuals)
            theta = theta - alpha * gradient # new weights/ thetas
            thetas = np.vstack((thetas, theta)) # store the result in a matrix
            costs.append(cost(theta,X[i,:],y[i]))
            emax = np.max(np.abs(thetas[j+1,:]-thetas[j,:]))

    return theta,thetas,costs, j

# testing gradient descents
# Learning Rate
# alpha = 0.0001
# # Number of iterations
# iterations = 3000
# # Initializing a random value to give algorithm a base value.
# p_distributed = P_line[(0,1)]/(len(P_Load_state))
# p_states = np.zeros((len(P_Load_state))) + p_distributed

# q_distributed = Q_line[(0,1)]/(len(P_Load_state))
# q_states = np.zeros((len(P_Load_state))) + q_distributed

# v0 = 1 # slack bus

# x_est = np.concatenate((p_states, q_states))
# x_est = np.insert(x_est, len(x_est), v0) # initialized state vars
# X,y,theta,W,alpha,iterations = jacobian_matrix,z,x_est,W,alpha,iterations

# # Running Batch Gradient Descent
# x_est,thetas,costs, counts = batch_gradient_descent(jacobian_matrix,z,x_est,W,alpha,iterations)

# # Running Stochastic Gradient Descent
# x_est,thetas,costs, counts = stochastic_gradient_descent(jacobian_matrix,z,x_est,W,alpha,iterations)
# # printing final values.
# # print('Final Theta 0 value: {:0.3f}\nFinal Theta 1 value: {:0.3f}'.format(theta[0][0],theta[1][0]))
# print('Final Cost/MSE(L2 Loss) Value: {:0.3f}'.format(costs[-1]))
