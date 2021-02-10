import numpy as np
import time
import torch

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
    delta_mat = np.zeros((jacobian_matrix.shape[1], 5000)) # delta in states
    residuals_mat = np.zeros((jacobian_matrix.shape[0], 5000)) # meas residuals
    tol = tol if tol is not None else 10e-12
    results = x_est
    emax = 100 # chosen higher than the tol

    while emax > tol and count < 2:

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

def cost(theta, H, y, W):
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
    Error = H.dot(theta)-y
    c = (1/(2*m)) * np.sum(np.square(Error) * W)
    # c = np.sum(np.square(Error) * np.diag(W))/ (m)
    return c

def batch_gradient_descent(H, y, theta, W, lr, iterations, tol = None):
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
    m = len(y)
    tol = tol if tol is not None else 10e-12
    # Initializing cost and theta's arrays with zeroes.
    
    thetas = theta
    costs = []
    count = 0
    emax = 100 # chosen higher than the tol
    # Calculating theta for every iteration.
    # for i in range(iterations):
    while emax > tol and count < iterations :
        # print(count)
        costs.append(cost(theta, H, y, W))
        residuals = H.dot(theta) -y
        w_residuals = np.dot(W, residuals) # weighted residuals
        gradient = 1/m*(np.dot(H.T, w_residuals)) # this is correct
        # another way to check grad, just to make more sense
        # w_residuals = w_residuals.reshape((m,1))
        # gradient2 = H * w_residuals
        # gradient2 = 1/m*np.sum(gradient2, axis = 0)
        # another way
        # # gradient = 1/m*(np.matmul(np.matmul(np.matmul(H.T, W), H), theta) - np.matmul(np.matmul(H.T, W), y))
        # print(gradient, gradient2)
        theta = theta - lr * gradient # new weights/ thetas
        thetas = np.vstack((thetas, theta)) # store the result in a matrix
        emax = np.max(np.abs(thetas[count+1,:]-thetas[count,:]))
        count+=1
        if count % 30000==0:
            print(count, emax)
    return theta, thetas, costs, count, emax

def stochastic_gradient_descent(H, y, theta, W, lr, iterations, tol = None):
    ''' implements SGD '''
    tol = tol if tol is not None else 10e-12
    # Initializing cost and theta's arrays with zeroes.

    thetas = theta
    costs = []
    # count = 0
    emax = 100 # chosen higher than the tol
    j = 0 # iteration count    
    # parameters are updated with every new measurement
    # for j in range(iterations):
    while emax > tol and j < iterations: # num iters and change in x    
        if j%300 == 0:
            print(j, emax)
        for i in range(H.shape[0]):
            # yhat = np.matmul(, x2)
            residuals = H[i,:].dot(theta)-y[i] # scalar in SGD
            w_residuals = np.dot(W[i,i], residuals) # weighted residuals
            gradient = H[i,:].T.dot(w_residuals)
            theta = theta - lr * gradient # new weights/ thetas
            thetas = np.vstack((thetas, theta)) # store the result in a matrix
            costs.append(cost(theta, H[i,:], y[i]))
            emax = np.max(np.abs(thetas[-1,:]-thetas[-2,:])) # chane in x
        j+=1
    return theta, thetas, costs, j

def stochastic_gradient_descent2(H, y, theta, W, lr, iterations, tol = None):
    ''' implements SGD with shuffle
        mathematically same as above implementation: checked
    '''
    tol = tol if tol is not None else 10e-12
    thetas = theta
    costs = []
    emax = 100 # chosen higher than the tol
    j = 0 # iteration count
    # for j in range(iterations): # num iters
    while emax > tol and j < iterations: # num iters and change in x
        if j%300 == 0:
            print(j, emax)
            shuffled_set = np.random.permutation(len(y)) # shuffle the meas
        for i in shuffled_set:
        # chose meas randomly rather than in an order
            # can iterate over thetas individually to double check
            estimate = np.sum(np.multiply(H[i,:], theta))
            w_residual = (y[i] - estimate) * W[i,i] # common grad term for each parameter
            # grad calculated for (1/2)*(y-Hx)^2*W
            gradient = -1 * w_residual * H[i,:]
            theta = theta - (lr * gradient) # update parameters
            thetas = np.vstack((thetas, theta)) # store the result in a matrix
            costs.append(cost(theta,H[i,:],y[i]))
            emax = np.max(np.abs(thetas[-1,:]-thetas[-2,:])) # change in x
        j+=1
    return theta, thetas, costs, j, emax

class WLeastSquaresRegressorTorch():

    def __init__(self, n_iter=10, eta=0.1, batch_size=10, tol = None, x_est = None):
        self.n_iter = n_iter
        self.eta = eta
        self.batch_size = batch_size
        self.tol = tol if tol is not None else 10e-12

    def fit(self, H, y, W, x_est=None):

        n_instances, n_features = H.shape
        
        # we need to "wrap" the NumPy arrays H and y as PyTorch tensors
        Ht = torch.tensor(H, dtype=torch.float)
        Yt = torch.tensor(y, dtype=torch.float)
        Wt = torch.tensor(W, dtype=torch.float)

        # initialize the weight vector to all zeros
        # self.x_est = torch.zeros(n_features, requires_grad=True, dtype=torch.float)
        if x_est is None:
            torch.manual_seed(0)
            x_est = torch.rand(n_features, requires_grad=True)
            print('None', x_est)
        else:
            x_est = torch.tensor(x_est, requires_grad=True).float()
            print(x_est)

        self.history = [] # to store the cost function

        # we select an optimizer, in this case (minibatch) SGD.
        # it needs to be told what parameters to optimize, and what learning rate (lr) to use
        # print(self.eta)
        # gradient descent algo
        optimizer = torch.optim.SGD([x_est], lr=self.eta) #, momentum =0.9)
        # adagrad descent
        # optimizer = torch.optim.Adagrad([self.x_est], lr=self.eta, 
        #                                 lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        
        # optimizer = torch.optim.RMSprop([self.x_est], lr=self.eta, 
        #     alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # adam decent
        # optimizer = torch.optim.Adam([self.x_est], lr=self.eta)
        
        for i in range(self.n_iter):
            
            total_loss = 0
            
            for batch_start in range(0, n_instances, self.batch_size):
                batch_end = batch_start + self.batch_size

                # pick out the batch again, as in the other notebook
                Hbatch = Ht[batch_start:batch_end, :]
                Ybatch = Yt[batch_start:batch_end]
                Wbatch = Wt[batch_start:batch_end,batch_start:batch_end]
                # mv = matrix-vector multiplication in Torch
                G = Hbatch.mv(x_est)
              
                # Loss
                Error = (G - Ybatch)
                # loss_batch = torch.sum((Error**2) * torch.diagonal(Wbatch)) / len(Ybatch)
                loss_batch = (1/(2*len(Ybatch))) * (torch.sum(Wbatch * (Error ** 2)))

                # we sum up the loss values for all the batches.
                # the item() here is to convert the tensor into a single number
                total_loss += loss_batch.item()

                # reset all gradients
                optimizer.zero_grad()                  

                # compute the gradients for the loss for this batch
                loss_batch.backward()
                # print(max(abs(self.x_est.grad))) # prints the gradient
                if max(abs(x_est.grad)) < self.tol:
                    emax = max(abs(x_est.grad))

                # for SGD, this is equivalent to x_est -= learning_rate * gradient as we saw before
                optimizer.step()

            emax = max(abs(x_est.grad))
            self.history.append(total_loss)

        print('SGD-minibatch final loss: {:.4f}'.format(total_loss))
        return x_est, emax
