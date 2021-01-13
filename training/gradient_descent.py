import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import math
import warnings
warnings.filterwarnings("error")

DO_TRAINING = False
DO_ANALYSIS = True


####################
# HELPER FUNCTIONS #
####################

def choose(a, b):
    """
    A short helper function to calculate binomial coefficients.
    """
    assert int(a) + int(b) == a + b, f'Choose only takes ints, not {a} and {b}'
    assert a >= 0 and b >= 0, f'Choose only takes nonnegative ints, not {a} and {b}'
    if a < b:
        return 0
    prod = 1
    faster_b = min(b, a-b)
    for i in range(faster_b):
        prod *= (a-i) / (i+1)
    return round(prod)

def off_patterns(pat, p_off, num_pats):
    """
    Returns all patterns which are off from pat by a proportion 
    specified by p_off, with a minimum of 1 element changed.

    Args:
        pat (np.array): Binary column vector of length d.
        p_off (float in [0,1]): Probability of each node being corrupted.
        num_pats (int): Number of corrupted patterns to return.

    Returns:
        len(pat) x num_pats 2darray (dtype=float) with corrupted patterns as columns.
    """
    d = len(pat)

    corrupt_mask = np.random.default_rng().random((d, num_pats))
    out = np.tile(pat, num_pats)

    out[corrupt_mask < p_off] = out[corrupt_mask < p_off] ^ 1
    return out

##################
# LOSS FUNCTIONS #
##################

def nll(pat, pred):
    """
    Negative Log Loss, evaluated elementwise and then summed over two arrays of the same shape.
    """
    total = 0
    for i in range(pat.size):
        if pat[i,0] == 1:
            total -= np.log(pred[i,0]) 
        elif pat[i,0] == 0:
            total -= np.log(1-pred[i,0])
        else: 
            raise ValueError()
    return total

def dnll(pat, pred):
    """
    Analytical gradient of the NLL function, elementwise.
    """
    out = np.zeros(pat.T.shape)
    for i in range(pat.size):
        if pat[i,0] == 1:
            out[0,i] = -1/pred[i,0]
        elif pat[i,0] == 0:
            out[0,i] = 1/(1-pred[i,0])
        else:
            raise ValueError()
    return out

def rmse(pat, pred):
    """
    Root Mean Square Error function.
    """
    return np.mean((pat - pred) ** 2) ** 0.5

def drmse(pat, pred):
    """
    Analytical gradient of RMSE, elementwise.
    """
    return ( 1/rmse(pat, pred) * 1/len(pat) * (pred - pat) ).T

########################
# GRADIENT CALCULATION #
########################

def _forward_pass(rate_matrix, pattern, output_rates):
    """
    Performs a 'forward pass' by analytically finding the steady-state probability
    that each node is excited.

    Args:
        rate_matrix (np.array): The weights (rate constants, arbitrary units) between nodes in the system.
            Should be square and symmetric.
        train_pattern (np.array): The (binary) pixel pattern to be trained on.
            Should be a column.
        output_rates (np.array): The intrinsic output rate constants of each node. 
            Should be a column.
    
    Returns:
        Ainv (np.array): The inverse of the matrix representing the linear system of equations.
            Should be square.
        pred (np.array): The predicted values of each node's output.
    """
    num_nodes = len(pattern)
    A = -rate_matrix
    diagonal_terms = rate_matrix.sum(axis=1) + pattern.T + output_rates.T

    if rate_matrix[range(num_nodes), range(num_nodes)].any():
        raise ValueError(f'diagonal terms not 0 in the rate matrix: \n {rate_matrix}')

    A[range(num_nodes), range(num_nodes)] = diagonal_terms

    try:
        Ainv = np.linalg.inv(A)
    except:
        raise ValueError(f'Singular matrix for rates={rate_matrix}, inputs={pattern}, outputs={output_rates}')

    pred = Ainv @ pattern
    return Ainv, pred

def gradient(loss_grad, pat, pred, Ainv, output_rates, verbose=False):
    """
    Finds the gradient of a loss function with respect to rates in a FRETnet.

    Args:
        loss_grad: The gradient of the loss function with respect to pred.
            Should take pat and pred as parameters and return a array of same size.
        pat (np.array): The given training pattern.
            Should be a dx1 binary column vector.
        pred (np.array): The predicted output of each node.
            Should be a dx1 column of probabilities.
        Ainv (np.array): The inverse of the matrix representing the linear system.
            Should be a nxn square matrix. 
        output_rates (np.array): The intrinsic output rate constants assigned to each node.
            Should be a dx1 column vector.
        verbose (bool): If True, returns a dict with keys:
            Ainv, pred, dL_dpred, dpred_dAinv, dAinv_dA, dA_dK, dL_dK

    Returns: 
        A dict of all the quantities computed, if verbose is True.
        dL_dK (np.array): The gradient of the loss with respect to the weights of the network.
            Should be a dxd matrix, reshaped from 1xd^2.

    """
    # don't actually have to compute the NLL, just its gradient
    num_nodes = len(pat)

    dL_dpred = loss_grad(pat, pred) # 1 x d

    dpred_dAinv = np.kron(pat, np.identity(num_nodes)).T # d x d^2
    
    dAinv_dA = -np.kron(Ainv, Ainv) # d^2 x d^2

    dA_dK = np.zeros((num_nodes**2, choose(num_nodes, 2))) # d^2 x (d choose 2)
    idx = 0
    for i,j in it.combinations(range(num_nodes), 2):
        dA_dK[i*num_nodes + i, idx] = 1
        dA_dK[j*num_nodes + j, idx] = 1
        dA_dK[i*num_nodes + j, idx] = -1
        dA_dK[j*num_nodes + i, idx] = -1
        idx += 1

    dL_dK = (dL_dpred @ dpred_dAinv @ dAinv_dA @ dA_dK).flatten()

    reshaped_dL_dK = np.zeros((num_nodes, num_nodes))
    idx = 0
    for i,j in it.combinations(range(num_nodes), 2):
        reshaped_dL_dK[i,j] = dL_dK[idx]
        idx += 1
    reshaped_dL_dK = reshaped_dL_dK + reshaped_dL_dK.T
    # 1 x (d choose 2) -> d x d, duplicating items for symmetry
    # and making diagonal terms 0

    if verbose:
        return {'Ainv':Ainv, 'pred':pred, 'dL_dpred':dL_dpred, 'dpred_dAinv':dpred_dAinv, 
            'dAinv_dA':dAinv_dA, 'dA_dK':dA_dK, 'reshaped_dL_dK':reshaped_dL_dK}
    else:
        return reshaped_dL_dK
    
############
# TRAINING #
############
    
def train(train_data, loss_fn, loss_grad, output_rates, step_size, iters, epsilon = None, noise = 0.1, num_corrupted=1, report_every=0):
    """
    Trains a new network on given training patterns using batch gradient descent.
    Patterns with specified amount of noise are used as input to a simulated FRETnet, 
    then steady-state behavior is compared to original patterns using a loss function.
    Stops either when gradient step dK is less than epsilon or after iters iterations.

    Args:
        train_data (np.array): 2d float array where each column is a training pattern.
            Should be dxn, where d is number of nodes and n is number of patterns.
        ouput_rates (np.array): The intrinsic outputs of each node.
            Should be a dx1 column vector. 
        step_size (float): Multiplier for each gradient descent update.
        iters (int): Number of times to pass through the training data.
        epsilon (float): Size of gradient at which training should stop.
        noise (float in [0,1]): Extend of deviation of training patterns 
            from the given train_data.
        num_corrupted (int): Num of corrupted patterns to generate 
            from each template pattern. 
        report_every (int): If True, prints all rate arrays while training.

    Returns:
        weights (np.array): Weights between nodes in the network. 
            Should be dxd, NONNEGATIVE and symmetric.
        err_over_time (list): Average error over training patterns,
            recorded for each iteration loop.
            Should have length n.
    """
    d, n = train_data.shape
    K = np.random.rand(d, d) / 10 + 0.45  # weights initialized at 0.5 +/- 0.05
    
    K = (K + K.T) / 2  # ensure starting weights are symmetric
    np.fill_diagonal(K, 0)
    err_over_time = []
    K_over_time = np.array(K).reshape((1, d, d))

    for i in range(iters):
        avg_err = 0
        dK = 0

        for j in range(n):  
            template = train_data[:, j:j+1]
            train_patterns = off_patterns(template, noise, num_corrupted)

            for k in range(num_corrupted):
                pat = train_patterns[:, k:k+1].astype(float) # ensure float

                # TODO Find a better way to deal with singular matrices
                #       arising from all-zero patterns
                
                pat[pat==0] = 0.1

                # Compute the prediction based on the off patterns,
                # But evaluate error relative to original template pattern.
                Ainv, pred = _forward_pass(K, pat, output_rates)
                # print(f'pat: \n{pat} \n pred: \n{pred}')
                dL_dK = gradient(loss_grad, template, pred, Ainv, output_rates)
                dK += dL_dK
                avg_err += loss_fn(template, pred)/(n * num_corrupted)
        
        new_K = K.copy()
        new_K -= step_size * dK
        new_K[new_K<0] = 0 # NOTE ReLU; ensures all weights are positive.
        err_over_time.append(avg_err)
        K_over_time = np.append(K_over_time, new_K.reshape(1, d, d), axis=0)

        # Stopping condition based on epsilon
        if epsilon and np.linalg.norm(new_K - K) < epsilon:
            print(f'Stopped at iteration {i+1}\n'
                    f'K: {new_K}\n'
                    f'error: {avg_err}\n'
                    f'< epsilon: {np.linalg.norm(new_K - K)}')
            return new_K, err_over_time, K_over_time
    
        if (i) % report_every == 0:
            print(f'Rates on iteration {i}: \n{new_K}')

        K = new_K

    return K, err_over_time, K_over_time

def grid_search(num_nodes, pat, outs, loss_fn, k_domain, resolution=10, noise=0.1):
    """
    Grid-searches FRET rates with given intrinsic output rates, noise amt for the configuration with the lowest loss_fn.
    """
    def f(params):
        K = np.reshape(params, (num_nodes, num_nodes))
        _, pred = _forward_pass(K, pat, outs)
        num_corrupted = 10
        off_pats = off_patterns(pat, noise, num_corrupted)
        avg_loss = 0
        for i in range(num_corrupted):
            off_pat = off_pats[:, i:i+1]
            avg_loss += loss_fn(off_pat, pred) / num_corrupted
        #TODO handle all zeros in off_pat
        return avg_loss
    
    k_ranges = [tuple(k_domain) for _ in range(num_nodes**2)]
    diag_indices = [(d*num_nodes + d) for d in range(num_nodes)]
    for i in diag_indices:
        k_ranges[i] = (0, 0)
    print(k_ranges)

    resbrute = brute(f, k_ranges, Ns=resolution, finish=None)
    K_min = np.reshape(resbrute, (num_nodes, num_nodes))
    return K_min

if __name__ == '__main__':

    num_nodes = 10
    num_patterns = 4
    train_data = np.random.randint(2, size=(num_nodes, num_patterns))

    # hyperparameters
    outs = np.full(num_nodes, 0.5)
    step_size = 0.001
    iters = int(100 / step_size)

    if DO_TRAINING:
        K, err_over_time, K_over_time = train(train_data, rmse, drmse, outs, step_size, iters, 
            epsilon=0.001 * step_size, noise=0.25, report_every=100)

        print (f'Training data:\n{train_data}')
        
        plt.plot(np.arange(len(err_over_time)), err_over_time, label='Error averaged over dataset')

        # track_i = 0
        # track_j = 1
        # plt.plot(np.arange(len(K_over_time)), K_over_time[:, track_i, track_j], label=f'K_{track_i},{track_j}')
        plt.legend()
        plt.show()
        
        print (nll(train_data[:, 1:2], _forward_pass(K, train_data[:, 1:2], outs)[1]))



# ADDED tests for helper fns, gradient
# ADDED RMSE error fn
# TODO RECORD check that gd still converges to all zeros if greater noise (USE BOTH ERROR FNS)
# TODO RECORD grid search on rate matrix + amt of noise for nontrivial global error min 
# TODO look into volatility of RMSE gd
# TODO Optimize k_out thru gradient descent
# TODO dual-rail
# TODO complex optimization packages
# TODO Change corruption of Hebbian (copy over off_patterns)
# TODO compare error of gradient descent to hebbian training
# TODO LOSS DOESN'T DEPEND ON RATES; test all rates instead of [1,0]

# TODO fix k_out values
# TODO weights tend to 0. why? do other hopfield networks/RBMs run into the same problem?
# TODO better starting rate parameters in train
# TODO adagrad? other step size optimizer?
# TODO add regularization
# cd Documents/'0 mit'/'year 1'/urop/dna-ml/'jberlean simulator'/FRETNET2/FRETNET