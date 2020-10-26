import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import warnings
warnings.filterwarnings("error")

def forward_pass(rate_matrix, pattern, output_rates):
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
    # try:
    #     assert not A[range(num_nodes), range(num_nodes)].any()
    # except:
    #     # print (f'Assertion Error: diagonal terms not 0 in the rate matrix: \n {A}')
    A[range(num_nodes), range(num_nodes)] = diagonal_terms

    try:
        Ainv = np.linalg.inv(A)
    except:
        print (f'Singular matrix for rates={rate_matrix}, inputs={pattern}, outputs={output_rates}')
        raise   

    pred = Ainv @ pattern
    return Ainv, pred
    
def gradient(pat, pred, Ainv, output_rates, verbose=False):
    """
    Performs backpropagation on a given training pattern and network weight matrix. 

    Args:
        pat (np.array): The given training pattern.
            Should be a dx1 binary column vector.
        pred (np.array): The predicted output of each node.
            Should be a dx1 column of probabilities.
        Ainv (np.array): The inverse of the matrix representing the linear system.
            Should be a nxn square matrix. 
        output_rates (np.array): The intrinsic output rate constants assigned to each node.
            Should be a dx1 column vector.
        verbose (bool): If True, returns all quantities computed, in a dict with keys:
            Ainv, pred, dNLL_dpred, dpred_dAinv, dAinv_dA, dA_dK, dNLL_dAinv, dNLL_dA, dNLL_dK

    Returns: 
        A dict of all the quantities computed, if verbose is True.
        dNLL_dK (np.array): The gradient of the NLL loss with respect to the weights of the network.
            Should be a dxd matrix, reshaped from 1xd^2.

    """
    # don't actually have to compute the NLL, just its gradient
    num_nodes = len(pat)
    dNLL_dpred = np.zeros(pat.T.shape)
    for i in range(pat.size):
        dNLL_dpred[0,i] = -pat[i,0]/pred[i,0] if pat[i,0] else (1-pat[i,0])/(1-pred[i,0])


    dpred_dAinv = np.kron(pat, np.identity(num_nodes)).T # d x d^2
    
    dAinv_dA = -np.kron(Ainv, Ainv) # d^2 x d^2

    dA_dK = -np.identity(num_nodes**2) # d^2 x d^2
    for i,j in it.product(range(num_nodes), range(num_nodes)):
        dA_dK[(num_nodes+1)*i, num_nodes*j + i] += 1
    try:
        dNLL_dAinv = dNLL_dpred @ dpred_dAinv
    except:
        print(dNLL_dpred, dpred_dAinv)
        return 
    dNLL_dA = dNLL_dAinv @ dAinv_dA
    dNLL_dK = dNLL_dA @ dA_dK
    dNLL_dK = dNLL_dK.reshape((num_nodes, num_nodes), order='F') # 1 x d^2 -> d x d

    if verbose:
        return {'Ainv':Ainv, 'pred':pred, 'dNLL_dpred':dNLL_dpred, 'dpred_dAinv':dpred_dAinv, 'dAinv_dA':dAinv_dA, 'dA_dK':dA_dK, 
        'dNLL_dAinv':dNLL_dAinv, 'dNLL_dA':dNLL_dA, 'dNLL_dK':dNLL_dK}
    else:
        return dNLL_dK
    
def nll(pat, pred):
    """
    Negative Log Loss, evaluated elementwise and then summed over two arrays of the same shape.
    """
    total = 0
    for i in range(pat.size):
        total -= pat[i,0] * np.log(pred[i,0]) if pat[i,0] else (1-pat[i,0]) * np.log(1-pred[i,0])
    return total 


def train(train_data, output_rates, step_size, iters, report_every=0):
    """
    Trains a new network on given training patterns.

    Args:
        train_data (np.array): 2d array where each column is a training pattern.
            Should be dxn, where d is number of nodes and n is number of patterns.
        ouput_rates (np.array): The intrinsic outputs of each node.
            Should be a dx1 column vector. 
        step_size (float): Multiplier for each gradient descent update.
        iters (int): Number of times to pass through the training data.
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
    K = (K + K.T) / 2  # ensure weights are symmetric
    np.fill_diagonal(K, 0)
    err_over_time = []

    for i in range(iters):
        avg_err = 0
        dK = 0
        for j in range(n):  
            train_pattern = train_data[:, j:j+1]
            Ainv, pred = forward_pass(K, train_pattern, output_rates)
            dNLL_dK = gradient(train_pattern, pred, Ainv, output_rates)

            dK += dNLL_dK
            avg_err += nll(train_pattern, pred)/n


        K -= step_size * dK
        # K = (K + K.T)/2 # TODO this seems wrong
        K[K<0] = 0 # NOTE ReLU; ensures all weights are positive.
        err_over_time.append(avg_err)
    
        if report_every and not i % report_every:
            print(f'Rates on iteration {i}: \n{K}')
    return K, err_over_time
    
def test(rate_matrix, test_data, output_rates):
    """
    Tests a network on given test patterns.

    Args:
        weights (np.array): The rate matrix of the network to test.
            Should be square.
        D_test (np.array): The data to test on.
    
    Returns:
        avg_err (float): The averaged NLL of the network over all patterns in test_data.
    """
    n = test_data.shape[1]
    total_err = 0
    for i in range(n):
        test_pattern = test_data[:, i:i+1]
        pred = forward_pass(rate_matrix, test_pattern, output_rates)[1]
        total_err += nll(test_pattern, pred)
    avg_err = total_err/n
    return avg_err

if __name__ == '__main__':
    num_nodes = 5
    num_patterns = 5
    train_data = np.random.randint(2, size = (num_nodes, num_patterns))
    iters = 10000
    step_size = 0.002
    outs = np.full(num_nodes, 0.5)
    K, err_over_time = train(train_data, outs, step_size, iters, report_every=1000)
    print (K)
    print (train_data)
    plt.plot(np.arange(iters), err_over_time)
    plt.ion()
    plt.show()

#TODO add regularization
#TODO why does K become asymmetric? use lagrange multipliers
#TODO train on off by 1
#TODO weights tend to 0. why? do other hopfield networks/RBMs run into the same problem?
#TODO properties of diagonally dominant matrices