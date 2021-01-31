import os, sys
import itertools as it
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import scipy.special

import warnings
warnings.filterwarnings("error")

# INTRAPACKAGE IMPORTS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
from objects import utils
from training.utils.loss import RMSE
from training.utils.helpers import off_patterns, Ainv_from_rates, k_in_from_input_data


########################
# GRADIENT CALCULATION #
########################

def calc_network_output_sr(rate_matrix, input_rates, output_rates, Ainv = None):
    """
    Computes the single-rail network output given all rate parameters (k_in, k_out, k_ij).
    The single-rail network output is the output fluorescence from each node, given by p_i * k^i_out.

    Args:
        rate_matrix (np.array): The weights (rate constants, arbitrary units) between nodes in the system.
            Should be square and symmetric, with all diagonal entries equal to 0.
        input_rates (np.array): The intrinsic excitation rates of each node (k_in).
            Should be length-n 1d array, for network of n nodes.
        output_rates (np.array): The intrinsic emission rate constants of each node (k_out). 
            Should be a length-n 1d array, for network of n nodes.
        Ainv (np.array): [optional] The precomputed inverse A matrix. If not given, will be computed.
            Should be a square nxn matrix.
    
    Returns:
        pred (np.array): The predicted values of each node's output.
    """
    if Ainv is None:  Ainv = Ainv_from_rates(rate_matrix, input_rates, output_rates)

    pred = Ainv @ input_rates
    
    return pred * output_rates

def calc_network_output_dr(input_pattern, rate_matrix_sr, output_rates_sr, Ainv = None):
    """
    Computes the dual-rail network output given an input pattern and parameters of the single-rail system
    (k_ij and k_out). The intrinsic excitation rate constants for the single-rail system are derived
    from the input pattern. The number of nodes in the dual-rail system (n) should be half the number
    in the single-rail system (2n).

    It is assumed that node i in the dual-rail network has nodes 2i and 2i+1 from the single-rail network
    as its + and - fluorophores, respectively.

    Args:
        input_pattern (np.array): The binary input data. 
            Should be length-n 1d array with each element equal to -1 or +1.
        rate_matrix_sr (np.array): A matrix of the FRET rate constants between nodes in the single-rail system. 
            Should be a square and symmetric 2nx2n matrix, with diagonal entries equal to 0.
        output_rates_sr (np.array): The intrinsic emission rate constants of each node in the single-rail system (k_out).
            Should be a length-2n 1d nonnegative array
        Ainv (np.array): [optional] The precomputed inverse A matrix, which may be given as an optimization. 
            If not given, will be computed. Should be a square nxn matrix.

    Returns:
        output_dr (np.array): The dual-rail outputs, defined as the difference between the output fluorescence from
            each dual-rail node's + and - fluorophores in the single-rail network. Should be length-n 1d array.
        output_sr (np.array): The single-rail outputs, defined as the output fluorescence from each single-rail node.
            Should be a length-2n 1d array.
        
    """
    num_nodes_dr = len(input_pattern)
    num_nodes_sr = 2*num_nodes_dr

    # Determine equivalent single-rail input rates based on dual-rail input pattern
    input_rates_sr = k_in_from_input_data(input_pattern)

    output_sr = calc_network_output_sr(rate_matrix_sr, input_rates_sr, output_rates_sr, Ainv = Ainv)

    output_dr = output_sr[range(0, num_nodes_sr, 2)] - output_sr[range(1, num_nodes_sr, 2)]
#    print(output_sr)
#    print(output_dr)

    return 100*output_dr, output_sr
    

# note: not yet modified for use with dual-rail
def gradient(loss_grad, pat, pred, Ainv, verbose=False):
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

    dA_dK = np.zeros((num_nodes**2, scipy.special.binom(num_nodes, 2))) # d^2 x (d choose 2)
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
    
#def train(train_data, loss_fn, loss_grad, output_rates, step_size, iters, epsilon = None, noise = 0.1, num_corrupted=1, report_every=0):
#    """
#    Trains a new network on given training patterns using batch gradient descent.
#    Patterns with specified amount of noise are used as input to a simulated FRETnet, 
#    then steady-state behavior is compared to original patterns using a loss function.
#    Stops either when gradient step dK is less than epsilon or after iters iterations.
#
#    Args:
#        train_data (np.array): 2d float array where each column is a training pattern.
#            Should be dxn, where d is number of nodes and n is number of patterns.
#        ouput_rates (np.array): The intrinsic outputs of each node.
#            Should be a dx1 column vector. 
#        step_size (float): Multiplier for each gradient descent update.
#        iters (int): Number of times to pass through the training data.
#        epsilon (float): Size of gradient at which training should stop.
#        noise (float in [0,1]): Extend of deviation of training patterns 
#            from the given train_data.
#        num_corrupted (int): Num of corrupted patterns to generate 
#            from each template pattern. 
#        report_every (int): If True, prints all rate arrays while training.
#
#    Returns:
#        weights (np.array): Weights between nodes in the network. 
#            Should be dxd, NONNEGATIVE and symmetric.
#        err_over_time (list): Average error over training patterns,
#            recorded for each iteration loop.
#            Should have length n.
#    """
#    d, n = train_data.shape
#    K = np.random.rand(d, d) / 10 + 0.45  # weights initialized at 0.5 +/- 0.05
#    
#    K = (K + K.T) / 2  # ensure starting weights are symmetric
#    np.fill_diagonal(K, 0)
#    err_over_time = []
#    K_over_time = np.array(K).reshape((1, d, d))
#
#    for i in range(iters):
#        avg_err = 0
#        dK = 0
#
#        for j in range(n):  
#            template = train_data[:, j:j+1]
#            train_patterns = off_patterns(template, noise, num_corrupted)
#
#            for k in range(num_corrupted):
#                pat = train_patterns[:, k:k+1].astype(float) # ensure float
#
#                # TODO Find a better way to deal with singular matrices
#                #       arising from all-zero patterns
#                
#                pat[pat==0] = 0.1
#
#                # Compute the prediction based on the off patterns,
#                # But evaluate error relative to original template pattern.
#                Ainv, pred = _forward_pass(K, pat, output_rates)
#                # print(f'pat: \n{pat} \n pred: \n{pred}')
#                dL_dK = gradient(loss_grad, template, pred, Ainv, output_rates)
#                dK += dL_dK
#                avg_err += loss_fn(template, pred)/(n * num_corrupted)
#        
#        new_K = K.copy()
#        new_K -= step_size * dK
#        new_K[new_K<0] = 0 # NOTE ReLU; ensures all weights are positive.
#        err_over_time.append(avg_err)
#        K_over_time = np.append(K_over_time, new_K.reshape(1, d, d), axis=0)
#
#        # Stopping condition based on epsilon
#        if epsilon and np.linalg.norm(new_K - K) < epsilon:
#            print(f'Stopped at iteration {i+1}\n'
#                    f'K: {new_K}\n'
#                    f'error: {avg_err}\n'
#                    f'< epsilon: {np.linalg.norm(new_K - K)}')
#            return new_K, err_over_time, K_over_time
#    
#        if (i) % report_every == 0:
#            print(f'Rates on iteration {i}: \n{new_K}')
#
#        K = new_K
#
#    return K, err_over_time, K_over_time

def train_dr_hebbian(stored_data):
    num_nodes_dr = len(stored_data[0])
    num_nodes_sr = 2*num_nodes_dr

    K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
    for d in stored_data:
        for i,j in it.combinations(range(num_nodes_dr), 2):
            i_p = 2*i
            i_n = 2*i+1
            j_p = 2*j
            j_n = 2*j+1
            if (d[i],d[j]) == (1,1):
                K_fret[i_p,j_p] += 1
            elif (d[i],d[j]) == (1,-1):
                K_fret[i_p,j_n] += 1
            elif (d[i],d[j]) == (-1,1):
                K_fret[i_n,j_p] += 1
            elif (d[i],d[j]) == (-1,-1):
                K_fret[i_n,j_n] += 1
    K_fret = K_fret + K_fret.T

    k_out = np.ones(num_nodes_sr)

    print(K_fret)

    return K_fret, k_out

def train_dr(stored_data, loss_func, duplication = 5, noise = 0.1, init = 'random', seed = None):
    def rates_to_params(K_fret, k_out):
        idxs = np.triu_indices(num_nodes_sr, 1)
        params = np.concatenate((K_fret[idxs], k_out))
        return params
    def params_to_rates(p):
        K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
        for idx, (i,j) in enumerate(it.combinations(range(num_nodes_sr),2)):
            K_fret[i,j] = p[idx]
            K_fret[j,i] = p[idx]
#        k_out = 10*np.ones(num_nodes_sr) # use this line for fixed, uniform k_out
#        k_out = p[-1]*np.ones(num_nodes_sr) # use this line for optimized, uniform k_out
        k_out = p[-num_nodes_sr:] # use this line for optimized, non-uniform k_out
        return K_fret, k_out

    def loss_func_scipy(params):
        K_fret, k_out = params_to_rates(params)

#        resid = np.empty((num_nodes_dr, len(train_data)))
#        for i, (input_data, output_data_cor) in enumerate(train_data):
#            output_data,_,_ = calc_network_output_dr(input_data, K_fret, k_out)
#            resid[:,i] = (output_data - output_data_cor).flatten()
#
#        return resid.flatten()

        resid = np.array([
            loss_func(
                calc_network_output_dr(
                    input_data,
                    K_fret,
                    k_out
                )[0],
                output_data_cor
            ) 
            for input_data,output_data_cor in train_data
        ])

        return resid

    num_nodes_dr = len(stored_data[0])
    num_nodes_sr = 2*num_nodes_dr
    train_data = [
        (input_data, output_data)
            for output_data in stored_data 
            for input_data in off_patterns(output_data, noise, duplication)
    ]

    num_params = num_nodes_sr*(num_nodes_sr-1)//2 + num_nodes_sr

    if init == 'one':
        init_params = np.ones(num_params)
    elif init == 'zero':
      init_params = np.zeros(num_params)
    elif init == 'hebbian':
      init_params = 5*rates_to_params(*train_dr_hebbian(stored_data)) + 1
    else:
        init_params = RNG.uniform(.01, 1, num_params)
    print(params_to_rates(init_params))
    res = scipy.optimize.least_squares(loss_func_scipy, init_params, bounds=(0,1), jac='3-point')

    print(res)

    return (*params_to_rates(res.x), res)
    

# NOTE: following code not modified for dual-rail
#def grid_search(num_nodes, pat, outs, loss_fn, k_domain, resolution=10, noise=0.1):
#    """
#    Grid-searches FRET rates with given intrinsic output rates, noise amt for the configuration with the lowest loss_fn.
#    """
    # triu_idx = np.triu_indices(num_nodes, 1)
    # num_cols = pats.shape[1]
    # def f(params):
    #     K = np.zeros((num_nodes, num_nodes), dtype=float)
    #     K[triu_idx] = params
    #     K += K.T

    #     loss_sum = 0

    #     for c in range(num_cols):
    #         pat = pats[:, c:c+1]
    #         Ainv = Ainv_from_rates(K, pat, outs)
    #         pred = Ainv @ pat
    #         num_corrupted = 5
    #         off_pats = off_patterns(pat, noise, num_corrupted)
    #         for i in range(num_corrupted):
    #             off_pat = off_pats[:, i:i+1]
    #             loss_sum += loss_fn(off_pat, pred)
    #     #TODO handle all zeros in off_pat
    #     return loss_sum
    
    # k_ranges = [tuple(k_domain) for _ in range(choose(num_nodes, 2))]

    # resbrute = brute(f, k_ranges, Ns=resolution, finish=None)
    # K_min = np.zeros((num_nodes, num_nodes), dtype=float)
    # K_min[triu_idx] = resbrute
    # K_min += K_min.T
    # return K_min

def rates_to_network(K_fret, k_out, k_in):
    num_nodes = len(k_out)
    nodes = [utils.InputNode('node{}'.format(i), production_rate=k_in_i, emit_rate=k_out_i) for i,(k_out_i,k_in_i) in enumerate(zip(k_out, k_in))]
    for i,j in it.product(range(num_nodes), range(num_nodes)):
        if i==j:  continue
        nodes[j].add_input(nodes[j], K_fret[i,j])

    return utils.Network(nodes)

if __name__ == '__main__':

    if not os.path.exists('tmp'):
      os.mkdir('tmp')

    SEED = np.random.randint(0,10**6)
#    SEED = 67331
    RNG = np.random.default_rng(SEED)
    print(f'SEED: {SEED}')
    loss = RMSE

    num_nodes = 3
    num_patterns = 2
#    stored_data_ints = RNG.permutation(2**num_nodes)[:num_patterns]
#    stored_data = [
#        np.array([int(v)*2-1 for v in format(i,'0{}b'.format(num_nodes))])
#        for i in stored_data_ints
#    ]
    stored_data = list(map(np.array, [[-1,-1,1],[1,-1,-1]]))
    init_method = 'random'


    print('Training data:', stored_data)

    trained_K_fret, trained_k_out, res = train_dr(stored_data, loss.fn, noise = 0.1, duplication=5, init=init_method)

    # this output is probably too much for bigger networks
    for i in range(2*num_nodes):
        print(trained_k_out[i], np.round(trained_K_fret[i,:],3).tolist())
    for d in map(np.array, it.product(*([[-1,1]]*num_nodes))):
        pre = '*' if list(d) in map(list, stored_data) else ' '
        print(pre, d.flatten(), calc_network_output_dr(d, trained_K_fret, trained_k_out)[0].flatten())

    # generate Network object with these connections
    k_in = np.array([
        kin for bit in stored_data[0] for kin in (max(bit, 0), -min(bit, 0))
    ])
    trained_network = rates_to_network(trained_K_fret, trained_k_out, k_in)

    output = {
      'seed': SEED,
      'init_method': init_method,
      'num_nodes': num_nodes,
      'num_patterns': num_patterns,
      'stored_data': stored_data,
      'trained_K_fret': trained_K_fret,
      'trained_k_out': trained_k_out,
      'scipy_output': res,
      'trained_network': trained_network
    }

    with open(f'tmp/output_nodes={num_nodes}_pat={num_patterns}_seed={SEED}.p','wb') as outfile:
      pickle.dump(output, outfile)
