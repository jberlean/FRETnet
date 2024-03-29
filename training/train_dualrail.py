import sys, pathlib
import itertools as it
import pickle
import math
import multiprocessing as mp
import time

import numpy as np
import scipy.optimize
import tqdm

# INTRAPACKAGE IMPORTS
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.append(pkg_path)
from objects import utils as objects

from train_utils import off_patterns, Ainv_from_rates, A_from_rates, k_in_from_input_data, network_from_rates, rate_from_positions, random_point_on_sphere, rates_to_positions

np.set_printoptions(precision=2, suppress=True)

# TODO: use functions in train_singlerail.py rather than reimplementing them here

########################
# GRADIENT CALCULATION #
########################

def adjust_Ainv_kfret(Ainv, dx, i, j):
  # As an optimization, if only a single rate has changed we may recompute
  # A^-1 quickly, using the Sherman-Morrison formula for rank-1 matrix perturbations:
  #   dAinv = -dx/(1 + dx*(Ainv[i,i] - 2*Ainv[i,j] + Ainv[j,j]) * ((A[:,i] - A[:,j]) @ (A[i,:] - A[j,:]))
  coef = -dx / (1 + dx * (Ainv[i,i] + Ainv[j,j] - 2*Ainv[i,j]))
  dAinv = np.outer(coef*(Ainv[i,:] - Ainv[j,:]), Ainv[i,:] - Ainv[j,:])
  return Ainv + dAinv

def adjust_Ainv_koff(Ainv, dx, i):
  dAinv = -dx * np.outer(Ainv[i,:], Ainv[i,:]) / (1 + dx * Ainv[i,i])
  return Ainv + dAinv

def calc_network_output_sr(rate_matrix, input_rates, output_rates, decay_rates = 0, Ainv = None):
    """
    Computes the single-rail network output given all rate parameters (k_in, k_emit, k_decay, k_ij).
    The single-rail network output is the output fluorescence from each node, given by p_i * k^i_emit.

    Args:
        rate_matrix (np.array): The weights (rate constants, arbitrary units) between nodes in the system.
            Should be square and symmetric, with all diagonal entries equal to 0.
        input_rates (np.array): The intrinsic excitation rates of each node (k_in).
            Should be length-n 1d array, for network of n nodes.
        output_rates (np.array): The intrinsic emission rate constants of each node (k_out). 
            Should be a length-n 1d array, for network of n nodes.
        decay_rates (np.array): The intrinsic decay rate constants of each node (k_decay). 
            Should be a length-n 1d array, for network of n nodes.
        Ainv (np.array): [optional] The precomputed inverse A matrix. If not given, linear system of eqns will be
            solved without explicitly computing the inverse A matrix.
            If given, should be a square nxn matrix.
    
    Returns:
        pred (np.array): The predicted values of each node's output.
    """
#    if Ainv is None:  Ainv = Ainv_from_rates(rate_matrix, output_rates, input_rates)

#    pred = Ainv @ input_rates
    if Ainv is None:
      num_nodes = len(input_rates)
      A = -rate_matrix
      A[np.diag_indices(num_nodes)] = rate_matrix.sum(axis=1) + input_rates + output_rates + decay_rates
      pred = np.linalg.solve(A, input_rates)
    else:
      pred = Ainv @ input_rates

#    num_nodes = len(input_rates)
#    A = -rate_matrix
#    A[np.diag_indices(num_nodes)] = rate_matrix.sum(axis=1) + input_rates + output_rates
#    # 0.025s (154/1500 - 1/12.9)
#
#    pred = A @ input_rates
#    # 0.033s (166/1500 - 1/12.9)

#    A = A_from_rates(rate_matrix, output_rates, input_rates) # 0.030s
#    pred = np.ones(len(output_rates))

#    print(pred)
#    print(output_rates)
    return pred * output_rates

def calc_network_output_dr(input_pattern, rate_matrix_sr, output_rates_sr, decay_rates_sr = 0, input_magnitude = 1, output_magnitude = 1, Ainv = None):
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
        decay_rates_sr (np.array): The intrinsic decay rate constants of each node in the single-rail system (k_decay).
            Should be a length-2n 1d nonnegative array
        input_magnitude (float or np.array): [default=1] The magnitude of input fluorescence into each node. 
            For each pixel, one of its corresponding fluorophores will have k_in=input_magnitude and the other k_in=0.
            If a float, all pixels have the same magnitude of input. If np.array, pixels have individually specified
            input magnitudes.
        output_magnitude (float): [default=1] The magnitude of output fluorescence expected from an "on" node.
            The output pixel value will be scaled so that if the range of (f_pos - f_neg) is 
              [-output_magnitude, +output_magnitude]
            then the range of pixel values will be
              [-1, +1].
        Ainv (np.array): [optional] The precomputed inverse A matrix, which may be given as an optimization. 
            If not given, sytem of equations will be solved without explicitly computing this. 
            If given, should be a square nxn matrix.

    Returns:
        output_dr (np.array): The dual-rail outputs, defined as the difference between the output fluorescence from
            each dual-rail node's + and - fluorophores in the single-rail network. Should be length-n 1d array.
        output_sr (np.array): The single-rail outputs, defined as the output fluorescence from each single-rail node.
            Should be a length-2n 1d array.
        
    """
    num_nodes_dr = len(input_pattern)
    num_nodes_sr = 2*num_nodes_dr

    # Determine equivalent single-rail input rates based on dual-rail input pattern
    input_rates_sr = input_magnitude * k_in_from_input_data(input_pattern)

    output_sr = calc_network_output_sr(
        rate_matrix_sr, 
        input_rates_sr, 
        output_rates_sr, 
        decay_rates = decay_rates_sr, 
        Ainv = Ainv
    )

    output_dr = (output_sr[0::2] - output_sr[1::2]) / output_magnitude
#    print(output_sr)
#    print(output_dr)

    return output_dr, output_sr

def calc_real_network_output_sr(K_fret_CC, K_fret_IC, K_fret_CO, K_fret_CQ = None, C_k0 = 1, I_kin = None, I_k0 = 1, Ainv = None, verbose=False):
    """
    Computes the single-rail network output given all rate parameters (k_in, k_emit, k_decay, k_ij).
    The single-rail network output is the output fluorescence from each node, given by p_i * k^i_emit.

    Args:
        rate_matrix (np.array): The weights (rate constants, arbitrary units) between nodes in the system.
            Should be square and symmetric, with all diagonal entries equal to 0.
        input_rates (np.array): The intrinsic excitation rates of each node (k_in).
            Should be length-n 1d array, for network of n nodes.
        output_rates (np.array): The intrinsic emission rate constants of each node (k_out). 
            Should be a length-n 1d array, for network of n nodes.
        decay_rates (np.array): The intrinsic decay rate constants of each node (k_decay). 
            Should be a length-n 1d array, for network of n nodes.
        Ainv (np.array): [optional] The precomputed inverse A matrix. If not given, linear system of eqns will be
            solved without explicitly computing the inverse A matrix.
            If given, should be a square nxn matrix.
    
    Returns:
        pred (np.array): The predicted values of each node's output.
    """
    # estimate effective k_in into each compute fluorophore, based on excitation of each input
    I_prob_0 = I_kin / (K_fret_IC.sum(axis=1) + I_k0 + I_kin) # excitation probability of each input
    C_kin = I_prob_0 @ K_fret_IC

    # estimate effective k_out, and k_decay for each compute fluor
    C_kout = K_fret_CO.sum(axis=1)
    C_kdecay = K_fret_CQ.sum(axis=1) + C_k0

    if Ainv is None:
      num_nodes = len(C_kin)
      A = -K_fret_CC
      A[np.diag_indices(num_nodes)] = K_fret_CC.sum(axis=1) + C_kin + C_kout + C_kdecay
      C_prob = np.linalg.solve(A, C_kin)
    else:
      C_prob = Ainv @ C_kin

    return C_prob @ K_fret_CO

   
def calc_real_network_output_dr(input_pattern, K_fret_CC, K_fret_IC, K_fret_CO, K_fret_CQ = None, C_k0 = 1, I_k0 = 1, input_magnitude = 1, output_magnitude = 1, Ainv = None, verbose=False):
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
        decay_rates_sr (np.array): The intrinsic decay rate constants of each node in the single-rail system (k_decay).
            Should be a length-2n 1d nonnegative array
        input_magnitude (float or np.array): [default=1] The magnitude of input fluorescence into each node. 
            For each pixel, one of its corresponding fluorophores will have k_in=input_magnitude and the other k_in=0.
            If a float, all pixels have the same magnitude of input. If np.array, pixels have individually specified
            input magnitudes.
        output_magnitude (float): [default=1] The magnitude of output fluorescence expected from an "on" node.
            The output pixel value will be scaled so that if the range of (f_pos - f_neg) is 
              [-output_magnitude, +output_magnitude]
            then the range of pixel values will be
              [-1, +1].
        Ainv (np.array): [optional] The precomputed inverse A matrix, which may be given as an optimization. 
            If not given, sytem of equations will be solved without explicitly computing this. 
            If given, should be a square nxn matrix.

    Returns:
        output_dr (np.array): The dual-rail outputs, defined as the difference between the output fluorescence from
            each dual-rail node's + and - fluorophores in the single-rail network. Should be length-n 1d array.
        output_sr (np.array): The single-rail outputs, defined as the output fluorescence from each single-rail node.
            Should be a length-2n 1d array.
        
    """
    num_nodes_dr = len(input_pattern)
    num_nodes_sr = 2*num_nodes_dr

    if K_fret_CQ is None:
      K_fret_CQ = np.zeros_like(K_fret_CO)

    output_sr = calc_real_network_output_sr(
        K_fret_CC = K_fret_CC, 
        K_fret_IC = K_fret_IC, 
        K_fret_CO = K_fret_CO, 
        K_fret_CQ = K_fret_CQ,
        C_k0 = C_k0,
        I_kin = input_magnitude * k_in_from_input_data(input_pattern),
        I_k0 = I_k0,
        Ainv = Ainv,
        verbose=verbose
    )

    output_dr = (output_sr[0::2] - output_sr[1::2]) / output_magnitude

    return output_dr, output_sr
   

############
# TRAINING #
############
    
def generate_training_data(stored_data, noise=.1, duplication=10, mode = 'flip', filter_zero = False, rng = None):
    train_data = [
        (input_data, output_data)
            for output_data in stored_data 
            for input_data in off_patterns(output_data, noise, duplication, mode, rng = rng)
            if not filter_zero or np.any(input_data != 0)
    ]

    return train_data

def train_dr_hebbian(train_data):

    stored_data = list(set(train_pt[1] for train_pt in train_data))

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

    return K_fret, k_out

def train_dr(train_data, loss, train_data_weights = None, init = 'random', input_magnitude = 1, output_magnitude = None, k_out_value = 1, seed = None):
    def rates_to_params(K_fret, k_out):
        idxs = np.triu_indices(num_nodes_sr, 1)
        params = np.concatenate((K_fret[idxs], k_out))
        return params
    def params_to_rates(p):
        K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
        for idx, (i,j) in enumerate(it.combinations(range(num_nodes_sr),2)):
            K_fret[i,j] = p[idx]
            K_fret[j,i] = p[idx]
        k_out = k_out_value*np.ones(num_nodes_sr) # use this line for fixed, uniform k_out
#        k_out = p[-1]*np.ones(num_nodes_sr) # use this line for optimized, uniform k_out
#        k_out = p[-num_nodes_sr:] # use this line for optimized, non-uniform k_out
        return K_fret, k_out

    def loss_func_scipy(params):
        K_fret, k_out = params_to_rates(params)

        resid = np.array([
            loss.fn(
                calc_network_output_dr(
                    input_data,
                    K_fret,
                    k_out,
                    input_magnitude = input_magnitude,
                    output_magnitude = output_magnitude
                )[0],
                output_data_cor
            ) 
            for input_data,output_data_cor in train_data
        ])

        return resid

    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr

    num_params = num_nodes_sr*(num_nodes_sr-1)//2 + num_nodes_sr

    if output_magnitude is None:
      output_magnitude = input_magnitude * k_out_value / (input_magnitude + k_out_value)

    if init == 'one':
        init_params = np.ones(num_params)
    elif init == 'zero':
      init_params = np.zeros(num_params)
#    elif init == 'hebbian':
#      init_params = 5*rates_to_params(*train_dr_hebbian(stored_data)) + 1
    else:
      init_params = rng.uniform(1e-10, 100, num_params)
    res = scipy.optimize.least_squares(loss_func_scipy, init_params, bounds=(1e-10,10**5), jac='3-point')

#    print(res)

    K_fret, k_out = params_to_rates(params_cur)
    output = {
      'K_fret': K_fret,
      'k_out': k_out,
      'k_decay': np.zeros(num_nodes_sr),
      'cost': 2*res.cost,
      'raw': res
    }
   
    return output
    
def train_dr_MC(train_data, loss, train_data_weights = None, low_bound = 1e-2, high_bound = 1e4, input_magnitude = 1, output_magnitude = None, k_out_value = 1, anneal_protocol = None, goal_accept_rate = 0.3, init_noise = 2, seed = None, verbose=False):
    def rates_to_params(K_fret, k_out):
        idxs = np.triu_indices(num_nodes_sr, 1)
        params = np.concatenate((K_fret[idxs], k_out))
        return params
    def params_to_rates(p):
        K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
        for idx, (i,j) in enumerate(it.combinations(range(num_nodes_sr),2)):
            K_fret[i,j] = p[idx]
            K_fret[j,i] = p[idx]
        k_out = k_out_value*np.ones(num_nodes_sr) # use this line for fixed, uniform k_out
#        k_out = p[-1]*np.ones(num_nodes_sr) # use this line for optimized, uniform k_out
#        k_out = p[-num_nodes_sr:] # use this line for optimized, non-uniform k_out
        return K_fret, k_out

    def loss_func(params):
        K_fret, k_out = params_to_rates(params)

        resid = np.array([
            loss.fn(
                calc_network_output_dr(
                    input_data,
                    K_fret,
                    k_out,
                    input_magnitude = input_magnitude,
                    output_magnitude = output_magnitude
                )[0],
                output_data_cor
            ) 
            for input_data,output_data_cor in train_data
        ])

        return resid

    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr

    num_params = num_nodes_sr*(num_nodes_sr-1)//2 #+ num_nodes_sr

    if output_magnitude is None: 
      output_magnitude = input_magnitude * k_out_value / (input_magnitude + k_out_value)

    if anneal_protocol is None:
      anneal_protocol = np.concatenate((.0025*np.ones(500), np.arange(.0025, 0, -1e-6)))
    accept_hist_len = 500

    init_params = np.exp(rng.uniform(np.log(low_bound), np.log(high_bound), num_params))
#    print(params_to_rates(init_params))

    params_cur = init_params
    f_cur = np.sum(loss_func(params_cur)**2)
    noise = init_noise

    params_hist = []
    accept_hist = []
    for i, T in enumerate(anneal_protocol):
      params_new = np.exp(np.log(params_cur) + rng.normal(0, noise, num_params))

      if any(params_new > high_bound) or any(params_new < low_bound):
        f_new = np.inf # disallow moving outside the bounds
      else:
        f_new = np.sum(loss_func(params_new)**2)

      df = max(f_new - f_cur, T*np.log(1e-100)) # avoid overflows
#      accept_prob = np.exp(-df/T) * np.product(params_new/params_cur)  # use correction factor for uniform prior
      accept_prob = np.exp(-df/T)  # removing the correction factor uses a log-uniform prior
      accept = False
      if rng.uniform(0,1) < accept_prob:
        params_cur = params_new
        f_cur = f_new
        accept = True

      accept_hist.append(accept)
      if len(accept_hist) > accept_hist_len:
        accept_hist = accept_hist[-accept_hist_len:]

#      if np.mean(accept_hist) > goal_accept_rate:
#        noise *= 1.002
#      elif np.mean(accept_hist) < goal_accept_rate:
#        noise /= 1.002

      if verbose and i%500 == 0:
        print(i, T, f_cur, np.mean(accept_hist))

      params_hist.append((T, f_cur, params_cur))

  
  #    print(f'Monte Carlo optimization results: {f_cur}')
    K_fret, k_out = params_to_rates(params_cur)
    output = {
      'K_fret': K_fret,
      'k_out': k_out,
      'k_decay': np.zeros(num_nodes_sr),
      'cost': f_cur,
      'raw': params_hist
    }
   
    return output

def train_dr_MCGibbs(train_data, loss, anneal_protocol, train_data_weights = None, k_fret_bounds = (1e-2, 1e4), k_decay_bounds = (1, 1e4), k_fret_toggle_prob = None, input_magnitude = 1, output_magnitude = None, k_out_value = 1, accept_rate_min = .4, accept_rate_max = .6, init_K_fret = None, init_k_out = None, init_k_decay = None, init_step_size = 2, seed = None, history_output_interval = None, pbar_file = None, verbose=False):
    def rates_to_params(K_fret, k_out, k_decay):
        idxs = np.triu_indices(num_nodes_sr, 1)
        params_k_fret_toggle = (K_fret[idxs] > 0)
        params_k_fret_rate = K_fret[idxs] + (~params_k_fret_toggle)*k_fret_bounds[0]

        params_k_decay = k_decay

        params = (params_k_fret_rate, params_k_fret_toggle, params_k_decay)

        return params
    def params_to_rates(params):
        params_k_fret_rate, params_k_fret_toggle, params_k_decay = params
        K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
        K_fret[np.triu_indices(num_nodes_sr, 1)] = params_k_fret_rate * params_k_fret_toggle
        K_fret += K_fret.T
        k_decay = params_k_decay
        k_out = k_out_value*np.ones(num_nodes_sr) # use this line for fixed, uniform k_out
#        k_out = p[-1]*np.ones(num_nodes_sr) # use this line for optimized, uniform k_out
        return K_fret, k_out, k_decay

    def loss_func(params, Ainvs, verbose=False):
        K_fret, k_out, k_decay = params_to_rates(params)

        output_data_all = [
            calc_network_output_dr(
                input_data,
                K_fret,
                k_out,
                decay_rates_sr = k_decay,
                input_magnitude = input_magnitude,
                output_magnitude = output_magnitude,
                Ainv = Ainv
            )
            for (input_data, _), Ainv in zip(train_data, Ainvs)
        ]

        resid = np.sqrt(np.array([
            loss.fn(output_data[0], output_data_cor)**2 * weight
            for output_data, (input_data, output_data_cor), weight in zip(output_data_all, train_data, train_data_weights)
        ]).sum() / train_data_weight_sum)

        if verbose:
          for output_data, (input_data,output_data_cor), w in zip(output_data_all, train_data, train_data_weights):
            print(input_data, output_data_cor, output_data[0], output_data[1], loss.fn(output_data[0], output_data_cor), f'x{w}')

        return resid

    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr

    if output_magnitude is None:
      output_magnitude = input_magnitude*k_out_value/(input_magnitude + k_out_value)

    if train_data_weights is None:
      train_data_weights = np.ones(len(train_data))
    train_data_weight_sum = sum(train_data_weights)

    num_params_k_fret = num_nodes_sr*(num_nodes_sr-1)//2
    num_params_k_decay = num_nodes_sr
    num_params = num_params_k_fret + num_params_k_decay
    train_k_fret = k_fret_bounds[0] != k_fret_bounds[1]
    train_k_decay = k_decay_bounds[0] != k_decay_bounds[1]

    k_fret_node_idxs = list(zip(*np.triu_indices(num_nodes_sr, 1)))
 
    accept_hist_len = 50

    if init_K_fret is None:
      init_K_fret = np.zeros((num_nodes_sr, num_nodes_sr))
      init_K_fret[np.triu_indices(num_nodes_sr, 1)] = np.exp(rng.uniform(np.log(k_fret_bounds[0]), np.log(k_fret_bounds[1]), num_params_k_fret))
      init_K_fret += init_K_fret.T
    if init_k_out is None:
      init_k_out = k_out_value * np.ones(num_nodes_sr)
    if init_k_decay is None:
      init_k_decay = np.exp(rng.uniform(np.log(k_decay_bounds[0]), np.log(k_decay_bounds[1]), num_params_k_decay))
    init_params = rates_to_params(init_K_fret, init_k_out, init_k_decay)

    params_cur = init_params
    Ainvs_cur = [
        Ainv_from_rates(
            init_K_fret, 
            input_magnitude * k_in_from_input_data(input_data), 
            init_k_out+init_k_decay
        ) 
        for input_data,_ in train_data
    ]
    f_cur = loss_func(params_cur, Ainvs_cur)

    step_size = init_step_size*np.ones(num_params)
    step_size_adjust = 1.02

    params_train_protocol = [ # list of info needed to train each parameter
        (idx, 'k_fret', idx) for idx in range(num_params_k_fret) if train_k_fret
    ] + [
        (idx + num_params_k_fret, 'k_decay', idx) for idx in range(num_params_k_decay) if train_k_decay
    ]

    params_hist = []
    accept_hist = -1*np.ones((accept_hist_len, num_params), dtype=int)
    if pbar_file is None:
      temps_iter = enumerate(anneal_protocol)
    else:
      temps_iter = tqdm.tqdm(enumerate(anneal_protocol), total=len(anneal_protocol), file=pbar_file)

    time_last_update = time.time()

    for i, T in temps_iter:
      rate_steps = rng.normal(0, step_size, num_params)
      if k_fret_toggle_prob is not None:
        toggle_steps = rng.binomial(n=1, p=k_fret_toggle_prob, size=num_params_k_fret)
      else:
        toggle_steps = [False]*num_params_k_fret
      for param_idx, param_type, idx in params_train_protocol:
#      for p_idx, low_bound, high_bound, adjust_Ainv_func, adjust_Ainv_args in params_train_protocol:
        params_k_fret_rate, params_k_fret_toggle, params_k_decay = params_cur

        if param_type == 'k_fret':
          lb,hb = k_fret_bounds

          rate_cur = params_k_fret_rate[idx]
          toggle_cur = params_k_fret_toggle[idx]
          rate_new = rate_cur * np.exp(rate_steps[param_idx])
          toggle_new = ~toggle_cur if toggle_steps[idx] else toggle_cur

          params_k_fret_rate_new = params_k_fret_rate.copy()
          params_k_fret_rate_new[idx] = rate_new
          params_k_fret_toggle_new = params_k_fret_toggle.copy()
          params_k_fret_toggle_new[idx] = toggle_new
          params_new = (params_k_fret_rate_new, params_k_fret_toggle_new, params_k_decay)

          d_rate = toggle_new*rate_new - toggle_cur*rate_cur
          adjust_Ainv_func = lambda Ainv: adjust_Ainv_kfret(Ainv, d_rate, *k_fret_node_idxs[idx])
  
        elif param_type == 'k_decay':
          lb,hb = k_decay_bounds

          rate_cur = params_k_decay[idx]
          rate_new = rate_cur * np.exp(rate_steps[param_idx])

          params_k_decay_new = params_k_decay.copy()
          params_k_decay_new[idx] = rate_new
          params_new = (params_k_fret_rate, params_k_fret_toggle, params_k_decay_new)

          d_rate = rate_new - rate_cur
          adjust_Ainv_func = lambda Ainv: adjust_Ainv_koff(Ainv, d_rate, idx)
  
        else:
          raise ValueError(f'Unknown parameter type during Monte Carlo optimization: {param_type}')

        if rate_new > hb or rate_new < lb: # throw out any moves outside the bounding box
          f_new = np.inf
        elif d_rate == 0: # do not recalculate if rate didn't change (e.g. if toggle stayed set to OFF)
          Ainvs_new = Ainvs_cur
          f_new = f_cur
        else:
          Ainvs_new = [adjust_Ainv_func(Ainv) for Ainv in Ainvs_cur]
          f_new = loss_func(params_new, Ainvs_new)

        df = max(f_new - f_cur, 0)
        accept_prob = np.exp(-df/T) 
        accept = False
        if rng.uniform(0,1) < accept_prob:
          params_cur = params_new
          Ainvs_cur = Ainvs_new
          f_cur = f_new
          accept = True
  
        accept_hist[i%accept_hist_len, param_idx] = accept

      if i % accept_hist_len == accept_hist_len-1:
        accept_rate = np.mean(accept_hist, axis=0)
        accept_change = -1*(accept_rate - accept_rate_min < 0) + 1*(accept_rate - accept_rate_max > 0)
        step_size *= step_size_adjust ** accept_change
#        print(accept_rate, accept_change, step_size)

      if i%500 == 0: # every 500 iterations, recompute the A^-1 matrices in case of accumulated numerical errors
        K_fret, k_out, k_decay = params_to_rates(params_cur)
        Ainvs_new = [
            Ainv_from_rates(
                K_fret, 
                input_magnitude * k_in_from_input_data(input_data), 
                k_out + k_decay
            )
            for input_data,_ in train_data
        ]
        max_err = max(np.sum((Ainv_n - Ainv_c)**2) for Ainv_n, Ainv_c in zip(Ainvs_new, Ainvs_cur))
        if max_err > 1e-5:
          print(f'WARNING: Iteration {i}: Accumulated numerical error in computation of A^-1 led to discrepancies of {max_err}')

        Ainvs_cur = Ainvs_new

      if verbose and (i%500 == 0 or time.time() - time_last_update > 120):
        K_fret, k_out, k_decay = params_to_rates(params_cur)
        print(f'Iteration {i} (T={T}):')
        print(f'{params_cur}')
        print('K_fret:')
        print(K_fret)
        print(f'k_out: {k_out}')
        print(f'k_decay: {k_decay}')
        print(f'Iteration {i} (T={T}): Acceptance averages = {np.mean(accept_hist, axis=0)}')
        print(f'Iteration {i} (T={T}): Step sizes = {step_size}')
        print(f'Iteration {i} (T={T}): f_cur = {f_cur}')
        loss_func(params_cur, Ainvs_cur, verbose=True)
        time_last_update = time.time()

      if history_output_interval is not None and i%history_output_interval == 0:
        params_hist.append((T, f_cur, params_cur))

  
  #    print(f'Monte Carlo optimization results: {f_cur}')

    K_fret, k_out, k_decay = params_to_rates(params_cur)
    output = {
      'K_fret': K_fret,
      'k_out': k_out,
      'k_decay': k_decay,
      'cost': f_cur,
      'raw': params_hist
    }
   
    return output

def train_dr_MCGibbs_positions(train_data, loss, anneal_protocol, train_data_weights = None, k_0 = 1, r_0_cc = 1, position_bounds = (-1e2, 1e2), min_dist=1, dims=3, input_magnitude = 1, output_magnitude = None, k_out_value = 1, accept_rate_min = .4, accept_rate_max = .6, init_positions = None, init_step_size = 20, seed = None, history_output_interval = None, pbar_file = None, verbose=False):
    def rates_from_positions(positions):
        K_fret = np.array([
            [
                rate_from_positions(positions[n1,:], positions[n2,:], k_0, r_0_cc) if n1!=n2 else 0
                for n2 in range(num_nodes_sr)
            ] 
            for n1 in range(num_nodes_sr)
        ])
        k_decay = np.zeros(num_nodes_sr)
        k_out = k_out_value*np.ones(num_nodes_sr) # use this line for fixed, uniform k_out
#        k_out = p[-1]*np.ones(num_nodes_sr) # use this line for optimized, uniform k_out
#        k_out = p[-num_nodes_sr:] # use this line for optimized, non-uniform k_out
        return K_fret, k_out, k_decay

    def loss_func(positions, verbose = False):
        K_fret, k_out, k_decay = rates_from_positions(positions)

        output_data_all = [
            calc_network_output_dr(
                input_data,
                K_fret,
                k_out,
                decay_rates_sr = k_decay,
                input_magnitude = input_magnitude,
                output_magnitude = output_magnitude,
            )
            for input_data,_ in train_data
        ]

        resid = np.sqrt(np.array([
            loss.fn(output_data[0], output_data_cor)**2 * weight
            for output_data, (input_data, output_data_cor), weight in zip(output_data_all, train_data, train_data_weights)
        ]).sum() / train_data_weight_sum)

        if verbose:
          for output_data, (input_data,output_data_cor), w in zip(output_data_all, train_data, train_data_weights):
            print(input_data, output_data_cor, output_data[0], output_data[1], loss.fn(output_data[0], output_data_cor), f'x{w}')

        return resid


    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr

    if train_data_weights is None:  train_data_weights = np.ones(len(train_data))
    train_data_weight_sum = sum(train_data_weights)

    if output_magnitude is None:
      output_magnitude = input_magnitude*k_out_value/(input_magnitude + k_out_value)

    min_position, max_position = position_bounds

    if init_positions is None:
      init_positions = rng.uniform(min_position, max_position, (num_nodes_sr, dims))

    positions_cur = init_positions
    f_cur = loss_func(positions_cur)

    accept_hist_len = 50
    step_size = init_step_size*np.ones(num_nodes_sr)
    step_size_adjust = 1.05

    pos_hist = []
    accept_hist = -1*np.ones((accept_hist_len, num_nodes_sr), dtype=int)
    if pbar_file is None:
      temps_iter = enumerate(anneal_protocol)
    else:
      temps_iter = tqdm.tqdm(enumerate(anneal_protocol), total=len(anneal_protocol), file=pbar_file)
    for i, T in temps_iter:
      steps_mag = rng.normal(0, step_size, num_nodes_sr)
      steps = np.array([random_point_on_sphere(step_mag, dims=dims, rng=rng) for step_mag in steps_mag])
      for node in range(num_nodes_sr):
        pos_cur = positions_cur[node,:]
        pos_new = pos_cur + steps[node,:]

        positions_new = positions_cur.copy()
        positions_new[node,:] = pos_new

        if np.any(pos_new > max_position) or np.any(pos_new < min_position): # throw out any moves outside the bounding box
          f_new = np.inf
        elif (np.linalg.norm(pos_new.reshape((1,dims)) - positions_new, axis=1) < min_dist).sum() > 1: # throw out the move if two fluors are now below the minimum distance
          f_new = np.inf
        else:
          f_new = loss_func(positions_new)

        df = max(f_new - f_cur, 0)
        accept_prob = np.exp(-df/T) 
#        print(f_cur, f_new, accept_prob)
        accept = False
        if rng.uniform(0,1) < accept_prob:
          positions_cur = positions_new
          f_cur = f_new
          accept = True

  
        accept_hist[i%accept_hist_len, node] = accept

      if i % accept_hist_len == accept_hist_len-1:
        accept_rate = np.mean(accept_hist, axis=0)
        accept_change = -1*(accept_rate - accept_rate_min < 0) + 1*(accept_rate - accept_rate_max > 0)
        step_size *= step_size_adjust ** accept_change
#        print(accept_rate, accept_change, step_size)

      if verbose and i%500 == 0:
        K_fret, k_out, k_decay = rates_from_positions(positions_cur)
        print(f'Iteration {i} (T={T}):')
        print('K_fret:')
        print(K_fret)
        print(f'k_out: {k_out}')
        print(f'k_decay: {k_decay}')
        print(f'Iteration {i} (T={T}): Acceptance averages = {np.mean(accept_hist, axis=0)}')
        print(f'Iteration {i} (T={T}): Step sizes = {step_size}')
        print(f'Iteration {i} (T={T}): f_cur = {f_cur}')
        loss_func(positions_cur, verbose=True)



      if history_output_interval is not None and i%history_output_interval == 0:
        pos_hist.append((T, f_cur, positions_cur))

  
  #    print(f'Monte Carlo optimization results: {f_cur}')

    # Center positions around the origin
    positions_cur -= positions_cur.mean(axis=0)
    

    K_fret, k_out, k_decay = rates_from_positions(positions_cur)

    node_names_dr = list(map(str, range(1, num_nodes_dr+1)))
    node_names_sr = [f'{n_dr}{pm}' for n_dr in node_names_dr for pm in ['+','-']]
    fluor_names = node_names_sr
    fluor_types = ['C']*num_nodes_sr
    nodes_map = {n_dr: (node_names_sr[2*i], node_names_sr[2*i+1]) for i,n_dr in enumerate(node_names_dr)}
    fluor_map = dict(zip(node_names_sr, fluor_names))

    output = {
      'K_fret': K_fret,
      'k_out': k_out,
      'k_decay': k_decay,

      'positions': positions_cur,

#      'network': network,
      'num_nodes_dr': num_nodes_dr,
      'num_nodes_sr': num_nodes_sr,
      'num_fluorophores': num_nodes_sr,

      'node_names_dr': node_names_dr,
      'node_names_sr': node_names_sr,
      'fluorophore_names': fluor_names,
      'fluorophore_types': fluor_types,
      'dr_to_sr_map': nodes_map,
      'sr_to_fluor_map': fluor_map,

      'cost': f_cur,
      'raw': pos_hist
    }
   
    return output


def train_dr_MCGibbs_positions_full(train_data, loss, anneal_protocol, train_data_weights = None, input_fluor_info = {}, compute_fluor_info = {}, position_bounds = (-1e2, 1e2), min_dist=1, max_dist_CI = np.inf, max_dist_CO = np.inf, max_dist_CQ = np.inf, dims=3, input_magnitude = 100, output_magnitude = 1, accept_rate_min = .4, accept_rate_max = .6, init_positions = None, init_step_size = 20, seed = None, history_output_interval = None, pbar_file = None, verbose=False):
    def initialize_positions():
      init_pos = np.empty((num_fluorophores, dims))
      for node_idx in range(num_nodes_sr):
        C_idx = C_idxs[node_idx]
        C_pos = rng.uniform(min_position, max_position, (dims,))
        init_pos[C_idx, :] = C_pos

        for idx_lst, max_dist in zip([I_idxs, O_idxs, Q_idxs], [max_dist_CI, max_dist_CO, max_dist_CQ]):
          if max_dist < max_position - min_position:
            init_pos[idx_lst[node_idx], :] = random_point_on_sphere(max_dist * rng.uniform(0,1)**(1./dims), dims=dims, rng=rng) + C_pos
          else:
            init_pos[idx_lst[node_idx], :] = rng.uniform(min_position, max_position, (dims,))
      return init_pos

    def positions_from_params(params):
      pos = params.copy()
      C_pos = params[C_idxs,:]
      pos[I_idxs,:] += C_pos
      pos[O_idxs,:] += C_pos
      pos[Q_idxs,:] += C_pos
      return pos
    def params_from_positions(pos):
      params = pos.copy()
      C_pos = params[C_idxs,:]
      params[I_idxs,:] -= C_pos
      params[O_idxs,:] -= C_pos
      params[Q_idxs,:] -= C_pos
      return params
    def rates_from_positions(pos):
        # compute FRET rate constants between relevant fluorophore pairs
        K_fret_IC = np.array([
            [rate_from_positions(pos[n1,:], pos[n2,:], I_k0, IC_r0) for n2 in C_idxs] 
            for n1 in I_idxs
        ])
        K_fret_CC = np.array([
            [rate_from_positions(pos[n1,:], pos[n2,:], C_k0, CC_r0) if n1!=n2 else 0 for n2 in C_idxs] 
            for n1 in C_idxs
        ])
        K_fret_CO = np.array([
            [rate_from_positions(pos[n1,:], pos[n2,:], C_k0, CO_r0) for n2 in O_idxs] 
            for n1 in C_idxs
        ])
        K_fret_CQ = np.array([
            [rate_from_positions(pos[n1,:], pos[n2,:], C_k0, CQ_r0) for n2 in Q_idxs] 
            for n1 in C_idxs
        ])

        return K_fret_CC, K_fret_IC, K_fret_CO, K_fret_CQ

    def loss_func(positions, verbose=False):
        """ Compute network output assuming:
            * Each input fluorophore has k_in = <input_magnitude>, so that if the fluorophore was very close to its
              compute fluorophore then it would have k_in = <input_magnitude> also
            * Each input fluorophore can be analyzed as if its k_off = (sum of k_FRET with each compute fluor)
            * Each output fluorophore serves as an infinite sink that immediately emits any FRET energy it receives
            * Each quencher serves as an infinite sink (no emission though)
            * Only FRET interactions are
              input -> compute, compute <-> compute, compute -> output, and compute -> quencher
        """
        K_fret_CC, K_fret_IC, K_fret_CO, K_fret_CQ = rates_from_positions(positions)

        k_in = np.diag(K_fret_IC)
        k_out = np.diag(K_fret_CO)
        k_decay = np.diag(K_fret_CQ) + C_k0

#        resid = np.array([
#            loss.fn(
#                calc_real_network_output_dr(
#                    input_data,
#                    K_fret_CC = K_fret_CC,
#                    K_fret_IC = K_fret_IC,
#                    K_fret_CO = K_fret_CO,
#                    K_fret_CQ = K_fret_CQ,
#                    C_k0 = C_k0,
#                    I_k0 = I_k0,
#                    input_magnitude = input_magnitude,
#                    output_magnitude = output_magnitude,
#                )[0],
#                output_data_cor
#            )**2 * multiplicity
#            for (input_data,output_data_cor),multiplicity in train_data_opt
#        ]).sum()
        output_data_all = [
            calc_real_network_output_dr(
                input_data,
                K_fret_CC = K_fret_CC,
                K_fret_IC = K_fret_IC,
                K_fret_CO = K_fret_CO,
                K_fret_CQ = K_fret_CQ,
                C_k0 = C_k0,
                I_k0 = I_k0,
                input_magnitude = input_magnitude,
                output_magnitude = output_magnitude,
                verbose = (verbose and (i%500==0))
            )
            for input_data,_ in train_data
        ]
        resid = np.sqrt(np.array([
            loss.fn(output_data[0], output_data_cor)**2 * weight
            for output_data, (input_data,output_data_cor), weight in zip(output_data_all, train_data, train_data_weights)
        ]).sum() / train_data_weight_sum)

        if verbose:
          for output_data, (input_data,output_data_cor),w in zip(output_data_all, train_data, train_data_weights):
            print(input_data, output_data_cor, output_data[0], output_data[1], loss.fn(output_data[0], output_data_cor), f'x{w}')

        return resid

    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr
    num_fluorophores = 4*num_nodes_sr

    if train_data_weights is None:  train_data_weights = np.ones(len(train_data))
    train_data_weight_sum = sum(train_data_weights)

    I_k0 = input_fluor_info.get('k_0', 1)
    C_k0 = compute_fluor_info.get('k_0', 1)

    IC_r0 = input_fluor_info.get('r_0', {}).get('compute', 2)
    CC_r0 = compute_fluor_info.get('r_0', {}).get('compute', 7)
    CO_r0 = compute_fluor_info.get('r_0', {}).get('output', 2)
    CQ_r0 = compute_fluor_info.get('r_0', {}).get('quencher', 2)

    min_position, max_position = position_bounds

    I_idxs = list(range(0, num_nodes_sr))
    C_idxs = list(range(num_nodes_sr, 2*num_nodes_sr))
    O_idxs = list(range(2*num_nodes_sr, 3*num_nodes_sr))
    Q_idxs = list(range(3*num_nodes_sr, 4*num_nodes_sr))

    I_idx_set, C_idx_set, O_idx_set, Q_idx_set = map(set, [I_idxs, C_idxs, O_idxs, Q_idxs])

    if init_positions is None:  init_positions = initialize_positions()
    init_params = params_from_positions(init_positions)

    params_cur = init_params
    f_cur = loss_func(init_positions)

    accept_hist_len = 50
    step_size = init_step_size*np.ones(num_fluorophores)
    step_size_adjust = 1.05

    pos_hist = [(None, f_cur, init_positions)]
    accept_hist = -1*np.ones((accept_hist_len, num_fluorophores), dtype=int)
    if pbar_file is None:
      temps_iter = enumerate(anneal_protocol)
    else:
      temps_iter = tqdm.tqdm(enumerate(anneal_protocol), total=len(anneal_protocol), file=pbar_file)

    time_last_update = time.time()
    for i, T in temps_iter:
      steps_mag = rng.normal(0, step_size, num_fluorophores)
      steps = np.array([random_point_on_sphere(step_mag, dims=dims, rng=rng) for step_mag in steps_mag])
      for nodegroup_idx, fluor_idx in it.chain(*map(enumerate, [I_idxs, C_idxs, O_idxs, Q_idxs])):
        param_cur = params_cur[fluor_idx,:]
        param_new = param_cur + steps[fluor_idx,:]

        params_new = params_cur.copy()
        params_new[fluor_idx,:] = param_new

        positions_new = positions_from_params(params_new)
        pos_new = positions_new[fluor_idx,:]

        f_new = None
        
        param_new_norm = np.linalg.norm(param_new)
        if fluor_idx in I_idx_set and param_new_norm > max_dist_CI \
            or fluor_idx in O_idx_set and param_new_norm > max_dist_CO \
            or fluor_idx in Q_idx_set and param_new_norm > max_dist_CQ:
          # reject move if it brings an I/O/Q fluor too far from its C fluor
          f_new = np.inf
          fluors_mod = []
        elif fluor_idx in C_idx_set:
          fluors_mod = [I_idxs[nodegroup_idx], C_idxs[nodegroup_idx], O_idxs[nodegroup_idx], Q_idxs[nodegroup_idx]]
        else:
          fluors_mod = [fluor_idx]

        for f_idx in fluors_mod:
          pos_new = positions_new[f_idx,:]
          if np.any(pos_new > max_position) or np.any(pos_new < min_position):
            # reject moves outside the bounding box
            f_new = np.inf
          elif (np.linalg.norm(pos_new.reshape((1,dims)) - positions_new, axis=1) < min_dist).sum() > 1:
            # reject move if it brings two fluors too close together
            f_new = np.inf

        if f_new is None:
          f_new = loss_func(positions_new)

        df = max(f_new - f_cur, 0)
        accept_prob = np.exp(-df/T) 
#        print(f_cur, f_new, accept_prob)
        accept = False
        if rng.uniform(0,1) < accept_prob:
          params_cur = params_new
          f_cur = f_new
          accept = True

        accept_hist[i%accept_hist_len, fluor_idx] = accept

      if i % accept_hist_len == accept_hist_len-1:
        accept_rate = np.mean(accept_hist, axis=0)
        accept_change = -1*(accept_rate - accept_rate_min < 0) + 1*(accept_rate - accept_rate_max > 0)
        step_size *= step_size_adjust ** accept_change
#        print(accept_rate, accept_change, step_size)

      if verbose and (i%500 == 0 or time.time() - time_last_update > 120):
        positions_cur = positions_from_params(params_cur)
        K_fret, K_input, K_output, K_quench = rates_from_positions(positions_cur)
        print(f'Iteration {i} (T={T}):')
        print('params | positions:')
        print(np.hstack((params_cur, positions_cur)))
        print('K_fret:')
        print(K_fret)
#        print(i, K_input)
#        print(i, K_output)
#        print(i, K_quench)
        print(f'Iteration {i} (T = {T}): Acceptance averages = {np.mean(accept_hist, axis=0)}')
        print(f'Iteration {i} (T = {T}): Step sizes = {step_size}')
        print(f'Iteration {i} (T={T}): f_cur = {f_cur}')
        loss_func(positions_cur, verbose=True)
        time_last_update = time.time()

      if history_output_interval is not None and i%history_output_interval == 0:
        pos_hist.append((T, f_cur, positions_from_params(params_cur)))

  
  #    print(f'Monte Carlo optimization results: {f_cur}')

    # Center positions around the origin
    positions_cur = positions_from_params(params_cur)
    positions_cur -= positions_cur.mean(axis=0)

    K_fret_CC, K_fret_IC, K_fret_CO, K_fret_CQ = rates_from_positions(positions_cur)
    k_in = np.diag(K_fret_IC)
    k_out = np.diag(K_fret_CO)
    k_decay = np.diag(K_fret_CQ) + C_k0

    node_names_dr = list(map(str, range(1, num_nodes_dr+1)))
    node_names_sr = [f'{n_dr}{pm}' for n_dr in node_names_dr for pm in ['+','-']]
    fluor_names = [f'{node_name}_{role}' for role in ['I','C','O','Q'] for node_name in node_names_sr]
#    network = objects.network_from_rates(K_fret, k_out, np.zeros_like(k_out), k_decay = k_decay+k_0, node_names=node_names_sr)
    fluor_types = [role for role in ['I','C','O','Q'] for _ in range(num_nodes_sr)]
#    node_types_sr = ['0']*num_nodes_sr
    nodes_map = {n_dr: (node_names_sr[2*i], node_names_sr[2*i+1]) for i,n_dr in enumerate(node_names_dr)}
    fluor_map = {n_sr: tuple(fluor_names[4*i::num_nodes_sr]) for i,n_sr in enumerate(node_names_sr)}

    output = {
      'I_k0': I_k0,
      'C_k0': C_k0,
      'IC_r0': IC_r0,
      'CC_r0': CC_r0,
      'CO_r0': CO_r0,
      'CQ_r0': CQ_r0,

      'K_fret': K_fret_CC,
      'K_in': K_fret_IC,
      'K_out': K_fret_CO,
      'K_quench': K_fret_CQ,
      'k_in': k_in,
      'k_out': k_out,
      'k_decay': k_decay,

      'positions': positions_cur,

#      'network': network,
      'num_nodes_dr': num_nodes_dr,
      'num_nodes_sr': num_nodes_sr,
      'num_fluorophores': num_fluorophores,

      'node_names_dr': node_names_dr,
      'node_names_sr': node_names_sr,
      'fluorophore_names': fluor_names,
      'fluorophore_types': fluor_types,
      'dr_to_sr_map': nodes_map,
      'sr_to_fluor_map': fluor_map,

      'cost': f_cur,
      'raw': pos_hist
    }
   
    return output

def train_dr_MCGibbs_positions_full_2step(train_data, loss, train_data_weights = None, train_kwargs_MG = {}, train_kwargs_MGpf = {}, dims = 3, cluster_threshold = 1e-2, cluster_spacing = 20, seed = None, verbose = False):
    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr
    num_nodes_f = 4*num_nodes_sr

    # Initial training of K_fret rates directly
    train_kwargs_MG['seed'] = train_kwargs_MG.get('seed', rng.integers(0, 10**6))
    res_MG = train_dr_MCGibbs(train_data, loss, train_data_weights = train_data_weights, **train_kwargs_MG)
    K_fret_0 = res_MG['K_fret']

    # Run position solver for each cluster
    input_fluor_info = train_kwargs_MGpf.get('input_fluor_info', {})
    compute_fluor_info = train_kwargs_MGpf.get('compute_fluor_info', {})

    I_k0 = input_fluor_info.get('k_0', 1)
    C_k0 = compute_fluor_info.get('k_0', 1)

    IC_r0 = input_fluor_info.get('r_0', {}).get('compute', 2)
    CC_r0 = compute_fluor_info.get('r_0', {}).get('compute', 7)
    CO_r0 = compute_fluor_info.get('r_0', {}).get('output', 2)
    CQ_r0 = compute_fluor_info.get('r_0', {}).get('quencher', 2)

    # Use MDS to get estimate for initial positions
    init_pos_compute_fluors = rates_to_positions(K_fret_0, k_0 = C_k0, r_0 = CC_r0, dims=dims)
    init_pos = np.empty((num_nodes_f, dims))
    for node_idx in range(num_nodes_sr):
      I_idx, C_idx, O_idx, Q_idx = node_idx, num_nodes_sr+node_idx, 2*num_nodes_sr+node_idx, 3*num_nodes_sr+node_idx
      C_pos = init_pos_compute_fluors[node_idx, :]
      init_pos[C_idx, :] = C_pos

      for idx, ideal_dist in zip([I_idx, O_idx, Q_idx], [IC_r0, CO_r0, CQ_r0]):
        init_pos[idx, :] = random_point_on_sphere(.5*ideal_dist * rng.uniform(0,1)**(1./dims), dims=dims, rng=rng) + C_pos

    # Overwrite some arguments and run
    train_kwargs_MGpf['seed'] = train_kwargs_MGpf.get('seed', rng.integers(0, 10**6))
    train_kwargs_MGpf['dims'] = dims
    train_kwargs_MGpf['init_positions'] = init_pos
    res_MGpf = train_dr_MCGibbs_positions_full(train_data, loss, train_data_weights = train_data_weights, **train_kwargs_MGpf)

    if verbose:
      print(f'Positions optimization complete:')
      print(f'positions:')
      print(res_MGpf['positions'])
      print(f'K_fret:')
      print(res_MGpf['K_fret'])
    
    # Collect results
    node_names_dr = list(map(str, range(1, num_nodes_dr+1)))
    node_names_sr = [f'{n_dr}{pm}' for n_dr in node_names_dr for pm in ['+','-']]
    fluor_names = [f'{node_name}_{role}' for role in ['I','C','O','Q'] for node_name in node_names_sr]
    fluor_types = [role for role in ['I','C','O','Q'] for _ in range(num_nodes_sr)]
    dr_to_sr_map = {n_dr: (node_names_sr[2*i], node_names_sr[2*i+1]) for i,n_dr in enumerate(node_names_dr)}
    sr_to_fluor_map = {n_sr: tuple(fluor_names[4*i::num_nodes_sr]) for i,n_sr in enumerate(node_names_sr)}

    raw = {'res_MG': res_MG, 'res_MGpf': res_MGpf}

    output = {
      'I_k0': I_k0,
      'C_k0': C_k0,
      'IC_r0': IC_r0,
      'CC_r0': CC_r0,
      'CO_r0': CO_r0,
      'CQ_r0': CQ_r0,

      'K_fret': res_MGpf['K_fret'],
      'K_in': res_MGpf['K_in'],
      'K_out': res_MGpf['K_out'],
      'K_quench': res_MGpf['K_quench'],
      'k_in': res_MGpf['k_in'],
      'k_out': res_MGpf['k_out'],
      'k_decay': res_MGpf['k_decay'],

      'positions': res_MGpf['positions'],

      'num_nodes_dr': num_nodes_dr,
      'num_nodes_sr': num_nodes_sr,
      'num_fluorophores': num_nodes_f,

      'node_names_dr': node_names_dr,
      'node_names_sr': node_names_sr,
      'fluorophore_names': fluor_names,
      'fluorophore_types': fluor_types,
      'dr_to_sr_map': dr_to_sr_map,
      'sr_to_fluor_map': sr_to_fluor_map,

      'cost': res_MGpf['cost'],
      'raw': raw
    }
    
    return output


def train_dr_multiple_multiprocessing_aux(args):
    train_func, train_data, train_data_weights, loss, seed, train_kwargs = args
    res = train_func(train_data, loss, train_data_weights = train_data_weights, seed=seed, **train_kwargs)
    return res
def train_dr_multiple_multiprocessing(train_func, train_data, train_data_weights, loss, processes, reps, pbar_file, seed, **train_kwargs):
    rng = np.random.default_rng(seed)
    
    seeds = [rng.integers(0, 10**6) for _ in range(reps)]

    results = []
    with mp.Pool(processes=processes) as pool:
      args_lst = [(train_func, train_data, train_data_weights, loss, seed, train_kwargs) for seed in seeds]
      results_it = pool.imap(train_dr_multiple_multiprocessing_aux, args_lst)
      for res in tqdm.tqdm(results_it, total=reps, file=pbar_file):
        results.append(res)

    return results, seeds

def train_dr_multiple_singleprocessing(train_func, train_data, train_data_weights, loss, reps, pbar_file, seed, **train_kwargs):
    rng = np.random.default_rng(seed)
    
    seeds = [rng.integers(0, 10**6) for _ in range(reps)]

    results = []
    for train_seed in tqdm.tqdm(seeds, file=pbar_file):
      res = train_func(train_data, loss, train_data_weights = train_data_weights, seed=seed, **train_kwargs)
      results.append(res)

    return results, seeds

def train_dr_multiple(train_func, train_data, train_data_weights, loss, processes = None, reps = 10, pbar_file = None, seed = None, **train_kwargs):
    if pbar_file is None:  pbar_file = sys.stderr # default for tqdm

    if processes is None:
      results, seeds = train_dr_multiple_singleprocessing(train_func, train_data, train_data_weights, loss, reps, pbar_file, seed, **train_kwargs)
    else:
      results, seeds = train_dr_multiple_multiprocessing(train_func, train_data, train_data_weights, loss, processes, reps, pbar_file, seed, **train_kwargs)


    return results, seeds
  
    
def compare_train_funcs(funcs_lst, args_lst, num_nodes = 4, num_patterns = 3, noise = 0.1, duplication = 20, iters = 50):
    all_output = []
    for i in range(iters):
      seed = np.random.randint(0, 10**6)
      rng = np.random.default_rng(seed)

      stored_data_ints = rng.permutation(2**num_nodes)[:num_patterns]
      stored_data = [
          np.array([int(v)*2-1 for v in format(i,'0{}b'.format(num_nodes))])
          for i in stored_data_ints
      ]
  
      train_data = generate_training_data(stored_data, noise = noise, duplication=duplication, rng = rng)
  
      train_seed = rng.integers(0, 10**6)
      training_results = [f(train_data, seed = train_seed, **args) for f, args in zip(funcs_lst, args_lst)]
  
      # generate Network object with these connections
      k_in = np.array([
          kin for bit in stored_data[0] for kin in (max(bit, 0), -min(bit, 0))
      ])
      trained_networks = [
          network_from_rates(res['K_fret'], res['k_out'], k_in, k_decay=res['k_decay']) for res in training_results
      ]

      print('Iteration {}: {}'.format(i, '\t'.join(str(results[2]) for results in training_results)))
  
      output = {
        'seed': seed,
        'train_seed': train_seed,
        'functions': funcs_lst,
        'function_args': args_lst,
        'num_nodes': num_nodes,
        'num_patterns': num_patterns,
        'stored_data': stored_data,
        'train_noise': noise,
        'train_duplication': duplication,
        'train_data': train_data,
        'function_outputs': [{'K_fret': K, 'k_out': k_out, 'cost': cost, 'raw': res} for K, k_out, cost, res in training_results],
        'trained_networks': trained_networks
      }

      all_output.append(output)

    return all_output

