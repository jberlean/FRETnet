import os, sys
import itertools as it
import pickle
import math
import multiprocessing as mp

import numpy as np
import scipy.optimize
import tqdm

# INTRAPACKAGE IMPORTS
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_dir not in sys.path:
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path

from train_utils import off_patterns, Ainv_from_rates, A_from_rates, k_in_from_input_data, network_from_rates

# TODO: use functions in train_singlerail.py rather than reimplementing them here

########################
# GRADIENT CALCULATION #
########################

def adjust_Ainv(Ainv, dx, i, j):
  # As an optimization, if only a single rate has changed we may recompute
  # A^-1 quickly, using the Sherman-Morrison formula for rank-1 matrix perturbations:
  #   dAinv = -dx/(1 + dx*(Ainv[i,i] - 2*Ainv[i,j] + Ainv[j,j]) * ((A[:,i] - A[:,j]) @ (A[i,:] - A[j,:]))
  coef = -dx / (1 + dx * (Ainv[i,i] + Ainv[j,j] - 2*Ainv[i,j]))
  dAinv = np.outer(coef*(Ainv[i,:] - Ainv[j,:]), Ainv[i,:] - Ainv[j,:])
  return Ainv + dAinv

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
#    if Ainv is None:  Ainv = Ainv_from_rates(rate_matrix, output_rates, input_rates)

#    pred = Ainv @ input_rates
    if Ainv is None:
      num_nodes = len(input_rates)
      A = -rate_matrix
      A[np.diag_indices(num_nodes)] = rate_matrix.sum(axis=1) + input_rates + output_rates
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

def calc_network_output_dr(input_pattern, rate_matrix_sr, output_rates_sr, input_magnitude = 1, output_magnitude = 1, Ainv = None):
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
        input_magnitude (float): [default=1] The magnitude of input fluorescence into each node. 
            For each pixel, one of its corresponding fluorophores will have k_in=input_magnitude and the other k_in=0.
        output_magnitude (float): [default=1] The magnitude of output fluorescence expected from an "on" node.
            The output pixel value will be scaled so that if the range of (f_pos - f_neg) is 
              [-output_magnitude, +output_magnitude]
            then the range of pixel values will be
              [-1, +1].
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
    input_rates_sr = input_magnitude * k_in_from_input_data(input_pattern)

    output_sr = calc_network_output_sr(rate_matrix_sr, input_rates_sr, output_rates_sr, Ainv = Ainv)

    output_dr = (output_sr[0::2] - output_sr[1::2]) / output_magnitude
#    print(output_sr)
#    print(output_dr)

    return output_dr, output_sr
    

   

############
# TRAINING #
############
    
def generate_training_data(stored_data, noise=.1, duplication=10, rng = None):
    train_data = [
        (input_data, output_data)
            for output_data in stored_data 
            for input_data in off_patterns(output_data, noise, duplication, rng = rng)
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

def train_dr(train_data, loss, init = 'random', input_magnitude = 1, output_magnitude = None, k_out_value = 1, seed = None):
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

    return (*params_to_rates(res.x), 2*res.cost, res)
    
def train_dr_MC(train_data, loss, low_bound = 1e-10, high_bound = 1e5, input_magnitude = 1, output_magnitude = None, k_out_value = 1, anneal_protocol = None, goal_accept_rate = 0.3, init_noise = 2, seed = None, verbose=False):
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
   
    return (*params_to_rates(params_cur), f_cur, params_hist)

def train_dr_MCGibbs(train_data, loss, low_bound = 1e-10, high_bound = 1e5, input_magnitude = 1, output_magnitude = None, k_out_value = 1, anneal_protocol = None, warmup_iters = 2500, goal_accept_rate = 0.44, init_step_size = 2, seed = None, history_output_interval = None, pbar_file = None, verbose=False):
    def rates_to_params(K_fret, k_out):
        idxs = np.triu_indices(num_nodes_sr, 1)
        params = K_fret[idxs]
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

    def loss_func(params, Ainvs):
        K_fret, k_out = params_to_rates(params)

        resid = (np.array([
            loss.fn(
                calc_network_output_dr(
                    input_data,
                    K_fret,
                    k_out,
                    input_magnitude = input_magnitude,
                    output_magnitude = output_magnitude,
                    Ainv = Ainv
                )[0],
                output_data_cor
            ) 
            for (input_data,output_data_cor),Ainv in zip(train_data, Ainvs)
        ])**2).sum()

        return resid

    rng = np.random.default_rng(seed)

    num_nodes_dr = len(train_data[0][0])
    num_nodes_sr = 2*num_nodes_dr

    num_params = num_nodes_sr*(num_nodes_sr-1)//2

    if output_magnitude is None:
      output_magnitude = input_magnitude*k_out_value/(input_magnitude + k_out_value)
 
    if anneal_protocol is None:
      anneal_protocol = np.concatenate((.0025*np.ones(500), np.arange(.0025, 0, -1e-6)))

    accept_hist_len = 50

    init_params = np.exp(rng.uniform(np.log(low_bound), np.log(high_bound), num_params))
    init_K_fret, init_k_out = params_to_rates(init_params)

    params_cur = init_params
    Ainvs_cur = [Ainv_from_rates(init_K_fret, input_magnitude * k_in_from_input_data(input_data), init_k_out) for input_data,_ in train_data]
    f_cur = loss_func(params_cur, Ainvs_cur)

    step_size = init_step_size*np.ones(num_params)
    step_size_adjust = 1.02

    params_hist = []
    accept_hist = -1*np.ones((accept_hist_len, num_params), dtype=int)
    if pbar_file is None:
      temps_iter = enumerate(anneal_protocol)
    else:
      temps_iter = tqdm.tqdm(enumerate(anneal_protocol), total=len(anneal_protocol), file=pbar_file)
    for i, T in temps_iter:
      steps = rng.normal(0, step_size, num_params)
      for p_idx, (node_in, node_out) in enumerate(zip(*np.triu_indices(num_nodes_sr, 1))):
        param_cur = params_cur[p_idx]
        param_new = param_cur * np.exp(steps[p_idx])

        params_new = params_cur.copy()
        params_new[p_idx] = param_new
  
        if param_new > high_bound or param_new < low_bound: # throw out any moves outside the bounding box
          f_new = np.inf
        else:
          Ainvs_new = [adjust_Ainv(Ainv, param_new - param_cur, node_in, node_out) for Ainv in Ainvs_cur]
          f_new = loss_func(params_new, Ainvs_new)

        df = max(f_new - f_cur, 0)
        accept_prob = np.exp(-df/T) 
        accept = False
        if rng.uniform(0,1) < accept_prob:
          params_cur = params_new
          Ainvs_cur = Ainvs_new
          f_cur = f_new
          accept = True
  
        accept_hist[i%accept_hist_len, p_idx] = accept

#      print(accept_hist[i%accept_hist_len, :])
      if i % accept_hist_len == accept_hist_len-1 and i < warmup_iters:
        step_size *= step_size_adjust ** (np.sign(np.mean(accept_hist, axis=0)-goal_accept_rate))

#        print(step_size)
#      if np.mean(accept_hist) > goal_accept_rate:
#        noise *= 1.002
#      elif np.mean(accept_hist) < goal_accept_rate:
#        noise /= 1.002

      if i%500 == 0: # every 500 iterations, recompute the A^-1 matrices in case of accumulated numerical errors
        K_fret, k_out = params_to_rates(params_cur)
        Ainvs_new = [
            Ainv_from_rates(
                K_fret, 
                input_magnitude * k_in_from_input_data(input_data), 
                k_out
            )
            for input_data,_ in train_data
        ]
        max_err = max(np.sum((Ainv_n - Ainv_c)**2) for Ainv_n, Ainv_c in zip(Ainvs_new, Ainvs_cur))
        if max_err > 1e-5:
          print(f'WARNING: Accumulated numerical error in computation of A^-1 led to discrepancies of {max_err}')

        Ainvs_cur = Ainvs_new

      if verbose and i%500 == 0:
        print(i, T, f_cur, params_cur)
        print(i, np.mean(accept_hist, axis=0), step_size)

      if history_output_interval is not None and i%history_output_interval == 0:
        params_hist.append((T, f_cur, params_cur))

  
  #    print(f'Monte Carlo optimization results: {f_cur}')
   
    return (*params_to_rates(params_cur), f_cur, params_hist)


def train_dr_multiple_multiprocessing_aux(args):
    train_func, train_data, loss, seed, train_kwargs = args
    return train_func(train_data, loss, seed=seed, **train_kwargs)
def train_dr_multiple_multiprocessing(train_func, train_data, loss, processes, reps, pbar_file, seed, **train_kwargs):
    rng = np.random.default_rng(seed)
    
    seeds = [rng.integers(0, 10**6) for _ in range(reps)]

    results = []
    with mp.Pool(processes=processes) as pool:
      args_lst = [(train_func, train_data, loss, seed, train_kwargs) for seed in seeds]
      results_it = pool.imap(train_dr_multiple_multiprocessing_aux, args_lst)
      for res in tqdm.tqdm(results_it, total=reps, file=pbar_file):
        results.append(dict(zip(['K_fret', 'k_out', 'cost', 'raw'], res)))

    return results, seeds

def train_dr_multiple_singleprocessing(train_func, train_data, loss, reps, pbar_file, seed, **train_kwargs):
    rng = np.random.default_rng(seed)
    
    seeds = [rng.integers(0, 10**6) for _ in range(reps)]

    results = []
    for train_seed in tqdm.tqdm(seeds, file=pbar_file):
      res = train_func(train_data, loss, seed=seed, **train_kwargs)
      results.append(dict(zip(['K_fret', 'k_out', 'cost', 'raw'], res)))

    return results, seeds

def train_dr_multiple(train_func, train_data, loss, processes = None, reps = 10, pbar_file = None, seed = None, **train_kwargs):
    if pbar_file is None:  pbar_file = sys.stderr # default for tqdm

    if processes is None:
      results, seeds = train_dr_multiple_singleprocessing(train_func, train_data, loss, reps, pbar_file, seed, **train_kwargs)
    else:
      results, seeds = train_dr_multiple_multiprocessing(train_func, train_data, loss, processes, reps, pbar_file, seed, **train_kwargs)


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
          network_from_rates(trained_K_fret, trained_k_out, k_in) for trained_K_fret, trained_k_out, _, _ in training_results
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


