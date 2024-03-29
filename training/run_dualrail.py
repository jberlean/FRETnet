import os, sys, pathlib
import itertools as it
import pickle
import math
import multiprocessing as mp
import importlib.util

import numpy as np
from scipy.optimize import brute
import scipy.special
import tqdm

# INTRAPACKAGE IMPORTS
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.append(pkg_path)

import train_dualrail
import train_utils
from objects import IO


## Command-line arguments

train_mode = sys.argv[1] # a bitstring ABC; A = MC, B = GD, C = MC-Gibbs
train_MC = train_mode[0]=='1'
train_GD = train_mode[1]=='1'
train_MG = train_mode[2]=='1'
train_MGp = train_mode[3:4] == '1'
train_MGpf = train_mode[4:5] == '1'
train_MGpf2 = train_mode[5:6] == '1'
train_str = f'{"MC" if train_MC else ""}{"GD" if train_GD else ""}{"MG" if train_MG else ""}{"MGp" if train_MGp else ""}{"MGpf" if train_MGpf else ""}{"MGpf2" if train_MGpf2 else ""}' # for file paths

if len(sys.argv) >= 3: 
  user_args_path = sys.argv[2]
  user_args_spec = importlib.util.spec_from_file_location('user_args', user_args_path)
  user_args = importlib.util.module_from_spec(user_args_spec)
  user_args_spec.loader.exec_module(user_args)
  user_args = user_args.__dict__
else:
  user_args = {}


## Training parameters

# Random seed for training
seed = user_args.get('seed', np.random.randint(0, 10**6))
rng = np.random.default_rng(seed)

# Parameters for generating stored memories
stored_data_user = user_args.get('stored_data', None)
num_nodes_dualrail = user_args.get('num_nodes', 4)
num_patterns = user_args.get('num_patterns', 3)

# Parameters for generating training data
noise = user_args.get('train_data_noise', 0.1)
duplication = user_args.get('train_data_duplication', 20)
corruption_mode = user_args.get('train_data_corruption_mode', 'flip')
train_data_user = user_args.get('train_data', None)
train_data_weights = user_args.get('train_data_weights', None)

# System parameters
input_fluor_info = {
  'k_0': 1,
  'r_0': {
    'compute': 2,
  },
}
compute_fluor_info = {
  'k_0': 1,
  'r_0': {
    'compute': 7,
    'output': 2,
    'quencher': 2,
  },
}
input_fluor_info.update(user_args.get('input_fluor_info', {}))
compute_fluor_info.update(user_args.get('compute_fluor_info', {}))

# Training parameters
reps = user_args.get('reps', 1)

train_kwargs_MC = dict(
    low_bound = 1e-2,
    high_bound = 1e4,
    input_magnitude = 1,
    output_magnitude = None,
    k_out_value = 100,
    anneal_protocol = None,
    goal_accept_rate = 0.3,
    init_noise = 2,
    verbose = False
)
train_kwargs_GD = dict(
    init = 'random',
    input_magnitude = 1,
    output_magnitude = None,
    k_out_value = 100
)
train_kwargs_MG = dict(
    k_fret_bounds = (1e-2, 1e4),
    k_decay_bounds = (1, 1e4),
    input_magnitude = 1,
    output_magnitude = None,
    k_out_value = 100,
    anneal_protocol = list(np.logspace(0, -5, 5000)),
    accept_rate_min = 0.4,
    accept_rate_max = 0.6,
    init_step_size = 2,
    verbose = False
)
train_kwargs_MGp = dict(
    k_0 = 1,
    r_0_cc = 7,
    position_bounds = (-1e2, 1e2),
    min_dist = 1,
    dims = 3,
    input_magnitude = 1,
    output_magnitude = None,
    k_out_value = 100,
    anneal_protocol = list(np.logspace(0, -5, 5000)),
    accept_rate_min = 0.4,
    accept_rate_max = 0.6,
    init_step_size = 20,
    verbose = False
)
train_kwargs_MGpf = dict(
    input_fluor_info = input_fluor_info, 
    compute_fluor_info = compute_fluor_info, 
    position_bounds = (-1e2, 1e2), 
    min_dist = 1, 
    max_dist_CI = 6,
    max_dist_CO = 6,
    max_dist_CQ = 8,
    dims = 3, 
    input_magnitude = 100, 
    output_magnitude = 1, 
    anneal_protocol = list(np.logspace(0, -5, 5000)), 
    accept_rate_min = 0.4, 
    accept_rate_max = 0.6, 
    init_step_size = 20, 
    verbose = False
)
train_kwargs_MGpf2 = dict(
    train_kwargs_MG = train_kwargs_MG,
    train_kwargs_MGpf = train_kwargs_MGpf,
    dims = 3,
    cluster_threshold = 1e-2,
    cluster_spacing = 20,
    verbose = False
)

train_kwargs_MC.update(user_args.get('train_kwargs_MC', {}))
train_kwargs_GD.update(user_args.get('train_kwargs_GD', {}))
train_kwargs_MG.update(user_args.get('train_kwargs_MG', {}))
train_kwargs_MGp.update(user_args.get('train_kwargs_MGp', {}))
train_kwargs_MGpf.update(user_args.get('train_kwargs_MGpf', {}))
train_kwargs_MGpf2.update(user_args.get('train_kwargs_MGpf2', {}))
# messy :(
train_kwargs_MGpf2['train_kwargs_MG'] = {**train_kwargs_MG, **train_kwargs_MGpf2['train_kwargs_MG']}
train_kwargs_MGpf2['train_kwargs_MGpf'] = {**train_kwargs_MGpf, **train_kwargs_MGpf2['train_kwargs_MGpf']}

processes = user_args.get('processes', None)

# Output parameters
outputdir = user_args.get('outputdir', f'tmp/{seed}/')
pbarpath = os.path.join(outputdir, f'pbar{train_str}_seed={seed}')
outfile_prefix = user_args.get('outfile_prefix', f'train{train_str}_{reps}x_seed={seed}')
outpath_prefix = os.path.join(outputdir, outfile_prefix)
outpath_full = f'{outpath_prefix}.p'
outpath_best = f'{outpath_prefix}_best.p'
outpath_best_excel = f'{outpath_prefix}_best.xlsx'
outpath_best_mol2 = f'{outpath_prefix}_best.mol2'

os.makedirs(os.path.dirname(outpath_prefix), exist_ok=True)

print(f'Output directory: {outputdir}')


## Run code

# Generate stored memories
if stored_data_user is None:
  stored_data_ints = rng.permutation(2**num_nodes_dualrail)[:num_patterns]
  stored_data = [
      np.array([int(v)*2-1 for v in format(i,'0{}b'.format(num_nodes_dualrail))])
      for i in stored_data_ints
  ]
else:
  stored_data_ints = rng.permutation(2**num_nodes_dualrail)[:num_patterns] # to make sure RNG is in the same state
  stored_data = list(map(np.array, stored_data_user))

print('Stored data:')
for d in stored_data:
  print(list(d))

# Generate training data
if train_data_user is None:
  train_data = train_dualrail.generate_training_data(stored_data, noise = noise, duplication=duplication, mode = corruption_mode, filter_zero = True, rng = rng)
  train_data_lst = [(tuple(d_in), tuple(d_out)) for d_in,d_out in train_data]
  train_data = list(set(train_data_lst))
  train_data_weights = [train_data_lst.count(d) for d in train_data]
else:
  train_dualrail.generate_training_data(stored_data, noise = noise, duplication=duplication, mode = corruption_mode, filter_zero = True, rng = rng) # to make sure RNG is in the same state regardless of whether training data is user-specified
  train_data = [
      (np.array(input_data), np.array(output_data))
      for input_data, output_data in train_data_user
  ]

print('Training data:')
for d_in, d_out in train_data:
  print(f'  {d_in}\t{d_out}')
  

# Perform training
train_seed_base = rng.integers(0, 10**6)
pbar_file = open(pbarpath, 'w')

if train_MC:
  results_MC, train_seeds_MC = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MC, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MC)
if train_GD:
  results_GD, train_seeds_GD = train_dualrail.train_dr_multiple(train_dualrail.train_dr, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_GD)
if train_MG:
  results_MG, train_seeds_MG = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MCGibbs, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MG)
if train_MGp:
  results_MGp, train_seeds_MGp = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MCGibbs_positions, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MGp)
if train_MGpf:
  results_MGpf, train_seeds_MGpf = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MCGibbs_positions_full, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MGpf)
if train_MGpf2:
  results_MGpf2, train_seeds_MGpf2 = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MCGibbs_positions_full_2step, train_data, train_data_weights, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MGpf2)
  
  

pbar_file.close()
 
# Output comprehensive results
output = {
  'seed': seed,
  'num_nodes': num_nodes_dualrail,
  'num_patterns': num_patterns,
  'stored_data': stored_data,
  'train_data': train_data,
  'train_data_noise': noise,
  'train_data_duplication': duplication,
  'train_data_corruption_mode': corruption_mode,
}
if train_MC:
  output['train_args_MC'] = train_kwargs_MC
  output['train_seeds_MC'] = train_seeds_MC
  output['results_MC'] = results_MC
if train_GD:
  output['train_args_GD'] = train_kwargs_GD
  output['train_seeds_GD'] = train_seeds_GD
  output['results_GD'] = results_GD
if train_MG:
  output['train_args_MG'] = train_kwargs_MG
  output['train_seeds_MG'] = train_seeds_MG
  output['results_MG'] = results_MG
if train_MGp:
  output['train_args_MGp'] = train_kwargs_MGp
  output['train_seeds_MGp'] = train_seeds_MGp
  output['results_MGp'] = results_MGp
if train_MGpf:
  output['train_args_MGpf'] = train_kwargs_MGpf
  output['train_seeds_MGpf'] = train_seeds_MGpf
  output['results_MGpf'] = results_MGpf
if train_MGpf2:
  output['train_args_MGpf2'] = train_kwargs_MGpf2
  output['train_seeds_MGpf2'] = train_seeds_MGpf2
  output['results_MGpf2'] = results_MGpf2
 
with open(outpath_full,'wb') as outfile:
  pickle.dump(output, outfile)

# Output results for best network, and XLSX/MOL2 representations of this network
best_results_key, best_idx, best_cost = None, None, np.inf
for results_key in ['results_MC', 'results_GD', 'results_MG', 'results_MGp', 'results_MGpf', 'results_MGpf2']:
  if results_key not in output:  continue
  idx = np.argmin([res['cost'] for res in output[results_key]])
  cost = output[results_key][idx]['cost']
  if cost < best_cost:
    best_results_key = results_key
    best_idx = idx
    best_cost = cost

best_method = best_results_key[len('results_'):]
best_args = output[f'train_args_{best_method}']
best_result = output[best_results_key][best_idx]

input_magnitude = best_args.get('input_magnitude', 1)
k_out_value = best_args.get('k_out_value', 100)
output_magnitude = best_args.get('output_magnitude', None)
if output_magnitude is None:  output_magnitude = input_magnitude * k_out_value / (input_magnitude + k_out_value)

K_fret = best_result['K_fret']
k_out = best_result['k_out']
k_in = best_result.get('k_in', input_magnitude * np.ones_like(k_out))
k_decay = best_result.get('k_decay', np.zeros_like(k_out))
K_out = best_result.get('K_out', np.diag(k_out))
K_in = best_result.get('K_in', np.diag(k_in))
K_quench = best_result.get('K_quench', np.zeros_like(K_out))

num_nodes_dr = best_result.get('num_nodes_dr', num_nodes_dualrail)
node_names_dr = best_result.get('node_names_dr', list(map(str, range(1, num_nodes_dr+1))))

num_nodes_sr = best_result.get('num_nodes_sr', 2*num_nodes_dr)
node_names_sr = best_result.get('node_names_sr', [f'{n_dr}{pm}' for n_dr in node_names_dr for pm in ['+','-']])
dr_to_sr_map = best_result.get('dr_to_sr_map', None)

num_fluors = best_result.get('num_fluorophores', num_nodes_sr)
fluor_names = best_result.get('fluorophore_names', node_names_sr)
fluor_types = best_result.get('fluorophore_types', ['C']*num_fluors)
sr_to_fluor_map = best_result.get('sr_to_fluor_map', None)
positions = best_result.get('positions', None)

output_best = {
  'index': best_idx,
  'cost': best_cost,

  'num_nodes_dualrail': num_nodes_dr,
  'node_names_dualrail': node_names_dr,

  'num_nodes_singlerail': num_nodes_sr,
  'node_names_singlerail': node_names_sr,
  'dualrail_to_singlerail_map': dr_to_sr_map,

  'num_fluorophores': num_fluors,
  'fluorophore_names': fluor_names,
  'fluorophore_types': fluor_types,
  'singlerail_to_fluorophore_map': sr_to_fluor_map,

  'positions': positions,

  'input_magnitude': input_magnitude,
  'output_magnitude': output_magnitude,

  'K_fret': K_fret,
  'k_in': k_in,
  'k_out': k_out,
  'k_decay': k_decay,
  'K_in': K_in,
  'K_out': K_out,
  'K_quench': K_quench,

  'args': best_args,
  'method': best_method,
  'training_metadata': {k:output[k] for k in output if not k.startswith('results')}
}

with open(outpath_best, 'wb') as outfile:
  pickle.dump(output_best, outfile)

if positions is not None:
  positions_lst = [positions[i,:] for i in range(num_fluors)]

  if num_fluors == num_nodes_sr:
    IO.output_network_excel(outpath_best_excel, node_names_sr, K_fret, positions_lst)
  else:
    zeros = np.zeros((num_nodes_sr, num_nodes_sr))
    K_fret_all = np.block([[zeros, K_in, zeros, zeros], [zeros, K_fret, K_out, K_quench], [zeros]*4, [zeros]*4])
    IO.output_network_excel(outpath_best_excel, fluor_names, K_fret_all, positions_lst)
  
  mol2_comments = [
      f'# Source: {outpath_best}',
      f'# Created by: Joseph Berleant',
  ]
  IO.output_network_mol2(outpath_best_mol2, fluor_names, positions_lst, fluor_types, outpath_prefix, mol2_comments)
