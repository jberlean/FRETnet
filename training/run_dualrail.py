import os, sys
import itertools as it
import pickle
import math
import multiprocessing as mp
import importlib.util

import numpy as np
from scipy.optimize import brute
import scipy.special
import tqdm

import train_dualrail
import train_utils


## Command-line arguments

train_mode = sys.argv[1] # a bitstring ABC; A = MC, B = GD, C = MC-Gibbs
train_MC = train_mode[0]=='1'
train_GD = train_mode[1]=='1'
train_MG = train_mode[2]=='1'
train_str = f'{"MC" if train_MC else ""}{"GD" if train_GD else ""}{"MG" if train_MG else ""}' # for file paths

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
num_nodes = user_args.get('num_nodes', 4)
num_patterns = user_args.get('num_patterns', 3)

# Parameters for generating training data
noise = user_args.get('train_data_noise', 0.1)
duplication = user_args.get('train_data_duplication', 20)

# Training parameters
reps = user_args.get('reps', 1)

train_kwargs_MC = dict(low_bound = 1e-2, high_bound = 1e4, input_magnitude = 1, output_magnitude = None, k_out_value = 100, anneal_protocol = None, goal_accept_rate = 0.3, init_noise = 2, verbose = False)
train_kwargs_GD = dict(init = 'random', input_magnitude = 1, output_magnitude = None, k_out_value = 100)
train_kwargs_MG = dict(low_bound = 1e-2, high_bound = 1e4, input_magnitude = 1, output_magnitude = None, k_out_value = 100, anneal_protocol = None, goal_accept_rate = 0.44, init_step_size = 2, verbose = False)

train_kwargs_MC.update(user_args.get('train_kwargs_MC', {}))
train_kwargs_GD.update(user_args.get('train_kwargs_GD', {}))
train_kwargs_MG.update(user_args.get('train_kwargs_MG', {}))

processes = user_args.get('processes', 1)

# Output parameters
#outputdir = user_args.get('outputdir', f'/scratch/jberlean/tmp/{seed}/train{train_str}_{reps}x_seed={seed}.p')
outputdir = f'tmp/{seed}'
pbarpath = os.path.join(outputdir, f'pbar{train_str}_seed={seed}')
outpath = os.path.join(outputdir, f'train{train_str}_{reps}x_seed={seed}.p')

os.makedirs(os.path.dirname(outpath), exist_ok=True)

print(f'Output directory: {outputdir}')


## Run code

# Generate stored memories
if stored_data_user is None:
  stored_data_ints = rng.permutation(2**num_nodes)[:num_patterns]
  stored_data = [
      np.array([int(v)*2-1 for v in format(i,'0{}b'.format(num_nodes))])
      for i in stored_data_ints
  ]
else:
  stored_data_ints = rng.permutation(2**num_nodes)[:num_patterns] # to make sure RNG is in the same state
  stored_data = list(map(np.array, stored_data_user))

print('Stored data:')
for d in stored_data:
  print(list(d))

# Generate training data
train_data = train_dualrail.generate_training_data(stored_data, noise = noise, duplication=duplication, rng = rng)

# Perform training
train_seed_base = rng.integers(0, 10**6)
pbar_file = open(pbarpath, 'w')

if train_MC:
  results_MC, train_seeds_MC = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MC, train_data, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MC)
if train_GD:
  results_GD, train_seeds_GD = train_dualrail.train_dr_multiple(train_dualrail.train_dr, train_data, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_GD)
if train_MG:
  results_MG, train_seeds_MG = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MCGibbs, train_data, train_utils.RMSE, processes = processes, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MG)
  

pbar_file.close()
 
# Output results
output = {
  'seed': seed,
  'num_nodes': num_nodes,
  'num_patterns': num_patterns,
  'stored_data': stored_data,
  'train_data': train_data,
  'train_data_noise': noise,
  'train_data_duplication': duplication,
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
 
with open(outpath,'wb') as outfile:
  pickle.dump(output, outfile)


