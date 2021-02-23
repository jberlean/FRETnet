import os, sys
import itertools as it
import pickle
import math
import multiprocessing as mp

import numpy as np
from scipy.optimize import brute
import scipy.special
from statsmodels.tsa.stattools import acf
import tqdm

import train_dualrail
import train_utils


## Command-line arguments

train_mode = int(sys.argv[1]) # 1 = MC, 2 = GD, 3 = MC+GD
train_MC = bool(train_mode&1)
train_GD = bool(train_mode&2)
train_str = f'{"MC" if train_MC else ""}{"GD" if train_GD else ""}' # for file paths

if len(sys.argv) >=3:  seed = int(sys.argv[2])
else:  seed = np.random.randint(0, 10**6)
rng = np.random.default_rng(seed)


## Training parameters

# Parameters for generating stored memories
num_nodes = 4
num_patterns = 3

# Parameters for generating training data
noise = 0.1
duplication = 20

# Training parameters
reps = 1
#train_kwargs_MC = dict(low_bound = 1e-10, high_bound = 1e5, anneal_protocol = None, goal_accept_rate = 0.3, init_noise = 2)
train_kwargs_MC = dict(low_bound = 1e-10, high_bound = 1e5, anneal_protocol = [.01]*100000, goal_accept_rate = 0.3, init_noise = 2, verbose=True)
train_kwargs_GD = {}

# Output parameters
#outpath = f'/scratch/jberlean/tmp/train{train_str}_{reps}x_seed={seed}.p'
outputdir = f'tmp/{seed}'
pbarpath = os.path.join(outputdir, f'pbar{train_str}_seed={seed}')
outpath = os.path.join(outputdir, f'train{train_str}_{reps}x_seed={seed}.p')

os.makedirs(os.path.dirname(outpath), exist_ok=True)


## Run code

# Generate stored memories
stored_data_ints = rng.permutation(2**num_nodes)[:num_patterns]
stored_data = [
    np.array([int(v)*2-1 for v in format(i,'0{}b'.format(num_nodes))])
    for i in stored_data_ints
]

print('Stored data:')
for d in stored_data:
  print(list(d))

# Generate training data
train_data = train_dualrail.generate_training_data(stored_data, noise = noise, duplication=duplication, rng = rng)

# Perform training
train_seed_base = rng.integers(0, 10**6)
pbar_file = open(pbarpath, 'w')

if train_MC:
  results_MC, train_seeds_MC = train_dualrail.train_dr_multiple(train_dualrail.train_dr_MC, train_data, train_utils.RMSE, processes = 3, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_MC)
if train_GD:
  results_GD, train_seeds_GD = train_dualrail.train_dr_multiple(train_dualrail.train_dr, train_data, train_utils.RMSE, processes = 3, seed = train_seed_base, pbar_file=pbar_file, reps=reps, **train_kwargs_GD)

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
 
with open(outpath,'wb') as outfile:
  pickle.dump(output, outfile)


# temp plotting code
_,_,params = zip(*results_MC[0]['raw'])
params_np = np.log(np.array(params))
#autocorr = acf(params0, adjusted=True, nlags=len(params0), fft=True)
autocorr = np.array([acf(params_np[1000:,i], adjusted=True, nlags=len(params)//2, fft=True) for i in range(len(params[0]))])
autocorr_mean = autocorr.mean(axis=0)
import matplotlib.pyplot as plt
plt.ion()
#plt.plot(range(len(autocorr)), autocorr)
plt.plot(range(len(autocorr[0])), list(zip(*autocorr)))
plt.plot(range(len(autocorr_mean)), autocorr_mean, 'k-')
plt.plot([0, len(autocorr_mean)], [0,0], 'k--')
plt.xlabel('iteration')
plt.ylabel('ACF')
plt.savefig(os.path.join(outputdir, 'autocorrelation.pdf'))
