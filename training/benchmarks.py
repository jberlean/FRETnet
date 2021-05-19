import os, sys
import pickle

import numpy as np
from statsmodels.tsa.stattools import acf

import train_dualrail
import train_utils


def calculate_ACF(num_pixels, num_nodes, train_data, temperature, num_iters, init_K_fret, init_k_out, init_k_decay):
  seed = np.random.randint(0, 10**6)
  outputdir = f'tmp/{seed}'
  os.makedirs(outputdir)

  extra_iters = num_iters//10
  res = train_dualrail.train_dr_MCGibbs(train_data, train_utils.RMSE, anneal_protocol = [temperature]*(extra_iters + num_iters), history_output_interval = 1, seed = seed, pbar_file=sys.stderr, init_K_fret = init_K_fret, init_k_out = init_k_out, init_k_decay = init_k_decay)

  _,_,params = zip(*res['raw'])
  params_np = np.log(np.array(params))

  _, num_params = params_np.shape

  #autocorr = acf(params0, adjusted=True, nlags=len(params0), fft=True)
  autocorr = np.array([acf(params_np[extra_iters:,i], adjusted=True, nlags=num_iters//2, fft=True) for i in range(num_params)])
  autocorr_mean = autocorr.mean(axis=0)
  autocorr_std = autocorr.std(axis=0)

  output = {
    'seed': seed,
    'num_pixels': num_pixels,
    'num_nodes': num_nodes,
    'train_data': train_data,
    'temperature': temperature,
    'num_iters': num_iters,
    'init_K_fret': init_K_fret,
    'init_k_out': init_k_out,
    'init_k_decay': init_k_decay,
    'training_output': res,
    'autocorrelation': autocorr,
    'autocorrelation_mean': autocorr_mean,
    'autocorrelation_std': autocorr_std,
  }
  
  with open(os.path.join(outputdir, 'output.p'),'wb') as outfile:
    pickle.dump(output, outfile)


  import matplotlib.pyplot as plt
  plt.ion()
  plt.figure()
  #plt.plot(range(len(autocorr)), autocorr)
  plt.plot(range(len(autocorr[0])), list(zip(*autocorr)))
  plt.plot(range(len(autocorr_mean)), autocorr_mean, 'k-')
  plt.plot(range(len(autocorr_std)), autocorr_mean + autocorr_std, 'k--')
  plt.plot(range(len(autocorr_std)), autocorr_mean - autocorr_std, 'k--')
  plt.plot([0, len(autocorr_mean)], [0,0], 'k:')
  plt.xlabel('iteration')
  plt.ylabel('ACF')
  plt.title(f'T = {temperature}')
  plt.savefig(os.path.join(outputdir, f'autocorrelation_T={temperature}.pdf'))

def benchmark_ACF(inpath, results_key, temperature = 1, num_iters = 10000):
  with open(inpath, 'rb') as infile:
    data = pickle.load(infile)
    results = data[results_key]

  best_idx = np.argmin([res['cost'] for res in results])
  best_K_fret = results[best_idx]['K_fret']
  best_k_out = results[best_idx]['k_out']
  best_k_decay = results[best_idx].get('k_decay', np.zeros_like(best_k_out))

  num_nodes = len(best_k_out)
  num_pixels = num_nodes//2
  train_data = data['train_data']

  calculate_ACF(num_pixels, num_nodes, train_data, temperature, num_iters, init_K_fret = best_K_fret, init_k_out = best_k_out, init_k_decay = best_k_decay)
  
