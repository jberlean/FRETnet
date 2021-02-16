import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import warnings
from datetime import datetime
import os, sys
warnings.filterwarnings("error")

# INTRAPACKAGE IMPORTS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
from train_utils import NLL, RMSE
from train_singlerail import train

num_nodes = 5
num_patterns = 4
train_data = np.random.randint(2, size=(num_nodes, num_patterns))
DO_TRAINING = False
DO_ANALYSIS = True

# hyperparameters
outs = np.full(num_nodes, 0.5)
step_size = 1e-3
max_iters = int(100 / step_size)

if DO_TRAINING:
    K_train, err_over_time, K_over_time = train(train_data, RMSE, outs, step_size, max_iters, 
        epsilon=0.001 * step_size, noise=0, num_corrupted=1, report_freq=250)

    print (f'Training data:\n{train_data}')
    
    # plt.plot(np.arange(len(err_over_time)), err_over_time, label='Error averaged over dataset')
    # plt.legend()

    fig, axs = plt.subplots(num_nodes, num_nodes, sharex=True, sharey=True, \
        gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(f'FRET rates over time')
    for r, c in it.product(range(num_nodes), range(num_nodes)):
        axs[r,c].plot(np.arange(len(K_over_time)), K_over_time[:, r, c], label = f'K[{r},{c}]')
        axs[r,c].legend()
    
    plt.show()
    

if DO_ANALYSIS:
    total = 0
    step_size = 1e-3
    max_iters = int(100 / step_size)
    num_corrupted = 3
    max_nodes = 5
    loss = RMSE

    start_time = str(datetime.utcnow())[:19].replace(':', '-').replace(' ', '_')
    with open(f'analysis_output/converge-to-zeros_{start_time}.out', 'w') as f:
        allzero_diff, trained_allzero, searched_allzero = 0, 0, 0
        for num_nodes in range(3, max_nodes + 1, 2):
            for num_patterns in range(2, num_nodes, 2):
                    for out_val in np.linspace(0.2, 1, 3):
                        outs = np.full(num_nodes, out_val)
                        # for noise in np.linspace(0.1, 0.5, 3):
                        noise = 0.01
                        print(f'Analyzing {num_nodes} nodes, {num_patterns} patterns, outs={round(out_val, 1)}, noise={round(noise, 1)}...')
                        train_data = np.random.randint(2, size=(num_nodes, num_patterns))
                        f.write(f'Training Data:\n{train_data}\n')

                        K_train, _, _ = train(train_data, loss, np.full(num_nodes, outs), step_size, max_iters, \
                            epsilon=0.0001*step_size, noise=noise, num_corrupted=num_corrupted, report_freq=-1)
                        f.write(f'Trained weights:\n{K_train}\n')
                        trained_allzero += np.count_nonzero(K_train) == 0
                        
                        K_search = grid_search(num_nodes, train_data, outs, loss, [0, 1], \
                            resolution=3, noise=noise, num_corrupted=num_corrupted)
                        f.write(f'Searched weights:\n{K_search}\n')
                        searched_allzero += np.count_nonzero(K_search) == 0

                        f.write('\n')

                        allzero_diff += int(trained_allzero != searched_allzero)
                        total += 1

        f.write(f'{trained_allzero}/{total} training runs resulted in all zeros\n'
            f'{searched_allzero}/{total} grid searches found optimum to be all zeros\n'
            f'Training and grid searching disagreed on {allzero_diff}/{total} param sets')

        

# TODO refine grid search; effect of noise?
# TODO RECORD check that gd still converges to all zeros if greater noise (USE BOTH ERROR FNS)
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

