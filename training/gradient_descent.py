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
from training.utils.loss import NLL, RMSE
from training.utils.helpers import choose, off_patterns, sliding_window, Ainv_from_rates


def gradient(loss, pat, pred, Ainv, verbose=False):
    """
    Finds the gradient of a loss function with respect to rates in a FRETnet.

    Args:
        loss: The type of loss (usually RMSE).
        pat (np.array): The given training pattern.
            Should be a dx1 binary column vector.
        pred (np.array): The predicted output of each node.
            Should be a dx1 column of probabilities.
        Ainv (np.array): The inverse of the matrix representing the linear system.
            Should be a nxn square matrix. 
        verbose (bool): If True, returns a dict with keys:
            Ainv, pred, dL_dpred, dpred_dAinv, dAinv_dA, dA_dK, dL_dK

    Returns: 
        if verbose: A dict of all the quantities computed
        else: dL_dK (np.array): The gradient of the loss with respect to the weights of the network.
            Should be a dxd matrix, reshaped from 1xd^2.

    """
    # don't actually have to compute the NLL, just its gradient
    num_nodes = len(pat)

    dL_dpred = loss.grad(pat, pred) # 1 x d

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
    
def train(train_data, loss, output_rates, step_size, max_iters, epsilon=None, noise=0.1, num_corrupted=1, report_freq=0):
    """
    Trains a new network on given training patterns using batch gradient descent.
    Stops either when gradient step dK is less than epsilon or after max_iters iterations.

    Args:
        train_data (np.array): 2d float array where each column is a training pattern.
            Should be dxn, where d is number of nodes and n is number of patterns.
        loss (class<utils.LossType>): usually RMSE.
        ouput_rates (np.array): The intrinsic outputs of each node.
            Should be a dx1 column vector. 
        step_size (float): Multiplier for each gradient descent update.
        max_iters (int): Number of times to pass through the training data.
        epsilon (float or None): Size of gradient at which training should stop. 
            None if training should go to max_iters.
        noise (float [0,1]): Extend of deviation of training patterns 
            from the given train_data.
        num_corrupted (int): Num of corrupted patterns to generate 
            from each template pattern, per training iter. 
        report_freq (int): Rate matrix is printed every iteration multiple of this param.
            0 to only report final rates, -1 to report nothing.

    Returns:
        weights (np.array): Weights between nodes in the network. 
            Should be dxd, NONNEGATIVE and symmetric.
        err_over_time (list): Average error over training patterns,
            recorded for each iteration loop.
            Should have length n.
    """
    d, n = train_data.shape
    K = np.random.randn(d, d) / 10 + 0.5  # weights initialized normally around 0.5
    
    K = (K + K.T) / 2  # ensure starting weights are symmetric
    np.fill_diagonal(K, 0)
    err_over_time = []
    K_over_time = np.array(K).reshape((1, d, d))

    for i in range(max_iters):
        avg_err = 0
        dK = 0

        for j in range(n):  
            template = train_data[:, j:j+1]
            train_patterns = off_patterns(template, noise, num_corrupted)

            for k in range(num_corrupted):
                pat = train_patterns[:, k:k+1].astype(float) # ensure float

                # Compute the prediction based on the off patterns,
                # But evaluate error relative to original template pattern.
                Ainv = Ainv_from_rates(K, pat, output_rates)
                pred = Ainv @ pat
                dL_dK = gradient(loss, template, pred, Ainv)
                dK += dL_dK
                avg_err += loss.fn(template, pred)/(n * num_corrupted)
        
        new_K = K.copy()
        new_K -= step_size * dK
        new_K[new_K<0] = 0 # ReLU; ensures all weights nonnegative.
        err_over_time.append(avg_err)
        K_over_time = np.append(K_over_time, new_K.reshape(1, d, d), axis=0)

        # Stopping condition based on epsilon
        if epsilon is not None and np.linalg.norm(new_K - K) < epsilon and report_freq >= 0:
            print(f'Stopped at iteration {i+1}\n'
                    f'K: {new_K}\n'
                    f'error: {avg_err}\n'
                    f'Change in loss for last iter: {np.linalg.norm(new_K - K)}')
            return new_K, err_over_time, K_over_time
    
        if report_freq > 0 and (i) % report_freq == 0:
            print(f'Rates on iteration {i}: \n{new_K}')

        K = new_K

    return K, err_over_time, K_over_time

def grid_search(num_nodes, pats, outs, loss, k_domain=(0,1), resolution=3, noise=0.1, num_corrupted=1):
    """
    Grid-searches FRET rates with given intrinsic output rates, noise amt for the configuration with the lowest loss.

    Args:
        num_nodes (int)
        pats (np.array): a column array, template of num_nodes bits.
        outs (np.array): a column array, the intrinsic outputs for each node.
        loss (class<training.utils.LossType>)
        k_domain (2-tuple): the domain of rates to search in.
        resolution: number of points to sample in the domain. 
            ex: k_domain=(0,1), resolution=3 -> [0, 0.5, 1]
        noise: probability of each input bit being corrupted during evaluation.
        num_corrupted: The number of corrupted patterns to generate from each true pattern.

    Returns:
        K_min (np.array): the rates producing the lowest loss.
    """
    triu_idx = np.triu_indices(num_nodes, 1)
    num_cols = pats.shape[1]

    def f(params):      # The function to be brute-force optimized
        K = np.zeros((num_nodes, num_nodes), dtype=float)
        K[triu_idx] = params
        K += K.T

        loss_sum = 0
        for c in range(num_cols):
            template = pats[:, c:c+1]
            off_pats = off_patterns(template, noise, num_corrupted)
            for i in range(num_corrupted):
                off_pat = off_pats[:, i:i+1]
                Ainv = Ainv_from_rates(K, off_pat, outs)
                pred = Ainv @ off_pat
                loss_sum += loss.fn(template, pred)
        return loss_sum
    
    k_ranges = [tuple(k_domain) for _ in range(choose(num_nodes, 2))]

    resbrute = brute(f, k_ranges, Ns=resolution, finish=None)
    K_min = np.zeros((num_nodes, num_nodes), dtype=float)
    K_min[triu_idx] = resbrute
    K_min += K_min.T
    return K_min

if __name__ == '__main__':

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