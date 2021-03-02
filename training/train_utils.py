import sys, os
import itertools as it

import numpy as np
import scipy.special


## Intrapackage imports

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_dir not in sys.path:
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
import objects.utils as object_utils


def choose(a, b):
    """
    Calculate binomial coefficients
    """
    return scipy.special.comb(a, b, exact=True)

def sliding_window(iterable, k):
    iters = it.tee(iterable, k)
    for num_skips, iter in enumerate(iters):
        for _ in range(num_skips):
            next(iter, None)
    
    return zip(*iters)

def off_patterns(pat, p_off, num_pats, rng = None):
    """
    Returns num_pat patterns, each generated by duplicating pat
    and randomly flipping each value with probability p_off.

    Args:
        pat (np.array): Binary vector of length d. Each element is -1 or +1
        p_off (float in [0,1]): Probability of each node being corrupted.
        num_pats (int): Number of corrupted patterns to return.
        rng (np.random.Generator): [optional] the random number generator to use.
            If not specified, will use np.random.default_rng()

    Returns:
        len(pat) x num_pats 2darray (dtype=float) with corrupted patterns as columns.
    """

    if rng is None:
        rng = np.random.default_rng()

    d = len(pat)

    corrupt_mask = rng.random((num_pats, d))
    out = np.tile(pat, (num_pats,1))

    out[corrupt_mask < p_off] = -out[corrupt_mask < p_off]
    return [out[i,:] for i in range(num_pats)]

def A_from_rates(K_fret, k_in, k_out):
    num_nodes = len(k_in)
    # 0.0048s

    A = -K_fret
    # 0.0088s

    A[np.diag_indices(num_nodes)] = K_fret.sum(axis=1) + k_in + k_out
    # 0.030s (161/1500 - 1/12.9)

    return A

def Ainv_from_rates(K_fret, k_in, k_out):
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
    A = A_from_rates(K_fret, k_in, k_out)

    try:
        ret = np.linalg.inv(A)
        return ret
    except:
        raise ValueError(f'Singular matrix for rates={K_fret}, inputs={k_in}, outputs={k_out}')

def k_in_from_input_data(input_data): # 0.018s/it (98/1000 - 80/1000)
    k_in_sr = np.empty(len(input_data)*2)
    k_in_sr[0::2] = np.maximum(input_data, 0)
    k_in_sr[1::2] = -np.minimum(input_data, 0)
    return k_in_sr

def network_from_rates(K_fret, k_out, k_in):
    num_nodes = len(k_out)
    nodes = [object_utils.InputNode('node{}'.format(i), production_rate=k_in_i, emit_rate=k_out_i) for i,(k_out_i,k_in_i) in enumerate(zip(k_out, k_in))]
    for i,j in it.product(range(num_nodes), range(num_nodes)):
        if i==j:  continue
        nodes[j].add_input(nodes[j], K_fret[i,j])

    return object_utils.Network(nodes)




class LossFunc():
    def fn(pat, pred):
        pass
    def grad(pat, pred):
        pass
    

class RMSE(LossFunc):
    @staticmethod
    def fn(pat, pred):
        return np.mean((pat-pred)**2) ** 0.5
    @staticmethod
    def grad(pat, pred):
        return np.zeros(pat.shape).T if (pred == pat).all() else ( np.mean((pat-pred)**2)**-0.5 * 1/len(pat) * (pred-pat) ).T
        
class NLL(LossFunc):
    @staticmethod
    def fn(pat, pred):
        total = 0
        for i in range(pat.size):
            if pat[i,0] == 1:
                total -= np.log(pred[i,0]) 
            elif pat[i,0] == 0:
                total -= np.log(1-pred[i,0])
            else: 
                raise ValueError()
        return total
    @staticmethod
    def grad(pat, pred):
        out = np.zeros(pat.T.shape)
        for i in range(pat.size):
            if pat[i,0] == 1:
                out[0,i] = -1/pred[i,0]
            elif pat[i,0] == 0:
                out[0,i] = 1/(1-pred[i,0])
            else:
                raise ValueError()
        return out
