import sys, os
import itertools as it

import numpy as np
import scipy.special
import sklearn.manifold


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

def off_patterns(pat, p_off, num_pats, mode = 'flip', rng = None):
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

    if mode == 'flip':
      out[corrupt_mask < p_off] = -out[corrupt_mask < p_off]
    elif mode == 'erase':
      out[corrupt_mask < p_off] = 0
    else: # use default "flip" behavior
      print(f'WARNING: Unknown bit corruption mode {mode}')
      out[corrupt_mask < p_off] = -out[corrupt_mask < p_off]
  
    return [out[i,:] for i in range(num_pats)]

def A_from_rates(K_fret, k_in, k_off):
    num_nodes = len(k_in)
    # 0.0048s

    A = -K_fret
    # 0.0088s

    A[np.diag_indices(num_nodes)] = K_fret.sum(axis=1) + k_in + k_off
    # 0.030s (161/1500 - 1/12.9)

    return A

def Ainv_from_rates(K_fret, k_in, k_off):
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
    A = A_from_rates(K_fret, k_in, k_off)

    try:
        ret = np.linalg.inv(A)
        return ret
    except:
        raise ValueError(f'Singular matrix for rates={K_fret}, inputs={k_in}, outputs={k_off}')

def k_in_from_input_data(input_data): # 0.018s/it (98/1000 - 80/1000)
    k_in_sr = np.empty(len(input_data)*2)
    k_in_sr[0::2] = np.maximum(input_data, 0)
    k_in_sr[1::2] = -np.minimum(input_data, 0)
    return k_in_sr

def network_from_rates(K_fret, k_out, k_in, k_decay = None):
    num_nodes = len(k_out)

    if k_decay is None:  k_decay = np.zeros(num_nodes)

    nodes = [object_utils.InputNode('node{}'.format(i), production_rate=k_in_i, emit_rate=k_out_i, decay_rate=k_decay_i) for i,(k_out_i,k_in_i, k_decay_i) in enumerate(zip(k_out, k_in, k_decay))]
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


def rate_to_distance(k_fret, k_0 = 1, r_0 = 1):
  return (k_0 / k_fret) ** (1/6.) * r_0

def rate_from_positions(pos1, pos2, k_0, r_0):
  return k_0 * (r_0 / np.linalg.norm(pos1-pos2))**6

def rates_to_positions(K_fret, k_off = None, k_0 = 1, r_0 = 1, max_k = 1e3, max_dist = 20, dims=3):
  num_nodes = K_fret.shape[0]

  if k_off is None:
    k_off = np.ones(num_nodes)

  dists = rate_to_distance(K_fret + np.eye(num_nodes), k_0, r_0)
  dists[np.diag_indices(num_nodes)] = 0.
  dists[dists > max_dist] = max_dist

  print(dists)
  
  # get a rough guess using MDS
  mds_embedding = sklearn.manifold.MDS(n_components = dims, dissimilarity = 'precomputed').fit_transform(dists)
  embedding = mds_embedding - mds_embedding.mean(axis=0)
  mds_dists = np.array([[np.linalg.norm(mds_embedding[i,:] - mds_embedding[j,:]) for j in range(num_nodes)] for i in range(num_nodes)])

  print(mds_dists)

#  # perform additional optimization
#  params_ideal = np.array([K_fret[i,j] / (K_fret[i,:].sum() + k_off[i]) for i,j in it.permutations(range(num_nodes),2)])
#  def func(embedding):
#    embedding = embedding.reshape((-1, dims))
#    embedding_dists = np.array([[np.linalg.norm(embedding[i,:] - embedding[j,:]) for j in range(num_nodes)] for i in range(num_nodes)])
#    embedding_rates = k_0 * (r_0 / (embedding_dists + np.eye(num_nodes)))**6
#    embedding_rates[np.diag_indices(num_nodes)] = 0.
#    params = np.array([embedding_rates[i,j] / (embedding_rates[i,:].sum() + k_off[i]) for i,j in it.permutations(range(num_nodes),2)])
#    bounds_penalty = np.sum(1. / (1 + (max_k/embedding_rates[embedding_rates>max_k*.99])**1000)) # log-sigmoid
#    return np.sum((params - params_ideal)**2) + bounds_penalty
#  scipy_res = scipy.optimize.minimize(func, mds_embedding.flatten())
#
#  embedding = scipy_res.x.reshape((-1,dims))
#  embedding -= embedding.mean(axis=0)
#  embedding_dists = np.array([[np.linalg.norm(embedding[i,:] - embedding[j,:]) for j in range(num_nodes)] for i in range(num_nodes)])
#  embedding_rates = k_0 * (r_0 / (embedding_dists + np.eye(num_nodes)))**6
#  embedding_rates[np.diag_indices(num_nodes)] = 0
#  params = np.array([embedding_rates[i,j] / (embedding_rates[i,:].sum() + k_off[i]) for i,j in it.permutations(range(num_nodes),2)])

#  import matplotlib.pyplot as plt
#  plt.ion()
#  plt.figure()
##  plt.scatter(dists.flatten(), embedding_dists.flatten(), color='k')
#  plt.scatter(params_ideal, params, color='k')
#  plt.plot([0, max(params_ideal)], [0, max(params_ideal)], 'k--')
#  plt.axis('square')
#
#  from mpl_toolkits import mplot3d
#  plt.figure()
#  ax = plt.axes(projection='3d')
#  colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
#  ax.scatter3D(embedding[:,0], embedding[:,1], embedding[:,2], color=colors[:num_nodes], alpha=1)
#
  return embedding

def random_point_on_sphere(radius, center = None, dims=3, rng=None):
  if center is None:  center = [0]*dims

  if rng is None:
    pts_gauss = np.random.normal(0, 1, dims)
  else:
    pts_gauss = rng.normal(0, 1, dims)

  pts_norm = pts_gauss / np.linalg.norm(pts_gauss) * radius
  return pts_norm + center

def rates_to_positions_full(K_fret, k_in = None, k_out = None, k_decay = None, k_0 = 1, R_0 = None, max_k = 1e3, max_dist=1e3, dims=3, node_names = None):
  num_nodes = K_fret.shape[0]
  num_fluors = 4*num_nodes

  if node_names is None:
    node_names = [f'{v}{pm}' for v in range(num_nodes//2) for pm in '+-']
  fluor_names = [f'{n}{suffix}' for suffix in ['_IN','','_OUT','_QUENCH'] for n in node_names]

  if k_in is None: # rate constant for transfer from input fluor to compute fluor
    k_in = np.ones(num_nodes)
  if k_out is None: # rate constant for transfer to output fluor
    k_out = np.ones(num_nodes)
  if k_decay is None: # rate constant for transfer to quencher
    k_decay = np.ones(num_nodes)
  k_off = k_0 + k_out + k_decay # rate constant for transition from high to low

  if R_0 is None: # matrix of Forster radii between input, compute, and output fluors and quenchers
    R_0 = np.ones((4,4))

  f_in_start, f_in_end = 0, num_nodes
  f_comp_start, f_comp_end = num_nodes, 2*num_nodes
  f_out_start, f_out_end = 2*num_nodes, 3*num_nodes
  quench_start, quench_end = 3*num_nodes, 4*num_nodes
  idx_lst = [(f_in_start, f_in_end), (f_comp_start, f_comp_end), (f_out_start, f_out_end), (quench_start, quench_end)]

  K = np.zeros((num_fluors, num_fluors))
  K[f_in_start:f_in_end, f_comp_start:f_comp_end] = np.diag(k_in)
  K[f_comp_start:f_comp_end, f_comp_start:f_comp_end] = K_fret
  K[f_comp_start:f_comp_end, f_out_start:f_out_end] = np.diag(k_out)
  K[f_comp_start:f_comp_end, quench_start:quench_end] = np.diag(k_decay)

  R_0_expanded = np.repeat(np.repeat(R_0, num_nodes, axis=0), num_nodes, axis=1)

  K_0 = np.ones((num_fluors, num_fluors))
  K_0[f_comp_start:f_comp_end, :] = k_0

  weights = np.array([[0,1,0,0], [0,1,1,1], [0,0,0,0], [0,0,0,0]])

  dists = np.zeros((num_fluors, num_fluors))
  for i,j in it.combinations(range(num_fluors), r=2):
    k, k_0, r_0 = K[i,j], K_0[i,j], R_0_expanded[i,j]
    if r_0 == 0 or k == 0:
      dists[i,j] = 0
    else:  
      dists[i,j] = rate_to_distance(k, k_0, r_0)
  dists += dists.T
  
  # get a rough guess using the basic optimizer for compute fluors and randomly placing other fluors
  init_embedding_comp = rates_to_positions(K_fret, k_off = k_off, k_0 = k_0, r_0 = R_0[1,1], max_k = max_k, dims=dims)
  init_embedding = np.zeros((num_fluors, dims))
  init_embedding[f_in_start:f_in_end, :] = [random_point_on_sphere(rate_to_distance(k_in[i], k_0, R_0[0,1]), init_embedding_comp[i,:], dims=dims) for i in range(num_nodes)]
  init_embedding[f_comp_start:f_comp_end, :] = init_embedding_comp
  init_embedding[f_out_start:f_out_end, :] = [random_point_on_sphere(rate_to_distance(k_out[i], k_0, R_0[1,2]), init_embedding_comp[i,:], dims=dims) for i in range(num_nodes)]
  init_embedding[quench_start:quench_end, :] = [random_point_on_sphere(rate_to_distance(k_decay[i], k_0, R_0[1,3]), init_embedding_comp[i,:], dims=dims) for i in range(num_nodes)]
  init_dists = np.array([[np.linalg.norm(init_embedding[i,:] - init_embedding[j,:]) for j in range(num_fluors)] for i in range(num_fluors)])

  # perform additional optimization
#  params_ideal = np.array([K_fret[i,j] / (K_fret[i,:].sum() + k_off[i]) for i,j in it.permutations(range(num_nodes),2)])
  params_ideal = K / (k_0 + np.tile(K.sum(axis=1), (num_fluors,1)).T)
  def func(embedding):
    embedding = embedding.reshape((-1, dims))
    embedding_dists = np.array([[np.linalg.norm(embedding[i,:] - embedding[j,:]) for j in range(num_fluors)] for i in range(num_fluors)])
    embedding_K = K_0 * (R_0_expanded / (embedding_dists + np.eye(num_fluors)))**6
    embedding_K[np.diag_indices(num_fluors)] = 0.
    resids = embedding_K / (k_0 + np.tile(embedding_K.sum(axis=1), (num_fluors,1)).T) - params_ideal
    scores = np.array([[(resids[i1:i2,j1:j2]**2).sum() for j1,j2 in idx_lst] for i1,i2 in idx_lst])
    bounds_penalty = np.sum(1. / (1 + (max_k/embedding_K[embedding_K>max_k*.99])**1000)) # log-sigmoid
    return (scores * weights).sum() + bounds_penalty
  scipy_res = scipy.optimize.minimize(func, init_embedding.flatten())

  embedding = scipy_res.x.reshape((-1,dims))
  #embedding = init_embedding
  embedding_dists = np.array([[np.linalg.norm(embedding[i,:] - embedding[j,:]) for j in range(num_fluors)] for i in range(num_fluors)])
  embedding_K = K_0 * (R_0_expanded / (embedding_dists + np.eye(num_fluors)))**6
  embedding_K[np.diag_indices(num_fluors)] = 0
  embedding_K_fret = embedding_K[f_comp_start:f_comp_end, f_comp_start:f_comp_end]
  embedding_k_in = np.diag(embedding_K[f_in_start:f_in_end, f_comp_start:f_comp_end])
  embedding_k_out = np.diag(embedding_K[f_comp_start:f_comp_end, f_out_start:f_out_end])
  embedding_k_decay = np.diag(embedding_K[f_comp_start:f_comp_end, quench_start:quench_end])
  params = embedding_K / (k_0 + np.tile(embedding_K.sum(axis=1), (num_fluors,1)).T)

  print(embedding_dists)

#  import matplotlib.pyplot as plt
#  plt.ion()
#  plt.figure()
##  plt.scatter(dists.flatten(), embedding_dists.flatten(), color='k')
#  plt.scatter(params_ideal, params, color='k')
#  plt.plot([0, max(params_ideal)], [0, max(params_ideal)], 'k--')
#  plt.axis('square')
#
#  from mpl_toolkits import mplot3d
#  plt.figure()
#  ax = plt.axes(projection='3d')
#  colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
#  ax.scatter3D(embedding[:,0], embedding[:,1], embedding[:,2], color=colors[:num_nodes], alpha=1)
#

  embedding_dict = dict(zip(fluor_names, [embedding[i,:] for i in range(num_fluors)]))
  output = {
    'positions': embedding_dict,
    'K': embedding_K,
    'k_in': embedding_k_in,
    'k_out': embedding_k_out,
    'K_fret': embedding_K_fret,
    'k_decay': embedding_k_decay,
    'fluorophore_names': fluor_names,
    'node_names': node_names,
  }
    
  return output
