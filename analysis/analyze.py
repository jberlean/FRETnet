import sys
import pathlib
import itertools as it

import numpy as np
import scipy.linalg

# INTRAPACKAGE IMPORTS
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.append(pkg_path)
from objects import utils as objects

# Basic functions for computing behavior of FRETnets
# Currently we have the ability to analytically compute the following steady-state quantities:
#   - Probability of each node being in the excited state
#   - Probability of each network state (excited/ground state of each node)
#   - Flux from each node out of the network
#   - Total flux out of each node (including both transfers within and out of the network)


##################################
# GENERIC STEADY-STATE BEHAVIOR ANALYSIS
##################################

def probability_by_node(network, hanlike=None, full=None):
  """ Computes the probability of each node being excited at steady state.
      Calculation may be very inefficient for large non-HAN-like networks.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.
        hanlike (bool): Whether the network is HAN-like, which greatly simplifies analysis.

      Returns:
        probabilities (dict): Maps each Node object in the Network to its probability of being excited
  """
  if hanlike is None:  hanlike = isinstance(network, objects.HANlikeNetwork)
  if full is None:  full = isinstance(network, objects.FullHANlikeNetwork)

  if hanlike and full:
    return _probability_by_node_hanlike_full(network)
  elif hanlike:
    return _probability_by_node_hanlike(network)
  else:
    return _probability_by_node_general(network)

def probability_by_network_state(network, eps = 10**-10):
  """ Computes the probability of each network state, defined as a tuple of 0/1 values
      describing the whether each Node is ground state (0) or excited state (1).
      The size of the output is exponential in the number of nodes, and the method requires solving
      a linear system of 2^n equations, so this becomes intractable very quickly.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.
        epsilon (float): precision of eigenvalue calculation (only determines when warnings will be printed)

      Returns:
        probabilities (dict): Maps tuples describing the network state (see above) to probabilities of occurring.
  """

  def state_to_idx(state):
    idx = 0
    for b in state:
      idx<<=1
      idx+=b
    return idx

  nodes = network.nodes
  num_nodes = len(nodes)
  node_idxs = {n: i for i,n in enumerate(nodes)}

  # compute k_in and k_off vectors for convenience
  k_in = np.array([n.production_rate if isinstance(n, objects.InputNode) else 0.0 for n in nodes])
  k_off = np.array([n.emit_rate+n.decay_rate for n in nodes])

  # compute matrix of coefficients for the linear system of equations
  # network state X has corresponding index in A equal to the binary number corresponding to X's bits 
  A = np.zeros((2**num_nodes, 2**num_nodes))
  for state in it.product([0,1], repeat=num_nodes):
    idx = state_to_idx(state)
    for node_idx, node in enumerate(nodes):
      # handle k_in, k_off
      if not state[node_idx]:
        state_node_on = list(state)
        state_node_on[node_idx] = 1

        A[idx, idx] -= k_in[node_idx]
        A[idx, state_to_idx(state_node_on)] += k_off[node_idx]
      else:
        state_node_off = list(state)
        state_node_off[node_idx] = 0
        
        A[idx, state_to_idx(state_node_off)] += k_in[node_idx]
        A[idx, idx] -= k_off[node_idx]

      # handle fret interactions
      for e in node.out_edges:
        output_node_idx = node_idxs[e.output]
        bits = (state[node_idx], state[output_node_idx])
        if bits == (0,1):
          prev_state = list(state)
          prev_state[node_idx] = 0
          prev_state[output_node_idx] = 1
          A[idx, state_to_idx(prev_state)] += e.rate
        elif bits == (1,0):
          A[idx, idx] -= e.rate

      for e in node.in_edges:
        input_node_idx = node_idxs[e.input]
        bits = (state[node_idx], state[input_node_idx])
        if bits == (0,1):
          A[idx, idx] -= e.rate
        elif bits == (1,0):
          prev_state = list(state)
          prev_state[node_idx] = 0
          prev_state[input_node_idx] = 1
          A[idx, state_to_idx(prev_state)] += e.rate

  # Analyze A matrix to determine null space, which represents the (stable) steady-state
  # vector of network state probabilities.
  # In well-behaved systems, there should be a single eigenvalue of magnitude 0, 
  # with all other eigenvalues having negative real part.
  # Pathological systems may exhibit the following:
  #   - multiple 0 eigenvalues, which may result in a non-unique steady state; any may be returned by this function
  #   - no 0 eigenvalues, which may result in all probabilities going to 0 over time (this should be impossible)
  #   - any number of positive eigenvalues, which result in probabilities exploding (also should be impossible)
  # This function will print a warning except in the last case
  nullspace = scipy.linalg.null_space(A)
  num_0_evals = nullspace.shape[1]
  if num_0_evals == 0:
    print(f'WARNING: No zero eigenvalues encountered when analyzing network {network}. Analysis failed')
  elif num_0_evals > 1:
    print(f'WARNING: Null-space has dimension >1, leading to non-unique steady state behavior. An arbitrary steady state will be returned')

  prob_ss = nullspace / nullspace.sum() # this is the vector of state probabilities

  output = {}
  for state in it.product([0,1], repeat=num_nodes):
    output[state] = prob_ss[state_to_idx(state)].item()

  return output

  

def node_outputs(network, hanlike=None, full=None):
  """ Computes the network output from each node, which is defined as the flux of emission from each node.
      Energy flux out of the network due to non-radiative decay (k_decay) is not included.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.
        hanlike (bool): Whether the network is HAN-like, which greatly simplifies analysis.

      Returns:
        outputs (dict): Maps each Node object in the Network to its flux out of the network.
  """
  if hanlike is None:  hanlike = isinstance(network, objects.HANlikeNetwork)
  if full is None:  full = isinstance(network, objects.FullHANlikeNetwork)

  if hanlike and full:
    return _node_outputs_hanlike_full(network)
  elif hanlike:
    return _node_outputs_hanlike(network)
  else:
    return _node_outputs_general(network)

def node_fluxes(network, hanlike=None, full=None):
  """ Computes the total flux out of each node, combining flux due to FRET as well as emission/decay. This
      should be equal to the flux into each node.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        fluxes (dict): Maps each Node object in the Network to its total flux within and out of the network
  """
  if hanlike is None:  hanlike = isinstance(network, objects.HANlikeNetwork)
  if full is None: full = isinstance(network, objects.FullHANlikeNetwork)

  if hanlike and full:
    return _node_fluxes_hanlike_full(network)
  elif hanlike:
    return _node_fluxes_hanlike(network)
  else:
    return _node_fluxes_general(network)


## Auxiliary functions, shouldn't need to be called from outside this module
def _probability_by_node_hanlike(network):
  """ Computes the probability of each node being excited at steady state in a HAN-like FRETnet.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        probabilities (dict): Maps each Node object in the Network to its probability of being excited
  """

  nodes = network.nodes
  num_nodes = len(nodes)

  node_idxs = {n: i for i,n in enumerate(nodes)}

  # Compute A matrix and k_in
  A = np.zeros((num_nodes, num_nodes))
  k_in = np.zeros(num_nodes)
  for i,node in enumerate(nodes):
    out_edges = node.out_edges

    if isinstance(node, objects.InputNode):
      k_in[i] = node.production_rate
    else:
      k_in[i] = 0.0

    Di = sum(e.rate for e in out_edges) + node.decay_rate + node.emit_rate + k_in[i]
    A[i,i] = Di

    for e in out_edges:
      j = node_idxs[e.output]
      A[i,j] = -e.rate

  # Check symmetric (but try to do the computation regardless)
  if not (A.T == A).all():
    print(f'WARNING: Matrix A for network {network} is not symmetric. Network may not be symmetric.')

  # Compute probabilities dictionary
  probs_vec = np.linalg.solve(A, k_in)
  probs_dict = {node: p for node,p in zip(nodes, probs_vec)}

  return probs_dict

def _probability_by_node_hanlike_full(network):
  """ Computes the probability of each node being excited at steady state in a HAN-like FRETnet.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        probabilities (dict): Maps each Node object in the Network to its probability of being excited
  """
  input_nodes = network.input_nodes
  compute_nodes = network.compute_nodes
  output_nodes = network.output_nodes
  quencher_nodes = network.quencher_nodes
  node_groups = network.node_groups
  num_nodes = len(node_groups)

  # Get relevant FRET matrices
  K_fret_IC = network.get_K_fret_IC()
  K_fret_CC = network.get_K_fret_CC()
  K_fret_CO = network.get_K_fret_CO()
  K_fret_CQ = network.get_K_fret_CQ()

  # Estimate effective k_in into each compute fluorophore, based on excitation of each input
  I_prob = np.array([n.production_rate / (K_fret_IC[i,:].sum() + n.production_rate + n.decay_rate + n.emit_rate) for i,n in enumerate(input_nodes)])
  C_kin = I_prob @ K_fret_IC

  # Estimate effective k_out and k_decay for each compute fluor
  C_kout = network.get_k_out()
  C_kdecay = network.get_k_decay()

  # Compute A matrix and k_in
  A = -K_fret_CC
  A[np.diag_indices(num_nodes)] = K_fret_CC.sum(axis=1) + C_kin + C_kout + C_kdecay

  # Check symmetric (but try to do the computation regardless)
  if not (A.T == A).all():
    print(f'WARNING: Matrix A for network {network} is not symmetric. Network may not be symmetric.')

  # Compute probabilities dictionary
  C_prob = np.linalg.solve(A, C_kin)
  O_prob = np.zeros_like(C_prob)
  Q_prob = np.zeros_like(O_prob)
  probs_dict = dict(it.chain(
      zip(input_nodes, I_prob), zip(compute_nodes, C_prob),
      zip(output_nodes, O_prob), zip(quencher_nodes, Q_prob),
      zip(node_groups, C_prob)
  ))

  return probs_dict


def _probability_by_node_general(network):
  nodes = network.nodes
  num_nodes = len(nodes)

  state_probs = probability_by_network_state(network)

  node_probs = {node: 0 for node in nodes}
  for state in state_probs:
    for node_idx, node in enumerate(nodes):
      if state[node_idx] == 1:
        node_probs[node] += state_probs[state]

  return node_probs

def _node_outputs_hanlike(network):
  """ Computes the network output from each node, defined as the rate of exciton loss due to radiative decay.
      Exciton loss due to non-radiative decay does not contribute to network output.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        outputs (dict): Maps each Node object in the Network to its flux out of the network.
  """
  nodes = network.nodes

  probs = _probability_by_node_hanlike(network)
  prob_vector = np.array([probs[n] for n in nodes])
  k_out = np.array([n.emit_rate for n in nodes])

  node_outputs = dict(zip(nodes, prob_vector*k_out))
  return node_outputs

def _node_outputs_hanlike_full(network):
  """ Computes the network output from each node, defined as the rate of exciton loss due to radiative decay.
      Exciton loss due to non-radiative decay does not contribute to network output.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        outputs (dict): Maps each Node object in the Network to its flux out of the network.
  """
  input_nodes = network.input_nodes
  compute_nodes = network.compute_nodes
  output_nodes = network.output_nodes
  quencher_nodes = network.quencher_nodes
  node_groups = network.node_groups

  probs = _probability_by_node_hanlike_full(network)

  I_prob = np.array([probs[n] for n in input_nodes])
  C_prob = np.array([probs[n] for n in compute_nodes])
  K_compute = network.get_K_fret_CC()
  K_in = network.get_K_fret_IC()
  K_out = network.get_K_fret_CO()
  K_quench = network.get_K_fret_CQ()

  I_outputs = I_prob * K_in.sum(axis=1)
  C_outputs = C_prob * (K_compute.sum(axis=1) + K_out.sum(axis=1) + K_quench.sum(axis=1))
  O_outputs = C_prob @ K_out
  Q_outputs = C_prob @ K_quench

  node_outputs = dict(it.chain(
      zip(input_nodes, I_outputs), zip(compute_nodes, C_outputs),
      zip(output_nodes, O_outputs), zip(quencher_nodes, Q_outputs),
      zip(node_groups, O_outputs)
  ))
  return node_outputs


def _node_outputs_general(network):
  nodes = network.nodes
  node_probs = _probability_by_node_general(network)

  node_outputs = {node:node_probs[node]*(node.emit_rate) for node in nodes}
  return node_outputs


def _node_fluxes_hanlike(network):
  nodes = network.nodes
  num_nodes = len(nodes)

  node_probs = _probability_by_node_hanlike(network)
  K_fret = network.get_K_fret()

  pairs = list(it.combinations(range(num_nodes), 2))
  pair_to_idx = {p:i for i,(j,k) in enumerate(pairs) for p in [(j,k),(k,j)]}
  num_pairs = len(pairs)

  B = np.zeros((num_pairs, num_pairs))
  c = np.zeros(num_pairs)
  for pair_idx, (i,j) in enumerate(pairs):
    n_i, n_j = nodes[i], nodes[j]

    c[pair_idx] = n_i.production_rate*node_probs[n_j] + n_j.production_rate*node_probs[n_i]

    B[pair_idx, pair_idx] -= n_i.production_rate + n_i.emit_rate + n_i.decay_rate \
        + n_j.production_rate + n_j.emit_rate + n_j.decay_rate
    for k in range(num_nodes):
      if k==i or k==j:  continue
      B[pair_idx, pair_to_idx[(j,k)]] = K_fret[k,i]
      B[pair_idx, pair_to_idx[(i,k)]] = K_fret[k,j]
      B[pair_idx, pair_idx] -= K_fret[i,k] + K_fret[j,k]

  pair_probs = np.linalg.solve(B, -c)

  fluxes = np.array([node_probs[n]*(n.emit_rate+n.decay_rate) for n in nodes])
  for (i,j), prob in zip(pairs, pair_probs):
    n_i,n_j = nodes[i], nodes[j]
    fluxes[i] += (node_probs[n_i] - prob)*K_fret[i,j]
    fluxes[j] += (node_probs[n_j] - prob)*K_fret[j,i]
  
  return dict(zip(nodes, fluxes))
  
def _node_fluxes_hanlike_full(network):
  input_nodes = network.input_nodes
  compute_nodes = network.compute_nodes
  output_nodes = network.output_nodes
  quencher_nodes = network.quencher_nodes
  num_node_groups = len(compute_nodes)

  # Get relevant FRET matrices
  K_fret_IC = network.get_K_fret_IC()
  K_fret_CC = network.get_K_fret_CC()
  K_fret_CO = network.get_K_fret_CO()
  K_fret_CQ = network.get_K_fret_CQ()

  node_probs = _probability_by_node_hanlike_full(network)
  I_prob = np.array([node_probs[n] for n in input_nodes])
  C_prob = np.array([node_probs[n] for n in compute_nodes])
  O_prob = np.array([node_probs[n] for n in output_nodes])
  Q_prob = np.array([node_probs[n] for n in quencher_nodes])

  # Estimate effective k_in, k_out, and k_decay for each compute fluorophore
  C_kin = I_prob @ K_fret_IC
  C_kout = network.get_k_out()
  C_kdecay = network.get_k_decay()

  # Enumerate pairs in order corresponding to matrix of transition rates
  pairs = list(it.combinations(range(num_node_groups), 2))
  pair_to_idx = {p:i for i,(j,k) in enumerate(pairs) for p in [(j,k),(k,j)]}
  num_pairs = len(pairs)

  # Construct rate matrix/vector for linear system of equations
  B = np.zeros((num_pairs, num_pairs))
  c = np.zeros(num_pairs)
  for pair_idx, (i,j) in enumerate(pairs):
    c[pair_idx] = C_kin[i]*C_prob[j] + C_kin[j]*C_prob[i]

    B[pair_idx, pair_idx] -= C_kin[i] + C_kout[i] + C_kdecay[i] + C_kin[j] + C_kout[j] + C_kdecay[j]
    for k in range(num_node_groups):
      if k==i or k==j:  continue
      B[pair_idx, pair_to_idx[(j,k)]] = K_fret_CC[k,i]
      B[pair_idx, pair_to_idx[(i,k)]] = K_fret_CC[k,j]
      B[pair_idx, pair_idx] -= K_fret_CC[i,k] + K_fret_CC[j,k]

  # Solve linear system of equations
  pair_probs = np.linalg.solve(B, -c)

  # Compute fluxes
  C_fluxes = C_prob * (C_kout+C_kdecay)
  for (i,j), prob in zip(pairs, pair_probs):
    C_fluxes[i] += (C_prob[i] - prob)*K_fret_CC[i,j]
    C_fluxes[j] += (C_prob[j] - prob)*K_fret_CC[j,i]

  I_fluxes = I_prob * (K_fret_IC @ (1-C_prob))  # approximate, doesn't account for correlation between I/C/O/Q states
  O_fluxes = (1-O_prob) * (C_prob @ K_fret_CO)
  Q_fluxes = (1-Q_prob) * (C_prob @ K_fret_CQ)

  flux_dict = dict(it.chain(
      zip(input_nodes, I_fluxes), zip(compute_nodes, C_fluxes),
      zip(output_nodes, O_fluxes), zip(quencher_nodes, Q_fluxes)
  ))
 
  return flux_dict
   

def _node_fluxes_general(network):
  nodes = network.nodes
  num_nodes = len(nodes)
  node_idxs = {n:i for i,n in enumerate(nodes)}

  state_probs = probability_by_network_state(network)
  
  output = {n: 0 for n in nodes}
  for node_idx, node in enumerate(nodes):
    flux = 0
    for state in state_probs:
      if state[node_idx] == 0:  continue

      flux += state_probs[state]*(node.emit_rate+node.decay_rate)

      for e in node.out_edges:
        out_node_idx = node_idxs[e.output]   
        if state[out_node_idx] == 0:
          flux += state_probs[state]*e.rate

    output[node] = flux

  return output


