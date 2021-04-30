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

def probability_by_node(network, hanlike=True):
  """ Computes the probability of each node being excited at steady state.
      Calculation may be very inefficient for large non-HAN-like networks.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.
        hanlike (bool): Whether the network is HAN-like, which greatly simplifies analysis.

      Returns:
        probabilities (dict): Maps each Node object in the Network to its probability of being excited
  """
  if hanlike:
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

  # compute k_in and k_out vectors for convenience
  k_in = np.array([n.production_rate if isinstance(n, objects.InputNode) else 0.0 for n in nodes])
  k_out = np.array([n.emit_rate+n.decay_rate for n in nodes])

  # compute matrix of coefficients for the linear system of equations
  # network state X has corresponding index in A equal to the binary number corresponding to X's bits 
  A = np.zeros((2**num_nodes, 2**num_nodes))
  for state in it.product([0,1], repeat=num_nodes):
    idx = state_to_idx(state)
    for node_idx, node in enumerate(nodes):
      # handle k_in, k_out
      if not state[node_idx]:
        state_node_on = list(state)
        state_node_on[node_idx] = 1

        A[idx, idx] -= k_in[node_idx]
        A[idx, state_to_idx(state_node_on)] += k_out[node_idx]
      else:
        state_node_off = list(state)
        state_node_off[node_idx] = 0
        
        A[idx, state_to_idx(state_node_off)] += k_in[node_idx]
        A[idx, idx] -= k_out[node_idx]

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
    output[state] = prob_ss[state_to_idx(state)]

  return output

  

def node_outputs(network, hanlike=True):
  """ Computes the flux out of the network from each node, which includes exciton loss due to emission and decay.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.
        hanlike (bool): Whether the network is HAN-like, which greatly simplifies analysis.

      Returns:
        outputs (dict): Maps each Node object in the Network to its flux out of the network.
  """
  if hanlike:
    return _node_outputs_hanlike(network)
  else:
    return _node_outputs_general(network)

def node_fluxes(network, hanlike=True):
  """ Computes the total flux out of each node, combining flux due to FRET as well as emission/decay. This
      should be equal to the flux into each node.
      TODO: Optimize computation in the case of HAN-like networks, where we can avoid computing explicitly the
      probability of each network state.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        fluxes (dict): Maps each Node object in the Network to its total flux within and out of the network
  """
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

  # Calculate inverse of A
  try:
    Ainv = np.linalg.inv(A)
  except:
    raise ValueError(f'Singular matrix A computed from network {network}')

  # Compute probabilities dictionary
  # TODO: switch to using numpy's linear system of equations solver (more efficient than explicitly finding inverse)
  probs_vec = Ainv @ k_in
  probs_dict = {node: p for node,p in zip(nodes, probs_vec)}

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
  """ Computes the flux out of the network from each node, which includes exciton loss due to emission and decay.

      Arguments:
        network (objects.utils.Network): The Network object to be analyzed.

      Returns:
        outputs (dict): Maps each Node object in the Network to its flux out of the network.
  """
  nodes = network.nodes

  probs = _probability_by_node_hanlike(network)
  prob_vector = np.array([probs[n] for n in nodes])
  k_out = np.array([n.emit_rate+n.decay_rate for n in nodes])

  node_outputs = dict(zip(nodes, prob_vector*k_out))
  return node_outputs

def _node_outputs_general(network):
  nodes = network.nodes
  node_probs = _probability_by_node_general(network)

  node_outputs = {node:node_probs[node]*(node.emit_rate+node.decay_rate) for node in nodes}
  return node_outputs


