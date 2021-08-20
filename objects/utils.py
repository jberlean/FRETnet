import itertools as it

import numpy as np

class Reaction(object):
  def __init__(self, mode, input = None, output = None, rate = 0.0):
    self.mode = mode
    self.input = input
    self.output = output
    self.rate = rate

  def propensity(self):
    if self.mode == 'production':
      return self.rate if self.output.is_off() else 0.0
    elif self.mode == 'transfer':
      return self.rate if self.input.is_on() and self.output.is_off() else 0.0
    elif self.mode == 'decay' or self.mode == 'emit':
      return self.rate if self.input.is_on() else 0.0
    else:
      print('WARNING: Can\'t calculate propensity for unknown reaction type {} (in: {}, out: {}, rate: {})'.format(self.mode, self.input, self.output, self.rate))
      return 0.0

  def execute(self):
    if self.mode == 'production':
      assert(self.output.is_off())
      self.output.set_on()
    elif self.mode == 'transfer':
      assert(self.input.is_on() and self.output.is_off())
      self.input.set_off()
      self.output.set_on()
    elif self.mode == 'decay' or self.mode == 'emit':
      assert(self.input.is_on())
      self.input.set_off()
    else:
      print('WARNING: Can\'t execute unknown reaction type {} (in: {}, out: {}, rate: {})'.format(self.mode, self.input, self.output, self.rate))


class Node(object):
  def __init__(self, name, in_edges = [], out_edges = [], production_rate = 0.0, decay_rate = 0.0, emit_rate = 1.0, status = False):
    self._name = name
    self._in_edges = tuple(in_edges)
    self._out_edges = tuple(out_edges)

    self._reactions = None

    self._production_rate = production_rate
    self._decay_rate = decay_rate
    self._emit_rate = emit_rate

    self._status = False
    self._propensity = None

  @property
  def name(self):
    return self._name

  @property
  def in_edges(self):
    return self._in_edges
  @property
  def out_edges(self):
    return self._out_edges

  @property
  def production_rate(self):
    return self._production_rate
  @production_rate.setter
  def production_rate(self, k):
    self._production_rate = k
    self.update()
  @property
  def decay_rate(self):
    return self._decay_rate
  @decay_rate.setter
  def decay_rate(self, k):
    self._decay_rate = k
    self.update()
  @property
  def emit_rate(self):
    return self._emit_rate
  @emit_rate.setter
  def emit_rate(self, k):
    self._emit_rate = k
    self.update()

  @property
  def status(self):
    if self._status:  return 'ON'
    else:  return 'OFF'
  def is_on(self):
    return self._status
  def is_off(self):
    return not self._status
  def set_on(self):
    self._status = True
  def set_off(self):
    self._status = False

  def add_input(self, n, rate):
    e = Edge(n, self, rate)
    self._in_edges += (e,)
    n._out_edges += (e,)

  def _make_reaction(self, edge):
    return Reaction('transfer', edge.input, edge.output, edge.rate)
  def _update_reactions(self):
    # populate list of reactions
    rxns = [
        Reaction('production', output=self, rate=self.production_rate),
        Reaction('decay', input=self, rate=self.decay_rate), 
        Reaction('emit', input=self, rate=self.emit_rate)
    ] + [self._make_reaction(edge) for edge in self.out_edges]
    self._reactions = rxns
  def _update_propensity(self):
    # calculate propensity
    p = 0.0
    rxn_p = np.empty(len(self._reactions))
    for i,rxn in enumerate(self._reactions):
      rxn_p[i] = rxn.propensity()
      p += rxn_p[i]

    self._reaction_propensities = rxn_p
    self._propensity = p
  def update(self):
    self._update_reactions()
    self._update_propensity()
 
  def propensity(self):
    if self._propensity is None:
      self.update()

    return self._propensity

  def choose_reaction(self):
    # chooses a reaction based on each reaction rate and returns it
    if self._reactions is None:
      self.update()

    reaction_probs = self._reaction_propensities/self._propensity
    rxn_idx = np.random.choice(len(self._reactions), p=reaction_probs)
    rxn = self._reactions[rxn_idx]

#    print('Node {} reactions (p = {}):'.format(self.name, self._propensity))
#    for r,p in zip(self._reactions, self._reaction_propensities):
#      rxn_str = '{} -> {}'.format(r.input.name if r.input is not None else '', r.output.name if r.output is not None else '')
#      print('{}: {}{}'.format(rxn_str, p,'*' if r==rxn else ''))

    return rxn

  def __str__(self):
    return f"Node('{self._name}')"

  def __repr__(self):
    return str(self)
   

class InputNode(Node):  # Included for legacy reasons. Node objects have the same functionality as InputNodes now.
  pass

class Edge(object):
  def __init__(self, input, output, rate):
    self._input = input
    self._output = output
    self._rate = rate

  @property
  def input(self):  return self._input
  @property
  def output(self):  return self._output
  @property
  def rate(self):  return self._rate

  def __str__(self):
    return f"Edge({self._input} -> {self._output})"

  def __repr__(self):
    return str(self)


class Network(object):
  def __init__(self, nodes = []):
    self._nodes = tuple(nodes)
    self._stats = {}
    self._node_stats = {n: {} for n in nodes}

    self._time = 0.0

  @property
  def nodes(self):
    return self._nodes
  @property
  def num_nodes(self):
    return len(self._nodes)

  def set_k_in(self, k_in):
    for n,v in zip(self._nodes, k_in):
      n.production_rate = v
  def get_k_in(self):
    return np.fromiter((n.production_rate for n in self._nodes), dtype=np.float)

  def set_k_out(self, k_out):
    for n,v in zip(self._nodes, k_out):
      n.emit_rate = v
  def get_k_out(self):
    return np.fromiter((n.emit_rate for n in self._nodes), dtype=np.float)

  def set_k_decay(self, k_decay):
    for n,v in zip(self._nodes, k_decay):
      n.decay_rate = v
  def get_k_decay(self):
    return np.fromiter((n.decay_rate for n in self._nodes), dtype=np.float)
 
  def get_K_fret(self):
    num_nodes = len(self._nodes)
    node_idxs = {n:i for i,n in enumerate(self._nodes)}

    K = np.zeros((num_nodes, num_nodes))
    for i,node in enumerate(self._nodes):
      for e in node.out_edges:
        j = node_idxs[e.output]
        K[i,j] = e.rate

    return K

  def compute_kfret_matrix(self):
    return self.get_K_fret()

  def choose_reaction(self):
    node_propensities = np.array([n.propensity() for n in self._nodes])
    propensity = node_propensities.sum()
    if propensity == 0:
      print('WARNING: No valid reaction found at time {}'.format(self._time))
      return None

    dt = np.random.exponential(1./propensity)
    node = np.random.choice(self._nodes, p=node_propensities/propensity)

    rxn = node.choose_reaction()
    return rxn, dt

  def step_forward(self):
    rxn, dt = self.choose_reaction()
    for node in self._nodes:
      self._node_stats[node][node.status] = self._node_stats[node].get(node.status, 0) + dt
    rxn.execute()
    self._time += dt

    keys = it.product(['*', rxn.mode], ['*', rxn.input], ['*', rxn.output])
    for key in keys:
      self._stats[key] = self._stats.get(key, 0) + 1

    changed_nodes = [rxn.input, rxn.output]
    for node in changed_nodes:
      if node is None:  continue
      node.update()
      for e in node.in_edges:  e.input.update()

    return rxn, dt, changed_nodes

  def activation(self, node):
    if self._time == 0.0:  return 0.0

    cts = self._stats.get(('*', node, '*'), 0)

    return cts/self._time

class HANlikeNetwork(Network):
  pass     

class FullHANlikeNetwork(HANlikeNetwork):
  def __init__(self, input_nodes = [], compute_nodes = [], output_nodes = [], quencher_nodes = [], node_group_names = None):
    self._nodes = tuple(input_nodes + compute_nodes + output_nodes + quencher_nodes)
    self._input_nodes = tuple(input_nodes)
    self._compute_nodes = tuple(compute_nodes)
    self._output_nodes = tuple(output_nodes)
    self._quencher_nodes = tuple(quencher_nodes)

    num_node_groups = len(input_nodes)
    self._input_node_idxs = tuple(range(num_node_groups))
    self._compute_node_idxs = tuple(range(num_node_groups, 2*num_node_groups))
    self._output_node_idxs = tuple(range(2*num_node_groups, 3*num_node_groups))
    self._quencher_node_idxs = tuple(range(3*num_node_groups, 4*num_node_groups))

    if node_group_names is None:
      node_group_names = [f'NodeGroup{i}' for i in range(num_node_groups)]
    self._num_node_groups = num_node_groups
    self._node_group_idxs = tuple(zip(self._input_node_idxs, self._compute_node_idxs, self._output_node_idxs, self._quencher_node_idxs))
    self._node_groups = tuple(tuple(map(self._nodes.__getitem__, idxs)) for idxs in self._node_group_idxs)
    self._node_to_node_group_idx = {n: i for i,ng in enumerate(self._node_groups) for n in ng}
    self._node_group_names = tuple(node_group_names)

    self._stats = {}
    self._node_stats = {n: {} for n in self._nodes}

    self._time = 0.0

  @property
  def input_nodes(self):
    return self._input_nodes
  @property
  def compute_nodes(self):
    return self._compute_nodes
  @property
  def output_nodes(self):
    return self._output_nodes
  @property
  def quencher_nodes(self):
    return self._quencher_nodes

  @property
  def num_node_groups(self):
    return self._num_node_groups
  @property
  def node_groups(self):
    return self._node_groups
  @property
  def node_group_names(self):
    return self._node_group_names

  def get_node_group(index = None, node = None):
    if index is not None:
      return self._node_groups[index]
    elif node is not None:
      return self._node_groups[self._node_to_node_group_idx[node]]
    else:
      raise ValueError('One of index or node should be specified to get_node_group()')

  def set_k_in(self, k_in):
    for n,v in zip(self._input_nodes, k_in):
      n.production_rate = v
  def get_k_in(self):
    return np.fromiter((n.production_rate for n in self._input_nodes),dtype=np.float)

  def set_k_out(self, k_out):
    raise ValueError('Cannot directly set k_out for full HAN-like FRETnets')
  def get_k_out(self):
    k_out = np.fromiter(
        (sum(e.rate for e in n.out_edges if e.output==o) for n,o in zip(self._compute_nodes, self._output_nodes))
    , dtype = np.float)
    return k_out

  def set_k_decay(self, k_decay):
    raise ValueError('Cannot directly set k_decay for full HAN-like FRETnets')
  def get_k_decay(self):
    compute_nodes = self._compute_nodes
    off_nodes = set(self._output_nodes) | set(self._quencher_nodes)
    k_decay = np.fromiter(
        (sum(e.rate for e in n.out_edges if e.output in off_nodes)+n.emit_rate+n.decay_rate for n in self._compute_nodes)
    , dtype = np.float) - self.get_k_out()
    return k_decay
  def get_k_decay_intrinsic(self):
    return np.fromiter((n.emit_rate + n.decay_rate for n in self._nodes), dtype = np.float)
 
  def _get_K_fret_nodesubset(self, donor_nodes, acceptor_nodes):
    D_num_nodes = len(donor_nodes)
    A_num_nodes = len(acceptor_nodes)

    A_node_idxs_dict = {n: i for i,n in enumerate(acceptor_nodes)}

    K = np.zeros((D_num_nodes, A_num_nodes))
    for i,donor in enumerate(donor_nodes):
      for e in donor.out_edges:
        if e.output in A_node_idxs_dict:
          j = A_node_idxs_dict[e.output]
          K[i,j] = e.rate

    return K
    

  def get_K_fret_IC(self):
    return self._get_K_fret_nodesubset(self._input_nodes, self._compute_nodes)

  def get_K_fret_CC(self):
    return self._get_K_fret_nodesubset(self._compute_nodes, self._compute_nodes)

  def get_K_fret_CO(self):
    return self._get_K_fret_nodesubset(self._compute_nodes, self._output_nodes)

  def get_K_fret_CQ(self):
    return self._get_K_fret_nodesubset(self._compute_nodes, self._quencher_nodes)

  def get_K_fret(self):
    Kfret = np.zeros((self.num_nodes, self.num_nodes))
    Kfret[np.ix_(self._input_node_idxs, self._compute_node_idxs)] = self.get_K_fret_IC()
    Kfret[np.ix_(self._compute_node_idxs, self._compute_node_idxs)] = self.get_K_fret_CC()
    Kfret[np.ix_(self._compute_node_idxs, self._output_node_idxs)] = self.get_K_fret_CO()
    Kfret[np.ix_(self._compute_node_idxs, self._quencher_node_idxs)] = self.get_K_fret_CQ()
    return Kfret

######
# CONVENIENCE FUNCTIONS
######

def network_from_rates(K_fret, k_out, k_in, k_decay = None, node_names = None):
    num_nodes = len(k_out)

    if k_decay is None:
      k_decay = np.zeros_like(k_out)

    if node_names is None:
      node_names = [f'node{i}' for i in range(num_nodes)]
    elif len(node_names) < num_nodes:
      node_names.extend([f'node{i}' for i in range(len(node_names), num_nodes)])

    nodes = [InputNode(name, production_rate=k_in_i, emit_rate=k_out_i, decay_rate = k_decay_i) for name,k_out_i,k_in_i,k_decay_i in zip(node_names, k_out, k_in, k_decay)]
    for i,j in it.product(range(num_nodes), range(num_nodes)):
        if i==j:  continue
        nodes[j].add_input(nodes[i], K_fret[i,j])

    return Network(nodes)


def full_HANlike_network_from_rates(K_fret_IC, K_fret_CC, K_fret_CO, K_fret_CQ, I_kin, I_k0=1, I_kdecay=0, C_k0=1, C_kdecay=0, node_names=None):
    num_nodes = K_fret_IC.shape[0]

    if node_names is None:
      node_names = [f'node{i}' for i in range(num_nodes)]
    elif len(node_names) < num_nodes:
      node_names.extend([f'node{i}' for i in range(len(node_names), num_nodes)])

    input_nodes = [InputNode(f'{name}_I', production_rate=k_in, emit_rate=I_k0, decay_rate=I_kdecay) for name,k_in in zip(node_names, I_kin)]
    compute_nodes = [Node(f'{name}_C', emit_rate=C_k0, decay_rate = C_kdecay) for name in node_names]
    output_nodes = [Node(f'{name}_O', emit_rate=0, decay_rate=0) for name in node_names]
    quencher_nodes = [Node(f'{name}_Q', emit_rate=0, decay_rate=0) for name in node_names]

    for i,j in it.product(range(num_nodes), repeat=2):
        compute_nodes[j].add_input(input_nodes[i], K_fret_IC[i,j])
        if i!=j:  compute_nodes[j].add_input(compute_nodes[i], K_fret_CC[i,j])
        output_nodes[j].add_input(compute_nodes[i], K_fret_CO[i,j])
        quencher_nodes[j].add_input(compute_nodes[i], K_fret_CQ[i,j])

    return FullHANlikeNetwork(input_nodes, compute_nodes, output_nodes, quencher_nodes, node_group_names = node_names)

