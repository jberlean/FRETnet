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
  def __init__(self, name, in_edges = [], out_edges = [], decay_rate = 0.0, emit_rate = 1.0, status = False):
    self._name = name
    self._in_edges = list(in_edges)
    self._out_edges = list(out_edges)

    self._reactions = None

    self._decay_rate = decay_rate
    self._emit_rate = emit_rate

    self._status = False
    self._propensity = None

  @property
  def name(self):
    return self._name

  @property
  def in_edges(self):
    return self._in_edges[:]
  @property
  def out_edges(self):
    return self._out_edges[:]

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
    self._in_edges.append(e)
    n._out_edges.append(e)

  def _make_reaction(self, edge):
    return Reaction('transfer', edge.input, edge.output, edge.rate)
  def _update_reactions(self):
    # populate list of reactions
    rxns = [
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
   

class InputNode(Node):
  def __init__(self, name, in_edges = [], out_edges = [], decay_rate = 0.0, emit_rate = 1.0, production_rate = 0.0, status = False):
    super().__init__(name, in_edges, out_edges, decay_rate, emit_rate, status)

    self._production_rate = production_rate

  @property
  def production_rate(self):
    return self._production_rate
  @production_rate.setter
  def production_rate(self, k):
    self._production_rate = k
    self.update()

  def _update_reactions(self):
    # populate list of reactions
    rxns = [
        Reaction('production', output=self, rate=self.production_rate),
        Reaction('decay', input=self, rate=self.decay_rate), 
        Reaction('emit', input=self, rate=self.emit_rate)
    ] + [self._make_reaction(edge) for edge in self.out_edges]
    self._reactions = rxns


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
    self._nodes = nodes[:]
    self._stats = {}
    self._node_stats = {n: {} for n in nodes}

    self._time = 0.0

  @property
  def nodes(self):
    return self._nodes[:]

  def compute_kfret_matrix(self):
    num_nodes = len(self._nodes)
    node_idxs = {n:i for i,n in enumerate(self._nodes)}

    K = np.zeros((num_nodes, num_nodes))
    for i,node in enumerate(self._nodes):
      for e in node.out_edges:
        j = node_idxs[e.output]
        K[i,j] = e.rate

    return K

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

######
# CONVENIENCE FUNCTIONS
######

def network_from_rates(K_fret, k_out, k_in, node_names = None):
    num_nodes = len(k_out)

    if node_names is None:
      node_names = [f'node{i}' for i in range(num_nodes)]
    elif len(node_names) < num_nodes:
      node_names.extend([f'node{i}' for i in range(len(node_names), num_nodes)])

    nodes = [InputNode(name, production_rate=k_in_i, emit_rate=k_out_i) for name,k_out_i,k_in_i in zip(node_names, k_out, k_in)]
    for i,j in it.product(range(num_nodes), range(num_nodes)):
        if i==j:  continue
        nodes[j].add_input(nodes[i], K_fret[i,j])

    return Network(nodes)


