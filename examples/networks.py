import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
import objects.utils as objects

def linear(n=2, k_in = 1, k_out = 1, k_fret = 1):
  nodes = [objects.InputNode('node1', production_rate=k_in, emit_rate = 0.0, decay_rate=0.0)] + \
      [objects.Node(f'node{i}', emit_rate = 0.0, decay_rate = 0.0) for i in range(2, n)] + \
      [objects.Node(f'node{n}', emit_rate=k_out, decay_rate = 0.0)]

  for i in range(n-1):
    nodes[i].add_input(nodes[i+1], k_fret)
    nodes[i+1].add_input(nodes[i], k_fret)

  return objects.Network(nodes)
