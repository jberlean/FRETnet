import sys
import pathlib

## Import FRETnet package
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.insert(0,pkg_path)

import objects.utils as objects

def linear(n=2, k_in = 1, k_out = 1, k_fret = 1):
  nodes = [objects.InputNode('node1', production_rate=k_in, emit_rate = 0.0, decay_rate=0.0)] + \
      [objects.Node(f'node{i}', emit_rate = 0.0, decay_rate = 0.0) for i in range(2, n)] + \
      [objects.Node(f'node{n}', emit_rate=k_out, decay_rate = 0.0)]

  for i in range(n-1):
    nodes[i].add_input(nodes[i+1], k_fret)
    nodes[i+1].add_input(nodes[i], k_fret)

  return objects.Network(nodes)
