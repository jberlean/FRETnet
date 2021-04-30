import sys
import pathlib
import pickle

import numpy as np

## Import FRETnet package
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.insert(0,pkg_path)

import objects.utils as fretnet_objects
from analysis.interactive import test_dualrail_network


if len(sys.argv) != 2:
  print ('Example usage: python -i test_network_dualrail.py data/9px.p')
  sys.exit(0)

## arguments
inpath = sys.argv[1]

## data import
with open(inpath, 'rb') as infile:
  data = pickle.load(infile)

## Select network to test, and construct fretnet_objects.Network instance
input_magnitude = data['train_args_MG'].get('input_magnitude', 1)
k_out_value = data['train_args_MG'].get('k_out_value', 100)
output_magnitude = input_magnitude * k_out_value / (input_magnitude + k_out_value)

results = data['results_MG']
best_idx = np.argmin([res['cost'] for res in results])
best_kfret = results[best_idx]['K_fret']
best_kout = results[best_idx]['k_out']

num_nodes = len(best_kout)
num_pixels = num_nodes//2
node_names = [f'{i}{pm}' for i in range(1,num_pixels+1) for pm in ['+','-']]
network = fretnet_objects.network_from_rates(best_kfret, best_kout, np.zeros_like(best_kout), node_names=node_names)

## Test network using test_dualrail_network function
pixels = list(map(str, range(1, num_pixels+1)))
pixel_to_node_map = {px: (network.nodes[2*i], network.nodes[2*i+1]) for i,px in enumerate(pixels)}

plot = test_dualrail_network(
  network = network,
  pixels = pixels,
  pixel_to_node_map = pixel_to_node_map,
  input_magnitude = input_magnitude,
  output_magnitude = output_magnitude
)
  
plot.display()
