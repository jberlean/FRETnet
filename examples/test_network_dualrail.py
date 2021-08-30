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
input_magnitude = data['input_magnitude']
output_magnitude = data['output_magnitude']

num_pixels = data['num_nodes_dualrail']
num_fluorophores = data['num_fluorophores']
pixel_names = data['node_names_dualrail']
fluorophore_names = data['fluorophore_names']
pixel_to_fluorophore_map = data['dualrail_to_singlerail_map']

K_fret = data['K_fret']
k_out = data['k_out']
k_decay = data.get('k_decay', np.zeros_like(k_out))

network = fretnet_objects.network_from_rates(K_fret, k_out, np.zeros_like(k_out), k_decay = k_decay, node_names = fluorophore_names, hanlike=True)

if pixel_to_fluorophore_map is None:
  pixel_to_node_sr_index_map = {i: (2*i,2*i+1) for i in range(num_pixels)}
else:
  px_to_idx = {px:i for i,px in enumerate(pixel_names)}
  fluor_to_idx = {f:i for i,f in enumerate(fluorophore_names)}
  pixel_to_node_sr_index_map = {px_to_idx[px]: (fluor_to_idx[fp], fluor_to_idx[fn]) for px, (fp, fn) in pixel_to_fluorophore_map.items()}

## Test network using test_dualrail_network function
plot = test_dualrail_network(
  network = network,
  pixels = pixel_names,
  pixel_to_node_sr_index_map = pixel_to_node_sr_index_map,
  input_magnitude = input_magnitude,
  output_magnitude = output_magnitude
)
  
plot.display()
