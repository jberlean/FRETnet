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

num_pixels = data['num_pixels']
num_fluorophores = data['num_fluorophores']
pixel_names = data['pixel_names']
fluorophore_names = data['fluorophore_names']
pixel_to_fluorophore_map = data['pixel_to_fluorophore_map']

network = data['network']
pixel_to_node_map = {px: (network.nodes[fluorophore_names.index(node_p)], network.nodes[fluorophore_names.index(node_n)]) for px, (node_p, node_n) in pixel_to_fluorophore_map.items()}

## Test network using test_dualrail_network function
plot = test_dualrail_network(
  network = network,
  pixels = pixel_names,
  pixel_to_node_map = pixel_to_node_map,
  input_magnitude = input_magnitude,
  output_magnitude = output_magnitude
)
  
plot.display()
