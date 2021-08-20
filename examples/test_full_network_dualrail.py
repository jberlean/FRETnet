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
  print ('Example usage: python -i test_full_network_dualrail.py data/2px_full.p')
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
num_nodes_sr = data['num_nodes_singlerail']
num_fluorophores = data['num_fluorophores']
pixel_names = data['node_names_dualrail']
node_names_sr = data['node_names_singlerail']
fluorophore_names = data['fluorophore_names']

dr_to_sr_map = data['dualrail_to_singlerail_map']
sr_to_fluor_map = data['singlerail_to_fluorophore_map']

K_fret = data['K_fret']
K_in = data['K_in']
K_out = data['K_out']
K_quench = data['K_quench']
I_k0 = data['args']['input_fluor_info'].get('k_0', 1)
I_kdecay = data['args']['input_fluor_info'].get('k_decay', 0)
C_k0 = data['args']['compute_fluor_info']['k_0']
C_kdecay = data['args']['compute_fluor_info'].get('k_decay', 0)

network = fretnet_objects.full_HANlike_network_from_rates(K_fret_IC = K_in, K_fret_CC = K_fret, K_fret_CO = K_out, K_fret_CQ = K_quench, I_kin = np.zeros(num_nodes_sr), I_k0 = I_k0, I_kdecay = I_kdecay, C_k0=C_k0, C_kdecay = C_kdecay, node_names = node_names_sr)

pixel_to_node_sr_index_map = {pixel_names.index(px): (node_names_sr.index(node_p), node_names_sr.index(node_n)) for px, (node_p, node_n) in dr_to_sr_map.items()}

## Test network using test_dualrail_network function
plot = test_dualrail_network(
  network = network,
  pixels = pixel_names,
  pixel_to_node_sr_index_map = pixel_to_node_sr_index_map,
  input_magnitude = input_magnitude,
  output_magnitude = output_magnitude
)
  
plot.display()
