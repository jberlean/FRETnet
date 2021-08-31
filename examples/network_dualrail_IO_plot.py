import sys, os
import pathlib
import pickle
import itertools as it

import numpy as np
from sklearn.manifold import TSNE
import sklearn.cluster
import networkx as nx
import tqdm

import matplotlib.pyplot as plt
plt.ion()

## Import FRETnet package
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.insert(0,pkg_path)

import objects.utils as fretnet_objects
from analysis import analyze


if len(sys.argv) < 2 or len(sys.argv) > 4:
  print ('Example usage: python -i network_dualrail_IO_plot.py data/4px_2.p')
  sys.exit(0)

## arguments
inpath = os.path.abspath(sys.argv[1])
tsne_perplexity = 50 if len(sys.argv)<=2 else int(sys.argv[2])
qstyle = True if len(sys.argv)<=3 else (sys.argv[3] == 'True')
seed = np.random.randint(0,10**6)

## data import
with open(inpath, 'rb') as infile:
  data = pickle.load(infile)

# prep data export
outputdir = f'tmp/{seed}'
os.makedirs(outputdir)

## Select network to test, and construct fretnet_objects.Network instance
input_magnitude = data['input_magnitude']
output_magnitude = data['output_magnitude']

stored_data = data['training_metadata']['stored_data']

num_pixels = data['num_nodes_dualrail']
num_nodes_singlerail = data['num_nodes_singlerail']
num_fluorophores = data['num_fluorophores']
pixel_names = data['node_names_dualrail']
fluorophore_names = data['fluorophore_names']
pixel_to_fluorophore_map = data['dualrail_to_singlerail_map']

K_fret = data['K_fret']
k_out = data['k_out']
k_decay = data.get('k_decay', np.zeros_like(k_out))

network = fretnet_objects.network_from_rates(K_fret, k_out, np.zeros_like(k_out), k_decay = k_decay, node_names = fluorophore_names, hanlike=True)

if pixel_to_fluorophore_map is None:
  pixel_to_node_map = {i: (2*i,2*i+1) for i in range(num_pixels)}
else:
  px_to_idx = {px:i for i,px in enumerate(pixel_names)}
  fluor_to_idx = {f:i for i,f in enumerate(fluorophore_names)}
  pixel_to_node_map = {px_to_idx[px]: (fluor_to_idx[fp], fluor_to_idx[fn]) for px, (fp, fn) in pixel_to_fluorophore_map.items()}


## Test network
if qstyle:
  inputs = np.array(list(it.product([-1, 0, 1], repeat=num_pixels)))
else:
  inputs = np.array(list(it.product([-1, 1], repeat=num_pixels)))

input_idxs = {tuple(inputs[idx,:]): idx for idx in range(inputs.shape[0])}

outputs = []
for img_input in tqdm.tqdm(inputs):
  k_in = np.zeros(num_nodes_singlerail)
  for px_idx, px_val in enumerate(img_input):
    node_sr_pos, node_sr_neg = pixel_to_node_map[px_idx]
    k_in[node_sr_pos] = max(0, input_magnitude*px_val)
    k_in[node_sr_neg] = max(0, -input_magnitude*px_val)
  network.set_k_in(k_in)

  network_output_dict = analyze.node_outputs(network, hanlike=True)
  network_output = [network_output_dict[n] for n in network.nodes]
  img_output = [(network_output[pixel_to_node_map[px_idx][0]] - network_output[pixel_to_node_map[px_idx][1]])/output_magnitude for px_idx in range(num_pixels)]

  outputs.append(img_output)
outputs = np.array(outputs)
  
#clusters = sklearn.cluster.OPTICS(min_samples=.06, metric='euclidean').fit_predict(outputs)
#clusters = sklearn.cluster.SpectralClustering(n_clusters = 8).fit_predict(outputs)
clusters = sklearn.cluster.MeanShift(bandwidth=np.sqrt(num_pixels*.2**2)).fit_predict(outputs)
#clusters = sklearn.cluster.AffinityPropagation().fit_predict(outputs)
num_clusters = len(set(clusters))

tsne_res = TSNE(n_components=2, perplexity=tsne_perplexity, init='pca', verbose=5).fit_transform(outputs)


good_clusters = sorted(set(clusters[input_idxs[tuple(stored_img)]] for stored_img in stored_data) - {-1})

## Plot results
colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
inputs_str = [''.join({-1: '0', 0: '?', 1: '1'}[v] for v in img_input) for img_input in inputs]

node_pos = dict(zip(inputs_str, tsne_res))
network_nx = nx.Graph()
network_nx.add_nodes_from(inputs_str)
#network_nx.add_edges_from([(in1,in2) for in1,in2 in it.combinations(inputs_str,2) if np.sum([v1==v2 for v1,v2 in zip(in1,in2)])==1])

plt.figure()
ax = plt.gca()


cluster_info = {}
for cluster in sorted(set(clusters)):
  is_good_cluster = cluster in good_clusters
  cluster_nodes_idx = [i for i in range(len(clusters)) if clusters[i]==cluster]
  cluster_nodes_str = list(map(inputs_str.__getitem__, cluster_nodes_idx))
  cluster_nodes_int = np.array(list(map(inputs.__getitem__, cluster_nodes_idx)))
  input_mean = list(np.mean(cluster_nodes_int, axis = 0))
  output_mean = list(np.mean(list(map(outputs.__getitem__, cluster_nodes_idx)), axis=0))

  if cluster == -1:
    node_color = 'white'
    node_alpha = 1.0
    linewidths = 1
    edgecolors = 'k'
  elif is_good_cluster:
    node_color = colors[good_clusters.index(cluster)]
    node_alpha = 1.0
    linewidths = 1
    edgecolors = node_color
  else:
    node_color = 'gray'
    node_alpha = .3
    linewidths = 1
    edgecolors = node_color

  print(f'{"*" if is_good_cluster else ""} Cluster {cluster} [{node_color}]')
  print(f'  Input Mean: {input_mean}')
  print(f'  Output Mean: {output_mean}')
  print(f'  Nodes: {cluster_nodes_str}')

  nx.draw_networkx(network_nx, nodelist = cluster_nodes_str, pos=node_pos, node_color=node_color, alpha=node_alpha, node_size = 10, linewidths=linewidths, edgecolors=edgecolors, ax=ax, with_labels=False)

  cluster_info[cluster] = {
    'color': node_color,
    'img_inputs': cluster_nodes_int,
    'img_inputs_str': cluster_nodes_str,
    'img_input_mean': input_mean,
    'img_output_mean': output_mean,
    'is_good_cluster': is_good_cluster,
  }

node_labels = {node: '' for node in inputs_str}
node_labels.update({inputs_str[input_idxs[tuple(stored_img)]]:inputs_str[input_idxs[tuple(stored_img)]] for stored_img in stored_data})
nx.draw_networkx_labels(network_nx, pos=node_pos, labels=node_labels)

plt.savefig(os.path.join(outputdir, 'image_output_clustering.pdf'))

output = {
  'datapath': inpath,
  'qstyle': qstyle,
  'input_magnitude': input_magnitude,
  'output_magnitude': output_magnitude,
  'stored_data': stored_data,
  'num_fluorophores': num_fluorophores,
  'num_pixels': num_pixels,
  'network': network,
  'img_inputs': inputs,
  'img_inputs_str': inputs_str,
  'img_outputs': outputs,
  'cluster_assignments': clusters,
  'num_clusters': num_clusters,
  'tsne_perplexity': tsne_perplexity,
  'tsne_results': tsne_res,
  'cluster_info': cluster_info,
}

with open(os.path.join(outputdir, 'output.p'), 'wb') as outfile:
  pickle.dump(output, outfile)

