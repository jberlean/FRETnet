import sys
import pathlib
import pickle

import numpy as np

## Import FRETnet package
pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.insert(0,pkg_path)
from analysis.interactive import plot_dualrail_tsne

datafile = sys.argv[1]
if len(sys.argv) >= 3:  perplexity = int(sys.argv[2])
else:  perplexity = 50

with open(datafile, 'rb') as instream:
  data = pickle.load(instream)

num_nodes = data['kfret'][0].shape[0]
num_pixels = num_nodes//2
kfret = data['kfret']
koff = data['koff']
tsne_res = data['tsne'][perplexity]

colors = data['colors']

plot_dualrail_tsne(num_nodes, num_pixels, kfret, koff, tsne_res, colors)


