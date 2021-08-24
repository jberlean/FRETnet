import sys
import pathlib
import math
import copy
import itertools as it

import networkx as nx
import numpy as np
import scipy.special
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()

pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.append(pkg_path)
import analysis.analyze as analyze
import objects.utils as objects

##################################
# DUAL-RAIL NETWORK VISUALIZATION
##################################

## TODO:
##  - Allow toggling of node states directly
##  - Allow toggling of network edges

def test_dualrail_network(network, pixels = None, pixel_to_node_sr_index_map = None, input_magnitude = 1, output_magnitude = 1, img_rows = None, img_cols = None):
  num_nodes = len(network.nodes)
  num_pixels = num_nodes // 2

  if pixels is None:
    pixels = list(map(str, range(1, num_pixels+1)))

  plot = DualRailNetworkPlot(network, pixels, pixel_to_node_sr_index_map, input_magnitude, output_magnitude, img_rows, img_cols)
  return plot

class DualRailNetworkPlot(object):
  def __init__(self, network, pixels, pixel_to_node_sr_index_map = None, input_magnitude = 1, output_magnitude = 1, img_rows = None, img_cols = None):
    self._network = copy.deepcopy(network)

    # Node (i.e. fluorophore) level properties
    self._nodes = self._network.nodes
    self._node_indices = {n:i for i,n in enumerate(self._nodes)}
    self._node_names = tuple(n.name for n in self._nodes)

    # Single-rail node (for full networks, these are node groups; otherwise these are simply nodes) level properties
    self._is_full_network = isinstance(network, objects.FullHANlikeNetwork)
    if self._is_full_network:
      self._nodes_sr = self._network.node_groups
      self._node_names_sr = self._network.node_group_names
      self._input_nodes = self._network.input_nodes
      self._output_nodes = self._network.output_nodes
    else:
      self._nodes_sr = self._nodes
      self._node_names_sr = self._node_names
      self._input_nodes = self._nodes
      self._output_nodes = self._nodes
    self._node_indices_sr = {n:i for i,n in enumerate(self._nodes_sr)}

    # Pixel (i.e. "dual-rail node") level properties
    if pixel_to_node_sr_index_map is None:
      pixel_to_node_sr_index_map = {i: (2*i,2*i+1) for i in range(len(pixels))}
    self._pixels = tuple(pixels)
    self._pixel_indices = {p:i for i,p in enumerate(self._pixels)}
    self._pixel_to_node_sr_map = {self._pixels[px_i]: (self._nodes_sr[npos_i], self._nodes_sr[nneg_i]) for px_i, (npos_i, nneg_i) in pixel_to_node_sr_index_map.items()}
    self._node_sr_to_pixel_map = {n: (px, plusminus) for px, (n1,n2) in self._pixel_to_node_sr_map.items() for n,plusminus in [(n1,True),(n2,False)]}
    self._input_magnitude = input_magnitude
    self._output_magnitude = output_magnitude

    if img_rows is None and img_cols is not None:
      self._img_rows = int(math.ceil(num_pixels / img_cols))
      self._img_cols = img_cols
    elif img_rows is not None and img_cols is None:
      self._img_rows = img_rows
      self._img_cols = int(math.ceil(num_pixels / img_rows))
    elif img_rows is None and img_cols is None:
      self._img_rows = int(math.ceil(math.sqrt(len(self._pixels))))
      self._img_cols = int(math.ceil(len(self._pixels)/self._img_rows))

    self._figure = None
    self._ax_img_in, self._ax_img_out = None, None
    self._ax_img_in_cbar, self._ax_img_out_cbar = None, None
    self._ax_net_in, self._ax_net_out = None, None
    self._ax_net_in_cbar, self._ax_net_out_cbar = None, None
    self._image_plot_mode = 'continuous'

    self._network_nx, self._network_nx_node_pos = self._init_network_nx(self._network)
    self._network_nx_node_pos = self._calc_network_node_pos_spring()
    self._network_activation_mode = 'output'

    # network inputs/outputs
    self._network_input = np.zeros(len(self._nodes))
    self._network_flux = None
    self._network_output = None
    self._image_input = np.zeros(len(self._pixels))
    self._image_output = None
    for px in self._pixels:  self.set_pixel_input(px, 0)

  def _setup_event_handlers(self):
    def click_handler(event):
      if event.inaxes == self._ax_img_in:
        # determine which pixel was clicked
        x,y = event.xdata, event.ydata
        row,col = int(round(y)), int(round(x))
        idx = row*self.image_cols + col

        if idx >= len(self._pixels):  return

        if 'shift' in str(event.key):
          new_state = 0
        else:
          new_state = -1 if self.pixel_input(self._pixels[idx])==1 else 1
        self.set_pixel_input(self._pixels[idx], new_state)

        self.update_plot()

    self._figure.canvas.mpl_connect('button_press_event', click_handler)

  def _init_network_nx(self, network):
    node_names = [n.name for n in network.nodes]
    num_nodes = len(node_names)

    network_nx = nx.Graph()
    network_nx.add_nodes_from(node_names)

    network.set_k_in(np.ones(network.num_node_groups))

    K_fret = network.get_K_fret()
    kout = np.array([n.emit_rate for n in network.nodes])
    kout[kout == 0] = 1

    for i1,i2 in zip(*np.triu_indices(num_nodes,1)):
      ko1, ko2 = kout[i1], kout[i2]
      kf1, kf2 = K_fret[i1,i2], K_fret[i2,i1]
      if kf1 > 0 and kf2 > 0:  w = (max(kf1/ko1, kf2/ko2))**(1/6)
      elif kf1 > 0:           w = (kf1/ko1)**(1/6)
      elif kf2 > 0:           w = (kf2/ko2)**(1/6)
      else:                   continue

      network_nx.add_edge(node_names[i1], node_names[i2], weight=2*(w**3*(w<1) + w*(w>=1)), spring_weight=w)

    node_angles = [np.pi-i/(num_nodes) for i in range(num_nodes)]
    node_pos = {n: (np.cos(theta), np.sin(theta)) for n,theta in zip(node_names, node_angles)}

    return network_nx, node_pos

  def _calc_network_node_pos_circular(self):
    node_angles = [np.pi-i/(self.num_nodes) for i in range(self.num_nodes)]
    circular_pos = {n: (np.cos(theta), np.sin(theta)) for n,theta in zip(self._node_names, node_angles)}
    return circular_pos
  def _calc_network_node_pos_spring(self):
    node_pos = self._calc_network_node_pos_circular()
    spring_pos = nx.drawing.layout.spring_layout(self._network_nx, pos=node_pos, fixed=None, weight='spring_weight')
    return spring_pos
    

  # Network properties
  @property
  def network(self):  return self._network

  @property
  def num_nodes(self):  return len(self._nodes)
  @property
  def nodes(self):  return self._nodes
  @property
  def node_names(self):  return self._node_names

  @property
  def num_nodes_singlerail(self):  return len(self._nodes_sr)
  @property
  def nodes_singlerail(self):  return self._nodes_sr
  @property
  def node_names_singlerail(self):  return self._node_names_sr

  @property
  def input_magnitude(self): return self._input_magnitude
  @input_magnitude.setter
  def input_magnitude(self, v):
    self._input_magnitude = v

  @property
  def output_magnitude(self):  return self._output_magnitude

  def _node_idx(self, node):
    return self._node_indices[node]
  def _node_idx_sr(self, node_sr):
    return self._node_indices_sr[node_sr]


  # Image properties
  @property
  def num_pixels(self):  return len(self._pixels)
  @property
  def pixels(self):  return self._pixels[:]

  def pixel_nodes(pixel):
    return self._pixel_to_node_map[pixel][:]

  @property
  def image_rows(self):  return self._img_rows
  @property
  def image_cols(self):  return self._img_cols

  def _pixel_idx(self, pixel):
    return self._pixel_indices[pixel]


  # Network methods
  def set_node_input(self, node_sr, k_in):
    # update node k_in
    input_node = self._input_nodes[self._node_idx_sr(node_sr)]
    self._network_input[self._node_idx(input_node)] = k_in

    # update corresponding pixel value
    pixel_idx, _ = self._pixel_idx(self._node_sr_to_pixel_map[node_sr])
    node_sr_pos, node_sr_neg = self._pixel_to_node_sr_map[self._node_sr_to_pixel_map[node]]
    self._image_input[pixel_idx] = (self.node_input(node_sr_pos) - self.node_input(node_sr_neg))/self.input_magnitude

    # indicate that network output and image output are out-of-date
    self._network_output = None
    self._image_output = None
  def node_input(self, node_sr):
    node_idx_sr = self._node_idx_sr(node_sr)
    node_idx = self._node_idx(self._input_nodes[node_idx_sr])
    if self._network_input[node_idx] == True:
      return self.input_magnitude
    else:
      return self._network_input[node_idx]
  def node_output(self, node_sr):
    if self._network_output is None:
      self._compute_network_output()
    out_node = self._output_nodes[self._node_idx_sr(node_sr)]
    return self._network_output[self._node_idx(out_node)]

  def _compute_network_output(self):
    # Update network inputs
    input_node_idxs = [self._node_idx(in_node) for in_node in self._input_nodes]
    k_in = self._network_input[input_node_idxs]
    k_in[k_in==True] = self._input_magnitude
    self._network.set_k_in(k_in)

    # Use analysis module to compute network output
    node_outputs = analyze.node_outputs(self._network)
    self._network_output = [node_outputs[n] for n in self._nodes]

    self._image_output = [0]*self.num_pixels
    for px_idx, px in enumerate(self._pixels):
      nodep_sr, noden_sr = self._pixel_to_node_sr_map[px]
      nodep_sr_idx, noden_sr_idx = self._node_idx_sr(nodep_sr), self._node_idx_sr(noden_sr)
      nodep, noden = self._output_nodes[nodep_sr_idx], self._output_nodes[noden_sr_idx]
      nodep_idx, noden_idx = self._node_idx(nodep), self._node_idx(noden)
      self._image_output[px_idx] = (self._network_output[nodep_idx] - self._network_output[noden_idx])/self.output_magnitude

  # Image methods
  def set_pixel_input(self, pixel, value):
    pixel_idx = self._pixel_idx(pixel)
    node_pos_sr, node_neg_sr = self._pixel_to_node_sr_map[pixel]
    node_pos_sr_idx, node_neg_sr_idx = self._node_idx_sr(node_pos_sr), self._node_idx_sr(node_neg_sr)
    in_node_pos, in_node_neg = self._input_nodes[node_pos_sr_idx], self._input_nodes[node_neg_sr_idx]

    self._network_input[self._node_idx(in_node_pos)] = max(0, value*self.input_magnitude)
    self._network_input[self._node_idx(in_node_neg)] = max(0, -value*self.input_magnitude)
    self._image_input[pixel_idx] = value

    self._network_output = None
    self._image_output = None
  def pixel_input(self, pixel):
    return self._image_input[self._pixel_idx(pixel)]
  def pixel_output(self, pixel):
    if self._image_output is None:
      self._compute_network_output()
    return self._image_output[self._pixel_idx(pixel)]

  # Plotting methods
  def display(self):
    self._figure, [[self._ax_img_in, self._ax_img_out], [self._ax_net_in, self._ax_net_out]] = plt.subplots(2,2)
    self._ax_net_in.axis('equal')
    self._ax_net_out.axis('equal')

    self.update_plot()
    self._setup_event_handlers()
    
  def update_plot(self):
    mat_img_in, mat_img_out, vals_net_in, vals_net_out = self._collect_plot_data()

    # Plot input and output images
    self._ax_img_in_cbar = self._draw_image(mat_img_in, self._ax_img_in, self._ax_img_in_cbar)
    self._ax_img_out_cbar = self._draw_image(mat_img_out, self._ax_img_out, self._ax_img_out_cbar)

    # Plot input and ouptut networks
    self._ax_net_in_cbar = self._draw_network(vals_net_in/self.input_magnitude, self._ax_net_in, self._ax_net_in_cbar)
    self._ax_net_out_cbar = self._draw_network(vals_net_out/self.output_magnitude, self._ax_net_out, self._ax_net_out_cbar)

    # Update figure
    self._figure.canvas.draw()
    self._figure.canvas.flush_events()
    
  def set_network_plot_mode(self, mode = 'spring'):
    if mode == 'spring':
      self._network_nx_node_pos = self._calc_network_node_pos_spring()
      self.update_plot()
    elif mode == 'circular':
      self._network_nx_node_pos = self._calc_network_node_pos_circular()
      self.update_plot()
    else:
      print(f'WARNING: Unknown plot mode {mode}')

  def set_network_activation_mode(self, mode = 'output'):
    self._network_activation_mode = mode
    self.update_plot()

  def set_image_plot_mode(self, mode = 'threshold'):
    self._image_plot_mode = mode
    self.update_plot()

  def _collect_plot_data(self):
    if self._network_output is None or self._image_output is None:
      self._compute_network_output()

    nr,nc = self.image_rows, self.image_cols

    vals_img_in = list(map(self.pixel_input, self._pixels))
    mat_img_in = np.concatenate((vals_img_in, np.nan*np.ones(nr*nc - len(vals_img_in)))).reshape((nr,nc))

    vals_img_out = list(map(self.pixel_output, self._pixels))
    mat_img_out = np.concatenate((vals_img_out, np.nan*np.ones(nr*nc - len(vals_img_in)))).reshape((nr,nc))

    vals_net_in = np.zeros(self.num_nodes)
    for n,v in zip(self._input_nodes, map(self.node_input, self._nodes_sr)):
      vals_net_in[self._node_idx(n)] = v

    vals_net_out = np.array(self._network_output)

    return mat_img_in, mat_img_out, vals_net_in, vals_net_out


  def _image_plot_cmap(self):
    if self._image_plot_mode == 'threshold':
      cmap = matplotlib.colors.ListedColormap(['w','k'])
    else:
      cmap = cm.gray_r
    return cmap

     
  def _draw_image(self, img_data, ax, ax_cbar = None):
    ax.clear()
    mappable = ax.matshow(img_data, vmin = -1, vmax = 1, cmap = self._image_plot_cmap())
    ax.set_xticks(np.arange(self.image_cols))
    ax.set_yticks(np.arange(self.image_rows))

    if ax_cbar is None:
      ax_cbar = self._figure.colorbar(mappable, ax=ax)
    else:
      ax_cbar.update_normal(mappable)

    return ax_cbar

  def _draw_network(self, node_values, ax, ax_cbar = None):
    vmin, vmax = 0.0, max(node_values)
    vmax = vmax if vmax > 0 else 1
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mappable = cm.ScalarMappable(norm = norm, cmap=cm.gray_r)

    node_colors = [[1 - norm(np.where(v<vmin, vmin, v))]*3 for v in node_values]

    _,_,edge_weights = zip(*self._network_nx.edges.data('weight'))

    ax.clear()
    nx.draw_networkx(self._network_nx, pos=self._network_nx_node_pos, ax=ax, with_labels=True, node_color=node_colors, node_size=400, font_color=(.5,.5,.5), font_weight='bold', edgecolors='k', width=edge_weights, font_size = 6)

    if ax_cbar is None:
      ax_cbar = self._figure.colorbar(mappable, ax=ax)
    else:
      ax_cbar.update_normal(mappable)

    return ax_cbar


##################################
# t-SNE ANALYSIS VISUALIZATION
##################################

def plot_dualrail_tsne(num_nodes, num_pixels, kfret, koff, data_tsne, colors):
  num_pts = len(kfret)
  node_names = [f'{i+1}{pm}' for i in range(num_pixels) for pm in '+-']

  fig, axes = plt.subplots(1,3)
  scatter_ax, detail_ax, network_ax = axes

  scatter = scatter_ax.scatter(data_tsne[:,0], data_tsne[:,1], marker='.', c=colors, picker=True, ec=None)

  # make components for detail axes
  detail_ax.set_xlim(-.5, num_nodes-.5)
  detail_ax.set_ylim(num_nodes-.5, -.5)
  detail_ax.set_xticks(np.arange(num_nodes))
  detail_ax.set_yticks(np.arange(num_nodes))
  detail_ax.set_xticklabels(node_names)
  detail_ax.set_yticklabels(node_names)

  # make components for network axes
  network_ax.clear()
  network_ax.axis('square')

  network = nx.Graph()
  node_angles = [np.pi-(i*3+isneg) * 2*np.pi/(3*num_pixels) for i in range(num_pixels) for isneg in [0,1]]
  node_pos = {n: (np.cos(theta), np.sin(theta)) for n,theta in zip(node_names, node_angles)}
  edges = [(node_names[i1], node_names[i2]) for i1,i2 in zip(*np.triu_indices(num_nodes,1))]
  network.add_nodes_from(node_names)
  network.add_edges_from(edges)

  def identify_node_clusters(nodes, edge_weights, threshold = .5):
    #weights_dict = {(a,b): w for i1,i2,w in zip(*np.triu_indices(len(nodes),1), edge_weights) for a,b in [(nodes[i1],nodes[i2]), (nodes[i2],nodes[i1])]}

    nodes_left = set(nodes)
    clusters = []
    while len(nodes_left) > 0:
      node = nodes_left.pop()
      
      cur_cluster = {node}
      to_process = {node}
      while len(to_process) > 0:
        new_node = to_process.pop()
        for n in set(nodes_left):
          if edge_weights[(n, new_node)] >= threshold:
            to_process.add(n)
            cur_cluster.add(n)
            nodes_left.remove(n)
      clusters.append(cur_cluster)
    return clusters

  def reorder_mat(mat, rows, cols):
    node_idxs = {n:i for i,n in enumerate(node_names)}

    mat_reorder = np.zeros((num_nodes, num_nodes))
    for i,j in it.product(range(len(rows)), range(len(cols))):
      mat_reorder[i,j] = mat[node_idxs[rows[i]], node_idxs[cols[j]]]

    return mat_reorder
      
           
  def show_point_details(idx):
#    data_sel = data[idx, :]
    kfret_sel = kfret[idx]
    koff_sel = koff[idx]
    
    # highlight selected point
    scatter.set_edgecolors([(0,0,0,0)]*idx + ['k'] + [(0,0,0,0)]*(num_pts-idx-1))

    # plot on detail axes each parameter set
    mat_rows = [f'{n+1}{pm}' for pm in '-+' for n in range(num_pixels)]
    mat_cols = [f'{n+1}{pm}' for pm in '+-' for n in range(num_pixels)]
    mat_raw = kfret_sel / (kfret_sel + np.broadcast_to(koff_sel.reshape(num_nodes, 1), (num_nodes, num_nodes)))
    mat = reorder_mat(mat_raw, mat_rows, mat_cols)
#    mat = data_to_matrix(data_sel, mat_rows, mat_cols)
    detail_ax.clear()
    detail_ax.matshow(mat, vmin=0, vmax=1, origin='upper', cmap='gray_r')
    detail_ax.set_xticks(np.arange(num_nodes))
    detail_ax.set_yticks(np.arange(num_nodes))
    detail_ax.set_xticklabels(mat_cols)
    detail_ax.set_yticklabels(mat_rows)

    # plot on network axes the network for the first point in event.ind (could hcange to closest point)
    edge_weights_mat = kfret_sel / (kfret_sel + .1*np.broadcast_to(koff_sel.reshape(num_nodes, 1), (num_nodes, num_nodes)))
    edge_weights_mat += edge_weights_mat.T
    edge_weights_lst = edge_weights_mat[np.triu_indices(num_nodes,1)]
    edge_weights = {(node_names[i1], node_names[i2]): edge_weights_mat[i1,i2] for i1,i2 in it.permutations(range(num_nodes),2)}
    node_clusters = identify_node_clusters(node_names, edge_weights)
    colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    node_color_dict = {n: color for cluster, color in zip(node_clusters, colors) for n in cluster}
    node_color = [node_color_dict[n] for n in node_names]
    for i1,i2 in zip(*np.triu_indices(num_nodes,1)):
      network.add_edge(node_names[i1], node_names[i2], weight=(edge_weights_mat[i1,i2]))

    network_ax.clear()
#    nx.draw_networkx(network, pos = node_pos, ax = network_ax, with_labels=True, node_color=node_color, node_size=400, width=data_sel*4)
    spring_pos = nx.drawing.layout.spring_layout(network, pos=node_pos, fixed=['1+'])
    nx.draw_networkx(network, pos = spring_pos, ax = network_ax, with_labels=True, node_color=node_color, node_size=400, width=edge_weights_lst*2)
#    nx.draw_networkx(network, ax = network_ax, with_labels=True, node_color='#CCC', node_size=400, width=data_sel*4)

    fig.canvas.draw()
    fig.canvas.flush_events()


  def scatter_pick_handler(event):
    #if event.artist != scatter:  return

    clickx, clicky = event.mouseevent.xdata, event.mouseevent.ydata

    sel_idx = event.ind[np.argmin(np.hypot(data_tsne[event.ind,0]-clickx, data_tsne[event.ind,1]-clicky))]
    show_point_details(sel_idx)

  fig.canvas.mpl_connect('pick_event', scatter_pick_handler)
