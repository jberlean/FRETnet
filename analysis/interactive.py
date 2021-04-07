import sys
import pathlib
import math
import copy
import itertools as it

import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

pkg_path = str(pathlib.Path(__file__).absolute().parent.parent)
if pkg_path not in sys.path:
  sys.path.append(pkg_path)
import analysis.analyze as analyze

##################################
# DUAL-RAIL NETWORK VISUALIZATION
##################################

## TODO:
##  - Allow toggling of node states directly
##  - Allow toggling of network edges

def test_dualrail_network(network, pixels = None, pixel_to_node_map = None, input_magnitude = 1, output_magnitude = 1, img_rows = None, img_cols = None):
  num_nodes = len(network.nodes)
  num_pixels = num_nodes // 2

  if pixels is None:
    pixels = list(map(str, range(1, num_pixels+1)))
  if pixel_to_node_map is None:
    pixel_to_node_map = {px: (network.nodes[2*i], network.nodes[2*i+1]) for i,px in enumerate(pixels)}

  plot = DualRailNetworkPlot(network, pixels, pixel_to_node_map, input_magnitude, output_magnitude, img_rows, img_cols)
  return plot

class DualRailNetworkPlot(object):
  def __init__(self, network, pixels, pixel_to_node_map = None, input_magnitude = 1, output_magnitude = 1, img_rows = None, img_cols = None):
    self._network = copy.deepcopy(network)
    self._nodes = self._network.nodes
    self._node_indices = {n:i for i,n in enumerate(self._nodes)}
    self._node_names = [n.name for n in self._nodes]
    self._input_magnitude = input_magnitude
    self._output_magnitude = output_magnitude

    self._pixels = pixels[:]
    self._pixel_indices = {p:i for i,p in enumerate(self._pixels)}
    if img_rows is None and img_cols is not None:
      self._img_rows = int(math.ceil(num_pixels / img_cols))
      self._img_cols = img_cols
    elif img_rows is not None and img_cols is None:
      self._img_rows = img_rows
      self._img_cols = int(math.ceil(num_pixels / img_rows))
    elif img_rows is None and img_cols is None:
      self._img_rows = int(math.ceil(math.sqrt(len(self._pixels))))
      self._img_cols = int(math.ceil(len(self._pixels)/self._img_rows))
    node_mapping = {n1: n2 for n1,n2 in zip(network.nodes, self._nodes)}
    self._pixel_to_node_map = {px: (node_mapping[npos], node_mapping[nneg]) for px, (npos, nneg) in pixel_to_node_map.items()}
    self._node_to_pixel_map = {n: px for px, (n1,n2) in pixel_to_node_map.items() for n in [n1,n2]}

    self._figure = None
    self._ax_img_in, self._ax_img_out = None, None
    self._ax_net_in, self._ax_net_out = None, None
    self._image_plot_mode = 'threshold'

    self._network_nx, self._network_nx_node_pos = self._init_network_nx(self._network)
    self._network_nx_node_pos = self._calc_network_node_pos_spring()

    # network inputs/outputs
    self._network_input = np.zeros(len(self._nodes))
    self._network_output = None
    self._image_input = np.zeros(len(self._pixels))
    self._image_output = None
    for px in self._pixels:  self.set_pixel_input(px, -1)

  def _setup_event_handlers(self):
    def click_handler(event):
      if event.inaxes == self._ax_img_in:
        # determine which pixel was clicked
        x,y = event.xdata, event.ydata
        row,col = int(round(y)), int(round(x))
        idx = row*self.image_cols + col
        new_state = -1 if self.pixel_input(self._pixels[idx])==1 else 1
        self.set_pixel_input(self._pixels[idx], new_state)
        self.update_plot()

    self._figure.canvas.mpl_connect('button_press_event', click_handler)

  def _init_network_nx(self, network):
    node_names = [n.name for n in network.nodes]
    num_nodes = len(node_names)

    network_nx = nx.Graph()
    network_nx.add_nodes_from(node_names)

    kfret_matrix = network.compute_kfret_matrix()
    for i1,i2 in zip(*np.triu_indices(num_nodes,1)):
      network_nx.add_edge(node_names[i1], node_names[i2], weight=4*kfret_matrix[i1,i2]/(100+kfret_matrix[i1,i2]), pring_weight=4*kfret_matrix[i1,i2]/(1+kfret_matrix[i1,i2]))

    node_angles = [np.pi-(i*3+isneg) * 2*np.pi/(3*self.num_pixels) for i in range(self.num_pixels) for isneg in [0,1]]
    node_pos = {n: (np.cos(theta), np.sin(theta)) for n,theta in zip(self._node_names, node_angles)}

    return network_nx, node_pos

  def _calc_network_node_pos_circular(self):
    node_angles = [np.pi-(i*3+isneg) * 2*np.pi/(3*self.num_pixels) for i in range(self.num_pixels) for isneg in [0,1]]
    circular_pos = {n: (np.cos(theta), np.sin(theta)) for n,theta in zip(self._node_names, node_angles)}
    return circular_pos
  def _calc_network_node_pos_spring(self):
    node_pos = self._calc_network_node_pos_circular()
    spring_pos = nx.drawing.layout.spring_layout(self._network_nx, pos=node_pos, fixed=['1+'], weight='spring_weight')
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
  def input_magnitude(self): return self._input_magnitude
  @input_magnitude.setter
  def input_magnitude(self, v):
    self._input_magnitude = v

  @property
  def output_magnitude(self):  return self._output_magnitude

  def _node_idx(self, node):
    return self._node_indices[node]


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
  def set_node_input(self, node, k_in):
    # update node k_in
    self._network_input[self._node_idx(node)] = k_in

    # update corresponding pixel value
    pixel_idx = self._pixel_idx(self._node_to_pixel_map[node])
    node_pos, node_neg = self._pixel_to_node_map[self._node_to_pixel_map[node]]
    self._image_input[pixel_idx] = (self.node_input(node_pos) - self.node_input(node_neg))/self.input_magnitude

    # indicate that network output and image output are out-of-date
    self._network_output = None
    self._image_output = None
  def node_input(self, node):
    node_idx = self._node_idx(node)
    if self._network_input[node_idx] == True:
      return self.input_magnitude
    else:
      return self._network_input[self._node_idx(node)]
  def node_output(self, node):
    if self._network_output is None:
      self._compute_network_output()
    return self._network_output[self._node_idx(node)]

  def _compute_network_output(self):
    # Update network inputs
    for node, in_rate in zip(self._nodes, self._network_input):
      if in_rate == True:  node.production_rate = self._input_magnitude
      else:  node.production_rate = in_rate

    # Use analysis module to compute network output
    node_outputs = analyze.node_outputs(self._network, hanlike=True)
    self._network_output = [node_outputs[n] for n in self._nodes]
    self._image_output = [0]*self.num_pixels
    for px_idx, px in enumerate(self._pixels):
      nodep, noden = self._pixel_to_node_map[px]
      nodep_idx, noden_idx = self._node_idx(nodep), self._node_idx(noden)
      self._image_output[px_idx] = (self._network_output[nodep_idx] - self._network_output[noden_idx])/self.output_magnitude

  # Image methods
  def set_pixel_input(self, pixel, value):
    pixel_idx = self._pixel_idx(pixel)
    node_pos, node_neg = self._pixel_to_node_map[pixel]
    node_pos_idx, node_neg_idx = self._node_idx(node_pos), self._node_idx(node_neg)

    self._network_input[node_pos_idx] = max(0, value*self.input_magnitude)
    self._network_input[node_neg_idx] = max(0, -value*self.input_magnitude)
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
    self._draw_image(mat_img_in, self._ax_img_in)
    self._draw_image(mat_img_out, self._ax_img_out)

    # Plot input and ouptut networks
    self._draw_network(vals_net_in/self.input_magnitude, self._ax_net_in)
    self._draw_network(vals_net_out/self.output_magnitude, self._ax_net_out)

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

  def set_image_plot_mode(self, mode = 'threshold'):
    self._image_plot_mode = mode
    self.update_plot()

  def _collect_plot_data(self):
    vals_img_in = list(map(self.pixel_input, self._pixels))
    mat_img_in = np.array(vals_img_in).reshape((self.image_rows, self.image_cols))

    vals_img_out = list(map(self.pixel_output, self._pixels))
    mat_img_out = np.array(vals_img_out).reshape((self.image_rows, self.image_cols))

    vals_net_in = np.array(list(map(self.node_input, self._nodes)))
    vals_net_out = np.array(list(map(self.node_output, self._nodes)))

    return mat_img_in, mat_img_out, vals_net_in, vals_net_out


  def _draw_image(self, img_data, ax):
    if self._image_plot_mode == 'threshold':
      vmin, vmax = -1e-3,1e-3
    else:
      vmin, vmax = -1, 1

    ax.clear()
    ax.matshow(img_data, vmin = vmin, vmax = vmax, cmap = 'gray_r')
    ax.set_xticks(np.arange(self.image_cols))
    ax.set_yticks(np.arange(self.image_rows))

  def _draw_network(self, node_values, ax):
    node_colors = [[1-max(min(v,1),0)]*3 for v in node_values]

    _,_,edge_weights = zip(*self._network_nx.edges.data('weight'))
#    edge_weights = 4*np.array(edge_weights)

    ax.clear()
    nx.draw_networkx(self._network_nx, pos=self._network_nx_node_pos, ax=ax, with_labels=True, node_color=node_colors, node_size=400, font_color=(.5,.5,.5), font_weight='bold', edgecolors='k', width=edge_weights)


##################################
# t-SNE ANALYSIS VISUALIZATION
##################################

def plot_dualrail_tsne(num_nodes, num_pixels, data, data_tsne, colors):
  num_pts, num_params = data.shape
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
    weights_dict = {(a,b): w for i1,i2,w in zip(*np.triu_indices(len(nodes),1), edge_weights) for a,b in [(nodes[i1],nodes[i2]), (nodes[i2],nodes[i1])]}

    nodes_left = set(nodes)
    clusters = []
    while len(nodes_left) > 0:
      node = nodes_left.pop()
      
      cur_cluster = {node}
      to_process = {node}
      while len(to_process) > 0:
        new_node = to_process.pop()
        for n in set(nodes_left):
          if weights_dict[(n, new_node)] >= threshold:
            to_process.add(n)
            cur_cluster.add(n)
            nodes_left.remove(n)
      clusters.append(cur_cluster)
    return clusters

  def data_to_matrix(data, rows, cols):
    node_idxs = {n:i for i,n in enumerate(node_names)}

    mat = np.zeros((num_nodes, num_nodes))
    mat[np.triu_indices(num_nodes, 1)] = data
    mat = mat + mat.T

    mat_reorder = np.zeros((num_nodes, num_nodes))
    for i,j in it.product(range(len(rows)), range(len(cols))):
      mat_reorder[i,j] = mat[node_idxs[rows[i]], node_idxs[cols[j]]]

    return mat_reorder
      
           
  def show_point_details(idx):
    data_sel = data[idx, :]
    
    # highlight selected point
    scatter.set_edgecolors([(0,0,0,0)]*idx + ['k'] + [(0,0,0,0)]*(num_pts-idx-1))

    # plot on detail axes each parameter set
    mat_rows = [f'{n+1}{pm}' for pm in '-+' for n in range(num_pixels)]
    mat_cols = [f'{n+1}{pm}' for pm in '+-' for n in range(num_pixels)]
    mat = data_to_matrix(data_sel, mat_rows, mat_cols)
    detail_ax.clear()
    detail_ax.matshow(mat, vmin=0, vmax=1, origin='upper', cmap='gray_r')
    detail_ax.set_xticks(np.arange(num_nodes))
    detail_ax.set_yticks(np.arange(num_nodes))
    detail_ax.set_xticklabels(mat_cols)
    detail_ax.set_yticklabels(mat_rows)

    # plot on network axes the network for the first point in event.ind (could hcange to closest point)
    node_clusters = identify_node_clusters(node_names, data_sel)
    colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    node_color_dict = {n: color for cluster, color in zip(node_clusters, colors) for n in cluster}
    node_color = [node_color_dict[n] for n in node_names]
    for i1,i2,w in zip(*np.triu_indices(num_nodes,1), data_sel):
      network.add_edge(node_names[i1], node_names[i2], weight=np.sqrt(w))

    network_ax.clear()
#    nx.draw_networkx(network, pos = node_pos, ax = network_ax, with_labels=True, node_color=node_color, node_size=400, width=data_sel*4)
    spring_pos = nx.drawing.layout.spring_layout(network, pos=node_pos, fixed=['1+'])
    nx.draw_networkx(network, pos = spring_pos, ax = network_ax, with_labels=True, node_color=node_color, node_size=400, width=data_sel*4)
#    nx.draw_networkx(network, ax = network_ax, with_labels=True, node_color='#CCC', node_size=400, width=data_sel*4)

    fig.canvas.draw()
    fig.canvas.flush_events()


  def scatter_pick_handler(event):
    #if event.artist != scatter:  return

    clickx, clicky = event.mouseevent.xdata, event.mouseevent.ydata

    sel_idx = event.ind[np.argmin(np.hypot(data_tsne[event.ind,0]-clickx, data_tsne[event.ind,1]-clicky))]
    show_point_details(sel_idx)

  fig.canvas.mpl_connect('pick_event', scatter_pick_handler)
