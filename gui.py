import time

import tkinter as tk

import utils

class Simulator(tk.Frame):
  def __init__(self, master=None, nodes = [], node_pos = []):
    super().__init__(master)

    self.grid(sticky=tk.N+tk.S+tk.E+tk.W)

    # make application resize properly
    top = self.winfo_toplevel()
    top.rowconfigure(0, weight=1)
    top.columnconfigure(0, weight=1)
    self.rowconfigure(0, weight=1)
    self.columnconfigure(0, weight=1)

    self.display = NetworkCanvas(self, nodes = nodes, node_pos = node_pos)
    self.display.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

    self.bind_all('<space>', lambda e: self.autorun_toggle())
    self.bind_all('<Key-Return>', lambda e: self.run_once())
    self.bind_all('<Key-Left>', lambda e: self.autorun_slowdown())
    self.bind_all('<Key-Right>', lambda e: self.autorun_speedup())

    self.autorun = False
    self.autorun_speed = 10 # ideal number of steps per second
    self.autorun_delay = 100 # in ms (tk time)
    self.autorun_batchsize = 1 # number of steps to do each autorun cycle
    self.autorun_start = None
    self.autorun_step_count = 0
    self.update_delay = .08 # in seconds (Py time)
    self.update_time = None

    self._nodes = nodes
    self._network = utils.Network(nodes)

    self.display.update()

  def autorun_toggle(self):
    if self.autorun:
      print('Stopping autorun...')
      self.autorun = False
    else:
      print('Starting autorun...')
      self.autorun = True
      self.start_autorun_timer()
  def autorun_slowdown(self):
    self.autorun_speed /= 1.25
    self._update_autorun_params()
    print('Speed: {:.2f} steps/sec ({}steps/{}ms)'.format(self.autorun_speed, self.autorun_batchsize, self.autorun_delay))
  def autorun_speedup(self):
    self.autorun_speed *= 1.25
    self._update_autorun_params()
    print('Speed: {:.2f} steps/sec ({}steps/{}ms)'.format(self.autorun_speed, self.autorun_batchsize, self.autorun_delay))
  def _update_autorun_params(self):
    if self.autorun_speed < 20:
      self.autorun_delay = int(1000. / self.autorun_speed)
      self.autorun_batchsize = 1
    else:
      self.autorun_delay = 50
      self.autorun_batchsize = int(.05*self.autorun_speed + .5)
  def start_autorun_timer(self):
    def func():
      if not self.autorun:  return

      delay = self.autorun_delay
      steps = self.autorun_batchsize

      for _ in range(steps):
        self.step_forward()
      self.autorun_step_count += steps

      if self.update_time is None or time.time() - self.update_time > self.update_delay:
        self.display.update(force_all = True)
        steps_per_sec = self.autorun_step_count / (time.time() - self.autorun_start)
        self.winfo_toplevel().title('{} steps/s'.format(steps_per_sec))
        self.update_time = time.time()

      self.after(delay, func)

    self.autorun_step_count = 0
    self.autorun_start = time.time()
    self.after(int(self.autorun_delay), func)

  def run_once(self):
    if self.autorun:  return
    self.step_forward()
    self.display.update()
    
  def step_forward(self):
    _, _, changed_nodes = self._network.step_forward()

    for node in changed_nodes:
      self.display.mark_dirty(node)

    

class NetworkCanvas(tk.Canvas):
  def __init__(self, 
      master=None, 
      nodes = [], 
      node_pos = [], 
      node_size = 50, 
      node_color_inactive = '#ccc', 
      node_color_active = '#444', 
      edge_color_inactive = '#787878',
      edge_color_active = '#000',
      **kwargs
  ):
    super().__init__(master, **kwargs)

    self._nodes = nodes
    self._node_pos_dict = dict(zip(nodes, node_pos))
    self._gui_params = {
        'node_size': node_size, 
        'node_color_inactive': node_color_inactive, 
        'node_color_active': node_color_active,
        'node_label_color_inactive': node_color_active,
        'node_label_color_active': node_color_inactive,
        'edge_color_inactive': edge_color_inactive,
        'edge_color_active': edge_color_active,
    }

    self._obj_to_gui_element = {}

    self.active = set()
    self.dirty = set()

    self._init_gui()

  def _init_gui(self):
    node_pos_dict = self._node_pos_dict
    for n in self._nodes:
      pos = node_pos_dict[n]
      self._init_node(n, pos)

      weight_norm = (sum(e.rate for e in n.out_edges) + n.decay_rate + n.emit_rate)/25.
      for e in n.out_edges:
        n2 = e.output
        pos2 = node_pos_dict[n2]
        weight = e.rate/weight_norm
        self._init_edge(e, pos, pos2,weight)

    # formatting stuff
    self.itemconfigure('node_inactive', fill=self._gui_params['node_color_inactive'])
    self.itemconfigure('node_active', fill=self._gui_params['node_color_active'])
    self.itemconfigure('edge_inactive', fill=self._gui_params['edge_color_inactive'])
    self.tag_lower('edge')

  def _init_node(self, node, pos):
    node_size = self._gui_params['node_size']
    rad = node_size/2
    x0,y0 = int(pos[0]-rad), int(pos[1]-rad)
    x1,y1 = x0+node_size, y0+node_size
    tags = ('element', 'node')
    if node.is_on():  tags += ('node_active',)
    else:  tags += ('node_inactive',)

    elem = self.create_oval(x0,y0,x1,y1, fill='#ccc', tags=tags)
    self._obj_to_gui_element[node] = elem

    lbl_tags = tags + ('{}.label'.format(elem),)
    self.create_text(*pos, text='', tags=lbl_tags)
  def _init_edge(self, e, pos0, pos1, weight):
    x0,y0 = pos0
    x1,y1 = pos1
    elem = self.create_line(x0,y0,x1,y1, width=weight, tags=('element','edge','edge_inactive'))
    self._obj_to_gui_element[e] = elem

  def mark_active(self, obj):
    self.active.add(obj)
    self.mark_dirty(obj)
  def mark_inactive(self, obj = None):
    if obj is None:
      self.active = set()
      [self.mark_dirty(o) for o in self.active]
    else:
      self.active.remove(obj)
      self.mark_dirty(obj)
  def mark_dirty(self, obj):
    self.dirty.add(obj)

  def update(self, force_all = False):
    if force_all:
      self.dirty = set(self._obj_to_gui_element.keys())
    self.update_dirty()
  def update_dirty(self):
    for obj in self.dirty:
      if isinstance(obj, utils.Node):
        self.update_node(obj)
      elif isinstance(obj, utils.Edge):
        self.update_edge(obj)
  def update_node(self, node):
    node_size = self._gui_params['node_size']
    rad = node_size/2
    pos = self._node_pos_dict[node]

    x0,y0 = int(pos[0]-rad), int(pos[1]-rad)
    x1,y1 = x0+node_size, y0+node_size
    fill = self._gui_params['node_color_active'] if node.is_on() else self._gui_params['node_color_inactive']

    gui_elem = self._obj_to_gui_element[node]

    self.coords(gui_elem, x0, y0, x1, y1)
    if node.is_on():
      self.dtag(gui_elem, 'node_inactive')
      self.addtag_withtag('node_active', gui_elem)
    else:
      self.dtag(gui_elem, 'node_active')
      self.addtag_withtag('node_inactive', gui_elem)
    self.itemconfigure(gui_elem, fill=fill)

    # formatting for node label
    lbl_elem = '{}.label'.format(gui_elem)
    lbl_fill = self._gui_params['node_label_color_active'] if node.is_on() else self._gui_params['node_label_color_inactive']
    lbl_text = '{:.4f}'.format(self.master._network.activation(node))
    self.coords(lbl_elem, *pos)
    self.itemconfigure(lbl_elem, fill=lbl_fill, text=lbl_text)
      
  def update_edge(self, edge):
    x0,y0 = self._node_pos_dict[edge.input]
    x1,y1 = self._node_pos_dict[edge.output]

    active = edge in self.active
    fill = self._gui_params['edge_color_active'] if active else self._gui_params['edge_color_inactive']

    gui_elem = self._obj_to_gui_element[edge]

    self.coords(gui_elem, x0, y0, x1, y1)
    if active:
      self.dtag(gui_elem, 'edge_inactive')
      self.addtag_withtag('edge_active', gui_elem)
    else:
      self.dtag(gui_elem, 'edge_active')
      self.addtag_withtag('edge_inactive', gui_elem)
    self.itemconfigure(gui_elem, fill=fill)

