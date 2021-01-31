import time
import sys,os

import numpy as np

# INTRAPACKAGE IMPORTS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
from objects import utils

x_ON = sys.argv[1].upper() == 'ON'
y_ON = sys.argv[2].upper() == 'ON'
print('x: {}, y: {}'.format(x_ON, y_ON))
   
eps = 100
k_slow = 100
x = utils.InputNode('input1', production_rate = eps**5*k_slow*x_ON, emit_rate = 0.0)
y = utils.InputNode('input2', production_rate = eps**5*k_slow*y_ON, emit_rate = 0.0)
h = utils.Node('middle', emit_rate = 0.0)
w = utils.Node('waste', emit_rate = k_slow*eps)
#w2 = utils.Node('waste2', emit_rate = k_slow)
z = utils.Node('output', emit_rate = k_slow*eps**5)
h.add_input(x, k_slow)
h.add_input(y, k_slow)
w.add_input(h, k_slow*eps**4)
#w2.add_input(w, k_slow*eps)
z.add_input(h, k_slow*eps**2)
nodes = [x,y,h,w,z]

network = utils.Network(nodes)

def check_output():
  t = network._time
  cts = network._stats.get(('emit', z, '*'), 0)
  o = None if t == 0 else cts/t
  print('{}cts/s ({}cts/{}s)'.format(o, cts, t))

while True:
  network.step_forward()

  if network._stats.get('steps',0) % 10000 == 0:
    check_output()
