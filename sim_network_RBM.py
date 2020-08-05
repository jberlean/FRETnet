import sys
import itertools as it

import numpy as np

import utils
import gui

def generate_RBM_network(num_nodes, rate_matrix, input_rates, output_rates):
  rad = num_nodes*20
  center_x = rad + 50
  center_y = rad + 50
  node_pos = [
      (rad*np.cos(n/num_nodes*2*np.pi) + center_x, rad*np.sin(n/num_nodes*2*np.pi) + center_y)
      for n in range(num_nodes)
  ]

  nodes = [
      utils.InputNode('in{}'.format(n), production_rate=k_in, emit_rate=k_out)
      for n,(k_in,k_out) in enumerate(zip(input_rates, output_rates))
  ]

  # to be properly undirected, make sure the rate matrix is symmetric
  for i,j in it.product(range(num_nodes), range(num_nodes)):
    n1,n2 = nodes[i],nodes[j]
    n1.add_input(n2, rate_matrix[i,j])

  sim = gui.Simulator(nodes = nodes, node_pos = node_pos)
  return sim

num_nodes = 8
rate_matrix = np.random.uniform(0,1,(num_nodes, num_nodes))
input_rates = np.random.uniform(0,1,num_nodes)
output_rates = np.random.uniform(0,1,num_nodes)

sim = generate_RBM_network(num_nodes, rate_matrix, input_rates, output_rates)
sim.after(100, sim.quit)
sim.mainloop()
