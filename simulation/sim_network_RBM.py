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

if __name__ == '__main__':
  num_nodes = 8
  rate_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    ], dtype=float)
  input_rates = np.array([1, 0, 0, 0, 0, 0, 0, 0])
  output_rates = np.full(num_nodes, 0.1)

  # num_nodes = 3
  # rate_matrix = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
  # input_rates = np.array([1,0,0])
  # output_rates = np.full(num_nodes, 0.2)

  # k_{si} = p_i(\sum k_{ij} k_{si} + k_{is}) - \sum k_{ij}p_j
  solve_matrix = -rate_matrix
  diagonal_terms = rate_matrix.sum(axis=1) + input_rates + output_rates
  assert((solve_matrix[range(num_nodes), range(num_nodes)] == np.zeros(num_nodes)).all())
  solve_matrix[range(num_nodes), range(num_nodes)] = diagonal_terms

  # solve for p_i, the probability of being activated (unitless)
  sol = np.linalg.solve(solve_matrix, input_rates)
  print(sol)
  # convert to activation (inverse time units) by dividing by lifetime 
  # (multiplying by total output rate)
  activation = sol * ((rate_matrix * (1-sol)).sum(axis=1) + output_rates)

  sim = generate_RBM_network(num_nodes, rate_matrix, input_rates, output_rates)
  sim.mainloop()

# probability for input [1,0,0,0,0,0,0,0] and uniform output 0.1 at t = 853946.3856082442:
# [0.6771959229616344, 0.551819744808054, 0.5531842930873935, 0.0, 0.0, 0.0, 0.5154575631719575, 0.0]
# expected sol:
# [0.7576571735626008, 0.537345513164965, 0.537345513164965,  0.0, 0.0, 0.0, 0.5910800644814616, 0.0]
#
# activation for input [1,1,0,0,0,0,0,0] and uniform output 0.1 at t = 459167.0102827745:
# [0.28231341109557057, 0.2823461114906782, 0.23703862405593498, 0.0, 0.0, 0.0, 0.562701858932474, 0.0]
# expected activation:
# [0.27885418,          0.27885418,         0.23141426,          0. , 0. , 0. , 0.55074444,        0. ]
# 
# activation for input [1,1,1,0,0,0,0,0] and uniform output 0.1 at t = 136629.58987318014:
# [0.2198215132971919, 0.21636637806019463, 0.21827740418685243, 0.0, 0.0, 0.0, 0.3877319128898517, 0.0]
# expected activation:
# [0.21664597,         0.21664597,          0.21664597,          0. , 0.  ,0. , 0.38498064,         0. ]
