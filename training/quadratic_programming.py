import random
import itertools as it

import numpy as np
import numpy.linalg
import qpsolvers

def optimize_FRET(k_in, k_out, training_data):
  # Solves for FRET rate constant matrix K using QP
  #
  # k_in is a length-n 1D numpy array, storing the intrinsic excitation rates
  #   [currently ignored because it is overwritten by each x input training point]
  # k_out is a length-n 1D numpy array, storing the intrinsic decay rates
  # training_data is a list of tuples (x,y), where each x,y is an input-output pair
  #   x is a length-n 1D numpy array, storing the intrinsic excitation rates for this training data point
  #   y is a length-n 1D numpy array, storing the desired output of each node given input x

  N = len(k_out)

  # set up QP problem with constraints
  # QP problem takes the form:
  #   Minimize x^T P x + qx 
  #   subject to Ax = b
  #              Gx <= h
  # where we will be optimizing vec(B), B = diag(k_out) @ A^-1
  P = np.kron(sum(x @ x.T for x,_ in training_data), np.eye(N))
  q = -2 * sum(y.T @ np.kron(x.T, np.eye(N)) for x,y in training_data).T

  n_ineq_constraints = N*(N-1)//2
  QP_G = np.zeros((n_ineq_constraints, N**2))
  for row, (i,j) in enumerate(it.combinations(range(N), 2)):
    idx_ii = i*N + i # index of element b_ii within vec(B)
    idx_ij = j*N + i # index of element b_ij within vec(B)
    QP_G[row, idx_ii] = -1.
    QP_G[row, idx_ij] = 1.
  QP_h = np.zeros((n_ineq_constraints,))

  n_eq_constraints = N*(N-1)//2
  QP_A = np.zeros((n_eq_constraints, N**2))
  for row, (i,j) in enumerate(it.combinations(range(N), 2)):
    idx_ij = j*N + i # index of element b_ij within vec(B)
    idx_ji = i*N + j # index of element b_ji within vec(B)
    QP_A[row, idx_ij] = k_out[j]
    QP_A[row, idx_ji] = -k_out[i]
  QP_b = np.zeros((n_eq_constraints, ))

  print(P.shape, q.shape, QP_A.shape, QP_b.shape, np.zeros((N**2,1)).shape)
#  sol = qpsolvers.solve_qp(P, q=q, A=sym_constraints_coef, b=sym_constraints_RHS, lb=np.zeros((N**2,1)), solver='osqp')
  sol = qpsolvers.solve_qp(P, q=q, 
      G=QP_G, h=QP_h,
      A=QP_A, b=QP_b, 
      lb=np.zeros((N**2,)),
      solver='osqp')

  B = sol.reshape((N, N), order='F')
  print(B)
  print(numpy.linalg.inv(np.diag(k_out.flatten())) @ B)
  A = numpy.linalg.inv(B) @ np.diag(k_out.flatten())
  
  K = -A
  for i in range(N):
    K[i,i] = 0

  print(B)
  print(A)
  print(K)

  return K

def simulate_network(k_in, k_out, K):
  N = len(k_in)
  A = -K
  for i in range(N):
    A[i,i] = k_in[i] + k_out[i] + K[:,i].sum()

  p_ss = numpy.linalg.inv(A) @ k_in
  o = p_ss * k_out

  return o

def generate_training_data(images, flip_rate = .1, repetition = 1):
  training_data = []
  for img in images:
    for i in range(repetition):
      flips = np.random.binomial(1,flip_rate, len(img))
      train_pt = img.copy()
      train_pt[flips] = 1 - train_pt[flips]
      training_data.append((train_pt, img))
  return training_data
    

def train_network(N, images):
  training_data = generate_training_data(images)

  k_out = .5 * np.ones((N,1))
  k_in = np.zeros((N, 1))
  K = optimize_FRET(k_in, k_out, training_data)

  return k_in, k_out, K

def test_network(k_in, k_out, K, images, iters = 10):
  N = len(k_in)

  loss = 0

  data = generate_training_data(images, repetition=iters)
  for x,y in data:
    out = simulate_network(k_in + x, k_out, K)
    loss += ((y - loss)**2).sum()

  return np.sqrt(loss / (len(data) * N))
  

def test_network_size(N, num_images, iters = 20):
  if 2**N < num_images:  return None

  results = []
  for i in range(iters):
    # generate random unique images (not sure if there's a better way to do this with high values of N)
    # this is done by generating distinct integers and then converting them to binary form
    img_nums = random.sample(range(2**N), num_images)
    images = [np.array(list(map(int, format(num, '0'+str(N)+'b')))).reshape((N,1)) for num in img_nums]

    k_in, k_out, K = train_network(N, images)
    loss = test_network(k_in, k_out, K, images)

    results.append({
      'loss': loss,
      'images': images,
      'k_in': k_in,
      'k_out': k_out,
      'K': K
    })

  return results

results = test_network_size(5, 5)
    


