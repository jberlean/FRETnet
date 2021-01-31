import numpy as np

def choose(a, b):
    """
    A short helper function to calculate binomial coefficients.
    """
    assert int(a) + int(b) == a + b, f'Choose only takes ints, not {a} and {b}'
    assert a >= 0 and b >= 0, f'Choose only takes nonnegative ints, not {a} and {b}'
    if a < b:
        return 0
    prod = 1
    faster_b = min(b, a-b)
    for i in range(faster_b):
        prod *= (a-i) / (i+1)
    return round(prod)

def off_patterns(pat, p_off, num_pats):
    """
    Returns all patterns which are off from pat by a proportion 
    specified by p_off, with a minimum of 1 element changed.

    Args:
        pat (np.array): Binary column vector of length d.
        p_off (float in [0,1]): Probability of each node being corrupted.
        num_pats (int): Number of corrupted patterns to return.

    Returns:
        len(pat) x num_pats 2darray (dtype=float) with corrupted patterns as columns.
    """
    d = len(pat)

    corrupt_mask = np.random.default_rng().random((d, num_pats))
    out = np.tile(pat, num_pats)

    out[corrupt_mask < p_off] = out[corrupt_mask < p_off] ^ 1
    return out

def Ainv_from_rates(K_fret, k_in, k_out):
    """
    Performs a 'forward pass' by analytically finding the steady-state probability
    that each node is excited.

    Args:
        rate_matrix (np.array): The weights (rate constants, arbitrary units) between nodes in the system.
            Should be square and symmetric.
        train_pattern (np.array): The (binary) pixel pattern to be trained on.
            Should be a column.
        output_rates (np.array): The intrinsic output rate constants of each node. 
            Should be a column.
    
    Returns:
        Ainv (np.array): The inverse of the matrix representing the linear system of equations.
            Should be square.
        pred (np.array): The predicted values of each node's output.
    """
    num_nodes = len(k_in)
    A = -K_fret
    diagonal_terms = K_fret.sum(axis=1) + k_in.T + k_out.T

    if K_fret[range(num_nodes), range(num_nodes)].any():
        raise ValueError(f'diagonal terms not 0 in the rate matrix: \n {K_fret}')

    A[range(num_nodes), range(num_nodes)] = diagonal_terms

    try:
        ret = np.linalg.inv(A)
        return ret
    except:
        raise ValueError(f'Singular matrix for rates={K_fret}, inputs={k_in}, outputs={k_out}')

def k_in_from_input_data(input_data):
    k_in_sr = np.array([
        kin for bit in input_data for kin in (max(bit, 0), -min(bit, 0))
    ])
    return k_in_sr