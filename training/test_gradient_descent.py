import pytest
import numpy as np
import gradient_descent as gd 
import matplotlib.pyplot as plt

def test_choose():
    assert gd.choose(4,2) == 6
    assert gd.choose(10,0) == 1
    assert gd.choose(7,7) == 1
    assert gd.choose(1, 2) == 0
    assert type(gd.choose(1,1)) is int
    try: 
        gd.choose(1.1, 0)
        raise Exception()
    except AssertionError: 
        pass
    try: 
        gd.choose(-1, 0)
        raise Exception()
    except AssertionError: 
        pass

def test_off_patterns():
    iters = int(1e5)
    off_zeros = 0
    off_ones = 0
    for _ in range(iters):
        d = 10
        p_off = 0.1
        zeros = gd.off_patterns(np.zeros((d,1), dtype=int), p_off, 1)
        ones = gd.off_patterns(np.ones((d,1), dtype=int), p_off, 1)
        off_zeros += np.count_nonzero(zeros) / d
        off_ones += np.count_nonzero(ones) / d
    assert np.allclose([off_zeros, off_ones] / iters, [0.1, 0.9])


def test_forward_pass():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,1]]).T
    outs = np.full((2, 1), 0.5)
    pred = gd._forward_pass(rates, inps, outs)[1]
    assert np.allclose(pred, np.array([[2/3, 2/3]]).T)

def test_nll_gradient_small():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,0]]).T
    outs = np.full((2, 1), 0.5)
    
    Ainv, pred = gd._forward_pass(rates, inps, outs)
    grads = gd.gradient(gd.dnll, inps, pred, Ainv, outs, verbose=True)
    expected = {
        'Ainv': np.array([[0.54545455, 0.36363636],
            [0.36363636, 0.90909091]]), 
        'pred': np.array([[0.54545455],
            [0.36363636]]),
        'dL_dpred': np.array([[-1.83333333,  1.57142857]]), 
        'dpred_dAinv': np.array([[1., 0., 0., 0.],
            [0., 1., 0., 0.]]), 
        'dAinv_dA': -np.array([[0.29752066, 0.19834711, 0.19834711, 0.1322314 ],
            [0.19834711, 0.49586777, 0.1322314 , 0.33057851],
            [0.19834711, 0.1322314 , 0.49586777, 0.33057851],
            [0.1322314 , 0.33057851, 0.33057851, 0.82644628]]), 
        'dA_dK': np.array([[1.],
            [-1.],
            [-1.],
            [1.]]), 
        'reshaped_dL_dK': np.array([[ 0.        , 0.21645022],
            [0.21645022,  0.        ]])}
    for k in expected:
        assert np.allclose(grads[k], expected[k])



def test_training_update():
    assert True

if __name__ == '__main__':
    pass

#TODO take small step for single variable in direction of gradient