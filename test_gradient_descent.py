import pytest
import numpy as np
import gradient_descent as gd 
import matplotlib.pyplot as plt

def test_forward_pass():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,1]]).T
    outs = np.full((2, 1), 0.5)
    pred = gd.forward_pass(rates, inps, outs)[1]
    assert np.allclose(pred, np.array([[2/3, 2/3]]).T)

def test_gradient():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,0]]).T
    outs = np.full((2, 1), 0.5)
    # pat = np.array([[1., 0]])
    Ainv, pred = gd.forward_pass(rates, inps, outs)
    grads = gd.gradient(inps, pred, Ainv, outs, verbose=True)
    expected = {
        'Ainv': np.array([[0.54545455, 0.36363636],
            [0.36363636, 0.90909091]]), 
        'pred': np.array([[0.54545455],
            [0.36363636]]), 
        'dNLL_dpred': np.array([[-1.83333333,  1.57142857]]), 
        'dpred_dAinv': np.array([[1., 0., 0., 0.],
            [0., 1., 0., 0.]]), 
        'dAinv_dA': np.array([[0.29752066, 0.19834711, 0.19834711, 0.1322314 ],
            [0.19834711, 0.49586777, 0.1322314 , 0.33057851],
            [0.19834711, 0.1322314 , 0.49586777, 0.33057851],
            [0.1322314 , 0.33057851, 0.33057851, 0.82644628]]), 
        'dA_dK': np.array([[ 0., -0.,  1., -0.],
            [-0., -1., -0., -0.],
            [-0., -0., -1., -0.],
            [-0.,  1., -0.,  0.]]), 
        'dNLL_dAinv': np.array([[-1.83333333,  1.57142857,  0.        ,  0.        ]]), 
        'dNLL_dA': np.array([[-0.23376623,  0.41558442, -0.15584416,  0.27705628]]), 
        'dNLL_dK': np.array([[ 0.        , -0.07792208],
            [-0.13852814,  0.        ]])}
    for k in expected:
        assert np.allclose(grads[k], expected[k])

def test_training_update():
    assert True

if __name__ == '__main__':
    pass

#TODO take small step for single variable in direction of gradient