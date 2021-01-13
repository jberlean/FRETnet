import pytest
import numpy as np
import gradient_descent as gd 
import matplotlib.pyplot as plt
import scipy.optimize as so

####################
# HELPER FUNCTIONS #
####################

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
    iters = int(1e4)
    off_zeros = 0
    off_ones = 0
    for _ in range(iters):
        d = 10
        p_off = 0.1
        zeros = gd.off_patterns(np.zeros((d,1), dtype=int), p_off, 1)
        ones = gd.off_patterns(np.ones((d,1), dtype=int), p_off, 1)
        off_zeros += np.count_nonzero(zeros) / d
        off_ones += np.count_nonzero(ones) / d
    assert np.allclose([x / iters for x in (off_zeros, off_ones)], [0.1, 0.9], rtol=2e-2, atol = 1e-3)


##################
# LOSS FUNCTIONS #
##################

def test_nll():
    pass

def test_dnll():
    pass

def test_rmse():
    pass

def test_drmse():
    pass


def test_gradient_nll_small():
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

def test_forward_pass_small():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,1]]).T
    outs = np.full((2, 1), 0.5)
    pred = gd._forward_pass(rates, inps, outs)[1]
    assert np.allclose(pred, np.array([[2/3, 2/3]]).T)

def gradtest_random_helper(loss, loss_grad): 
    # checks the gradient at the [1,0] node (had trouble trying to check all nodes).
    for d in range(2,10):
        K = np.random.rand(d, d) / 10 + 0.45  # weights initialized at 0.5 +/- 0.05
        K = (K + K.T) / 2  # ensure starting weights are symmetric
        np.fill_diagonal(K, 0) # and diagonal terms are zero

        inps = np.random.randint(2, size=(d, 1))
        outs = np.full((d, 1), 0.5)

        def func(x): 
            rates = K.copy()
            rates[1,0] = x[0]
            rates[0,1] = x[0]
            _, pred = gd._forward_pass(rates, inps, outs)
            return loss(inps, pred)
        
        def grad(x):
            rates = K.copy()
            rates[1,0] = x[0]
            rates[0,1] = x[0]
            Ainv, pred = gd._forward_pass(rates, inps, outs)
            grad_array = gd.gradient(loss_grad, inps, pred, Ainv, outs)
            return grad_array

# def grad(x, lg):
#     rates = K.copy()
#     rates[1,0] = x[0]
#     rates[0,1] = x[0]
#     Ainv, pred = _forward_pass(rates, inps, outs)
#     grad_array = gradient(lg, inps, pred, Ainv, outs)
#     return grad_array

# K = np.array([[0., 1],[1, 0]])
# inps = np.array([[1.,0]]).T
# outs = np.full((2, 1), 0.5)

        # def func(x0): 
        #     _, pred = gd._forward_pass(x0, inps, outs)
        #     return gd.nll(inps, pred)

        # def grad(x0):
        #     Ainv, pred = gd._forward_pass(x0, inps, outs)
        #     grad_array = gd.gradient(gd.dnll, inps, pred, Ainv, outs)
        #     return grad_array
        
        # epsilon = np.full((d,d), 1e-8)
        # np.fill_diagonal(epsilon, 0)

        epsilon = 1e-10
        testval = np.random.rand()
        numgrad = so.approx_fprime([testval], func, epsilon)
        print(f'numgrad: \n{numgrad}')

        angrad = grad([testval])[1,0]
        print(f'angrad: \n{angrad}')

        mse = np.mean((numgrad - angrad)**2)
        assert mse < 1e-8

def test_gradient_nll_random():
    gradtest_random_helper(gd.nll, gd.dnll)

def test_gradient_mse_random():
    gradtest_random_helper(gd.rmse, gd.drmse)

def test_training_update():
    pass

if __name__ == '__main__':
    pass

#TODO take small step for single variable in direction of gradient