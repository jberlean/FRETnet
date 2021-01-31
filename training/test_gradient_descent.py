import pytest
import sys, os
import numpy as np
import scipy.optimize as so

# INTRAPACKAGE IMPORTS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to python path
import training.gradient_descent as srgd
from training.utils.loss import RMSE, NLL
from training.utils.helpers import Ainv_from_rates

####################
# HELPER FUNCTIONS #
####################

def test_choose():
    assert srgd.choose(4,2) == 6
    assert srgd.choose(10,0) == 1
    assert srgd.choose(7,7) == 1
    assert srgd.choose(1, 2) == 0
    assert type(srgd.choose(1,1)) is int
    try: 
        srgd.choose(1.1, 0)
        raise Exception()
    except AssertionError: 
        pass
    try: 
        srgd.choose(-1, 0)
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
        zeros = srgd.off_patterns(np.zeros((d,1), dtype=int), p_off, 1)
        ones = srgd.off_patterns(np.ones((d,1), dtype=int), p_off, 1)
        off_zeros += np.count_nonzero(zeros) / d
        off_ones += np.count_nonzero(ones) / d
    assert np.allclose([x / iters for x in (off_zeros, off_ones)], [0.1, 0.9], rtol=2e-2, atol = 1e-3)


#################
# GRADIENT CALC #
#################

def test_forward_pass_small():
    rates = np.array([[0., 1],[1, 0]])
    inps = np.array([[1.,1]]).T
    outs = np.full((2, 1), 0.5)
    Ainv = srgd.Ainv_from_rates(rates, inps, outs)
    pred = Ainv @ inps
    assert np.allclose(pred, np.array([[2/3, 2/3]]).T)

def gradtest_random_helper(loss): 
    # checks the gradient at the [1,0] node (had trouble trying to check all nodes).
    for d in range(2,10):
        K = np.random.randn(d, d) / 10 + 0.5  # weights initialized normally around 0.5
        K = (K + K.T) / 2  # ensure starting weights are symmetric
        np.fill_diagonal(K, 0) # and diagonal terms are zero
        inps = np.random.randint(2, size=(d, 1))
        outs = np.full((d, 1), 0.5)

        def func(x): 
            rates = K.copy()
            rates[1,0] = x[0]
            rates[0,1] = x[0]
            Ainv = Ainv_from_rates(rates, inps, outs)
            pred = Ainv @ inps
            return loss.fn(inps, pred)
        
        def grad(x):
            rates = K.copy()
            rates[1,0] = x
            rates[0,1] = x
            Ainv = Ainv_from_rates(rates, inps, outs)
            pred = Ainv @ inps
            grad_array = srgd.gradient(loss, inps, pred, Ainv)
            return grad_array

        epsilon = 1e-8
        testval = np.random.rand()
        numgrad = so.approx_fprime([testval], func, epsilon)
        angrad = grad(testval)[1,0]


        mse = np.mean((numgrad - angrad)**2)
        assert mse < 1e-8
def test_gradient_nll_random():
    gradtest_random_helper(NLL)

def test_gradient_rmse_random():
    gradtest_random_helper(RMSE)



def test_training_update():
    pass

if __name__ == '__main__':
    pass
