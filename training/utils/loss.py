import numpy as np


class LossFunc():
    def fn(pat, pred):
        pass
    def grad(pat, pred):
        pass

class RMSE(LossFunc):
    @staticmethod
    def fn(pat, pred):
        return np.mean((pat - pred) ** 2) ** 0.5
    @classmethod
    def grad(cls, pat, pred):
        return ( 1/cls.fn(pat, pred) * 1/len(pat) * (pred - pat) ).T
        
class NLL(LossFunc):
    @staticmethod
    def fn(pat, pred):
        total = 0
        for i in range(pat.size):
            if pat[i,0] == 1:
                total -= np.log(pred[i,0]) 
            elif pat[i,0] == 0:
                total -= np.log(1-pred[i,0])
            else: 
                raise ValueError()
        return total
    @staticmethod
    def grad(pat, pred):
        out = np.zeros(pat.T.shape)
        for i in range(pat.size):
            if pat[i,0] == 1:
                out[0,i] = -1/pred[i,0]
            elif pat[i,0] == 0:
                out[0,i] = 1/(1-pred[i,0])
            else:
                raise ValueError()
        return out


def nll(pat, pred):
    """
    Negative Log Loss, evaluated elementwise and then summed over two arrays of the same shape.
    """
    pass


def rmse(pat, pred):
    """
    Root Mean Square Error function.
    """

def drmse(pat, pred):
    """
    Analytical gradient of RMSE, elementwise.
    """