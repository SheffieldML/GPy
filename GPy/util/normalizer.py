'''
Created on Aug 27, 2014

@author: t-mazwie
'''
import logging
import numpy as np

class Norm(object):
    def __init__(self):
        pass
    def scale_by(self, Y):
        """
        Use data matrix Y as normalization space to work in.
        """
        raise NotImplementedError
    def normalize(self, Y):
        """
        Project Y into normalized space
        """
        raise NotImplementedError
    def inverse_mean(self, X):
        """
        Project the normalized object X into space of Y
        """
        raise NotImplementedError
    def inverse_variance(self, var):
        return var
    def scaled(self):
        """
        Whether this Norm object has been initialized.
        """
        raise NotImplementedError
class MeanNorm(Norm):
    def __init__(self):
        self.mean = None
    def scale_by(self, Y):
        Y = np.ma.masked_invalid(Y, copy=False)
        self.mean = Y.mean(0).view(np.ndarray)
    def normalize(self, Y):
        return Y-self.mean
    def inverse_mean(self, X):
        return X+self.mean
    def scaled(self):
        return self.mean is not None
