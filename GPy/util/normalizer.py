'''
Created on Aug 27, 2014

@author: Max Zwiessele
'''
import logging
import numpy as np

class _Norm(object):
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
        if not self.scaled():
            raise AttributeError("Norm object not initialized yet, try calling scale_by(data) first.")
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

class Standardize(_Norm):
    def __init__(self):
        self.mean = None
    def scale_by(self, Y):
        Y = np.ma.masked_invalid(Y, copy=False)
        self.mean = Y.mean(0).view(np.ndarray)
        self.std = Y.std(0).view(np.ndarray)
    def normalize(self, Y):
        super(Standardize, self).normalize(Y)
        return (Y-self.mean)/self.std
    def inverse_mean(self, X):
        return (X*self.std)+self.mean
    def inverse_variance(self, var):
        return (var*(self.std**2))
    def scaled(self):
        return self.mean is not None

# Inverse variance to be implemented, disabling for now
# If someone in the future want to implement this,
# we need to implement the inverse variance for 
# normalization. This means, we need to know the factor
# for the variance to be multiplied to the variance in
# normalized space. This is easy to compute for standardization
# (see above) but gets tricky here.
# class Normalize(_Norm):
#     def __init__(self):
#         self.ymin = None
#         self.ymax = None
#     def scale_by(self, Y):
#         Y = np.ma.masked_invalid(Y, copy=False)
#         self.ymin = Y.min(0).view(np.ndarray)
#         self.ymax = Y.max(0).view(np.ndarray)
#     def normalize(self, Y):
#         super(Normalize, self).normalize(Y)
#         return (Y - self.ymin) / (self.ymax - self.ymin) - .5
#     def inverse_mean(self, X):
#         return (X + .5) * (self.ymax - self.ymin) + self.ymin
#     def inverse_variance(self, var):
#         
#         return (var*(self.std**2))
#     def scaled(self):
#         return (self.ymin is not None) and (self.ymax is not None)