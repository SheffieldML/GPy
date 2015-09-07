# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import numpy as np
import GPy

class RegressionMethod(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.preprocess = True
        
    def _preprocess(self, data,  train):
        """Zero-mean, unit-variance normalization by default"""
        if train:
            inputs, labels = data
            self.data_mean = inputs.mean(axis=0)
            self.data_std = inputs.std(axis=0)
            self.labels_mean = labels.mean(axis=0)
            self.labels_std = labels.std(axis=0)
            return ((inputs-self.data_mean)/self.data_std, (labels-self.labels_mean)/self.labels_std)
        else:
            return (data-self.data_mean)/self.data_std
    
    def _reverse_trans_labels(self, labels):
        return labels*self.labels_std+self.labels_mean
        
    def fit(self, train_data):
        if self.preprocess:
            train_data = self._preprocess(train_data, True)
        return self._fit(train_data)
    
    def predict(self, test_data):
        if self.preprocess:
            test_data = self._preprocess(test_data, False)
        labels = self._predict(test_data)
        if self.preprocess:
            labels = self._reverse_trans_labels(labels)
        return labels
    
    @abc.abstractmethod
    def _fit(self, train_data):
        """Fit the model. Return True if successful"""
        return True
    
    @abc.abstractmethod
    def _predict(self, test_data):
        """Predict on test data"""
        return None
    
class GP_RBF(RegressionMethod):
    name = 'GP_RBF'
    
    def _fit(self, train_data):
        inputs, labels = train_data
        self.model = GPy.models.GPRegression(inputs, labels,kernel=GPy.kern.RBF(inputs.shape[-1],ARD=True) +GPy.kern.Linear(inputs.shape[1], ARD=True)   )
        self.model.likelihood.variance[:] = labels.var()*0.01
        self.model.optimize()
        return True
    
    def _predict(self, test_data):
        return self.model.predict(test_data)[0]
    
class SparseGP_RBF(RegressionMethod):
    name = 'SparseGP_RBF'
    
    def _fit(self, train_data):
        inputs, labels = train_data
        self.model = GPy.models.SparseGPRegression(inputs, labels,kernel=GPy.kern.RBF(inputs.shape[-1],ARD=True) +GPy.kern.Linear(inputs.shape[1], ARD=True) ,num_inducing=100)
        self.model.likelihood.variance[:] = labels.var()*0.01
        self.model.optimize()
        return True
    
    def _predict(self, test_data):
        return self.model.predict(test_data)[0]
    
# class MRD_RBF(RegressionMethod):
#     name = 'MRD_RBF'
#     
#     def _fit(self, train_data):
#         inputs, labels = train_data
#         Q = 5
#         self.model = GPy.models.MRD([inputs, labels],Q,kernel=GPy.kern.RBF(Q,ARD=True),num_inducing=50)
#         self.model.Y0.likelihood.variance[:] = inputs.var()*0.01
#         self.model.Y1.likelihood.variance[:] = labels.var()*0.01
#         self.model.optimize()
#         return True
#     
#     def _predict(self, test_data):
#         return self.model.predict(self.model.Y0.infer_newX(test_data)[0])[0]

class SVIGP_RBF(RegressionMethod):
    name = 'SVIGP_RBF'
    
    def _fit(self, train_data):
        X, Y = train_data
        
        Z = X[np.random.permutation(X.shape[0])[:100]]
        k = GPy.kern.RBF(X.shape[1], ARD=True) + GPy.kern.Linear(X.shape[1], ARD=True) + GPy.kern.White(X.shape[1],0.01) 

        lik = GPy.likelihoods.StudentT(deg_free=3.)
        self.model = GPy.core.SVGP(X, Y, Z=Z, kernel=k, likelihood=lik)
        [self.model.optimize('scg', max_iters=40, gtol=0, messages=0, xtol=0, ftol=0) for i in range(10)]
        self.model.optimize('bfgs', max_iters=1000, gtol=0, messages=0)
        return True
    
    def _predict(self, test_data):
        return self.model.predict(test_data)[0]    
