# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping

class MLP(Mapping):
    """
    Mapping based on a multi-layer perceptron neural network model.

    .. math::

       f(\\mathbf{x}*) = \\mathbf{W}^0\\boldsymbol{\\phi}(\\mathbf{W}^1\\mathbf{x}+\\mathbf{b}^1)^* + \\mathbf{b}^0

    where

    .. math::

      \\phi(\\cdot) = \\text{tanh}(\\cdot)

    :param X: input observations
    :type X: ndarray
    :param output_dim: dimension of output.
    :type output_dim: int
    :param hidden_dim: dimension of hidden layer. If it is an int, there is one hidden layer of the given dimension. If it is a list of ints there are as manny hidden layers as the length of the list, each with the given number of hidden nodes in it.
    :type hidden_dim: int or list of ints. 

    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=3):
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim)
        self.name = 'mlp'
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim
        self.activation = [None]*len(self.hidden_dim)
        self.W = []
        self._dL_dW = []
        self.bias = []
        self._dL_dbias = []
        self.W.append(np.zeros((self.input_dim, self.hidden_dim[0])))
        self._dL_dW.append(np.zeros((self.input_dim, self.hidden_dim[0])))
        self.bias.append(np.zeros(self.hidden_dim[0]))
        self._dL_dbias.append(np.zeros(self.hidden_dim[0]))
        self.num_params = self.hidden_dim[0]*(self.input_dim+1)
        for h1, h0 in zip(hidden_dim[1:], hidden_dim[0:-1]):
            self.W.append(np.zeros((h0, h1)))
            self._dL_dW.append(np.zeros((h0, h1)))
            self.bias.append(np.zeros(h1))
            self._dL_dbias.append(np.zeros(h1))
            self.num_params += h1*(h0+1)
        self.W.append(np.zeros((self.hidden_dim[-1], self.output_dim)))
        self._dL_dW.append(np.zeros((self.hidden_dim[-1], self.output_dim)))
        self.bias.append(np.zeros(self.output_dim))
        self._dL_dbias.append(np.zeros(self.output_dim))
        self.num_params += self.output_dim*(self.hidden_dim[-1]+1)
        self.randomize()

    def _get_param_names(self):
        return sum([['W%i_%i_%i' % (i, n, d)  for n in range(self.W[i].shape[0]) for d in range(self.W[i].shape[1])] + ['bias%i_%i' % (i, d) for d in range(self.W[i].shape[1])] for i in range(len(self.W))], [])

    def _get_params(self):
        param = np.array([])
        for W, bias in zip(self.W, self.bias):
            param = np.hstack((param, W.flatten(), bias))
        return param
    
    def _set_params(self, x):
        start = 0
        for W, bias in zip(self.W, self.bias):
            end = W.shape[0]*W.shape[1]+start
            W[:] = x[start:end].reshape(W.shape[0], W.shape[1]).copy()
            start = end
            end = W.shape[1]+end
            bias[:] = x[start:end].copy()
            start = end

    def randomize(self):
        for W, bias in zip(self.W, self.bias):
            W[:] = np.random.randn(W.shape[0], W.shape[1])/np.sqrt(W.shape[0]+1)
            bias[:] = np.random.randn(W.shape[1])/np.sqrt(W.shape[0]+1)

    def f(self, X):
        self._f_computations(X)
        return np.dot(np.tanh(self.activation[-1]), self.W[-1]) + self.bias[-1]

    def _f_computations(self, X):
        W = self.W[0]
        bias = self.bias[0]
        self.activation[0] = np.dot(X,W) + bias
        for W, bias, index in zip(self.W[1:-1], self.bias[1:-1], range(1, len(self.activation))):
            self.activation[index] = np.dot(np.tanh(self.activation[index-1]), W)+bias

    def df_dtheta(self, dL_df, X):
        self._df_computations(dL_df, X)
        g = np.array([])
        for gW, gbias in zip(self._dL_dW, self._dL_dbias):
            g = np.hstack((g, gW.flatten(), gbias))
        return g

    def _df_computations(self, dL_df, X):
        self._f_computations(X)
        a0 = self.activation[-1]
        W = self.W[-1]
        self._dL_dW[-1] = (dL_df[:, :, None]*np.tanh(a0[:, None, :])).sum(0).T
        dL_dta=(dL_df[:, None, :]*W[None, :, :]).sum(2)
        self._dL_dbias[-1] = (dL_df.sum(0))
        for dL_dW, dL_dbias, W, bias, a0, a1 in zip(self._dL_dW[-2:0:-1],
                                                    self._dL_dbias[-2:0:-1],
                                                    self.W[-2:0:-1],
                                                    self.bias[-2:0:-1],
                                                    self.activation[-2::-1],
                                                    self.activation[-1:0:-1]):
            ta = np.tanh(a1)
            dL_da = dL_dta*(1-ta*ta)
            dL_dW[:] = (dL_da[:, :, None]*np.tanh(a0[:, None, :])).sum(0).T
            dL_dbias[:] = (dL_da.sum(0))
            dL_dta = (dL_da[:, None, :]*W[None, :, :]).sum(2)
        ta = np.tanh(self.activation[0])
        dL_da = dL_dta*(1-ta*ta)
        W = self.W[0]
        self._dL_dW[0] = (dL_da[:, :, None]*X[:, None, :]).sum(0).T
        self._dL_dbias[0] = (dL_da.sum(0))
        self._dL_dX = (dL_da[:, None, :]*W[None, :, :]).sum(2)

        
    def df_dX(self, dL_df, X):
        self._df_computations(dL_df, X)
        return self._dL_dX
    
