# Copyright (c) 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
from ..core import Param

class MLPext(Mapping):
    """
    Mapping based on a multi-layer perceptron neural network model, with multiple hidden layers. Activation function
    is applied to all hidden layers. The output is a linear combination of the last layer features, i.e. the
    last layer is linear.
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dims=[3], prior=None, activation='tanh', name='mlpmap'):

        """
        :param input_dim: number of input dimensions
        :param output_dim: number of output dimensions
        :param hidden_dims: list of hidden sizes of hidden layers
        :param prior: variance of Gaussian prior on all variables. If None, no prior is used (default: None)
        :param activation: choose activation function. Allowed values are 'tanh' and 'sigmoid'
        :param name:
        """
        super(MLPext, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        assert activation in ['tanh', 'sigmoid', 'relu'], NotImplementedError('Only tanh, relu and sigmoid activations'
                                                                              'are implemented')
        self.hidden_dims = hidden_dims
        self.W_list = list()
        self.b_list = list()
        for i in np.arange(len(hidden_dims) + 1):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            out_dim = output_dim if i == len(hidden_dims) else hidden_dims[i]
            self.W_list.append(Param('W%d'%i, np.random.randn(in_dim, out_dim)))
            self.b_list.append(Param('b%d'%i, np.random.randn(out_dim)))

        if prior is not None:
            for W, b in zip(self.W_list, self.b_list):
                W.set_prior(Gaussian(0, prior))
                b.set_prior(Gaussian(0, prior))

        self.link_parameters(*self.W_list)
        self.link_parameters(*self.b_list)

        if activation == 'tanh':
            self.act = np.tanh
            self.grad_act = lambda x: 1. / np.square(np.cosh(x))

        elif activation == 'sigmoid':
            from scipy.special import expit
            from scipy.stats import logistic
            self.act = expit
            self.grad_act = logistic._pdf

        elif activation == 'relu':
            self.act = lambda x: x * (x > 0)
            self.grad_act = lambda x: 1. * (x > 0)

    def f(self, X):
        net = X
        for W, b, i in zip(self.W_list, self.b_list, np.arange(len(self.W_list))):
            net = np.dot(net, W)
            net = net + b
            if i < len(self.W_list)-1:
                # Don't apply nonlinearity to last layer outputs
                net = self.act(net)
        return net

    def _f_preactivations(self, X):
        """Computes the network preactivations, i.e. the results of all intermediate linear layers before applying the
        activation function on them
        :param X: input data
        :return: list of preactivations [X, XW+b, f(XW+b)W+b, ...]
        """

        preactivations_list = list()
        net = X
        preactivations_list.append(X)

        for W, b, i in zip(self.W_list, self.b_list, np.arange(len(self.W_list))):
            net = np.dot(net, W)
            net = net + b
            if i < len(self.W_list) - 1:
                preactivations_list.append(net)
                net = self.act(net)
        return preactivations_list

    def update_gradients(self, dL_dF, X):
        preactivations_list = self._f_preactivations(X)
        d_dact = dL_dF
        d_dlayer = d_dact
        for W, b, preactivation, i in zip(reversed(self.W_list), reversed(self.b_list), reversed(preactivations_list),
                                          reversed(np.arange(len(self.W_list)))):
            if i > 0:
                # Apply activation function to linear preactivations to get input from previous layer
                # (except for first layer where input is X)
                activation = self.act(preactivation)
            else:
                activation = preactivation
            W.gradient = np.dot(activation.T, d_dlayer)
            b.gradient = np.sum(d_dlayer, 0)

            if i > 0:
                # Don't need this computation if we are at the bottom layer
                d_dact = np.dot(d_dlayer, W.T)
                # d_dlayer = d_dact / np.square(np.cosh(preactivation))
                d_dlayer = d_dact * self.grad_act(preactivation)

    def fix_parameters(self):
        """Helper function that fixes all parameters"""
        for W, b in zip(self.W_list, self.b_list):
            W.fix()
            b.fix()

    def unfix_parameters(self):
        """Helper function that unfixes all parameters"""
        for W, b in zip(self.W_list, self.b_list):
            W.unfix()
            b.unfix()

    def gradients_X(self, dL_dF, X):
        preactivations_list = self._f_preactivations(X)
        d_dact = dL_dF
        d_dlayer = d_dact
        for W, preactivation, i in zip(reversed(self.W_list), reversed(preactivations_list),
                                       reversed(np.arange(len(self.W_list)))):

            # Backpropagation through hidden layer.
            d_dact = np.dot(d_dlayer, W.T)
            d_dlayer = d_dact * self.grad_act(preactivation)

        return d_dact
