# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
from ..core import Param

class MLP(Mapping):
    """
    Mapping based on a multi-layer perceptron neural network model, with a single hidden layer
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=3, name='mlpmap'):
        super(MLP, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        self.hidden_dim = hidden_dim
        self.W1 = Param('W1', np.random.randn(self.input_dim, self.hidden_dim))
        self.b1 = Param('b1', np.random.randn(self.hidden_dim))
        self.W2 = Param('W2', np.random.randn(self.hidden_dim, self.output_dim))
        self.b2 = Param('b2', np.random.randn(self.output_dim))
        self.link_parameters(self.W1, self.b1, self.W2, self.b2)


    def f(self, X):
        layer1 = np.dot(X, self.W1) + self.b1
        activations = np.tanh(layer1)
        return  np.dot(activations, self.W2) + self.b2

    def update_gradients(self, dL_dF, X):
        layer1 = np.dot(X,self.W1) + self.b1
        activations = np.tanh(layer1)

        #Evaluate second-layer gradients.
        self.W2.gradient = np.dot(activations.T, dL_dF)
        self.b2.gradient = np.sum(dL_dF, 0)

        # Backpropagation to hidden layer.
        dL_dact = np.dot(dL_dF, self.W2.T)
        dL_dlayer1 = dL_dact * (1 - np.power(activations, 2))

        # Finally, evaluate the first-layer gradients.
        self.W1.gradient = np.dot(X.T,dL_dlayer1)
        self.b1.gradient = np.sum(dL_dlayer1, 0)

    def gradients_X(self, dL_dF, X):
        layer1 = np.dot(X,self.W1) + self.b1
        activations = np.tanh(layer1)

        # Backpropagation to hidden layer.
        dL_dact = np.dot(dL_dF, self.W2.T)
        dL_dlayer1 = dL_dact * (1 - np.power(activations, 2))

        return np.dot(dL_dlayer1, self.W1.T)



