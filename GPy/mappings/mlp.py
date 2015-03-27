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
        super(MLP).__init__(self, input_dim=input_dim, output_dim=output_dim, name=name)
        self.hidden_dim = hidden_dim
        self.W1 = Param('W1', np.random.randn(self.input_dim, self.hidden_dim))
        self.b1 = Param('b1', np.random.randn(self.hidden_dim))
        self.W2 = Param('W2', np.random.randn(self.hidden_dim, self.output_dim))
        self.b2 = Param('b2', np.random.randn(self.output_dim))


    def f(self, X):
        N, D = X.shape
        activations = np.tanh(np.dot(X,self.W1) + self.b1)
        self.out = np.dot(self.activations,self.W2) + self.b2
        return self.output_fn(self.out)

    def update_gradients(self, dL_dF, X):
        activations = np.tanh(np.dot(X,self.W1) + self.b1)


        #Evaluate second-layer gradients.
        self.W2.gradient = np.dot(activations.T, dL_dF)
        self.b2.gradient = np.sum(dL_dF, 0)

        # Backpropagation to hidden layer.
        delta_hid = np.dot(dL_dF, self.W2.T) * (1.0 - activations**2)

        # Finally, evaluate the first-layer gradients.
        self.W1.gradients = np.dot(X.T,delta_hid)
        self.b1.gradients = np.sum(delta_hid, 0)

