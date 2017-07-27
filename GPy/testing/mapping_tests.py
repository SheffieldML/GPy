# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class MappingGradChecker(GPy.core.Model):
    """
    This class has everything we need to check the gradient of a mapping. It
    implement a simple likelihood which is a weighted sum of the outputs of the
    mapping. the gradients are checked against the parameters of the mapping
    and the input.
    """
    def __init__(self, mapping, X, name='map_grad_check'):
        super(MappingGradChecker, self).__init__(name)
        self.mapping = mapping
        self.link_parameter(self.mapping)
        self.X = GPy.core.Param('X',X)
        self.link_parameter(self.X)
        self.dL_dY = np.random.randn(self.X.shape[0], self.mapping.output_dim)
    def log_likelihood(self):
        return np.sum(self.mapping.f(self.X) * self.dL_dY)
    def parameters_changed(self):
        self.X.gradient = self.mapping.gradients_X(self.dL_dY, self.X)
        self.mapping.update_gradients(self.dL_dY, self.X)


class MappingTests(unittest.TestCase):

    def test_kernelmapping(self):
        X = np.random.randn(100,3)
        Z = np.random.randn(10,3)
        mapping = GPy.mappings.Kernel(3, 2, Z, GPy.kern.RBF(3))
        self.assertTrue(MappingGradChecker(mapping, X).checkgrad())

    def test_linearmapping(self):
        mapping = GPy.mappings.Linear(3, 2)
        X = np.random.randn(100,3)
        self.assertTrue(MappingGradChecker(mapping, X).checkgrad())

    def test_mlpmapping(self):
        mapping = GPy.mappings.MLP(input_dim=3, hidden_dim=5, output_dim=2)
        X = np.random.randn(100,3)
        self.assertTrue(MappingGradChecker(mapping, X).checkgrad())

    def test_mlpextmapping(self):
        for activation in ['tanh', 'relu', 'sigmoid']:
            mapping = GPy.mappings.MLPext(input_dim=3, hidden_dims=[5,5,5], output_dim=2, activation=activation)
            X = np.random.randn(100,3)
            self.assertTrue(MappingGradChecker(mapping, X).checkgrad())

    def test_addmapping(self):
        m1 = GPy.mappings.MLP(input_dim=3, hidden_dim=5, output_dim=2)
        m2 = GPy.mappings.Linear(input_dim=3, output_dim=2)
        mapping = GPy.mappings.Additive(m1, m2)
        X = np.random.randn(100,3)
        self.assertTrue(MappingGradChecker(mapping, X).checkgrad())

    def test_compoundmapping(self):
        m1 = GPy.mappings.MLP(input_dim=3, hidden_dim=5, output_dim=2)
        Z = np.random.randn(10,2)
        m2 = GPy.mappings.Kernel(2, 4, Z, GPy.kern.RBF(2))
        mapping = GPy.mappings.Compound(m1, m2)
        X = np.random.randn(100,3)
        self.assertTrue(MappingGradChecker(mapping, X).checkgrad())




if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
