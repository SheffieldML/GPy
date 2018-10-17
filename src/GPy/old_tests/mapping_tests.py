# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class MappingGradChecker(GPy.core.Model):
    """
    This class has everything we need to check the gradient of a mapping. It
    implement a simple likelihood which is the sum of the outputs of the
    mapping. the gradients are checked against the parameters of the mapping
    and the input.
    """
    def __init__(self, mapping, X, name):
        super(MappingChecker).__init__(self, name)
        self.mapping = mapping
        self.add_parameter(self.mapping)
        self.X = GPy.core.Param('X',X)
        self.add_parameter(self.X)
        self.dL_dY = np.ones((self.X.shape[0]. self.mapping.output_dim))
    def log_likelihood(self):
        return np.sum(self.mapping.f(X))
    def parameters_changed(self):
        self.X.gradient = self.mapping.gradients_X(self.dL_dY, self.X)
        self.mapping.update_gradients(self.dL_dY, self.X)







class MappingTests(unittest.TestCase):

    def test_kernelmapping(self):
        verbose = False
        mapping = GPy.mappings.Kernel(np.random.rand(10, 3), 2)
        self.assertTrue(GPy.core.mapping.Mapping_check_df_dtheta(mapping=mapping).checkgrad(verbose=verbose))
        self.assertTrue(GPy.core.mapping.Mapping_check_df_dX(mapping=mapping).checkgrad(verbose=verbose))

    def test_linearmapping(self):
        verbose = False
        mapping = GPy.mappings.Linear(3, 2)
        self.assertTrue(GPy.core.Mapping_check_df_dtheta(mapping=mapping).checkgrad(verbose=verbose))
        self.assertTrue(GPy.core.Mapping_check_df_dX(mapping=mapping).checkgrad(verbose=verbose))

    def test_mlpmapping(self):
        verbose = False
        mapping = GPy.mappings.MLP(input_dim=2, hidden_dim=[3, 4, 8, 2], output_dim=2)
        self.assertTrue(GPy.core.Mapping_check_df_dtheta(mapping=mapping).checkgrad(verbose=verbose))
        self.assertTrue(GPy.core.Mapping_check_df_dX(mapping=mapping).checkgrad(verbose=verbose))


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
