# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
    


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
