# Copyright (c) 2014, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy


class FITCtest:
    def setup(self):
        ######################################
        # # 1 dimensional example

        N = 20
        # sample inputs and outputs
        self.X1D = np.random.uniform(-3.0, 3.0, (N, 1))
        self.Y1D = np.sin(self.X1D) + np.random.randn(N, 1) * 0.05

        ######################################
        # # 2 dimensional example

        # sample inputs and outputs
        self.X2D = np.random.uniform(-3.0, 3.0, (N, 2))
        self.Y2D = (
            np.sin(self.X2D[:, 0:1]) * np.sin(self.X2D[:, 1:2])
            + np.random.randn(N, 1) * 0.05
        )

    def test_fitc_1d(self):
        self.setup()
        m = GPy.models.SparseGPRegression(self.X1D, self.Y1D)
        m.inference_method = GPy.inference.latent_function_inference.FITC()
        assert m.checkgrad(), "Gradient check failed!"

    def test_fitc_2d(self):
        self.setup()
        m = GPy.models.SparseGPRegression(self.X2D, self.Y2D)
        m.inference_method = GPy.inference.latent_function_inference.FITC()
        assert m.checkgrad(), "Gradient check failed!"
