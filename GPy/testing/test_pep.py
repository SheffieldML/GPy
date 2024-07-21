# Copyright (c) 2014, James Hensman, 2016, Thang Bui
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy


class TestPEPgradient:
    def setup(self):
        ######################################
        # # 1 dimensional example
        np.random.seed(10)

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

        #######################################
        # # more datapoints, check in alpha limits, the log marginal likelihood
        # # is consistent with FITC and VFE/Var_DTC
        M = 5
        np.random.seed(42)
        self.X1 = np.c_[np.linspace(-1.0, 1.0, N)]
        self.Y1 = np.sin(self.X1) + np.random.randn(N, 1) * 0.05
        self.kernel = GPy.kern.RBF(input_dim=1, lengthscale=0.5, variance=1)
        self.Z = np.random.uniform(-1, 1, (M, 1))
        self.lik_noise_var = 0.01

    def test_pep_1d_gradients(self):
        self.setup()
        m = GPy.models.SparseGPRegression(self.X1D, self.Y1D)
        m.inference_method = GPy.inference.latent_function_inference.PEP(
            alpha=np.random.rand()
        )
        assert m.checkgrad()

    def test_pep_2d_gradients(self):
        self.setup()
        m = GPy.models.SparseGPRegression(self.X2D, self.Y2D)
        m.inference_method = GPy.inference.latent_function_inference.PEP(
            alpha=np.random.rand()
        )
        assert m.checkgrad()

    def test_pep_vfe_consistency(self):
        self.setup()
        vfe_model = GPy.models.SparseGPRegression(
            self.X1, self.Y1, kernel=self.kernel, Z=self.Z
        )
        vfe_model.inference_method = GPy.inference.latent_function_inference.VarDTC()
        vfe_model.Gaussian_noise.variance = self.lik_noise_var
        vfe_lml = vfe_model.log_likelihood()

        pep_model = GPy.models.SparseGPRegression(
            self.X1, self.Y1, kernel=self.kernel, Z=self.Z
        )
        pep_model.inference_method = GPy.inference.latent_function_inference.PEP(
            alpha=1e-5
        )
        pep_model.Gaussian_noise.variance = self.lik_noise_var
        pep_lml = pep_model.log_likelihood()

        np.testing.assert_almost_equal(
            vfe_lml[0, 0], pep_lml[0], decimal=abs(0.01 * pep_lml[0])
        )

    def test_pep_fitc_consistency(self):
        self.setup()
        fitc_model = GPy.models.SparseGPRegression(
            self.X1D, self.Y1D, kernel=self.kernel, Z=self.Z
        )
        fitc_model.inference_method = GPy.inference.latent_function_inference.FITC()
        fitc_model.Gaussian_noise.variance = self.lik_noise_var
        fitc_lml = fitc_model.log_likelihood()

        pep_model = GPy.models.SparseGPRegression(
            self.X1D, self.Y1D, kernel=self.kernel, Z=self.Z
        )
        pep_model.inference_method = GPy.inference.latent_function_inference.PEP(
            alpha=1
        )
        pep_model.Gaussian_noise.variance = self.lik_noise_var
        pep_lml = pep_model.log_likelihood()

        np.testing.assert_almost_equal(
            fitc_lml, pep_lml[0], decimal=abs(0.001 * pep_lml[0])
        )
