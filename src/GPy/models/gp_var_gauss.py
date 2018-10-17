# Copyright (c) 2014, James Hensman, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from ..core.parameterization.param import Param
from ..inference.latent_function_inference import VarGauss

log_2_pi = np.log(2*np.pi)


class GPVariationalGaussianApproximation(GP):
    """
    The Variational Gaussian Approximation revisited

    @article{Opper:2009,
        title = {The Variational Gaussian Approximation Revisited},
        author = {Opper, Manfred and Archambeau, C{\'e}dric},
        journal = {Neural Comput.},
        year = {2009},
        pages = {786--792},
    }
    """
    def __init__(self, X, Y, kernel, likelihood, Y_metadata=None):

        num_data = Y.shape[0]
        self.alpha = Param('alpha', np.zeros((num_data,1))) # only one latent fn for now.
        self.beta = Param('beta', np.ones(num_data))

        inf = VarGauss(self.alpha, self.beta)
        super(GPVariationalGaussianApproximation, self).__init__(X, Y, kernel, likelihood, name='VarGP', inference_method=inf, Y_metadata=Y_metadata)

        self.link_parameter(self.alpha)
        self.link_parameter(self.beta)
