# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from ..models import GPLVM
from ..mappings import *


class BCGPLVM(GPLVM):
    """
    Back constrained Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param mapping: mapping for back constraint
    :type mapping: GPy.core.Mapping object

    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, normalize_Y=False, mapping=None):
        
        if mapping is None:
            mapping = Kernel(X=Y, output_dim=input_dim)
        self.mapping = mapping
        GPLVM.__init__(self, Y, input_dim, init, X, kernel, normalize_Y)
        self.X = self.mapping.f(self.likelihood.Y)

    def _get_param_names(self):
        return self.mapping._get_param_names() + GP._get_param_names(self)

    def _get_params(self):
        return np.hstack((self.mapping._get_params(), GP._get_params(self)))

    def _set_params(self, x):
        self.mapping._set_params(x[:self.mapping.num_params])
        self.X = self.mapping.f(self.likelihood.Y)
        GP._set_params(self, x[self.mapping.num_params:])

    def _log_likelihood_gradients(self):
        dL_df = self.kern.gradients_X(self.dL_dK, self.X)
        dL_dtheta = self.mapping.df_dtheta(dL_df, self.likelihood.Y)
        return np.hstack((dL_dtheta.flatten(), GP._log_likelihood_gradients(self)))

