# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

from ..core import GpSSM
from .. import likelihoods
from .. import kern

class GPRegressionSSM(GpSSM):
    """
    Gaussian Process State Space model (Gp-SSM) for one-dimensional inputs

    This is a thin wrapper around the models.GpSSM class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to Matern32_SSM
    :param Norm normalizer: [False]

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None):

        if kernel is None:
            kernel = kern.Matern32_SSM(1)   # no other kernels implemented so far

        likelihood = likelihoods.Gaussian()
        super(GPRegressionSSM, self).__init__(X, Y, kernel, likelihood, name='GP SSM regression', Y_metadata=Y_metadata, normalizer=normalizer)

