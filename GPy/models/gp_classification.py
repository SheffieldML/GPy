# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import GP
from .. import likelihoods
from .. import kern
import numpy as np
from ..inference.latent_function_inference.expectation_propagation import EP

class GPClassification(GP):
    """
    Gaussian Process classification

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values, can be None if likelihood is not None
    :param kernel: a GPy kernel, defaults to rbf
    :param likelihood: a GPy likelihood, defaults to Bernoulli
    :param inference_method: Latent function inference to use, defaults to EP
    :type inference_method: :class:`GPy.inference.latent_function_inference.LatentFunctionInference`

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None,Y_metadata=None, mean_function=None, inference_method=None,
                 likelihood=None, normalizer=False):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if likelihood is None:
            likelihood = likelihoods.Bernoulli()

        if inference_method is None:
            inference_method = EP()

        super(GPClassification, self).__init__(X=X, Y=Y,  kernel=kernel, likelihood=likelihood, inference_method=inference_method,
                                               mean_function=mean_function, name='gp_classification', normalizer=normalizer)

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        GPClassification(gp.X, gp.Y, gp.kern, gp.likelihood, gp.inference_method, gp.mean_function, name='gp_classification')

    def to_dict(self, save_data=True):
        model_dict = super(GPClassification,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPClassification"
        return model_dict

    @staticmethod
    def from_dict(input_dict, data=None):
        import GPy
        m = GPy.core.model.Model.from_dict(input_dict, data)
        return GPClassification.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        input_dict = GPClassification._format_input_dict(input_dict, data)
        input_dict.pop('name', None)  # Name parameter not required by GPClassification
        return GPClassification(**input_dict)
