# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from GPy.core.parameterization.variational import VariationalPosterior

import logging
logger = logging.getLogger("sparse gp")

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

    This is not for missing data, as the implementation for missing data involves
    some inefficient optimization routine decisions.
    See missing data SparseGP implementation in py:class:'~GPy.models.sparse_gp_minibatch.SparseGPMiniBatch'.

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x input_dim)
    :param num_inducing: Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int

    """

    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, X_variance=None, inference_method=None,
                 name='sparse gp', Y_metadata=None, normalizer=False):

        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=3)
            else:
                #inference_method = ??
                raise NotImplementedError("what to do what to do?")
            print(("defaulting to ", inference_method, "for latent function inference"))

        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        super(SparseGP, self).__init__(X, Y, kernel, likelihood, mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        self.posterior = None

    @property
    def _predictive_variable(self):
        return self.Z

    def has_uncertain_inputs(self):
        return isinstance(self.X, VariationalPosterior)

    def set_Z(self, Z, trigger_update=True):
        if trigger_update: self.update_model(False)
        self.unlink_parameter(self.Z)
        self.Z = Param('inducing inputs',Z)
        self.link_parameter(self.Z, index=0)
        if trigger_update: self.update_model(True)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = \
        self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood,
                                        self.Y_normalized, Y_metadata=self.Y_metadata,
                                        mean_function=self.mean_function)
        self._update_gradients()

    def _update_gradients(self):
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

        if isinstance(self.X, VariationalPosterior):
            #gradients wrt kernel
            dL_dKmm = self.grad_dict['dL_dKmm']
            self.kern.update_gradients_full(dL_dKmm, self.Z, None)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                    Z=self.Z,
                                                    dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                    dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                    dL_dpsi2=self.grad_dict['dL_dpsi2'])
            self.kern.gradient += kerngrad

            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += self.kern.gradients_Z_expectations(
                               self.grad_dict['dL_dpsi0'],
                               self.grad_dict['dL_dpsi1'],
                               self.grad_dict['dL_dpsi2'],
                               Z=self.Z,
                               variational_posterior=self.X)
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
            kerngrad += self.kern.gradient
            self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z, None)
            self.kern.gradient += kerngrad
            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
        self._Zgrad = self.Z.gradient.copy()

    def to_dict(self, save_data=True):
        """
        Convert the object into a json serializable dictionary.

        :param boolean save_data: if true, it adds the training data self.X and self.Y to the dictionary
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(SparseGP, self).to_dict(save_data)
        input_dict["class"] = "GPy.core.SparseGP"
        input_dict["Z"] = self.Z.tolist()
        return input_dict

    @staticmethod
    def _format_input_dict(input_dict, data=None):
        input_dict = GP._format_input_dict(input_dict, data)
        input_dict["Z"] = np.array(input_dict["Z"])
        return input_dict

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        input_dict = SparseGP._format_input_dict(input_dict, data)
        return SparseGP(**input_dict)
