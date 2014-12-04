# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.parameterization.param import Param
from ..core.gp import GP
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from ..core.parameterization.variational import VariationalPosterior

import logging
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.inference.optimization.stochastics import SparseGPStochastics,\
    SparseGPMissing
#no stochastics.py file added! from GPy.inference.optimization.stochastics import SparseGPStochastics,\
    #SparseGPMissing
logger = logging.getLogger("sparse gp")

class SparseGPMiniBatch(GP):
    """
    A general purpose Sparse GP model
'''
Created on 3 Nov 2014

@author: maxz
'''

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

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

    def __init__(self, X, Y, Z, kernel, likelihood, inference_method=None,
                 name='sparse gp', Y_metadata=None, normalizer=False,
                 missing_data=False, stochastic=False, batchsize=1):
        
        # pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=1 if not missing_data else Y.shape[1])
            else:
                #inference_method = ??
                raise NotImplementedError, "what to do what to do?"
            print "defaulting to ", inference_method, "for latent function inference"

        self.kl_factr = 1.
        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.missing_data = missing_data

        if stochastic and missing_data:
            self.missing_data = True
            self.ninan = ~np.isnan(Y)
            self.stochastics = SparseGPStochastics(self, batchsize)
        elif stochastic and not missing_data:
            self.missing_data = False
            self.stochastics = SparseGPStochastics(self, batchsize)
        elif missing_data:
            self.missing_data = True
            self.ninan = ~np.isnan(Y)
            self.stochastics = SparseGPMissing(self)
        else:
            self.stochastics = False

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        if self.missing_data:
            self.Ylist = []
            overall = self.Y_normalized.shape[1]
            m_f = lambda i: "Precomputing Y for missing data: {: >7.2%}".format(float(i+1)/overall)
            message = m_f(-1)
            print message,
            for d in xrange(overall):
                self.Ylist.append(self.Y_normalized[self.ninan[:, d], d][:, None])
                print ' '*(len(message)+1) + '\r',
                message = m_f(d)
                print message,
            print ''

        self.posterior = None

    def has_uncertain_inputs(self):
        return isinstance(self.X, VariationalPosterior)

    def _inner_parameters_changed(self, kern, X, Z, likelihood, Y, Y_metadata, Lm=None, dL_dKmm=None, subset_indices=None):
        """
        This is the standard part, which usually belongs in parameters_changed.

        For automatic handling of subsampling (such as missing_data, stochastics etc.), we need to put this into an inner
        loop, in order to ensure a different handling of gradients etc of different
        subsets of data.

        The dict in current_values will be passed aroung as current_values for
        the rest of the algorithm, so this is the place to store current values,
        such as subsets etc, if necessary.

        If Lm and dL_dKmm can be precomputed (or only need to be computed once)
        pass them in here, so they will be passed to the inference_method.

        subset_indices is a dictionary of indices. you can put the indices however you
        like them into this dictionary for inner use of the indices inside the
        algorithm.
        """
        try:
            posterior, log_marginal_likelihood, grad_dict = self.inference_method.inference(kern, X, Z, likelihood, Y, Y_metadata, Lm=Lm, dL_dKmm=None)
        except:
            posterior, log_marginal_likelihood, grad_dict = self.inference_method.inference(kern, X, Z, likelihood, Y, Y_metadata)
        current_values = {}
        likelihood.update_gradients(grad_dict['dL_dthetaL'])
        current_values['likgrad'] = likelihood.gradient.copy()
        if subset_indices is None:
            subset_indices = {}
        if isinstance(X, VariationalPosterior):
            #gradients wrt kernel
            dL_dKmm = grad_dict['dL_dKmm']
            kern.update_gradients_full(dL_dKmm, Z, None)
            current_values['kerngrad'] = kern.gradient.copy()
            kern.update_gradients_expectations(variational_posterior=X,
                                                    Z=Z,
                                                    dL_dpsi0=grad_dict['dL_dpsi0'],
                                                    dL_dpsi1=grad_dict['dL_dpsi1'],
                                                    dL_dpsi2=grad_dict['dL_dpsi2'])
            current_values['kerngrad'] += kern.gradient

            #gradients wrt Z
            current_values['Zgrad'] = kern.gradients_X(dL_dKmm, Z)
            current_values['Zgrad'] += kern.gradients_Z_expectations(
                               grad_dict['dL_dpsi0'],
                               grad_dict['dL_dpsi1'],
                               grad_dict['dL_dpsi2'],
                               Z=Z,
                               variational_posterior=X)
        else:
            #gradients wrt kernel
            kern.update_gradients_diag(grad_dict['dL_dKdiag'], X)
            current_values['kerngrad'] = kern.gradient.copy()
            kern.update_gradients_full(grad_dict['dL_dKnm'], X, Z)
            current_values['kerngrad'] += kern.gradient
            kern.update_gradients_full(grad_dict['dL_dKmm'], Z, None)
            current_values['kerngrad'] += kern.gradient
            #gradients wrt Z
            current_values['Zgrad'] = kern.gradients_X(grad_dict['dL_dKmm'], Z)
            current_values['Zgrad'] += kern.gradients_X(grad_dict['dL_dKnm'].T, Z, X)
        return posterior, log_marginal_likelihood, grad_dict, current_values, subset_indices

    def _inner_take_over_or_update(self, full_values=None, current_values=None, value_indices=None):
        """
        This is for automatic updates of values in the inner loop of missing
        data handling. Both arguments are dictionaries and the values in
        full_values will be updated by the current_gradients.

        If a key from current_values does not exist in full_values, it will be
        initialized to the value in current_values.

        If there is indices needed for the update, value_indices can be used for
        that. If value_indices has the same key, as current_values, the update
        in full_values will be indexed by the indices in value_indices.

        grads:
            dictionary of standing gradients (you will have to carefully make sure, that
            the ordering is right!). The values in here will be updated such that
            full_values[key] += current_values[key]  forall key in full_gradients.keys()

        gradients:
            dictionary of gradients in the current set of parameters.

        value_indices:
            dictionary holding indices for the update in full_values.
            if the key exists the update rule is:def df(x):
            full_values[key][value_indices[key]] += current_values[key]
        """
        for key in current_values.keys():
            if value_indices is not None and value_indices.has_key(key):
                index = value_indices[key]
            else:
                index = slice(None)
            if full_values.has_key(key):
                full_values[key][index] += current_values[key]
            else:
                full_values[key] = current_values[key]

    def _inner_values_update(self, current_values):
        """
        This exists if there is more to do with the current values.
        It will be called allways in the inner loop, so that
        you can do additional inner updates for the inside of the missing data
        loop etc. This can also be used for stochastic updates, when only working on
        one dimension of the output.
        """
        pass

    def _outer_values_update(self, full_values):
        """
        Here you put the values, which were collected before in the right places.
        E.g. set the gradients of parameters, etc.
        """
        self.likelihood.gradient = full_values['likgrad']
        self.kern.gradient = full_values['kerngrad']
        self.Z.gradient = full_values['Zgrad']

    def _outer_init_full_values(self):
        """
        If full_values has indices in values_indices, we might want to initialize
        the full_values differently, so that subsetting is possible.

        Here you can initialize the full_values for the values needed.

        Keep in mind, that if a key does not exist in full_values when updating
        values, it will be set (so e.g. for Z there is no need to initialize Zgrad,
        as there is no subsetting needed. For X in BGPLVM on the other hand we probably need
        to initialize the gradients for the mean and the variance in order to
        have the full gradient for indexing)
        """
        return {}

    def _outer_loop_for_missing_data(self):
        Lm = None
        dL_dKmm = None

        self._log_marginal_likelihood = 0
        self.full_values = self._outer_init_full_values()

        if self.posterior is None:
            woodbury_inv = np.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            woodbury_vector = np.zeros((self.num_inducing, self.output_dim))
        else:
            woodbury_inv = self.posterior._woodbury_inv
            woodbury_vector = self.posterior._woodbury_vector

        if not self.stochastics:
            m_f = lambda i: "Inference with missing_data: {: >7.2%}".format(float(i+1)/self.output_dim)
            message = m_f(-1)
            print message,

        for d in self.stochastics.d:
            ninan = self.ninan[:, d]

            if not self.stochastics:
                print ' '*(len(message)) + '\r',
                message = m_f(d)
                print message,

            posterior, log_marginal_likelihood, \
                grad_dict, current_values, value_indices = self._inner_parameters_changed(
                                self.kern, self.X[ninan],
                                self.Z, self.likelihood,
                                self.Ylist[d], self.Y_metadata,
                                Lm, dL_dKmm,
                                subset_indices=dict(outputs=d, samples=ninan))

            self._inner_take_over_or_update(self.full_values, current_values, value_indices)
            self._inner_values_update(current_values)

            Lm = posterior.K_chol
            dL_dKmm = grad_dict['dL_dKmm']
            woodbury_inv[:, :, d] = posterior.woodbury_inv
            woodbury_vector[:, d:d+1] = posterior.woodbury_vector
            self._log_marginal_likelihood += log_marginal_likelihood
        if not self.stochastics:
            print ''

        if self.posterior is None:
            self.posterior = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector,
                                   K=posterior._K, mean=None, cov=None, K_chol=posterior.K_chol)
        self._outer_values_update(self.full_values)

    def _outer_loop_without_missing_data(self):
        self._log_marginal_likelihood = 0

        if self.posterior is None:
            woodbury_inv = np.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            woodbury_vector = np.zeros((self.num_inducing, self.output_dim))
        else:
            woodbury_inv = self.posterior._woodbury_inv
            woodbury_vector = self.posterior._woodbury_vector

        d = self.stochastics.d
        posterior, log_marginal_likelihood, \
            grad_dict, self.full_values, _ = self._inner_parameters_changed(
                            self.kern, self.X,
                            self.Z, self.likelihood,
                            self.Y_normalized[:, d], self.Y_metadata)
        self.grad_dict = grad_dict

        self._log_marginal_likelihood += log_marginal_likelihood

        self._outer_values_update(self.full_values)

        woodbury_inv[:, :, d] = posterior.woodbury_inv[:, :, None]
        woodbury_vector[:, d] = posterior.woodbury_vector
        if self.posterior is None:
            self.posterior = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector,
                                   K=posterior._K, mean=None, cov=None, K_chol=posterior.K_chol)

    def parameters_changed(self):
        if self.missing_data:
            self._outer_loop_for_missing_data()
        elif self.stochastics:
            self._outer_loop_without_missing_data()
        else:
            self.posterior, self._log_marginal_likelihood, self.grad_dict, self.full_values, _ = self._inner_parameters_changed(self.kern, self.X, self.Z, self.likelihood, self.Y_normalized, self.Y_metadata)
            self._outer_values_update(self.full_values)

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values
        """

        if kern is None: kern = self.kern

        if not isinstance(Xnew, VariationalPosterior):
            Kx = kern.K(self.Z, Xnew)
            mu = np.dot(Kx.T, self.posterior.woodbury_vector)
            if full_cov:
                Kxx = kern.K(Xnew)
                if self.posterior.woodbury_inv.ndim == 2:
                    var = Kxx - np.dot(Kx.T, np.dot(self.posterior.woodbury_inv, Kx))
                elif self.posterior.woodbury_inv.ndim == 3:
                    var = Kxx[:,:,None] - np.tensordot(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx).T, Kx, [1,0]).swapaxes(1,2)
                var = var
            else:
                Kxx = kern.Kdiag(Xnew)
                var = (Kxx - np.sum(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T
        else:
            Kx = kern.psi1(self.Z, Xnew)
            mu = np.dot(Kx, self.posterior.woodbury_vector)
            if full_cov:
                raise NotImplementedError, "TODO"
            else:
                Kxx = kern.psi0(self.Z, Xnew)
                psi2 = kern.psi2(self.Z, Xnew)
                var = Kxx - np.sum(np.sum(psi2 * Kmmi_LmiBLmi[None, :, :], 1), 1)
        return mu, var
