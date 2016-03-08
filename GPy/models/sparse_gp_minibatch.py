# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
import numpy as np
from ..core.parameterization.param import Param
from GPy.core.parameterization.variational import VariationalPosterior
from ..core.sparse_gp import SparseGP
from ..core.gp import GP
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods

import logging
from ..inference.latent_function_inference.posterior import Posterior
from ..inference.optimization.stochastics import SparseGPStochastics, SparseGPMissing
logger = logging.getLogger("sparse gp")

class SparseGPMiniBatch(SparseGP):
    """
    A general purpose Sparse GP model, allowing missing data and stochastics across dimensions.

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
        self._update_stochastics = False

        # pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=3 if not missing_data else Y.shape[1])
            else:
                #inference_method = ??
                raise NotImplementedError("what to do what to do?")
            print("defaulting to ", inference_method, "for latent function inference")

        self.kl_factr = 1.
        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.missing_data = missing_data

        if stochastic and missing_data:
            self.missing_data = True
            self.stochastics = SparseGPStochastics(self, batchsize, self.missing_data)
        elif stochastic and not missing_data:
            self.missing_data = False
            self.stochastics = SparseGPStochastics(self, batchsize, self.missing_data)
        elif missing_data:
            self.missing_data = True
            self.stochastics = SparseGPMissing(self)
        else:
            self.stochastics = False

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        self.posterior = None
        
    def optimize(self, optimizer=None, start=None, **kwargs):
        try:
            self._update_stochastics = True
            SparseGP.optimize(self, optimizer=optimizer, start=start, **kwargs)
        finally:
            self._update_stochastics = False
            
    def has_uncertain_inputs(self):
        return isinstance(self.X, VariationalPosterior)

    def _inner_parameters_changed(self, kern, X, Z, likelihood, Y, Y_metadata, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None, **kwargs):
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
        return self.inference_method.inference(kern, X, Z, likelihood, Y, Y_metadata, Lm=Lm,
                                               dL_dKmm=dL_dKmm, psi0=psi0, psi1=psi1, psi2=psi2, **kwargs)

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
            if value_indices is not None and key in value_indices:
                index = value_indices[key]
            else:
                index = slice(None)
            if key in full_values:
                try:
                    full_values[key][index] += current_values[key]
                except:
                    full_values[key] += current_values[key]
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
        if self.has_uncertain_inputs():
            #gradients wrt kernel
            dL_dKmm = full_values['dL_dKmm']
            self.kern.update_gradients_full(dL_dKmm, self.Z, None)
            kgrad = self.kern.gradient.copy()
            self.kern.update_gradients_expectations(
                                                variational_posterior=self.X,
                                                Z=self.Z, dL_dpsi0=full_values['dL_dpsi0'],
                                                dL_dpsi1=full_values['dL_dpsi1'],
                                                dL_dpsi2=full_values['dL_dpsi2'])
            self.kern.gradient += kgrad


            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += self.kern.gradients_Z_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z, dL_dpsi0=full_values['dL_dpsi0'],
                                            dL_dpsi1=full_values['dL_dpsi1'],
                                            dL_dpsi2=full_values['dL_dpsi2'])
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(full_values['dL_dKdiag'], self.X)
            kgrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(full_values['dL_dKnm'], self.X, self.Z)
            kgrad += self.kern.gradient
            self.kern.update_gradients_full(full_values['dL_dKmm'], self.Z, None)
            self.kern.gradient += kgrad
            #kgrad += self.kern.gradient

            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(full_values['dL_dKmm'], self.Z)
            self.Z.gradient += self.kern.gradients_X(full_values['dL_dKnm'].T, self.Z, self.X)

        self.likelihood.update_gradients(full_values['dL_dthetaL'])

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
        retd = dict(dL_dKmm=np.zeros((self.Z.shape[0], self.Z.shape[0])))
        if self.has_uncertain_inputs():
            retd.update(dict(dL_dpsi0=np.zeros(self.X.shape[0]),
                             dL_dpsi1=np.zeros((self.X.shape[0], self.Z.shape[0])),
                             dL_dpsi2=np.zeros((self.X.shape[0], self.Z.shape[0], self.Z.shape[0]))))
        else:
            retd.update({'dL_dKdiag': np.zeros(self.X.shape[0]),
                         'dL_dKnm': np.zeros((self.X.shape[0], self.Z.shape[0]))})
        return retd

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

        #if not self.stochastics:
        #    m_f = lambda i: "Inference with missing_data: {: >7.2%}".format(float(i+1)/self.output_dim)
        #    message = m_f(-1)
        #    print(message, end=' ')

        for d, ninan in self.stochastics.d:
            #if not self.stochastics:
            #    print(' '*(len(message)) + '\r', end=' ')
            #    message = m_f(d)
            #    print(message, end=' ')

            psi0ni = self.psi0[ninan]
            psi1ni = self.psi1[ninan]
            if self.has_uncertain_inputs():
                psi2ni = self.psi2[ninan]
                value_indices = dict(outputs=d, samples=ninan, dL_dpsi0=ninan, dL_dpsi1=ninan, dL_dpsi2=ninan)
            else:
                psi2ni = None
                value_indices = dict(outputs=d, samples=ninan, dL_dKdiag=ninan, dL_dKnm=ninan)

            posterior, log_marginal_likelihood, grad_dict = self._inner_parameters_changed(
                                self.kern, self.X[ninan],
                                self.Z, self.likelihood,
                                self.Y_normalized[ninan][:, d], self.Y_metadata,
                                Lm, dL_dKmm,
                                psi0=psi0ni, psi1=psi1ni, psi2=psi2ni)

            # Fill out the full values by adding in the apporpriate grad_dict
            # values
            self._inner_take_over_or_update(self.full_values, grad_dict, value_indices)
            self._inner_values_update(grad_dict)  # What is this for? -> MRD

            woodbury_inv[:, :, d] = posterior.woodbury_inv[:,:,None]
            woodbury_vector[:, d] = posterior.woodbury_vector
            self._log_marginal_likelihood += log_marginal_likelihood

        #if not self.stochastics:
        #    print('')

        if self.posterior is None:
            self.posterior = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector,
                                   K=posterior._K, mean=None, cov=None, K_chol=posterior.K_chol)
        self._outer_values_update(self.full_values)
        if self.has_uncertain_inputs():
            self.kern.return_psi2_n = False

    def _outer_loop_without_missing_data(self):
        if self.posterior is None:
            woodbury_inv = np.zeros((self.num_inducing, self.num_inducing, self.output_dim))
            woodbury_vector = np.zeros((self.num_inducing, self.output_dim))
        else:
            woodbury_inv = self.posterior._woodbury_inv
            woodbury_vector = self.posterior._woodbury_vector

        d = self.stochastics.d[0][0]
        posterior, log_marginal_likelihood, grad_dict= self._inner_parameters_changed(
                            self.kern, self.X,
                            self.Z, self.likelihood,
                            self.Y_normalized[:, d], self.Y_metadata)
        self.grad_dict = grad_dict

        self._log_marginal_likelihood = log_marginal_likelihood

        self._outer_values_update(self.grad_dict)

        woodbury_inv[:, :, d] = posterior.woodbury_inv[:, :, None]
        woodbury_vector[:, d] = posterior.woodbury_vector
        if self.posterior is None:
            self.posterior = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector,
                                   K=posterior._K, mean=None, cov=None, K_chol=posterior.K_chol)

    def parameters_changed(self):
        #Compute the psi statistics for N once, but don't sum out N in psi2
        if self.has_uncertain_inputs():
            #psi0 = ObsAr(self.kern.psi0(self.Z, self.X))
            #psi1 = ObsAr(self.kern.psi1(self.Z, self.X))
            #psi2 = ObsAr(self.kern.psi2(self.Z, self.X))
            self.psi0 = self.kern.psi0(self.Z, self.X)
            self.psi1 = self.kern.psi1(self.Z, self.X)
            self.psi2 = self.kern.psi2n(self.Z, self.X)
        else:
            self.psi0 = self.kern.Kdiag(self.X)
            self.psi1 = self.kern.K(self.X, self.Z)
            self.psi2 = None

        if self.missing_data:
            self._outer_loop_for_missing_data()
        elif self.stochastics:
            if self._update_stochastics:
                self.stochastics.do_stochastics()
            self._outer_loop_without_missing_data()
        else:
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self._inner_parameters_changed(self.kern, self.X, self.Z, self.likelihood, self.Y_normalized, self.Y_metadata)
            self._outer_values_update(self.grad_dict)
        self._Zgrad = self.Z.gradient.copy()
