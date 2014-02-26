# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import sys
import warnings
from .. import kern
from ..util.linalg import dtrtrs
from model import Model
from parameterization import ObservableArray
from .. import likelihoods
from ..likelihoods.gaussian import Gaussian
from ..inference.latent_function_inference import exact_gaussian_inference
from parameterization.variational import VariationalPosterior

class GP(Model):
    """
    General purpose Gaussian process model

    :param X: input observations
    :param Y: output observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y


    """
    def __init__(self, X, Y, kernel, likelihood, inference_method=None, Y_metadata=None, name='gp'):
        super(GP, self).__init__(name)

        assert X.ndim == 2
        if isinstance(X, ObservableArray) or isinstance(X, VariationalPosterior):
            self.X = X
        else: self.X = ObservableArray(X)

        self.num_data, self.input_dim = self.X.shape

        assert Y.ndim == 2
        self.Y = ObservableArray(Y)
        assert Y.shape[0] == self.num_data
        _, self.output_dim = self.Y.shape

        if Y_metadata is not None:
            self.Y_metadata = ObservableArray(Y_metadata)
        else:
            self.Y_metadata = None

        assert isinstance(kernel, kern.Kern)
        assert self.input_dim == kernel.input_dim
        self.kern = kernel

        assert isinstance(likelihood, likelihoods.Likelihood)
        self.likelihood = likelihood

        #find a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation
                print "defaulting to ", inference_method, "for latent function inference"
        self.inference_method = inference_method

        self.add_parameter(self.kern)
        self.add_parameter(self.likelihood)
        if self.__class__ is GP:
            self.parameters_changed()

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y, Y_metadata=self.Y_metadata)
        self.kern.update_gradients_full(grad_dict['dL_dK'], self.X)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _raw_predict(self, _Xnew, full_cov=False):
        """
        Internal helper function for making predictions, does not account
        for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        """
        Kx = self.kern.K(_Xnew, self.X).T
        #LiKx, _ = dtrtrs(self.posterior.woodbury_chol, np.asfortranarray(Kx), lower=1)
        WiKx = np.dot(self.posterior.woodbury_inv, Kx)
        mu = np.dot(Kx.T, self.posterior.woodbury_vector)
        if full_cov:
            Kxx = self.kern.K(_Xnew)
            #var = Kxx - tdot(LiKx.T)
            var = np.dot(Kx.T, WiKx)
        else:
            Kxx = self.kern.Kdiag(_Xnew)
            #var = Kxx - np.sum(LiKx*LiKx, 0)
            var = Kxx - np.sum(WiKx*Kx, 0)
            var = var.reshape(-1, 1)
        return mu, var

    def predict(self, Xnew, full_cov=False, **likelihood_args):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :returns: mean: posterior mean,  a Numpy array, Nnew x self.input_dim
        :returns: var: posterior variance, a Numpy array, Nnew x 1 if
                       full_cov=False, Nnew x Nnew otherwise
        :returns: lower and upper boundaries of the 95% confidence intervals,
                  Numpy arrays,  Nnew x self.input_dim


           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        #predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov, **likelihood_args)
        return mean, var, _025pm, _975pm

    def posterior_samples_f(self,X,size=10, full_cov=True):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray, Nnew x self.input_dim.
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: Ysim: set of simulations, a Numpy array (N x samples).
        """
        m, v = self._raw_predict(X,  full_cov=full_cov)
        v = v.reshape(m.size,-1) if len(v.shape)==3 else v
        if not full_cov:
            Ysim = np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
        else:
            Ysim = np.random.multivariate_normal(m.flatten(), v, size).T

        return Ysim

    def posterior_samples(self,X,size=10, full_cov=True,noise_model=None):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray, Nnew x self.input_dim.
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations, a Numpy array (N x samples).
        """
        Ysim = self.posterior_samples_f(X, size, full_cov=full_cov)
        if isinstance(self.likelihood, Gaussian):
            noise_std = np.sqrt(self.likelihood._get_params())
            Ysim += np.random.normal(0,noise_std,Ysim.shape)
        elif isinstance(self.likelihood, Gaussian_Mixed_Noise):
            assert noise_model is not None, "A noise model must be specified."
            noise_std = np.sqrt(self.likelihood._get_params()[noise_model])
            Ysim += np.random.normal(0,noise_std,Ysim.shape)
        else:
            Ysim = self.likelihood.noise_model.samples(Ysim)

        return Ysim

    def plot_f(self, *args, **kwargs):
        """

        Plot the GP's view of the world, where the data is normalized and
        before applying a likelihood.

        This is a convenience function: arguments are passed to
        GPy.plotting.matplot_dep.models_plots.plot_f_fit

        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        models_plots.plot_fit_f(self,*args,**kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region
            identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted
            function
          - In higher dimensions, use fixed_inputs to plot the GP  with some of
            the inputs fixed.

        Can plot only part of the data and part of the posterior functions
        using which_data_rows which_data_ycols and which_parts

        This is a convenience function: arguments are passed to
        GPy.plotting.matplot_dep.models_plots.plot_fit

        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        models_plots.plot_fit(self,*args,**kwargs)

    def _getstate(self):
        """

        Get the current state of the class, here we return everything that is
        needed to recompute the model.

        """

        return Model._getstate(self) + [self.X,
                self.num_data,
                self.input_dim,
                self.kern,
                self.likelihood,
                self.output_dim,
                self._Xoffset,
                self._Xscale,
                ]

    def _setstate(self, state):
        self._Xscale = state.pop()
        self._Xoffset = state.pop()
        self.output_dim = state.pop()
        self.likelihood = state.pop()
        self.kern = state.pop()
        self.input_dim = state.pop()
        self.num_data = state.pop()
        self.X = state.pop()
        Model._setstate(self, state)
