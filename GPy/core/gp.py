# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import sys
import warnings
from .. import kern
from ..util.linalg import dtrtrs
from model import Model
from parameterization import ObsAr
from .. import likelihoods
from ..likelihoods.gaussian import Gaussian
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation, LatentFunctionInference
from parameterization.variational import VariationalPosterior
from scipy.sparse.base import issparse

class GP(Model):
    """
    General purpose Gaussian process model

    :param X: input observations
    :param Y: output observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :param :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference_method: The inference method to use for this GP
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y


    """
    def __init__(self, X, Y, kernel, likelihood, inference_method=None, name='gp', Y_metadata=None):
        super(GP, self).__init__(name)

        assert X.ndim == 2
        if isinstance(X, (ObsAr, VariationalPosterior)):
            self.X = X.copy()
        else: self.X = ObsAr(X)

        self.num_data, self.input_dim = self.X.shape

        assert Y.ndim == 2
        if issparse(Y): self.Y = Y
        else: self.Y = ObsAr(Y)
        assert Y.shape[0] == self.num_data
        _, self.output_dim = self.Y.shape

        #TODO: check the type of this is okay?
        self.Y_metadata = Y_metadata

        assert isinstance(kernel, kern.Kern)
        #assert self.input_dim == kernel.input_dim
        self.kern = kernel

        assert isinstance(likelihood, likelihoods.Likelihood)
        self.likelihood = likelihood

        #find a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian) or isinstance(likelihood, likelihoods.MixedNoise):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation.EP()
                print "defaulting to ", inference_method, "for latent function inference"
        self.inference_method = inference_method

        self.add_parameter(self.kern)
        self.add_parameter(self.likelihood)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _raw_predict(self, _Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        $$
        p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                       = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
        \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        $$

        """
        if kern is None:
            kern = self.kern

        Kx = kern.K(_Xnew, self.X).T
        WiKx = np.dot(self.posterior.woodbury_inv, Kx)
        mu = np.dot(Kx.T, self.posterior.woodbury_vector)
        if full_cov:
            Kxx = kern.K(_Xnew)
            var = Kxx - np.dot(Kx.T, WiKx)
        else:
            Kxx = kern.Kdiag(_Xnew)
            var = Kxx - np.sum(WiKx*Kx, 0)
            var = var.reshape(-1, 1)

        #force mu to be a column vector
        if len(mu.shape)==1: mu = mu[:,None]
        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :returns: mean: posterior mean,  a Numpy array, Nnew x self.input_dim
        :returns: var: posterior variance, a Numpy array, Nnew x 1 if
                       full_cov=False, Nnew x Nnew otherwise
        :returns: lower and upper boundaries of the 95% confidence intervals,
                  Numpy arrays,  Nnew x self.input_dim


           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        #predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var, full_cov, Y_metadata)
        return mean, var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None):
        m, v = self._raw_predict(X,  full_cov=False)
        return self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata)

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

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None):
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
        Ysim = self.likelihood.samples(Ysim, Y_metadata)

        return Ysim

    def plot_f(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=True,
        linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx'):
        """
        Plot the GP's view of the world, where the data is normalized and before applying a likelihood.
        This is a call to plot with plot_raw=True.
        Data will not be plotted in this, as the GP's view of the world
        may live in another space, or units then the data.
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        if linecol is not None:
            kw['linecol'] = linecol
        if fillcol is not None:
            kw['fillcol'] = fillcol
        return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     which_data_ycols, fixed_inputs,
                                     levels, samples, fignum, ax, resolution,
                                     plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     data_symbol=data_symbol, **kw)

    def plot(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=False,
        linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx'):
        """
        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, use fixed_inputs to plot the GP  with some of the inputs fixed.

        Can plot only part of the data and part of the posterior functions
        using which_data_rowsm which_data_ycols.

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_rows: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :type output: integer (first output is 0)
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol:
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        if linecol is not None:
            kw['linecol'] = linecol
        if fillcol is not None:
            kw['fillcol'] = fillcol
        return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     which_data_ycols, fixed_inputs,
                                     levels, samples, fignum, ax, resolution,
                                     plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     data_symbol=data_symbol, **kw)

    def input_sensitivity(self):
        """
        Returns the sensitivity for each dimension of this model
        """
        return self.kern.input_sensitivity()

    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer)
        :type optimizer: string

        TODO: valid args
        """
        self.inference_method.on_optimization_start()
        try:
            super(GP, self).optimize(optimizer, start, **kwargs)
        except KeyboardInterrupt:
            print "KeyboardInterrupt caught, calling on_optimization_end() to round things up"
            self.inference_method.on_optimization_end()
            raise
