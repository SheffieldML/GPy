# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import sys
from .. import kern
from .model import Model
from .parameterization import ObsAr
from .mapping import Mapping
from .. import likelihoods
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from .parameterization.variational import VariationalPosterior

import logging
import warnings
from GPy.util.normalizer import MeanNorm
logger = logging.getLogger("GP")

class GP(Model):
    """
    General purpose Gaussian process model

    :param X: input observations
    :param Y: output observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    :rtype: model object
    :param Norm normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is None, we will normalize using MeanNorm.
        If normalizer is False, no normalization will be done.

    .. Note:: Multiple independent outputs are allowed using columns of Y


    """
    def __init__(self, X, Y, kernel, likelihood, mean_function=None, inference_method=None, name='gp', Y_metadata=None, normalizer=False):
        super(GP, self).__init__(name)

        assert X.ndim == 2
        if isinstance(X, (ObsAr, VariationalPosterior)):
            self.X = X.copy()
        else: self.X = ObsAr(X)

        self.num_data, self.input_dim = self.X.shape

        assert Y.ndim == 2
        logger.info("initializing Y")

        if normalizer is True:
            self.normalizer = MeanNorm()
        elif normalizer is False:
            self.normalizer = None
        else:
            self.normalizer = normalizer

        if self.normalizer is not None:
            self.normalizer.scale_by(Y)
            self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
            self.Y = Y
        elif isinstance(Y, np.ndarray):
            self.Y = ObsAr(Y)
            self.Y_normalized = self.Y
        else:
            self.Y = Y

        if Y.shape[0] != self.num_data:
            #There can be cases where we want inputs than outputs, for example if we have multiple latent
            #function values
            warnings.warn("There are more rows in your input data X, \
                         than in your output data Y, be VERY sure this is what you want")
        _, self.output_dim = self.Y.shape

        assert ((Y_metadata is None) or isinstance(Y_metadata, dict))
        self.Y_metadata = Y_metadata

        assert isinstance(kernel, kern.Kern)
        #assert self.input_dim == kernel.input_dim
        self.kern = kernel

        assert isinstance(likelihood, likelihoods.Likelihood)
        self.likelihood = likelihood

        #handle the mean function
        self.mean_function = mean_function
        if mean_function is not None:
            assert isinstance(self.mean_function, Mapping)
            assert mean_function.input_dim == self.input_dim
            assert mean_function.output_dim == self.output_dim
            self.link_parameter(mean_function)

        #find a sensible inference method
        logger.info("initializing inference method")
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian) or isinstance(likelihood, likelihoods.MixedNoise):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation.EP()
                print("defaulting to ", inference_method, "for latent function inference")
        self.inference_method = inference_method

        logger.info("adding kernel and likelihood as parameters")
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior = None

        # The predictive variable to be used to predict using the posterior object's
        # woodbury_vector and woodbury_inv is defined as predictive_variable
        # as long as the posterior has the right woodbury entries.
        # It is the input variable used for the covariance between
        # X_star and the posterior of the GP.
        # This is usually just a link to self.X (full GP) or self.Z (sparse GP).
        # Make sure to name this variable and the predict functions will "just work"
        # In maths the predictive variable is:
        #         K_{xx} - K_{xp}W_{pp}^{-1}K_{px}
        #         W_{pp} := \texttt{Woodbury inv}
        #         p := _predictive_variable

    @property
    def _predictive_variable(self):
        return self.X

    def set_XY(self, X=None, Y=None):
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray
        """
        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            if self.X in self.parameters:
                # LVM models
                if isinstance(self.X, VariationalPosterior):
                    assert isinstance(X, type(self.X)), "The given X must have the same type as the X in the model!"
                    self.unlink_parameter(self.X)
                    self.X = X
                    self.link_parameter(self.X)
                else:
                    self.unlink_parameter(self.X)
                    from ..core import Param
                    self.X = Param('latent mean',X)
                    self.link_parameter(self.X)
            else:
                self.X = ObsAr(X)
        self.update_model(True)

    def set_X(self,X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        self.set_XY(X=X)

    def set_Y(self,Y):
        """
        Set the output data of the model

        :param X: output observations
        :type X: np.ndarray
        """
        self.set_XY(Y=Y)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        if kern is None:
            kern = self.kern

        Kx = kern.K(self._predictive_variable, Xnew)
        mu = np.dot(Kx.T, self.posterior.woodbury_vector)
        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)
        if full_cov:
            Kxx = kern.K(Xnew)
            if self.posterior.woodbury_inv.ndim == 2:
                var = Kxx - np.dot(Kx.T, np.dot(self.posterior.woodbury_inv, Kx))
            elif self.posterior.woodbury_inv.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],Kxx.shape[1],self.posterior.woodbury_inv.shape[2]))
                from ..util.linalg import mdot
                for i in range(var.shape[2]):
                    var[:, :, i] = (Kxx - mdot(Kx.T, self.posterior.woodbury_inv[:, :, i], Kx))
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            if self.posterior.woodbury_inv.ndim == 2:
                var = (Kxx - np.sum(np.dot(self.posterior.woodbury_inv.T, Kx) * Kx, 0))[:,None]
            elif self.posterior.woodbury_inv.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],self.posterior.woodbury_inv.shape[2]))
                for i in range(var.shape[1]):
                    var[:, i] = (Kxx - (np.sum(np.dot(self.posterior.woodbury_inv[:, :, i].T, Kx) * Kx, 0)))
            var = var
        #add in the mean function
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)

        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        Note: If you want the predictive quantiles (e.g. 95% confidence interval) use :py:func:"~GPy.core.gp.GP.predict_quantiles".
        """
        #predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)
        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(mu), self.normalizer.inverse_variance(var)

        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var, full_cov, Y_metadata=Y_metadata)
        return mean, var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param kern: optional kernel to use for prediction
        :type predict_kw: dict
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.output_dim), np.ndarray (Xnew x self.output_dim)]
        """
        m, v = self._raw_predict(X,  full_cov=False, kern=kern)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        return self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata=Y_metadata)

    def predictive_gradients(self, Xnew):
        """
        Compute the derivatives of the predicted latent function with respect to X*

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        derivatives of the mean and variance. Resulting arrays are sized:
         dmu_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).

        Note that this is not the same as computing the mean and variance of the derivative of the function!

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
        :param X: The points at which to get the predictive gradients
        :type X: np.ndarray (Xnew x self.input_dim)
        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        """
        dmu_dX = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))
        for i in range(self.output_dim):
            dmu_dX[:,:,i] = self.kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self.X)

        # gradients wrt the diagonal part k_{xx}
        dv_dX = self.kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        alpha = -2.*np.dot(self.kern.K(Xnew, self.X),self.posterior.woodbury_inv)
        dv_dX += self.kern.gradients_X(alpha, Xnew, self.X)
        return dmu_dX, dv_dX


    def predict_jacobian(self, Xnew, kern=None, full_cov=True):
        """
        Compute the derivatives of the posterior of the GP.

        Given a set of points at which to predict X* (size [N*,Q]), compute the
        mean and variance of the derivative. Resulting arrays are sized:

         dL_dX* -- [N*, Q ,D], where D is the number of output in this GP (usually one).
          Note that this is the mean and variance of the derivative,
          not the derivative of the mean and variance! (See predictive_gradients for that)

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
          If there is missing data, it is not implemented for now, but
          there will be one output variance per output dimension.

        :param X: The points at which to get the predictive gradients.
        :type X: np.ndarray (Xnew x self.input_dim)
        :param kern: The kernel to compute the jacobian for.
        :param boolean full_cov: whether to return the full covariance of the jacobian.

        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q,(D)) ]

        Note: We always return sum in input_dim gradients, as the off-diagonals
        in the input_dim are not needed for further calculations.
        This is a compromise for increase in speed. Mathematically the jacobian would
        have another dimension in Q.
        """
        if kern is None:
            kern = self.kern

        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)

        dK_dXnew_full = np.empty((self._predictive_variable.shape[0], Xnew.shape[0], Xnew.shape[1]))
        for i in range(self._predictive_variable.shape[0]):
            dK_dXnew_full[i] = kern.gradients_X([[1.]], Xnew, self._predictive_variable[[i]])

        if full_cov:
            dK2_dXdX = kern.gradients_XX([[1.]], Xnew)
        else:
            dK2_dXdX = kern.gradients_XX_diag([[1.]], Xnew)

        def compute_cov_inner(wi):
            if full_cov:
                # full covariance gradients:
                var_jac = dK2_dXdX - np.einsum('qnm,miq->niq', dK_dXnew_full.T.dot(wi), dK_dXnew_full)
            else:
                var_jac = dK2_dXdX - np.einsum('qim,miq->iq', dK_dXnew_full.T.dot(wi), dK_dXnew_full)
            return var_jac

        if self.posterior.woodbury_inv.ndim == 3: # Missing data:
            if full_cov:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[0],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
            else:
                var_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))
                for d in range(self.posterior.woodbury_inv.shape[2]):
                    var_jac[:, :, d] = compute_cov_inner(self.posterior.woodbury_inv[:, :, d])
        else:
            var_jac = compute_cov_inner(self.posterior.woodbury_inv)
        return mean_jac, var_jac

    def predict_wishard_embedding(self, Xnew, kern=None, mean=True, covariance=True):
        """
        Predict the wishard embedding G of the GP. This is the density of the
        input of the GP defined by the probabilistic function mapping f.
        G = J_mean.T*J_mean + output_dim*J_cov.

        :param array-like Xnew: The points at which to evaluate the magnification.
        :param :py:class:`~GPy.kern.Kern` kern: The kernel to use for the magnification.

        Supplying only a part of the learning kernel gives insights into the density
        of the specific kernel part of the input function. E.g. one can see how dense the
        linear part of a kernel is compared to the non-linear part etc.
        """
        if kern is None:
            kern = self.kern

        mu_jac, var_jac = self.predict_jacobian(Xnew, kern, full_cov=False)
        mumuT = np.einsum('iqd,ipd->iqp', mu_jac, mu_jac)
        Sigma = np.zeros(mumuT.shape)
        if var_jac.ndim == 3:
            Sigma[(slice(None), )+np.diag_indices(Xnew.shape[1], 2)] = var_jac.sum(-1)
        else:
            Sigma[(slice(None), )+np.diag_indices(Xnew.shape[1], 2)] = self.output_dim*var_jac
        G = 0.
        if mean:
            G += mumuT
        if covariance:
            G += Sigma
        return G

    def predict_magnification(self, Xnew, kern=None, mean=True, covariance=True):
        """
        Predict the magnification factor as

        sqrt(det(G))

        for each point N in Xnew
        """
        G = self.predict_wishard_embedding(Xnew, kern, mean, covariance)
        from ..util.linalg import jitchol
        mag = np.empty(Xnew.shape[0])
        for n in range(Xnew.shape[0]):
            try:
                mag[n] = np.sqrt(np.exp(2*np.sum(np.log(np.diag(jitchol(G[n, :, :]))))))
            except:
                mag[n] = np.sqrt(np.linalg.det(G[n, :, :]))
        return mag

    def posterior_samples_f(self,X,size=10, full_cov=True):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (N x samples)
        """
        m, v = self._raw_predict(X,  full_cov=full_cov)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        v = v.reshape(m.size,-1) if len(v.shape)==3 else v
        if not full_cov:
            fsim = np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
        else:
            fsim = np.random.multivariate_normal(m.flatten(), v, size).T

        return fsim

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations, a Numpy array (N x samples).
        """
        fsim = self.posterior_samples_f(X, size, full_cov=full_cov)
        Ysim = self.likelihood.samples(fsim, Y_metadata=Y_metadata)
        return Ysim

    def plot_f(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=True,
        linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx',
        apply_link=False):
        """
        Plot the GP's view of the world, where the data is normalized and before applying a likelihood.
        This is a call to plot with plot_raw=True.
        Data will not be plotted in this, as the GP's view of the world
        may live in another space, or units then the data.

        Can plot only part of the data and part of the posterior functions
        using which_data_rowsm which_data_ycols.

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_ycols: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param Y_metadata: additional data associated with Y which may be needed
        :type Y_metadata: dict
        :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        :param apply_link: if there is a link function of the likelihood, plot the link(f*) rather than f*
        :type apply_link: boolean
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
                                     data_symbol=data_symbol, apply_link=apply_link, **kw)

    def plot(self, plot_limits=None, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=[],
        levels=20, samples=0, fignum=None, ax=None, resolution=None,
        plot_raw=False, linecol=None,fillcol=None, Y_metadata=None,
        data_symbol='kx', predict_kw=None, plot_training_data=True, samples_y=0, apply_link=False):
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
        :type which_data_ycols: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        :type levels: int
        :param samples: the number of a posteriori samples to plot, p(f*|y)
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param Y_metadata: additional data associated with Y which may be needed
        :type Y_metadata: dict
        :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        :param plot_training_data: whether or not to plot the training points
        :type plot_training_data: boolean
        :param samples_y: the number of a posteriori samples to plot, p(y*|y)
        :type samples_y: int
        :param apply_link: if there is a link function of the likelihood, plot the link(f*) rather than f*, when plotting posterior samples f
        :type apply_link: boolean
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
                                     data_symbol=data_symbol, predict_kw=predict_kw,
                                     plot_training_data=plot_training_data, samples_y=samples_y, apply_link=apply_link, **kw)


    def plot_data(self, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        fignum=None, ax=None, data_symbol='kx'):
        """
        Plot the training data
          - For higher dimensions than two, use fixed_inputs to plot the data points with some of the inputs fixed.

        Can plot only part of the data
        using which_data_rows and which_data_ycols.

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_ycols: 'all' or a list of integers
        :param visible_dims: an array specifying the input dimensions to plot (maximum two)
        :type visible_dims: a numpy array
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        :type levels: int
        :param samples: the number of a posteriori samples to plot, p(f*|y)
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        return models_plots.plot_data(self, which_data_rows,
                                     which_data_ycols, visible_dims,
                                     fignum, ax, data_symbol, **kw)


    def errorbars_trainset(self, which_data_rows='all',
            which_data_ycols='all', fixed_inputs=[], fignum=None, ax=None,
            linecol=None, data_symbol='kx', predict_kw=None, plot_training_data=True,lw=None):

        """
        Plot the posterior error bars corresponding to the training data
          - For higher dimensions than two, use fixed_inputs to plot the data points with some of the inputs fixed.

        Can plot only part of the data
        using which_data_rows and which_data_ycols.

        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_rows: 'all' or a list of integers
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param plot_training_data: whether or not to plot the training points
        :type plot_training_data: boolean
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import models_plots
        kw = {}
        if lw is not None:
            kw['lw'] = lw
        return models_plots.errorbars_trainset(self, which_data_rows, which_data_ycols, fixed_inputs,
                                    fignum, ax, linecol, data_symbol,
                                    predict_kw, plot_training_data, **kw)


    def plot_magnification(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, legend=True,
                plot_limits=None,
                aspect='auto', updates=False, plot_inducing=True, kern=None, **kwargs):

        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_magnification(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, **kwargs)


    def input_sensitivity(self, summarize=True):
        """
        Returns the sensitivity for each dimension of this model
        """
        return self.kern.input_sensitivity(summarize=summarize)

    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        """
        self.inference_method.on_optimization_start()
        try:
            super(GP, self).optimize(optimizer, start, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, calling on_optimization_end() to round things up")
            self.inference_method.on_optimization_end()
            raise

    def infer_newX(self, Y_new, optimize=True):
        """
        Infer X for the new observed data *Y_new*.

        :param Y_new: the new observed data for inference
        :type Y_new: numpy.ndarray
        :param optimize: whether to optimize the location of new X (True by default)
        :type optimize: boolean
        :return: a tuple containing the posterior estimation of X and the model that optimize X
        :rtype: (:class:`~GPy.core.parameterization.variational.VariationalPosterior` and numpy.ndarray, :class:`~GPy.core.model.Model`)
        """
        from ..inference.latent_function_inference.inferenceX import infer_newX
        return infer_newX(self, Y_new, optimize=optimize)

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density(y_test, mu_star, var_star, Y_metadata=Y_metadata)

    def log_predictive_density_sampling(self, x_test, y_test, Y_metadata=None, num_samples=1000):
        """
        Calculation of the log predictive density by sampling

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        :param num_samples: number of samples to use in monte carlo integration
        :type num_samples: int
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density_sampling(y_test, mu_star, var_star, Y_metadata=Y_metadata, num_samples=num_samples)

