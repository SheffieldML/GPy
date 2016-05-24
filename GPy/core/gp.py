# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from GPy.core.model import Model
from paramz import ObsAr
from .mapping import Mapping
from .. import likelihoods
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from GPy.core.parameterization.variational import VariationalPosterior

import logging
import warnings
from GPy.util.normalizer import MeanNorm
from ..util.linalg import pdinv
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

        if self.kern._effective_input_dim != self.X.shape[1]:
            warnings.warn("Your kernel has a different input dimension {} then the given X dimension {}. Be very sure this is what you want and you have not forgotten to set the right input dimenion in your kernel".format(self.kern._effective_input_dim, self.X.shape[1]))

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
                print("defaulting to " + str(inference_method) + " for latent function inference")
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
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

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
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        """
        Predict the function(s) at the new point(s) Xnew. This includes the likelihood
        variance added to the predicted underlying function (usually referred to as f).

        In order to predict without adding in the likelihood give
        `include_likelihood=False`, or refer to self.predict_noiseless().

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :param bool include_likelihood: Whether or not to add likelihood noise to the predicted underlying latent function f.

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

        if include_likelihood:
            # now push through likelihood
            if likelihood is None:
                likelihood = self.likelihood
            mu, var = likelihood.predictive_values(mu, var, full_cov, Y_metadata=Y_metadata)
        return mu, var

    def predict_noiseless(self,  Xnew, full_cov=False, Y_metadata=None, kern=None):
        """
        Convenience function to predict the underlying function of the GP (often
        referred to as f) without adding the likelihood variance on the
        prediction function.

        This is most likely what you want to use for your predictions.

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
        return self.predict(Xnew, full_cov, Y_metadata, kern, None, False)

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None):
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
        if likelihood is None:
            likelihood = self.likelihood
        return likelihood.predictive_quantiles(m, v, quantiles, Y_metadata=Y_metadata)

    def predictive_gradients(self, Xnew, kern=None):
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
        if kern is None:
            kern = self.kern
        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],self.output_dim))

        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)

        # gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable), self.posterior.woodbury_inv)
        dv_dX += kern.gradients_X(alpha, Xnew, self._predictive_variable)
        return mean_jac, dv_dX


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
        one = np.ones((1,1))
        for i in range(self._predictive_variable.shape[0]):
            dK_dXnew_full[i] = kern.gradients_X(one, Xnew, self._predictive_variable[[i]])

        if full_cov:
            dK2_dXdX = kern.gradients_XX(one, Xnew)
        else:
            dK2_dXdX = kern.gradients_XX_diag(one, Xnew)

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

    def predict_wishart_embedding(self, Xnew, kern=None, mean=True, covariance=True):
        """
        Predict the wishart embedding G of the GP. This is the density of the
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

    def predict_wishard_embedding(self, Xnew, kern=None, mean=True, covariance=True):
        warnings.warn("Wrong naming, use predict_wishart_embedding instead. Will be removed in future versions!", DeprecationWarning)
        return self.predict_wishart_embedding(Xnew, kern, mean, covariance)

    def predict_magnification(self, Xnew, kern=None, mean=True, covariance=True, dimensions=None):
        """
        Predict the magnification factor as

        sqrt(det(G))

        for each point N in Xnew.

        :param bool mean: whether to include the mean of the wishart embedding.
        :param bool covariance: whether to include the covariance of the wishart embedding.
        :param array-like dimensions: which dimensions of the input space to use [defaults to self.get_most_significant_input_dimensions()[:2]]
        """
        G = self.predict_wishard_embedding(Xnew, kern, mean, covariance)
        if dimensions is None:
            dimensions = self.get_most_significant_input_dimensions()[:2]
        G = G[:, dimensions][:,:,dimensions]
        from ..util.linalg import jitchol
        mag = np.empty(Xnew.shape[0])
        for n in range(Xnew.shape[0]):
            try:
                mag[n] = np.sqrt(np.exp(2*np.sum(np.log(np.diag(jitchol(G[n, :, :]))))))
            except:
                mag[n] = np.sqrt(np.linalg.det(G[n, :, :]))
        return mag

    def posterior_samples_f(self,X, size=10, full_cov=True, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        m, v = self._raw_predict(X,  full_cov=full_cov, **predict_kwargs)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)

        def sim_one_dim(m, v):
            if not full_cov:
                return np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
            else:
                return np.random.multivariate_normal(m.flatten(), v, size).T

        if self.output_dim == 1:
            return sim_one_dim(m, v)
        else:
            fsim = np.empty((self.output_dim, self.num_data, size))
            for d in range(self.output_dim):
                if full_cov and v.ndim == 3:
                    fsim[d] = sim_one_dim(m[:, d], v[:, :, d])
                elif (not full_cov) and v.ndim == 2:
                    fsim[d] = sim_one_dim(m[:, d], v[:, d])
                else:
                    fsim[d] = sim_one_dim(m[:, d], v)
        return fsim

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None, likelihood=None, **predict_kwargs):
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
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        fsim = self.posterior_samples_f(X, size, full_cov=full_cov, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[0]):
                fsim[d] = likelihood.samples(fsim[d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim

    def input_sensitivity(self, summarize=True):
        """
        Returns the sensitivity for each dimension of this model
        """
        return self.kern.input_sensitivity(summarize=summarize)

    def get_most_significant_input_dimensions(self, which_indices=None):
        return self.kern.get_most_significant_input_dimensions(which_indices)

    def optimize(self, optimizer=None, start=None, messages=False, max_iters=1000, ipython_notebook=True, clear_after_finish=False, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        :param bool ipython_notebook: whether to use ipython notebook widgets or not.
        :param bool clear_after_finish: if in ipython notebook, we can clear the widgets after optimization.
        """
        self.inference_method.on_optimization_start()
        try:
            super(GP, self).optimize(optimizer, start, messages, max_iters, ipython_notebook, clear_after_finish, **kwargs)
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

    def CCD(self):
        """
        Code is based on implementation within GPStuff, INLA and the original Sanchez and Sanchez paper (2005)
        """
        modal_params = self.optimizer_array[:].copy()
        num_free_params = modal_params.shape[0]

        # Calculate the numerical hessian for *every* parameter
        H = self.numerical_parameter_hessian()

        try:
            curv = pdinv(H)[0]
        except np.linalg.linalg.LinAlgError:
            print("Hessian is not positive definite, ensure parameters are at their MAP solution by optimizing, if that doesn't work try modifying the step length parameter")
            return
        # CCD points are calculated for unit circle, so we take the principle components and scale them such that we have a unit sphere in our parameter space, which we can then map back to parameter space
        [w, V] = np.linalg.eig(curv)
        z = (V*np.sqrt(w[None,:])).T

        # Calculate points, we will do just CCD for now

        # First we build the hadamarx matrix from the paper that encodes *all* the points possible in the num_free_params dimensional space.
        grow = np.sum(num_free_params >= np.array([1, 2, 3, 4, 6, 7, 9, 12, 18, 22, 30, 39, 53, 70, 93]))

        H0 = 1

        #Build the design matrix that holds *every* corner on the num_free_params dimensional hyper-cube (the hadamard matrix), H in the paper
        for i in range(grow):
            H0 = np.vstack([np.hstack([H0, H0]), 
                            np.hstack([H0, -H0])])
        
        # For each additional parameter our dimensional space increases, as does the number of CCD points we require to do the approximate integration. CCD is a method of deciding which points should be included and which can be ignored without a significant effect. It would be ideal but too computationally expensive - when we have many parameters - to use all the corners of the hypercube in our integration
        # See Sanchez and Sanchez (2005) for details. on how the points are chosen
        walsh_inds = np.array([1, 2, 4, 8, 15, 16, 32, 51, 64, 85, 106, 128, 150, 171, 219, 237, 247, 256, 279, 297, 455, 512, 537, 557, 597, 643, 803, 863, 898, 1024, 1051, 1070, 1112, 1169, 1333, 1345, 1620, 1866, 2048, 2076, 2085, 2158, 2372, 2456, 2618, 2800, 2873, 3127, 3284, 3483, 3557, 3763, 4096, 4125, 4135, 4176, 4435, 4459, 4469, 4497, 4752, 5255, 5732, 5801, 5915, 6100, 6369, 6907, 7069, 8192, 8263, 8351, 8422, 8458, 8571, 8750, 8858, 9124, 9314, 9500, 10026, 10455, 10556, 11778, 11885, 11984, 13548, 14007, 14514, 14965, 15125, 15554, 16384, 16457, 16517, 16609, 16771, 16853, 17022, 17453, 17891, 18073, 18562, 18980, 19030, 19932, 20075, 20745, 21544, 22633, 23200, 24167, 25700, 26360, 26591, 26776, 28443, 28905, 29577, 32705])

        used_walsh_inds = walsh_inds[:num_free_params]

        # For every additional parameters, we get an additional number of points (which as we decide to drop points using Sanchezes rules, won't necessarily grow at the same rate)
        ccd_points = H0[:, used_walsh_inds]
        
        # ccd only gives corner points, so lets add an additional one for the centre point
        ccd_points = np.vstack([np.zeros((1,num_free_params)), ccd_points])

        # Now we add the points on the unit hyper-sphere that arent on the corners
        for i in range(num_free_params):
            top_edge = np.zeros(num_free_params)
            top_edge[i] = np.sqrt(num_free_params)
            bottom_edge = np.zeros(num_free_params)
            bottom_edge[i] = -np.sqrt(num_free_params)
            ccd_points = np.vstack([ccd_points, top_edge, bottom_edge])

        # Find the appropriate scaling such that the edges lie on a boundry of equal density
        # First find the density at the mode
        log_marginal = lambda p: -self._objective(p)
        mode_density = log_marginal(modal_params)  # FIXME: We really want to evaluate the log_marginal not the negative objective as it makes more sense, but they are equivalent in this case

        # Treating the posterior over hyperparameters as a standard normal, moving z*sqrt(2) should make the likelihood drop by 1, as z should be acting as a unit vector along the principle components of the posterior. 

        scalings = np.ones_like(ccd_points)
        # The below code makes me feel dirty. Pythonise it!
        # This is naive scaling, assuming that it is well approximated by a standard normal, in practice you might want to scale in different directions seperately (split normal approximation)
        for j in range(num_free_params*2):
            temp = np.zeros((1, num_free_params))
            if j % 2:
                direction = -1
            else:
                direction = 1
            ind = np.ceil(j // 2)  # This integer division is required, will not work with python 3
            temp[0, ind] = direction
            
            # This is the point mapped onto the contour of a unit gaussian, stretched by the eigenvectors over the marginal likelihood
            contour_point = modal_params + np.sqrt(2)*temp.dot(z)
            point_density = log_marginal(contour_point)

            if mode_density > point_density:
                # The scale is based on the amount this point must be scaled in order to be on the contour of the stretched standard normal (stretched by principle components)
                scale = np.sqrt(1.0 / (mode_density - point_density))
            else:
                print("Contour point is higher than mode, must be multimodal or mode not found")
                scale = 1  # Print a warning and say "dont scale in this direction"

            # Set the scaling for all dimensions where the point is in this direction, when looking at this dimension, clipped.
            scalings[ccd_points[:, ind]*direction > 0.0, ind] = np.maximum(np.minimum(scale, 10.0), 1/10.0)
            
        # Make the points *slightly* further out. Aki Vehtari has shown this to be more accurate
        over_scale = 1.1
        ccd_points = over_scale*scalings*ccd_points

        # We have the scaled ccd_points, now we need to map them into parameter space by projecting along principle components and shifting to mode
        param_points = ccd_points.dot(z) + modal_params

        # Evaluate log marginal at all parameter points
        point_densities = np.ones(param_points.shape[0])*np.nan
        for point_ind, param_point in enumerate(param_points):
            point_densities[point_ind] = log_marginal(param_point)
        
        # Remove nan densities
        non_nan_densities = np.isfinite(point_densities).flatten()
        point_densities = point_densities[non_nan_densities]
        param_points = param_points[non_nan_densities, :]

        point_densities = point_densities - np.max(point_densities)  # Why don't we have to deal with this shift?
        point_densities = np.exp(point_densities)

        if num_free_params > 0:
            # Each section that the ccd point represents needs an associated weight. Since every point is on the same contour, all weights are the same, except the centre weight
            point_weights = 1.0/((2.0*np.pi)**(-num_free_params*0.5) * np.exp(-0.5*num_free_params*over_scale**2) * (param_points.shape[0]-1)*over_scale**2)
            centre_weight = (2*np.pi)**(num_free_params*0.5) * (1 - 1.0/over_scale**2)
            # normalize
            point_weights = point_weights / centre_weight
            centre_weight = 1.0

            point_densities[1:] *= point_weights

        # Normalize density
        point_densities /= point_densities.sum()

        # Remove small density points
        non_small_densities = (point_densities > 0.01/point_densities.size).flatten()
        point_densities = point_densities[non_small_densities]
        param_points = param_points[non_small_densities, :]
        point_densities /= point_densities.sum()

        # for dimension_ind, dimension in enumerate(num_free_params):
            # # Due to the way we built the ccd_points up, every other one changes to the next direction
            # ind = np.ceil(dimension_ind / 2.0)
            # if dimension_ind % 2:
                # direction = 1
            # else:
                # direction = -1
            # print(z[param_dir,:])
        # for direction in [1, -1]:
            # for parameter in range(num_free_params):
                #If the log likelihood doesn't even drop when we move away from the mode, then we havent found the modal points, or there are multiple modes in the posterior (bad)
                # upper_density = self._objective(modal_params + np.sqrt(2)*np.dot(direction,z))

                # We want to work out the scaling for every point, so we first figure out which way the vector is scaling from the mode, and scale in that direction
                

        # fig = plt.figure()
        # if num_free_params == 3:
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(ccd_points[:,0], ccd_points[:,1], ccd_points[:,2]) 
        # elif num_free_params == 2:
            # plt.scatter(ccd_points[:,0], ccd_points[:,1])
        # plt.show()
        import paramz

        transformed_points = param_points.copy()

        # Need to transform the points to the space of parameters again
        f = np.ones(self.size).astype(bool)
        f[self.constraints[paramz.transformations.__fixed__]] = paramz.transformations.FIXED
        new_t_points = [c.f(transformed_points[:, ind[f[ind]]]) for c, ind in self.constraints.items() if c != paramz.transformations.__fixed__][0]

        return new_t_points, point_densities

    def numerical_parameter_hessian(self, step_length=1e-3):
        """
        Calculate the numerical hessian for the posterior of the parameters

        Often this will want to be done at the modal values of the parameters so the optimizer should be called first
        
        H = [d^2L_dp1dp1, d^2L_dp1dp2, d^2L_dp1dp3, ...;
             d^2L_dp2dp1, d^2L_dp2dp2, d^2L_dp3dp3, ...;
            ...
             ]
        Where the vector p is all of the parameters in param_array, not just parameters of this model class itself
        """
        # We use optimizer array as we want to ignore any parameters with fixed values
        num_child_params = self.optimizer_array.shape[0]
        
        H = np.eye(num_child_params)

        initial_params = self.optimizer_array[:].copy()
        # Calculate gradient of marginal likelihood moving one at a time
        for i in range(num_child_params):
            # Pick out the e vector with one in the parameter that we wish to move
            e = H[:,i]
            # Get parameters required to evaluate by nudging left and right
            left = initial_params - step_length*e
            right = initial_params + step_length*e

            left_grad = self._grads(left)
            right_grad = self._grads(right)

            finite_diff = (right_grad - left_grad) / (2*step_length)

            #Put in the correct row of the hessian
            H[:,i] = finite_diff

        return H
