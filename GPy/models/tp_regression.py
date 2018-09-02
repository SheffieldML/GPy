# Copyright (c) 2017 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import Model
from ..core.parameterization import Param
from ..core import Mapping
from ..kern import Kern, RBF
from ..inference.latent_function_inference import ExactStudentTInference
from ..util.normalizer import Standardize

import numpy as np
from scipy import stats
from paramz import ObsAr
from paramz.transformations import Logexp

import warnings


class TPRegression(Model):
    """
    Student-t Process model for regression, as presented in

       Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes.
       In Artificial Intelligence and Statistics (pp. 877-885).

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    :param deg_free: initial value for the degrees of freedom hyperparameter
    :param Norm normalizer: [False]

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, deg_free=5., normalizer=None, mean_function=None, name='TP regression'):
        super(TPRegression, self).__init__(name=name)
        # X
        assert X.ndim == 2
        self.set_X(X)
        self.num_data, self.input_dim = self.X.shape

        # Y
        assert Y.ndim == 2
        if normalizer is True:
            self.normalizer = Standardize()
        elif normalizer is False:
            self.normalizer = None
        else:
            self.normalizer = normalizer

        self.set_Y(Y)

        if Y.shape[0] != self.num_data:
            # There can be cases where we want inputs than outputs, for example if we have multiple latent
            # function values
            warnings.warn("There are more rows in your input data X, \
                                 than in your output data Y, be VERY sure this is what you want")
        self.output_dim = self.Y.shape[1]

        # Kernel
        kernel = kernel or RBF(self.X.shape[1])
        assert isinstance(kernel, Kern)
        self.kern = kernel
        self.link_parameter(self.kern)

        if self.kern._effective_input_dim != self.X.shape[1]:
            warnings.warn(
                "Your kernel has a different input dimension {} then the given X dimension {}. Be very sure this is "
                "what you want and you have not forgotten to set the right input dimenion in your kernel".format(
                    self.kern._effective_input_dim, self.X.shape[1]))

        # Mean function
        self.mean_function = mean_function
        if mean_function is not None:
            assert isinstance(self.mean_function, Mapping)
            assert mean_function.input_dim == self.input_dim
            assert mean_function.output_dim == self.output_dim
            self.link_parameter(mean_function)

        # Degrees of freedom
        self.nu = Param('deg_free', float(deg_free), Logexp())
        self.link_parameter(self.nu)

        # Inference
        self.inference_method = ExactStudentTInference()
        self.posterior = None
        self._log_marginal_likelihood = None

        # Insert property for plotting (not used)
        self.Y_metadata = None

    def _update_posterior_dof(self, dof, which):
        if self.posterior is not None:
            self.posterior.nu = dof

    @property
    def _predictive_variable(self):
        return self.X

    def set_XY(self, X, Y):
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray or ObsAr
        """
        self.update_model(False)
        self.set_Y(Y)
        self.set_X(X)
        self.update_model(True)

    def set_X(self, X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        assert isinstance(X, np.ndarray)
        state = self.update_model()
        self.update_model(False)
        self.X = ObsAr(X)
        self.update_model(state)

    def set_Y(self, Y):
        """
        Set the output data of the model

        :param Y: output observations
        :type Y: np.ndarray or ObsArray
        """
        assert isinstance(Y, (np.ndarray, ObsAr))
        state = self.update_model()
        self.update_model(False)
        if self.normalizer is not None:
            self.normalizer.scale_by(Y)
            self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
            self.Y = Y
        else:
            self.Y = ObsAr(Y) if isinstance(Y, np.ndarray) else Y
            self.Y_normalized = self.Y
        self.update_model(state)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in this class this method re-performs inference, recalculating the posterior, log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, grad_dict = self.inference_method.inference(self.kern,
                                                                                                   self.X,
                                                                                                   self.Y_normalized,
                                                                                                   self.nu + 2 + np.finfo(
                                                                                                       float).eps,
                                                                                                   self.mean_function)
        self.kern.update_gradients_full(grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(grad_dict['dL_dm'], self.X)
        self.nu.gradient = grad_dict['dL_dnu']

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood or self.inference()[1]

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = MVN\left(\nu + N,f*| K_{x*x}(K_{xx})^{-1}Y,
                        \frac{\nu + \beta - 2}{\nu + N - 2}K_{x*x*} - K_{xx*}(K_{xx})^{-1}K_{xx*}\right)
            \nu := \texttt{Degrees of freedom}
        """
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew,
                                              pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var

    def predict(self, Xnew, full_cov=False, kern=None, **kwargs):
        """
        Predict the function(s) at the new point(s) Xnew. For Student-t processes, this method is equivalent to
        predict_noiseless as no likelihood is included in the model.
        """
        return self.predict_noiseless(Xnew, full_cov=full_cov, kern=kern)

    def predict_noiseless(self, Xnew, full_cov=False, kern=None):
        """
        Predict the underlying function  f at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :param kern: The kernel to use for prediction (defaults to the model kern).

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim.
           If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.
        """
        # Predict the latent function values
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        # Un-apply normalization
        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(mu), self.normalizer.inverse_variance(var)

        return mu, var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), kern=None, **kwargs):
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
        mu, var = self._raw_predict(X, full_cov=False, kern=kern)
        quantiles = [stats.t.ppf(q / 100., self.nu + 2 + self.num_data) * np.sqrt(var) + mu for q in quantiles]

        if self.normalizer is not None:
            quantiles = [self.normalizer.inverse_mean(q) for q in quantiles]

        return quantiles

    def posterior_samples(self, X, size=10, full_cov=False, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X, equivalent to posterior_samples_f due to the absence of a likelihood.
        """
        return self.posterior_samples_f(X, size, full_cov=full_cov, **predict_kwargs)

    def posterior_samples_f(self, X, size=10, full_cov=True, **predict_kwargs):
        """
        Samples the posterior TP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        mu, var = self._raw_predict(X, full_cov=full_cov, **predict_kwargs)
        if self.normalizer is not None:
            mu, var = self.normalizer.inverse_mean(mu), self.normalizer.inverse_variance(var)

        def sim_one_dim(m, v):
            nu = self.nu + 2 + self.num_data
            v = np.diag(v.flatten()) if not full_cov else v
            Z = np.random.multivariate_normal(np.zeros(X.shape[0]), v, size).T
            g = np.tile(np.random.gamma(nu / 2., 2. / nu, size), (X.shape[0], 1))
            return m + Z / np.sqrt(g)

        if self.output_dim == 1:
            return sim_one_dim(mu, var)
        else:
            fsim = np.empty((self.output_dim, self.num_data, size))
            for d in range(self.output_dim):
                if full_cov and var.ndim == 3:
                    fsim[d] = sim_one_dim(mu[:, d], var[:, :, d])
                elif (not full_cov) and var.ndim == 2:
                    fsim[d] = sim_one_dim(mu[:, d], var[:, d])
                else:
                    fsim[d] = sim_one_dim(mu[:, d], var)
        return fsim
