# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import PosteriorExact as Posterior
import numpy as np
from . import LatentFunctionInference
from . import exact_gp_ops


class ExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    :param float jitter: constant added to the noise variance on the diagonal of
        ``K + noise`` before it is factorised. The default, 1e-8, is the historical
        GPy value. It makes the model GPy evaluates differ slightly from the one the
        parameters describe: the effective noise variance is ``variance + jitter``,
        a relative change of ``jitter / variance``, so it matters only when the noise
        variance is small (about 1e-3 or below). Pass ``jitter=0`` to evaluate the
        stated model exactly, at the cost of the extra numerical guard.

    """

    DEFAULT_JITTER = 1e-8

    def __init__(self, jitter=DEFAULT_JITTER):
        self.jitter = jitter  # self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        jitter = getattr(self, "jitter", self.DEFAULT_JITTER)
        if jitter != self.DEFAULT_JITTER:
            input_dict["jitter"] = jitter  # omitted at the default so older readers still load the file
        return input_dict

    def inference(
        self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None
    ):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        m = 0 if mean_function is None else mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y - m

        if K is None:
            K = kern.K(X)

        # CIP-000F Stage 2: pure ndarray LML / dL_dK path.
        log_marginal, dL_dK, alpha, LW, _Wi = exact_gp_ops.log_marginal_and_gradients(
            K, YYT_factor, variance, jitter=getattr(self, "jitter", self.DEFAULT_JITTER)
        )

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return (
            Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K),
            log_marginal,
            {"dL_dK": dL_dK, "dL_dthetaL": dL_dthetaL, "dL_dm": alpha},
        )

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5 * np.log(2 * np.pi) - 0.5 * np.log(c_diag) + 0.5 * (g**2) / c_diag
        # believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        # this is the negative marginal LOO
        return -neg_log_marginal_LOO
