"""Pure NumPy exact-GP inference primitives (CIP-000F Stage 2).

Array-in / array-out helpers with no ``Model`` / ``Kern`` / ``paramz`` coupling.
``ExactGaussianInference.inference`` unwraps parameters and delegates here.
"""

from __future__ import annotations

import numpy as np

from ...util.linalg import dpotrs, pdinv, tdot
from ...util import diag

log_2_pi = np.log(2 * np.pi)


def log_marginal_and_gradients(K, Y, noise_variance, jitter=1e-8):
    """Exact GP log marginal likelihood and kernel gradients from plain arrays.

    Parameters
    ----------
    K : array_like, shape (N, N)
        Prior covariance (without observation noise).
    Y : array_like, shape (N, D)
        Observations (mean already subtracted if a mean function is used).
    noise_variance : float or array_like broadcastable to the diagonal
        Gaussian observation noise variance.
    jitter : float
        Diagonal jitter added with the noise (matches historical GPy path).

    Returns
    -------
    log_marginal : float
    dL_dK : ndarray, shape (N, N)
        Gradient of the log marginal wrt ``K``.
    alpha : ndarray, shape (N, D)
        Posterior woodbury vector (``K_y^{-1} Y``).
    woodbury_chol : ndarray
        Cholesky factor of ``K + noise + jitter``.
    Wi : ndarray, shape (N, N)
        Inverse of ``K + noise + jitter``.
    """
    K = np.asarray(K, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    Ky = K.copy()
    diag.add(Ky, np.asarray(noise_variance, dtype=np.float64) + jitter)
    Wi, LW, _LWi, W_logdet = pdinv(Ky)
    alpha, _ = dpotrs(LW, Y, lower=1)
    log_marginal = 0.5 * (-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * Y))
    dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)
    return float(log_marginal), dL_dK, alpha, LW, Wi


def rbf_log_marginal(X, Y, kernel_variance, lengthscale, noise_variance, ARD=False, X2=None):
    """Exact GP LML for an RBF prior using only ndarray inputs.

    Suitable as a Stage 2 / Stage 2T seed: no GPy model or ``Param`` required.
    """
    from ...kern.src import rbf_ops

    if X2 is not None:
        raise ValueError("training LML uses a single design matrix X")
    K = rbf_ops.K(X, kernel_variance, lengthscale, X2=None, ARD=ARD)
    log_marginal, dL_dK, alpha, LW, Wi = log_marginal_and_gradients(K, Y, noise_variance)
    return log_marginal, {"K": K, "dL_dK": dL_dK, "alpha": alpha, "woodbury_chol": LW, "Wi": Wi}
