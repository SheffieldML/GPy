# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import Posterior
from ...util.linalg import tdot, dtrtrs, pdinv
from ...util.sparse_linalg import parse_sparse_linalg, factor_I_plus_AA
from ...util import diag
import numpy as np
from . import LatentFunctionInference

log_2_pi = np.log(2 * np.pi)


class FITC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    :param sparse_linalg: Inducing-side factor mode (CIP-0013): ``'stable'``
        (default until LQ cutover), ``'lq'``, or ``'legacy'`` (not yet).
    :param float const_jitter: constant added to the diagonal of ``Kuu`` (the covariance of the
        inducing values) before it is factorised. The default is 1e-6. The sparse
        approximations used to differ in this convention (VarDTC 1e-8, FITC 1e-6), and so do other
        frameworks, so results agree across them only to about the jitter's relative effect
        when ``Kuu`` is ill conditioned. Pass the same value to compare like with like.
    """

    const_jitter = 1e-6

    def __init__(self, sparse_linalg="stable", const_jitter=const_jitter):
        self.const_jitter = const_jitter
        self._sparse_linalg = parse_sparse_linalg(sparse_linalg)

    @property
    def sparse_linalg(self):
        return self._sparse_linalg

    @sparse_linalg.setter
    def sparse_linalg(self, mode):
        self._sparse_linalg = parse_sparse_linalg(mode)

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
        assert mean_function is None, "inference with a mean function not implemented"

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        # make sure the noise is not hetero
        sigma_n = likelihood.gaussian_variance(Y_metadata)
        if sigma_n.size > 1:
            raise NotImplementedError("no hetero noise with this implementation of FITC")

        Kmm = kern.K(Z)
        Knn = kern.Kdiag(X)
        Knm = kern.K(X, Z)
        U = Knm

        # factor Kmm
        diag.add(Kmm, self.const_jitter)
        Kmmi, L, Li, _ = pdinv(Kmm)

        # compute beta_star, the effective noise precision
        LiUT = np.dot(Li, U.T)
        sigma_star = Knn + sigma_n - np.sum(np.square(LiUT), 0)
        beta_star = 1.0 / sigma_star

        # Whitened features F; factor I + F F.T via sparse_linalg (CIP-0013).
        F = LiUT * np.sqrt(beta_star)
        LA, _Agram, _Q = factor_I_plus_AA(F, self._sparse_linalg)

        # back substutue to get b, P, v
        URiy = np.dot(U.T * beta_star, Y)
        tmp, _ = dtrtrs(L, URiy, lower=1)
        b, _ = dtrtrs(LA, tmp, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
        P = tdot(tmp.T)

        # compute log marginal
        log_marginal = (
            -0.5 * num_data * output_dim * np.log(2 * np.pi)
            + -np.sum(np.log(np.diag(LA))) * output_dim
            + 0.5 * output_dim * np.sum(np.log(beta_star))
            + -0.5 * np.sum(np.square(Y.T * np.sqrt(beta_star)))
            + 0.5 * np.sum(np.square(b))
        )
        # compute dL_dR
        Uv = np.dot(U, v)
        dL_dR = (
            0.5
            * (
                np.sum(U * np.dot(U, P), 1)
                - 1.0 / beta_star
                + np.sum(np.square(Y), 1)
                - 2.0 * np.sum(Uv * Y, 1)
                + np.sum(np.square(Uv), 1)
            )
            * beta_star**2
        )

        # Compute dL_dKmm
        vvT_P = tdot(v) + P
        dL_dK = 0.5 * (Kmmi - vvT_P)
        KiU = np.dot(Kmmi, U.T)
        dL_dK += np.dot(KiU * dL_dR, KiU.T)

        # Compute dL_dU
        vY = np.dot(v, Y.T)
        dL_dU = vY - np.dot(vvT_P, U.T)
        dL_dU *= beta_star
        dL_dU -= 2.0 * KiU * dL_dR

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)
        grad_dict = {"dL_dKmm": dL_dK, "dL_dKdiag": dL_dR, "dL_dKnm": dL_dU.T, "dL_dthetaL": dL_dthetaL}

        # construct a posterior object
        post = Posterior(woodbury_inv=Kmmi - P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)

        return post, log_marginal, grad_dict
