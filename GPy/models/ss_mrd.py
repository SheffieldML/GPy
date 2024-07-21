"""
The Maniforld Relevance Determination model with the spike-and-slab prior
"""

import numpy as np
from ..core import Model
from .ss_gplvm import SSGPLVM
from GPy.core.parameterization.variational import (
    SpikeAndSlabPrior,
    NormalPosterior,
    VariationalPrior,
)
from ..util.misc import param_to_array
from ..kern import RBF
from ..core import Param
from numpy.linalg.linalg import LinAlgError


class SSMRD(Model):
    def __init__(
        self,
        Ylist,
        input_dim,
        X=None,
        X_variance=None,
        Gammas=None,
        initx="PCA_concat",
        initz="permute",
        num_inducing=10,
        Zs=None,
        kernels=None,
        inference_methods=None,
        likelihoods=None,
        group_spike=True,
        pi=0.5,
        name="ss_mrd",
        Ynames=None,
        mpi_comm=None,
        IBP=False,
        alpha=2.0,
        taus=None,
    ):
        super(SSMRD, self).__init__(name)
        self.mpi_comm = mpi_comm
        self._PROPAGATE_ = False

        # initialize X for individual models
        X, X_variance, Gammas, fracs = self._init_X(
            Ylist, input_dim, X, X_variance, Gammas, initx
        )
        self.X = NormalPosterior(means=X, variances=X_variance)

        if kernels is None:
            kernels = [
                RBF(input_dim, lengthscale=1.0 / fracs, ARD=True)
                for i in range(len(Ylist))
            ]
        if Zs is None:
            Zs = [None] * len(Ylist)
        if likelihoods is None:
            likelihoods = [None] * len(Ylist)
        if inference_methods is None:
            inference_methods = [None] * len(Ylist)

        if IBP:
            self.var_priors = [
                IBPPrior_SSMRD(len(Ylist), input_dim, alpha=alpha)
                for i in range(len(Ylist))
            ]
        else:
            self.var_priors = [
                SpikeAndSlabPrior_SSMRD(
                    nModels=len(Ylist), pi=pi, learnPi=False, group_spike=group_spike
                )
                for i in range(len(Ylist))
            ]
        self.models = [
            SSGPLVM(
                y,
                input_dim,
                X=X.copy(),
                X_variance=X_variance.copy(),
                Gamma=Gammas[i],
                num_inducing=num_inducing,
                Z=Zs[i],
                learnPi=False,
                group_spike=group_spike,
                kernel=kernels[i],
                inference_method=inference_methods[i],
                likelihood=likelihoods[i],
                variational_prior=self.var_priors[i],
                IBP=IBP,
                tau=None if taus is None else taus[i],
                name="model_" + str(i),
                mpi_comm=mpi_comm,
                sharedX=True,
            )
            for i, y in enumerate(Ylist)
        ]
        self.link_parameters(*(self.models + [self.X]))

    def _propogate_X_val(self):
        if self._PROPAGATE_:
            return
        for m in self.models:
            m.X.mean.values[:] = self.X.mean.values
            m.X.variance.values[:] = self.X.variance.values
        varp_list = [m.X for m in self.models]
        [vp._update_inernal(varp_list) for vp in self.var_priors]
        self._PROPAGATE_ = True

    def _collate_X_gradient(self):
        self._PROPAGATE_ = False
        self.X.mean.gradient[:] = 0
        self.X.variance.gradient[:] = 0
        for m in self.models:
            self.X.mean.gradient += m.X.mean.gradient
            self.X.variance.gradient += m.X.variance.gradient

    def parameters_changed(self):
        super(SSMRD, self).parameters_changed()
        [m.parameters_changed() for m in self.models]
        self._log_marginal_likelihood = sum(
            [m._log_marginal_likelihood for m in self.models]
        )
        self._collate_X_gradient()

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _init_X(
        self, Ylist, input_dim, X=None, X_variance=None, Gammas=None, initx="PCA_concat"
    ):
        # Divide latent dimensions
        idx = np.empty((input_dim,), dtype=int)
        residue = (input_dim) % (len(Ylist))
        for i in range(len(Ylist)):
            if i < residue:
                size = input_dim / len(Ylist) + 1
                idx[i * size : (i + 1) * size] = i
            else:
                size = input_dim / len(Ylist)
                idx[i * size + residue : (i + 1) * size + residue] = i

        if X is None:
            if initx == "PCA_concat":
                X = np.empty((Ylist[0].shape[0], input_dim))
                fracs = np.empty((input_dim,))
                from ..util.initialization import initialize_latent

                for i in range(len(Ylist)):
                    Y = Ylist[i]
                    dim = (idx == i).sum()
                    if dim > 0:
                        x, fr = initialize_latent("PCA", dim, Y)
                        X[:, idx == i] = x
                        fracs[idx == i] = fr
            elif initx == "PCA_joint":
                y = np.hstack(Ylist)
                from ..util.initialization import initialize_latent

                X, fracs = initialize_latent("PCA", input_dim, y)
            else:
                X = np.random.randn(Ylist[0].shape[0], input_dim)
                fracs = np.ones(input_dim)
        else:
            fracs = np.ones(input_dim)

        if X_variance is None:  # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0, 0.1, X.shape)

        if Gammas is None:
            Gammas = []
            for x in X:
                gamma = np.empty_like(
                    X
                )  # The posterior probabilities of the binary variable in the variational approximation
                gamma[:] = 0.5 + 0.1 * np.random.randn(X.shape[0], input_dim)
                gamma[gamma > 1.0 - 1e-9] = 1.0 - 1e-9
                gamma[gamma < 1e-9] = 1e-9
                Gammas.append(gamma)
        return X, X_variance, Gammas, fracs

    @Model.optimizer_array.setter
    def optimizer_array(self, p):
        if self.mpi_comm != None:
            if self._IN_OPTIMIZATION_ and self.mpi_comm.rank == 0:
                self.mpi_comm.Bcast(np.int32(1), root=0)
            self.mpi_comm.Bcast(p, root=0)
        Model.optimizer_array.fset(self, p)

    def optimize(self, optimizer=None, start=None, **kwargs):
        self._IN_OPTIMIZATION_ = True
        if self.mpi_comm == None:
            super(SSMRD, self).optimize(optimizer, start, **kwargs)
        elif self.mpi_comm.rank == 0:
            super(SSMRD, self).optimize(optimizer, start, **kwargs)
            self.mpi_comm.Bcast(np.int32(-1), root=0)
        elif self.mpi_comm.rank > 0:
            x = self.optimizer_array.copy()
            flag = np.empty(1, dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag, root=0)
                if flag == 1:
                    try:
                        self.optimizer_array = x
                        self._fail_count = 0
                    except (LinAlgError, ZeroDivisionError, ValueError):
                        if self._fail_count >= self._allowed_failures:
                            raise
                        self._fail_count += 1
                elif flag == -1:
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False


class SpikeAndSlabPrior_SSMRD(SpikeAndSlabPrior):
    def __init__(
        self,
        nModels,
        pi=0.5,
        learnPi=False,
        group_spike=True,
        variance=1.0,
        name="SSMRDPrior",
        **kw
    ):
        self.nModels = nModels
        self._b_prob_all = 0.5
        super(SpikeAndSlabPrior_SSMRD, self).__init__(
            pi=pi,
            learnPi=learnPi,
            group_spike=group_spike,
            variance=variance,
            name=name,
            **kw
        )

    def _update_inernal(self, varp_list):
        """Make an update of the internal status by gathering the variational posteriors for all the individual models."""
        # The probability for the binary variable for the same latent dimension of any of the models is on.
        if self.group_spike:
            self._b_prob_all = 1.0 - param_to_array(varp_list[0].gamma_group)
            [
                np.multiply(self._b_prob_all, 1.0 - vp.gamma_group, self._b_prob_all)
                for vp in varp_list[1:]
            ]
        else:
            self._b_prob_all = 1.0 - param_to_array(varp_list[0].binary_prob)
            [
                np.multiply(self._b_prob_all, 1.0 - vp.binary_prob, self._b_prob_all)
                for vp in varp_list[1:]
            ]

    def KL_divergence(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.binary_prob[0]
        else:
            gamma = variational_posterior.binary_prob
        if len(self.pi.shape) == 2:
            idx = np.unique(gamma._raveled_index() / gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        var_mean = np.square(mu) / self.variance
        var_S = S / self.variance - np.log(S)
        var_gamma = (gamma * np.log(gamma / pi)).sum() + (
            (1 - gamma) * np.log((1 - gamma) / (1 - pi))
        ).sum()
        return var_gamma + (
            (1.0 - self._b_prob_all) * (np.log(self.variance) - 1.0 + var_mean + var_S)
        ).sum() / (2.0 * self.nModels)

    def update_gradients_KL(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        N = variational_posterior.num_data
        if self.group_spike:
            gamma = variational_posterior.binary_prob.values[0]
        else:
            gamma = variational_posterior.binary_prob.values
        if len(self.pi.shape) == 2:
            idx = np.unique(gamma._raveled_index() / gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        if self.group_spike:
            tmp = self._b_prob_all / (1.0 - gamma)
            variational_posterior.binary_prob.gradient -= (
                np.log((1 - pi) / pi * gamma / (1.0 - gamma)) / N
                + tmp
                * (
                    (np.square(mu) + S) / self.variance
                    - np.log(S)
                    + np.log(self.variance)
                    - 1.0
                )
                / 2.0
            )
        else:
            variational_posterior.binary_prob.gradient -= (
                np.log((1 - pi) / pi * gamma / (1.0 - gamma))
                + (
                    (np.square(mu) + S) / self.variance
                    - np.log(S)
                    + np.log(self.variance)
                    - 1.0
                )
                / 2.0
            )
        mu.gradient -= (1.0 - self._b_prob_all) * mu / (self.variance * self.nModels)
        S.gradient -= (
            (1.0 / self.variance - 1.0 / S)
            * (1.0 - self._b_prob_all)
            / (2.0 * self.nModels)
        )
        if self.learnPi:
            raise "Not Supported!"


class IBPPrior_SSMRD(VariationalPrior):
    def __init__(self, nModels, input_dim, alpha=2.0, tau=None, name="IBPPrior", **kw):
        super(IBPPrior_SSMRD, self).__init__(name=name, **kw)
        from paramz.transformations import Logexp, __fixed__

        self.nModels = nModels
        self._b_prob_all = 0.5
        self.input_dim = input_dim
        self.variance = 1.0
        self.alpha = Param("alpha", alpha, __fixed__)
        self.link_parameter(self.alpha)

    def _update_inernal(self, varp_list):
        """Make an update of the internal status by gathering the variational posteriors for all the individual models."""
        # The probability for the binary variable for the same latent dimension of any of the models is on.
        self._b_prob_all = 1.0 - param_to_array(varp_list[0].gamma_group)
        [
            np.multiply(self._b_prob_all, 1.0 - vp.gamma_group, self._b_prob_all)
            for vp in varp_list[1:]
        ]

    def KL_divergence(self, variational_posterior):
        mu, S, gamma, tau = (
            variational_posterior.mean.values,
            variational_posterior.variance.values,
            variational_posterior.gamma_group.values,
            variational_posterior.tau.values,
        )

        var_mean = np.square(mu) / self.variance
        var_S = S / self.variance - np.log(S)
        part1 = (
            (1.0 - self._b_prob_all) * (np.log(self.variance) - 1.0 + var_mean + var_S)
        ).sum() / (2.0 * self.nModels)

        ad = self.alpha / self.input_dim
        from scipy.special import betaln, digamma

        part2 = (
            (gamma * np.log(gamma)).sum()
            + ((1.0 - gamma) * np.log(1.0 - gamma)).sum()
            + (betaln(ad, 1.0) * self.input_dim - betaln(tau[:, 0], tau[:, 1]).sum())
            / self.nModels
            + (((tau[:, 0] - ad) / self.nModels - gamma) * digamma(tau[:, 0])).sum()
            + (
                ((tau[:, 1] - 1.0) / self.nModels + gamma - 1.0) * digamma(tau[:, 1])
            ).sum()
            + (
                ((1.0 + ad - tau[:, 0] - tau[:, 1]) / self.nModels + 1.0)
                * digamma(tau.sum(axis=1))
            ).sum()
        )
        return part1 + part2

    def update_gradients_KL(self, variational_posterior):
        mu, S, gamma, tau = (
            variational_posterior.mean.values,
            variational_posterior.variance.values,
            variational_posterior.gamma_group.values,
            variational_posterior.tau.values,
        )

        variational_posterior.mean.gradient -= (
            (1.0 - self._b_prob_all) * mu / (self.variance * self.nModels)
        )
        variational_posterior.variance.gradient -= (
            (1.0 / self.variance - 1.0 / S)
            * (1.0 - self._b_prob_all)
            / (2.0 * self.nModels)
        )
        from scipy.special import digamma, polygamma

        tmp = self._b_prob_all / (1.0 - gamma)
        dgamma = (
            np.log(gamma / (1.0 - gamma)) + digamma(tau[:, 1]) - digamma(tau[:, 0])
        ) / variational_posterior.num_data
        variational_posterior.binary_prob.gradient -= (
            dgamma
            + tmp
            * (
                (np.square(mu) + S) / self.variance
                - np.log(S)
                + np.log(self.variance)
                - 1.0
            )
            / 2.0
        )
        ad = self.alpha / self.input_dim
        common = ((1.0 + ad - tau[:, 0] - tau[:, 1]) / self.nModels + 1.0) * polygamma(
            1, tau.sum(axis=1)
        )
        variational_posterior.tau.gradient[:, 0] = -(
            ((tau[:, 0] - ad) / self.nModels - gamma) * polygamma(1, tau[:, 0]) + common
        )
        variational_posterior.tau.gradient[:, 1] = -(
            ((tau[:, 1] - 1.0) / self.nModels + gamma - 1.0) * polygamma(1, tau[:, 1])
            + common
        )
