# ## Copyright (c) 2014, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import sys

from .util import assert_mcmc_priors


class Metropolis_Hastings:
    """
    Metropolis–Hastings sampling of unfixed model parameters.

    Proposals are in unconstrained ``optimizer_array`` space. The acceptance
    ratio uses ``log_likelihood() + log_prior()``; for constrained parameters
    the prior term includes the transform Jacobian only when a prior is set.
    By default a prior is required on every unfixed parameter.

    :param model: the GPy model that will be sampled
    :param cov: proposal covariance in unconstrained space (identity by default)
    :param require_priors: if True (default), require a prior on every unfixed
        parameter so transform Jacobians enter the target density
    """

    def __init__(self, model, cov=None, require_priors=True):
        """Metropolis Hastings, with tunings according to Gelman et al."""
        assert_mcmc_priors(model, require_priors=require_priors)
        self.model = model
        self.require_priors = require_priors
        current = self.model.optimizer_array
        self.D = current.size
        self.chains = []
        if cov is None:
            self.cov = np.eye(self.D)
        else:
            self.cov = cov
        self.scale = 2.4 / np.sqrt(self.D)
        self.new_chain(current)

    def new_chain(self, start=None):
        self.chains.append([])
        if start is None:
            self.model.randomize()
        else:
            self.model.optimizer_array = start

    def sample(self, Ntotal=10000, Nburn=1000, Nthin=10, tune=True, tune_throughout=False, tune_interval=400):
        assert_mcmc_priors(self.model, require_priors=self.require_priors)
        current = self.model.optimizer_array
        fcurrent = self.model.log_likelihood() + self.model.log_prior()
        accepted = np.zeros(Ntotal, dtype=bool)
        for it in range(Ntotal):
            print(f"sample {it + 1} of {Ntotal}\r", end="")
            sys.stdout.flush()
            prop = np.random.multivariate_normal(current, self.cov * self.scale * self.scale)
            self.model.optimizer_array = prop
            fprop = self.model.log_likelihood() + self.model.log_prior()

            if fprop > fcurrent:  # sample accepted, going 'uphill'
                accepted[it] = True
                current = prop
                fcurrent = fprop
            else:
                u = np.random.rand()
                if np.exp(fprop - fcurrent) > u:  # sample accepted downhill
                    accepted[it] = True
                    current = prop
                    fcurrent = fprop

            # store current value
            if (it > Nburn) & ((it % Nthin) == 0):
                self.chains[-1].append(current)

            # tuning!
            if it & ((it % tune_interval) == 0) & tune & ((it < Nburn) | (tune_throughout)):
                pc = np.mean(accepted[it - tune_interval : it])
                self.cov = np.cov(np.vstack(self.chains[-1][-tune_interval:]).T)
                if pc > 0.25:
                    self.scale *= 1.1
                if pc < 0.15:
                    self.scale /= 1.1

    def predict(self, function, args):
        """Make a prediction for the function, to which we will pass the additional arguments"""
        param = self.model.param_array
        fs = []
        for p in self.chain:
            self.model.param_array = p
            fs.append(function(*args))
        # reset model to starting state
        self.model.param_array = param
        return fs
