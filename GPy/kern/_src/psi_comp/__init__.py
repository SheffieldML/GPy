# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ....core.parameterization.parameter_core import Pickleable
from GPy.util.caching import Cache_this
from ....core.parameterization import variational
import rbf_psi_comp
import ssrbf_psi_comp
import sslinear_psi_comp
import linear_psi_comp

class PSICOMP_RBF(Pickleable):
    @Cache_this(limit=2, ignore_args=(0,))
    def psicomputations(self, variance, lengthscale, Z, variational_posterior):
        if isinstance(variational_posterior, variational.NormalPosterior):
            return rbf_psi_comp.psicomputations(variance, lengthscale, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return ssrbf_psi_comp.psicomputations(variance, lengthscale, Z, variational_posterior)
        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    @Cache_this(limit=2, ignore_args=(0,1,2,3))
    def psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        if isinstance(variational_posterior, variational.NormalPosterior):
            return rbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior)
        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    def _setup_observers(self):
        pass

class PSICOMP_Linear(Pickleable):

    @Cache_this(limit=2, ignore_args=(0,))
    def psicomputations(self, variance, Z, variational_posterior):
        if isinstance(variational_posterior, variational.NormalPosterior):
            return linear_psi_comp.psicomputations(variance, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return sslinear_psi_comp.psicomputations(variance, Z, variational_posterior)
        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    @Cache_this(limit=2, ignore_args=(0,1,2,3))
    def psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
        if isinstance(variational_posterior, variational.NormalPosterior):
            return linear_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return sslinear_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior)
        else:
            raise ValueError, "unknown distriubtion received for psi-statistics"

    def _setup_observers(self):
        pass