# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from paramz.core.pickleable import Pickleable
from paramz.caching import Cache_this
from GPy.core.parameterization import variational
#from linear_psi_comp import LINEAr

class PSICOMP(Pickleable):
        
    def psicomputations(self, kern, Z, qX, return_psi2_n=False):
        raise NotImplementedError("Abstract method!")
    
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, qX):
        raise NotImplementedError("Abstract method!")

    def _setup_observers(self):
        pass

from .gaussherm import PSICOMP_GH
from . import rbf_psi_comp, linear_psi_comp, ssrbf_psi_comp, sslinear_psi_comp

class PSICOMP_RBF(PSICOMP):
    @Cache_this(limit=3, ignore_args=(0,))
    def psicomputations(self, kern, Z, variational_posterior, return_psi2_n=False):
        variance, lengthscale = kern.variance, kern.lengthscale
        if isinstance(variational_posterior, variational.NormalPosterior):
            return rbf_psi_comp.psicomputations(variance, lengthscale, Z, variational_posterior, return_psi2_n=return_psi2_n)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return ssrbf_psi_comp.psicomputations(variance, lengthscale, Z, variational_posterior)
        else:
            raise ValueError("unknown distriubtion received for psi-statistics")

    @Cache_this(limit=3, ignore_args=(0,2,3,4))
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        variance, lengthscale = kern.variance, kern.lengthscale
        if isinstance(variational_posterior, variational.NormalPosterior):
            return rbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return ssrbf_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior)
        else:
            raise ValueError("unknown distriubtion received for psi-statistics")

class PSICOMP_Linear(PSICOMP):

    @Cache_this(limit=3, ignore_args=(0,))
    def psicomputations(self, kern, Z, variational_posterior, return_psi2_n=False):
        variances = kern.variances
        if isinstance(variational_posterior, variational.NormalPosterior):
            return linear_psi_comp.psicomputations(variances, Z, variational_posterior, return_psi2_n=return_psi2_n)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return sslinear_psi_comp.psicomputations(variances, Z, variational_posterior)
        else:
            raise ValueError("unknown distriubtion received for psi-statistics")

    @Cache_this(limit=3, ignore_args=(0,2,3,4))
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        variances = kern.variances
        if isinstance(variational_posterior, variational.NormalPosterior):
            return linear_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variances, Z, variational_posterior)
        elif isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return sslinear_psi_comp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variances, Z, variational_posterior)
        else:
            raise ValueError("unknown distriubtion received for psi-statistics")


from . import ssrbf_psi_gpucomp
from .rbf_psi_gpucomp import PSICOMP_RBF_GPU
