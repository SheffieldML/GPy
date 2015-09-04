# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .stationary import Stationary
from .psi_comp import PSICOMP_RBF
from .psi_comp.rbf_psi_gpucomp import PSICOMP_RBF_GPU
from ...util.config import *

class RBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False):
        super(RBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r**2-1)*self.K_of_r(r)

    def __getstate__(self):
        dc = super(RBF, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        return super(RBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1 #TODO: higher dim spectra?
        return self.variance*np.sqrt(2*np.pi)*self.lengthscale*np.exp(-self.lengthscale*2*omega**2/2)

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

