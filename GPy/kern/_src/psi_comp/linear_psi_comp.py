# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for Bayesian GPLVM
"""

import numpy as np
from ....util.linalg import tdot

def psicomputations(variance, Z, variational_posterior):
    """
    Compute psi-statistics for ss-linear kernel
    """
    # here are the "statistics" for psi0, psi1 and psi2
    # Produced intermediate results:
    # psi0    N
    # psi1    NxM
    # psi2    MxM
    mu = variational_posterior.mean
    S = variational_posterior.variance

    psi0 = (variance*(np.square(mu)+S)).sum(axis=1)
    psi1 = np.dot(mu,(variance*Z).T)
    psi2 = np.dot(S.sum(axis=0)*np.square(variance)*Z,Z.T)+ tdot(psi1.T)

    return psi0, psi1, psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
    mu = variational_posterior.mean
    S = variational_posterior.variance

    dL_dvar, dL_dmu, dL_dS, dL_dZ = _psi2computations(dL_dpsi2, variance, Z, mu, S)

    # Compute for psi0 and psi1
    mu2S = np.square(mu)+S
    dL_dpsi0_var = dL_dpsi0[:,None]*variance[None,:]
    dL_dpsi1_mu = np.dot(dL_dpsi1.T,mu)
    dL_dvar += (dL_dpsi0[:,None]*mu2S).sum(axis=0)+ (dL_dpsi1_mu*Z).sum(axis=0)
    dL_dmu += 2.*dL_dpsi0_var*mu+np.dot(dL_dpsi1,Z)*variance
    dL_dS += dL_dpsi0_var
    dL_dZ += dL_dpsi1_mu*variance
    
    return dL_dvar, dL_dZ, dL_dmu, dL_dS

def _psi2computations(dL_dpsi2, variance, Z, mu, S):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi2_dvariance      Q
    # _psi2_dZ             MxQ
    # _psi2_dmu            NxQ
    # _psi2_dS             NxQ
    
    variance2 = np.square(variance)
    common_sum = np.dot(mu,(variance*Z).T)
    Z_expect = (np.dot(dL_dpsi2,Z)*Z).sum(axis=0)
    dL_dpsi2T = dL_dpsi2+dL_dpsi2.T
    common_expect = np.dot(common_sum,np.dot(dL_dpsi2T,Z))
    Z2_expect = np.inner(common_sum,dL_dpsi2T)
    Z1_expect = np.dot(dL_dpsi2T,Z)

    dL_dvar = 2.*S.sum(axis=0)*variance*Z_expect+(common_expect*mu).sum(axis=0)
            
    dL_dmu = common_expect*variance
    
    dL_dS = np.empty(S.shape)
    dL_dS[:] = Z_expect*variance2
    
    dL_dZ = variance2*S.sum(axis=0)*Z1_expect+np.dot(Z2_expect.T,variance*mu)

    return dL_dvar, dL_dmu, dL_dS, dL_dZ
