# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for SSGPLVM
"""

from ....util.linalg import tdot

import numpy as np

def psicomputations(variance, Z, variational_posterior, return_psi2_n=False):
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
    gamma = variational_posterior.binary_prob

    psi0 = (gamma*(np.square(mu)+S)*variance).sum(axis=-1)
    psi1 = np.inner(variance*gamma*mu,Z)
    psi2 = np.inner(np.square(variance)*(gamma*((1-gamma)*np.square(mu)+S)).sum(axis=0)*Z,Z)+tdot(psi1.T)

    return psi0, psi1, psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
    mu = variational_posterior.mean
    S = variational_posterior.variance
    gamma = variational_posterior.binary_prob

    dL_dvar, dL_dgamma, dL_dmu, dL_dS, dL_dZ = _psi2computations(dL_dpsi2, variance, Z, mu, S, gamma)

    # Compute for psi0 and psi1
    mu2S = np.square(mu)+S
    dL_dvar += (dL_dpsi0[:,None]*gamma*mu2S).sum(axis=0) + (dL_dpsi1.T.dot(gamma*mu)*Z).sum(axis=0)
    dL_dgamma += dL_dpsi0[:,None]*variance*mu2S+ dL_dpsi1.dot(Z)*mu*variance
    dL_dmu += dL_dpsi0[:,None]*2.*variance*gamma*mu + dL_dpsi1.dot(Z)*gamma*variance
    dL_dS += dL_dpsi0[:,None]*variance*gamma
    dL_dZ += dL_dpsi1.T.dot(gamma*mu)*variance
    
    return dL_dvar, dL_dZ, dL_dmu, dL_dS, dL_dgamma

def _psi2computations(dL_dpsi2, variance, Z, mu, S, gamma):
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
    # _psi2_dgamma         NxQ
    # _psi2_dmu            NxQ
    # _psi2_dS             NxQ
    
    mu2 = np.square(mu)
    gamma2 = np.square(gamma)
    variance2 = np.square(variance)
    mu2S = mu2+S # NxQ
    gvm = gamma*mu*variance
    common_sum = gvm.dot(Z.T)
    Z_expect = (np.dot(dL_dpsi2,Z)*Z).sum(axis=0)
    Z_expect_var2 = Z_expect*variance2
    dL_dpsi2T = dL_dpsi2+dL_dpsi2.T
    common_expect = common_sum.dot(dL_dpsi2T).dot(Z)
    Z2_expect = common_sum.dot(dL_dpsi2T)
    Z1_expect = dL_dpsi2T.dot(Z)
    
    dL_dvar = variance*Z_expect*2.*(gamma*mu2S-gamma2*mu2).sum(axis=0)+(common_expect*gamma*mu).sum(axis=0)
        
    dL_dgamma = Z_expect_var2*(mu2S-2.*gamma*mu2)+common_expect*mu*variance
                
    dL_dmu = Z_expect_var2*mu*2.*(gamma-gamma2) + common_expect*gamma*variance

    dL_dS = gamma*Z_expect_var2
    
    dL_dZ = (gamma*(mu2S-gamma*mu2)).sum(axis=0)*variance2*Z1_expect+ Z2_expect.T.dot(gamma*mu)*variance

    return dL_dvar, dL_dgamma, dL_dmu, dL_dS, dL_dZ
