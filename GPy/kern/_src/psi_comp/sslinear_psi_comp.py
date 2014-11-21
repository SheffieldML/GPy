# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for SSGPLVM
"""

from ....util.linalg import tdot

import numpy as np

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
    dL_dvar += np.einsum('n,nq,nq->q',dL_dpsi0,gamma,mu2S) + np.einsum('nm,nq,mq,nq->q',dL_dpsi1,gamma,Z,mu)
    dL_dgamma += np.einsum('n,q,nq->nq',dL_dpsi0,variance,mu2S) + np.einsum('nm,q,mq,nq->nq',dL_dpsi1,variance,Z,mu)
    dL_dmu += np.einsum('n,nq,q,nq->nq',dL_dpsi0,gamma,2.*variance,mu) + np.einsum('nm,nq,q,mq->nq',dL_dpsi1,gamma,variance,Z)
    dL_dS += np.einsum('n,nq,q->nq',dL_dpsi0,gamma,variance)
    dL_dZ +=  np.einsum('nm,nq,q,nq->mq',dL_dpsi1,gamma, variance,mu)
    
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
    gvm = np.einsum('nq,nq,q->nq',gamma,mu,variance)
    common_sum = np.einsum('nq,mq->nm',gvm,Z)
#     common_sum = np.einsum('nq,q,mq,nq->nm',gamma,variance,Z,mu) # NxM
    Z_expect = np.einsum('mo,mq,oq->q',dL_dpsi2,Z,Z)
    dL_dpsi2T = dL_dpsi2+dL_dpsi2.T
    tmp = np.einsum('mo,oq->mq',dL_dpsi2T,Z)
    common_expect = np.einsum('mq,nm->nq',tmp,common_sum)
#     common_expect = np.einsum('mo,mq,no->nq',dL_dpsi2+dL_dpsi2.T,Z,common_sum)
    Z2_expect = np.einsum('om,nm->no',dL_dpsi2T,common_sum)
    Z1_expect = np.einsum('om,mq->oq',dL_dpsi2T,Z)
    
    dL_dvar = np.einsum('nq,q,q->q',2.*(gamma*mu2S-gamma2*mu2),variance,Z_expect)+\
        np.einsum('nq,nq,nq->q',common_expect,gamma,mu)
        
    dL_dgamma = np.einsum('q,q,nq->nq',Z_expect,variance2,(mu2S-2.*gamma*mu2))+\
        np.einsum('nq,q,nq->nq',common_expect,variance,mu)
    
    dL_dmu = np.einsum('q,q,nq,nq->nq',Z_expect,variance2,mu,2.*(gamma-gamma2))+\
            np.einsum('nq,nq,q->nq',common_expect,gamma,variance)
                    
    dL_dS = np.einsum('q,nq,q->nq',Z_expect,gamma,variance2)
    
#     dL_dZ = 2.*(np.einsum('om,nq,q,mq,nq->oq',dL_dpsi2,gamma,variance2,Z,(mu2S-gamma*mu2))+np.einsum('om,nq,q,nq,nm->oq',dL_dpsi2,gamma,variance,mu,common_sum))
    dL_dZ = Z1_expect*np.einsum('nq,q,nq->q',gamma,variance2,(mu2S-gamma*mu2))+np.einsum('nq,q,nq,nm->mq',gamma,variance,mu,Z2_expect)

    return dL_dvar, dL_dgamma, dL_dmu, dL_dS, dL_dZ
