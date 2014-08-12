# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for Bayesian GPLVM
"""

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

    psi0 = np.einsum('q,nq->n',variance,np.square(mu)+S)
    psi1 = np.einsum('q,mq,nq->nm',variance,Z,mu)

    tmp = np.einsum('q,mq,nq->nm',variance,Z,mu)
    psi2 = np.einsum('q,mq,oq,nq->mo',np.square(variance),Z,Z,S) + np.einsum('nm,no->mo',tmp,tmp)

    return psi0, psi1, psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
    mu = variational_posterior.mean
    S = variational_posterior.variance

    dL_dvar, dL_dmu, dL_dS, dL_dZ = _psi2computations(dL_dpsi2, variance, Z, mu, S)

    # Compute for psi0 and psi1
    mu2S = np.square(mu)+S
    dL_dvar += np.einsum('n,nq->q',dL_dpsi0,mu2S) + np.einsum('nm,mq,nq->q',dL_dpsi1,Z,mu)
    dL_dmu += np.einsum('n,q,nq->nq',dL_dpsi0,2.*variance,mu) + np.einsum('nm,q,mq->nq',dL_dpsi1,variance,Z)
    dL_dS += np.einsum('n,q->nq',dL_dpsi0,variance)
    dL_dZ +=  np.einsum('nm,q,nq->mq',dL_dpsi1, variance,mu)
    
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
    common_sum = np.einsum('q,mq,nq->nm',variance,Z,mu) # NxM

    dL_dvar = np.einsum('mo,nq,q,mq,oq->q',dL_dpsi2,2.*S,variance,Z,Z)+\
        np.einsum('mo,mq,nq,no->q',dL_dpsi2,Z,mu,common_sum)+\
        np.einsum('mo,oq,nq,nm->q',dL_dpsi2,Z,mu,common_sum)
            
    dL_dmu = np.einsum('mo,q,mq,no->nq',dL_dpsi2,variance,Z,common_sum)+\
        np.einsum('mo,q,oq,nm->nq',dL_dpsi2,variance,Z,common_sum)
    
    dL_dS = np.empty(S.shape)
    dL_dS[:] = np.einsum('mo,q,mq,oq->q',dL_dpsi2,variance2,Z,Z)
    
    dL_dZ = 2.*(np.einsum('om,q,mq,nq->oq',dL_dpsi2,variance2,Z,S)+np.einsum('om,q,nq,nm->oq',dL_dpsi2,variance,mu,common_sum))

    return dL_dvar, dL_dmu, dL_dS, dL_dZ
