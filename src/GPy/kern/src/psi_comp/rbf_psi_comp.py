"""
The module for psi-statistics for RBF kernel
"""

import numpy as np
from paramz.caching import Cacher

def psicomputations(variance, lengthscale, Z, variational_posterior, return_psi2_n=False):
    # here are the "statistics" for psi0, psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    mu = variational_posterior.mean
    S = variational_posterior.variance

    psi0 = np.empty(mu.shape[0])
    psi0[:] = variance
    psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
    psi2 = _psi2computations(variance, lengthscale, Z, mu, S)
    if not return_psi2_n: psi2 = psi2.sum(axis=0)
    return psi0, psi1, psi2

def __psi1computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results:
    # _psi1                NxM

    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_logdenom = np.log(S/lengthscale2+1.).sum(axis=-1) # N
    _psi1_log = (_psi1_logdenom[:,None]+np.einsum('nmq,nq->nm',np.square(mu[:,None,:]-Z[None,:,:]),1./(S+lengthscale2)))/(-2.)
    _psi1 = variance*np.exp(_psi1_log)

    return _psi1

def __psi2computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced intermediate results:
    # _psi2                MxM

    N,M,Q = mu.shape[0], Z.shape[0], mu.shape[1]
    lengthscale2 = np.square(lengthscale)

    _psi2_logdenom = np.log(2.*S/lengthscale2+1.).sum(axis=-1)/(-2.) # N
    _psi2_exp1 = (np.square(Z[:,None,:]-Z[None,:,:])/lengthscale2).sum(axis=-1)/(-4.) #MxM
    Z_hat = (Z[:,None,:]+Z[None,:,:])/2. #MxMxQ
    denom = 1./(2.*S+lengthscale2)
    _psi2_exp2 = -(np.square(mu)*denom).sum(axis=-1)[:,None,None]+(2*(mu*denom).dot(Z_hat.reshape(M*M,Q).T) - denom.dot(np.square(Z_hat).reshape(M*M,Q).T)).reshape(N,M,M)
    _psi2 = variance*variance*np.exp(_psi2_logdenom[:,None,None]+_psi2_exp1[None,:,:]+_psi2_exp2)
    return _psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
    ARD = (len(lengthscale)!=1)

    dvar_psi1, dl_psi1, dZ_psi1, dmu_psi1, dS_psi1 = _psi1compDer(dL_dpsi1, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
    dvar_psi2, dl_psi2, dZ_psi2, dmu_psi2, dS_psi2 = _psi2compDer(dL_dpsi2, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)

    dL_dvar = np.sum(dL_dpsi0) + dvar_psi1 + dvar_psi2

    dL_dlengscale = dl_psi1 + dl_psi2
    if not ARD:
        dL_dlengscale = dL_dlengscale.sum()

    dL_dmu = dmu_psi1 + dmu_psi2
    dL_dS = dS_psi1 + dS_psi2
    dL_dZ = dZ_psi1 + dZ_psi2

    return dL_dvar, dL_dlengscale, dL_dZ, dL_dmu, dL_dS

def _psi1compDer(dL_dpsi1, variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results: dL_dparams w.r.t. psi1
    # _dL_dvariance     1
    # _dL_dlengthscale  Q
    # _dL_dZ            MxQ
    # _dL_dgamma        NxQ
    # _dL_dmu           NxQ
    # _dL_dS            NxQ

    lengthscale2 = np.square(lengthscale)

    _psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
    Lpsi1 = dL_dpsi1*_psi1
    Zmu = Z[None,:,:]-mu[:,None,:] # NxMxQ
    denom = 1./(S+lengthscale2)
    Zmu2_denom = np.square(Zmu)*denom[:,None,:] #NxMxQ
    _dL_dvar = Lpsi1.sum()/variance
    _dL_dmu = np.einsum('nm,nmq,nq->nq',Lpsi1,Zmu,denom)
    _dL_dS = np.einsum('nm,nmq,nq->nq',Lpsi1,(Zmu2_denom-1.),denom)/2.
    _dL_dZ = -np.einsum('nm,nmq,nq->mq',Lpsi1,Zmu,denom)
    _dL_dl = np.einsum('nm,nmq,nq->q',Lpsi1,(Zmu2_denom+(S/lengthscale2)[:,None,:]),denom*lengthscale)

    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS

def _psi2compDer(dL_dpsi2, variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced the derivatives w.r.t. psi2:
    # _dL_dvariance      1
    # _dL_dlengthscale   Q
    # _dL_dZ             MxQ
    # _dL_dgamma         NxQ
    # _dL_dmu            NxQ
    # _dL_dS             NxQ
    N,M,Q = mu.shape[0],Z.shape[0],mu.shape[1]
    lengthscale2 = np.square(lengthscale)
    denom = 1./(2*S+lengthscale2)
    denom2 = np.square(denom)

    if len(dL_dpsi2.shape)==2: dL_dpsi2 = (dL_dpsi2+dL_dpsi2.T)/2
    else: dL_dpsi2  = (dL_dpsi2+ np.swapaxes(dL_dpsi2, 1,2))/2
    _psi2 = _psi2computations(variance, lengthscale, Z, mu, S) # NxMxM
    Lpsi2 = dL_dpsi2*_psi2 # dL_dpsi2 is MxM, using broadcast to multiply N out
    Lpsi2sum = Lpsi2.reshape(N,M*M).sum(1) #N
    tmp = Lpsi2.reshape(N*M,M).dot(Z).reshape(N,M,Q)
    Lpsi2Z = tmp.sum(1)  #NxQ
    Lpsi2Z2 = Lpsi2.reshape(N*M,M).dot(np.square(Z)).reshape(N,M,Q).sum(1) #np.einsum('nmo,oq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Z2p = (tmp*Z[None,:,:]).sum(1) #np.einsum('nmo,mq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Zhat = Lpsi2Z
    Lpsi2Zhat2 = (Lpsi2Z2+Lpsi2Z2p)/2

    _dL_dvar = Lpsi2sum.sum()*2/variance
    _dL_dmu = (-2*denom) * (mu*Lpsi2sum[:,None]-Lpsi2Zhat)
    _dL_dS = (2*np.square(denom))*(np.square(mu)*Lpsi2sum[:,None]-2*mu*Lpsi2Zhat+Lpsi2Zhat2) - denom*Lpsi2sum[:,None]
#     _dL_dZ = -np.einsum('nmo,oq->oq',Lpsi2,Z)/lengthscale2+np.einsum('nmo,oq->mq',Lpsi2,Z)/lengthscale2+ \
#              2*np.einsum('nmo,nq,nq->mq',Lpsi2,mu,denom) - np.einsum('nmo,nq,mq->mq',Lpsi2,denom,Z) - np.einsum('nmo,oq,nq->mq',Lpsi2,Z,denom)
    Lpsi2_N = Lpsi2.sum(0)
    Lpsi2_M = Lpsi2.sum(2)
    _dL_dZ = -Lpsi2_N.sum(0)[:,None]*Z/lengthscale2+Lpsi2_N.dot(Z)/lengthscale2+ \
             2*Lpsi2_M.T.dot(mu*denom) - Lpsi2_M.T.dot(denom)*Z - (Lpsi2.reshape(N,M*M).T.dot(denom).reshape(M,M,Q)*Z[None,:,:]).sum(1)#np.einsum('nmo,oq,nq->mq',Lpsi2,Z,denom)
    _dL_dl = 2*lengthscale* ((S/lengthscale2*denom+np.square(mu*denom))*Lpsi2sum[:,None]+(Lpsi2Z2-Lpsi2Z2p)/(2*np.square(lengthscale2))-
                             (2*mu*denom2)*Lpsi2Zhat+denom2*Lpsi2Zhat2).sum(axis=0)

    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS

_psi1computations = Cacher(__psi1computations, limit=3)
_psi2computations = Cacher(__psi2computations, limit=3)
