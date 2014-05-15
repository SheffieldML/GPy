# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for SSGPLVM
"""

import numpy as np
from GPy.util.caching import Cache_this

class PSICOMP_SSLinear(object):
    #@Cache_this(limit=1, ignore_args=(0,))
    def psicomputations(self, variance, Z, mu, S, gamma):
        """
        Compute psi-statistics for ss-linear kernel
        """
        # here are the "statistics" for psi0, psi1 and psi2
        # Produced intermediate results:
        # psi0    N
        # psi1    NxM
        # psi2    MxM

        psi0 = np.einsum('q,nq,nq->n',variance,gamma,np.square(mu)+S)
        psi1 = np.einsum('nq,q,mq,nq->nm',gamma,variance,Z,mu)
        mu2 = np.square(mu)
        variances2 = np.square(variance)
        tmp = np.einsum('nq,q,mq,nq->nm',gamma,variance,Z,mu)
        psi2 = np.einsum('nq,q,mq,oq,nq->mo',gamma,variances2,Z,Z,mu2+S)+\
               np.einsum('nm,no->mo',tmp,tmp) - np.einsum('nq,q,mq,oq,nq->mo',np.square(gamma),variances2,Z,Z,mu2)

        return psi0, psi1, psi2
    
    #@Cache_this(limit=1, ignore_args=(0,1,2,3))
    def psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob

        dL_dvar, dL_dgamma, dL_dmu, dL_dS, dL_dZ = self._psi2computations(dL_dpsi2, variance, Z, mu, S, gamma)

        # Compute for psi0 and psi1
        mu2S = np.square(mu)+S
        dL_dvar += np.einsum('n,nq,nq->q',dL_dpsi0,gamma,mu2S) + np.einsum('nm,nq,mq,nq->q',dL_dpsi1,gamma,Z,mu)
        dL_dgamma += np.einsum('n,q,nq->nq',dL_dpsi0,variance,mu2S) + np.einsum('nm,q,mq,nq->nq',dL_dpsi1,variance,Z,mu)
        dL_dmu += np.einsum('n,nq,q,nq->nq',dL_dpsi0,gamma,2.*variance,mu) + np.einsum('nm,nq,q,mq->nq',dL_dpsi1,gamma,variance,Z)
        dL_dS += np.einsum('n,nq,q->nq',dL_dpsi0,gamma,variance)
        dL_dZ +=  np.einsum('nm,nq,q,nq->mq',dL_dpsi1,gamma, variance,mu)
        
        return dL_dvar, dL_dZ, dL_dmu, dL_dS, dL_dgamma
    
    def _psi2computations(self, dL_dpsi2, variance, Z, mu, S, gamma):
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
        common_sum = np.einsum('nq,q,mq,nq->nm',gamma,variance,Z,mu) # NxM

        dL_dvar = np.einsum('mo,nq,q,mq,oq->q',dL_dpsi2,2.*(gamma*mu2S-gamma2*mu2),variance,Z,Z)+\
            np.einsum('mo,nq,mq,nq,no->q',dL_dpsi2,gamma,Z,mu,common_sum)+\
            np.einsum('mo,nq,oq,nq,nm->q',dL_dpsi2,gamma,Z,mu,common_sum)
            
        dL_dgamma = np.einsum('mo,q,mq,oq,nq->nq',dL_dpsi2,variance2,Z,Z,(mu2S-2.*gamma*mu2))+\
            np.einsum('mo,q,mq,nq,no->nq',dL_dpsi2,variance,Z,mu,common_sum)+\
            np.einsum('mo,q,oq,nq,nm->nq',dL_dpsi2,variance,Z,mu,common_sum)
        
        dL_dmu = np.einsum('mo,q,mq,oq,nq,nq->nq',dL_dpsi2,variance2,Z,Z,mu,2.*(gamma-gamma2))+\
            np.einsum('mo,nq,q,mq,no->nq',dL_dpsi2,gamma,variance,Z,common_sum)+\
            np.einsum('mo,nq,q,oq,nm->nq',dL_dpsi2,gamma,variance,Z,common_sum)
                        
        dL_dS = np.einsum('mo,nq,q,mq,oq->nq',dL_dpsi2,gamma,variance2,Z,Z)
        
        dL_dZ = 2.*(np.einsum('om,nq,q,mq,nq->oq',dL_dpsi2,gamma,variance2,Z,mu2S)+np.einsum('om,nq,q,nq,nm->oq',dL_dpsi2,gamma,variance,mu,common_sum)
                    -np.einsum('om,nq,q,mq,nq->oq',dL_dpsi2,gamma2,variance2,Z,mu2))

        return dL_dvar, dL_dgamma, dL_dmu, dL_dS, dL_dZ
