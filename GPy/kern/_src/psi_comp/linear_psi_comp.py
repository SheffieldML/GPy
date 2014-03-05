# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the Psi statistics computation of the linear kernel for SSGPLVM
"""

import numpy as np
from GPy.util.caching import Cache_this

#@Cache_this(limit=1)
def _psi2computations(variance, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi2                NxMxM
    # _psi2_dvariance      NxMxMxQ
    # _psi2_dZ             NxMxQ
    # _psi2_dgamma         NxMxMxQ
    # _psi2_dmu            NxMxMxQ
    # _psi2_dS             NxMxMxQ
    
    mu2 = np.square(mu)
    gamma2 = np.square(gamma)
    variance2 = np.square(variance)
    mu2S = mu2+S # NxQ
    common_sum = np.einsum('nq,q,mq,nq->nm',gamma,variance,Z,mu) # NxM
            
    _dpsi2_dvariance = np.einsum('nq,q,mq,oq->nmoq',2.*(gamma*mu2S-gamma2*mu2),variance,Z,Z)+\
        np.einsum('nq,mq,nq,no->nmoq',gamma,Z,mu,common_sum)+\
        np.einsum('nq,oq,nq,nm->nmoq',gamma,Z,mu,common_sum)
        
    _dpsi2_dgamma = np.einsum('q,mq,oq,nq->nmoq',variance2,Z,Z,(mu2S-2.*gamma*mu2))+\
        np.einsum('q,mq,nq,no->nmoq',variance,Z,mu,common_sum)+\
        np.einsum('q,oq,nq,nm->nmoq',variance,Z,mu,common_sum)
    
    _dpsi2_dmu = np.einsum('q,mq,oq,nq,nq->nmoq',variance2,Z,Z,mu,2.*(gamma-gamma2))+\
        np.einsum('nq,q,mq,no->nmoq',gamma,variance,Z,common_sum)+\
        np.einsum('nq,q,oq,nm->nmoq',gamma,variance,Z,common_sum)
        
    _dpsi2_dS = np.einsum('nq,q,mq,oq->nmoq',gamma,variance2,Z,Z)
    
    _dpsi2_dZ = 2.*(np.einsum('nq,q,mq,nq->nmq',gamma,variance2,Z,mu2S)+np.einsum('nq,q,nq,nm->nmq',gamma,variance,mu,common_sum)
                    -np.einsum('nq,q,mq,nq->nmq',gamma2,variance2,Z,mu2))

    return _dpsi2_dvariance, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _dpsi2_dZ