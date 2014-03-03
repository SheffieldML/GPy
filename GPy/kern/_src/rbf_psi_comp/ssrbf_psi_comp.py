# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the psi statistics computation
"""

import numpy as np
from GPy.util.caching import Cache_this

@Cache_this(limit=1)
def _Z_distances(Z):
    Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
    Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
    return Zhat, Zdist

@Cache_this(limit=1)
def _psi1computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    # _dpsi1_dvariance     NxM
    # _dpsi1_dlengthscale  NxMxQ
    # _dpsi1_dZ            NxMxQ
    # _dpsi1_dgamma        NxMxQ
    # _dpsi1_dmu           NxMxQ
    # _dpsi1_dS            NxMxQ
    
    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_denom = S[:, None, :] / lengthscale2 + 1.  # Nx1xQ
    _psi1_denom_sqrt = np.sqrt(_psi1_denom) #Nx1xQ
    _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
    _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom) # NxMxQ
    _psi1_common = gamma[:,None,:] / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #Nx1xQ
    _psi1_exponent1 = np.log(gamma[:,None,:]) -0.5 * (_psi1_dist_sq + np.log(_psi1_denom)) # NxMxQ
    _psi1_exponent2 = np.log(1.-gamma[:,None,:]) -0.5 * (np.square(Z[None,:,:])/lengthscale2) # NxMxQ
    _psi1_exponent = np.log(np.exp(_psi1_exponent1) + np.exp(_psi1_exponent2)) #NxMxQ
    _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
    _psi1_exp_dist_sq = np.exp(-0.5*_psi1_dist_sq) # NxMxQ
    _psi1_exp_Z = np.exp(-0.5*np.square(Z[None,:,:])/lengthscale2) # 1xMxQ
    _psi1_q = variance * np.exp(_psi1_exp_sum[:,:,None] - _psi1_exponent) # NxMxQ
    _psi1 = variance * np.exp(_psi1_exp_sum) # NxM
    _dpsi1_dvariance = _psi1 / variance # NxM
    _dpsi1_dgamma = _psi1_q * (_psi1_exp_dist_sq/_psi1_denom_sqrt-_psi1_exp_Z) # NxMxQ
    _dpsi1_dmu = _psi1_q * (_psi1_exp_dist_sq * _psi1_dist * _psi1_common) # NxMxQ
    _dpsi1_dS = _psi1_q * (_psi1_exp_dist_sq * _psi1_common * 0.5 * (_psi1_dist_sq - 1.)) # NxMxQ
    _dpsi1_dZ = _psi1_q * (- _psi1_common * _psi1_dist * _psi1_exp_dist_sq - (1-gamma[:,None,:])/lengthscale2*Z[None,:,:]*_psi1_exp_Z) # NxMxQ
    _dpsi1_dlengthscale = 2.*lengthscale*_psi1_q * (0.5*_psi1_common*(S[:,None,:]/lengthscale2+_psi1_dist_sq)*_psi1_exp_dist_sq + 0.5*(1-gamma[:,None,:])*np.square(Z[None,:,:]/lengthscale2)*_psi1_exp_Z) # NxMxQ

    return _psi1, _dpsi1_dvariance, _dpsi1_dgamma, _dpsi1_dmu, _dpsi1_dS, _dpsi1_dZ, _dpsi1_dlengthscale

@Cache_this(limit=1)
def _psi2computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi2                NxMxM
    # _psi2_dvariance      NxMxM
    # _psi2_dlengthscale   NxMxMxQ
    # _psi2_dZ             NxMxMxQ
    # _psi2_dgamma         NxMxMxQ
    # _psi2_dmu            NxMxMxQ
    # _psi2_dS             NxMxMxQ
    
    lengthscale2 = np.square(lengthscale)
    
    _psi2_Zhat, _psi2_Zdist = _Z_distances(Z)
    _psi2_Zdist_sq = np.square(_psi2_Zdist / lengthscale) # M,M,Q
    _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

    # psi2
    _psi2_denom = 2.*S[:, None, None, :] / lengthscale2 + 1. # Nx1x1xQ
    _psi2_denom_sqrt = np.sqrt(_psi2_denom)
    _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
    _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom)
    _psi2_common = gamma[:,None,None,:]/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # Nx1x1xQ
    _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom)+np.log(gamma[:,None,None,:]) #N,M,M,Q
    _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
    _psi2_exponent = np.log(np.exp(_psi2_exponent1) + np.exp(_psi2_exponent2))
    _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
    _psi2_q = np.square(variance) * np.exp(_psi2_exp_sum[:,:,:,None]-_psi2_exponent) # NxMxMxQ 
    _psi2_exp_dist_sq = np.exp(-_psi2_Zdist_sq -_psi2_mudist_sq) # NxMxMxQ
    _psi2_exp_Z = np.exp(-0.5*_psi2_Z_sq_sum) # MxMxQ
    _psi2 = np.square(variance) * np.exp(_psi2_exp_sum) # N,M,M
    _dpsi2_dvariance = 2. * _psi2/variance # NxMxM
    _dpsi2_dgamma = _psi2_q * (_psi2_exp_dist_sq/_psi2_denom_sqrt - _psi2_exp_Z) # NxMxMxQ
    _dpsi2_dmu = _psi2_q * (-2.*_psi2_common*_psi2_mudist * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dS = _psi2_q * (_psi2_common * (2.*_psi2_mudist_sq - 1.) * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dZ = 2.*_psi2_q * (_psi2_common*(-_psi2_Zdist*_psi2_denom+_psi2_mudist)*_psi2_exp_dist_sq - (1-gamma[:,None,None,:])*Z[:,None,:]/lengthscale2*_psi2_exp_Z) # NxMxMxQ
    _dpsi2_dlengthscale = 2.*lengthscale* _psi2_q * (_psi2_common*(S[:,None,None,:]/lengthscale2+_psi2_Zdist_sq*_psi2_denom+_psi2_mudist_sq)*_psi2_exp_dist_sq+(1-gamma[:,None,None,:])*_psi2_Z_sq_sum*0.5/lengthscale2*_psi2_exp_Z) # NxMxMxQ

    return _psi2, _dpsi2_dvariance, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _dpsi2_dZ, _dpsi2_dlengthscale
