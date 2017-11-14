# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy, Arno Solin
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Classes in this module enhance several stationary covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .rbf import RBF
from .stationary import Exponential
from .stationary import RatQuad

import numpy as np
import scipy as sp
try:
    from scipy.linalg import solve_continuous_lyapunov as lyap
except ImportError:
    from scipy.linalg import solve_lyapunov as lyap

class sde_RBF(RBF):
    """

    Class provide extra functionality to transfer this covariance function into
    SDE form.

    Radial Basis Function kernel:

    .. math::

        k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """
    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """

        self.variance.gradient = gradients[0]
        self.lengthscale.gradient = gradients[1]

    def sde(self):
        """
        Return the state space representation of the covariance.
        """

        N = 10# approximation order ( number of terms in exponent series expansion)
        roots_rounding_decimals = 6

        fn = np.math.factorial(N)

        kappa = 1.0/2.0/self.lengthscale**2

        Qc = np.array((self.variance*np.sqrt(np.pi/kappa)*fn*(4*kappa)**N,),)

        pp = np.zeros((2*N+1,)) # array of polynomial coefficients from higher power to lower

        for n in range(0, N+1): # (2N+1) - number of polynomial coefficients
            pp[2*(N-n)] = fn*(4.0*kappa)**(N-n)/np.math.factorial(n)*(-1)**n

        pp = sp.poly1d(pp)
        roots = sp.roots(pp)

        neg_real_part_roots = roots[np.round(np.real(roots) ,roots_rounding_decimals) < 0]
        aa = sp.poly1d(neg_real_part_roots, r=True).coeffs

        F = np.diag(np.ones((N-1,)),1)
        F[-1,:] = -aa[-1:0:-1]

        L= np.zeros((N,1))
        L[N-1,0] = 1

        H = np.zeros((1,N))
        H[0,0] = 1

        # Infinite covariance:
        Pinf = lyap(F, -np.dot(L,np.dot( Qc[0,0],L.T)))
        Pinf = 0.5*(Pinf + Pinf.T)
        # Allocating space for derivatives
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2])
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2])

        # Derivatives:
        dFvariance = np.zeros(F.shape)
        dFlengthscale = np.zeros(F.shape)
        dFlengthscale[-1,:] = -aa[-1:0:-1]/self.lengthscale * np.arange(-N,0,1)

        dQcvariance = Qc/self.variance
        dQclengthscale = np.array(((self.variance*np.sqrt(2*np.pi)*fn*2**N*self.lengthscale**(-2*N)*(1-2*N,),)))

        dPinf_variance = Pinf/self.variance

        lp = Pinf.shape[0]
        coeff = np.arange(1,lp+1).reshape(lp,1) + np.arange(1,lp+1).reshape(1,lp) - 2
        coeff[np.mod(coeff,2) != 0] = 0
        dPinf_lengthscale = -1/self.lengthscale*Pinf*coeff

        dF[:,:,0]    = dFvariance
        dF[:,:,1]    = dFlengthscale
        dQc[:,:,0]   = dQcvariance
        dQc[:,:,1]   = dQclengthscale
        dPinf[:,:,0] = dPinf_variance
        dPinf[:,:,1] = dPinf_lengthscale

        P0 = Pinf.copy()
        dP0 = dPinf.copy()

        # Benefits of this are not very sound. Helps only in one case:
        # SVD Kalman + RBF kernel
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf,dP0, T) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )

        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

class sde_Exponential(Exponential):
    """

    Class provide extra functionality to transfer this covariance function into
    SDE form.

    Exponential kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r \\bigg) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """

        self.variance.gradient = gradients[0]
        self.lengthscale.gradient = gradients[1]

    def sde(self):
        """
        Return the state space representation of the covariance.
        """
        variance = float(self.variance.values)
        lengthscale = float(self.lengthscale)

        F  = np.array(((-1.0/lengthscale,),))
        L  = np.array(((1.0,),))
        Qc = np.array( ((2.0*variance/lengthscale,),) )
        H = np.array(((1.0,),))
        Pinf = np.array(((variance,),))
        P0 = Pinf.copy()

        dF = np.zeros((1,1,2));
        dQc = np.zeros((1,1,2));
        dPinf = np.zeros((1,1,2));

        dF[:,:,0] = 0.0
        dF[:,:,1] = 1.0/lengthscale**2

        dQc[:,:,0] = 2.0/lengthscale
        dQc[:,:,1] = -2.0*variance/lengthscale**2

        dPinf[:,:,0] = 1.0
        dPinf[:,:,1] = 0.0

        dP0 = dPinf.copy()

        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

class sde_RatQuad(RatQuad):
    """

    Class provide extra functionality to transfer this covariance function into
    SDE form.

    Rational Quadratic kernel:

    .. math::

       k(r) = \sigma^2 \\bigg( 1 + \\frac{r^2}{2} \\bigg)^{- \alpha} \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde(self):
        """
        Return the state space representation of the covariance.
        """

        assert False, 'Not Implemented'

        # Params to use:

        # self.lengthscale
        # self.variance
        #self.power

        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)
