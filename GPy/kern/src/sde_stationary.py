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
import warnings

class sde_RBF(RBF):
    """

    Class provide extra functionality to transfer this covariance function into
    SDE form.

    Radial Basis Function kernel:

    .. math::

        k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """
    def __init__(self, *args, **kwargs):
        """
        Init constructior.
        
        Two optinal extra parameters are added in addition to the ones in 
        RBF kernel.
        
        :param approx_order: approximation order for the RBF covariance. (Default 10)
        :type approx_order: int
        
        :param balance: Whether to balance this kernel separately. (Defaulf True). Model has a separate parameter for balancing.
        :type balance: bool
        """
        
        if 'balance' in kwargs:
            self.balance = bool( kwargs.get('balance') )
            del kwargs['balance']
        else:
            self.balance = True
        
        
        if 'approx_order' in kwargs:
            self.approx_order = kwargs.get('approx_order')
            del kwargs['approx_order']
        else:
            self.approx_order = 6
        
        
        
        super(sde_RBF, self).__init__(*args, **kwargs)
        
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
        
        Note! For Sparse GP inference too small or two high values of lengthscale
        lead to instabilities. This is because Qc are too high or too low
        and P_inf are not full rank. This effect depends on approximatio order.
        For N = 10. lengthscale must be in (0.8,8). For other N tests must be conducted.
        N=6: (0.06,31)
        Variance should be within reasonable bounds as well, but its dependence is linear.
        
        The above facts do not take into accout regularization.
        """
        #import pdb; pdb.set_trace()
        if self.approx_order is not None:
            N = self.approx_order
        else:
            N = 10# approximation order ( number of terms in exponent series expansion)
            
        roots_rounding_decimals = 6

        fn = np.math.factorial(N)

        p_lengthscale = float( self.lengthscale )
        p_variance = float(self.variance)
        kappa = 1.0/2.0/p_lengthscale**2

        Qc = np.array( ((p_variance*np.sqrt(np.pi/kappa)*fn*(4*kappa)**N,),) )
        
        eps = 1e-12
        if (float(Qc) > 1.0/eps) or (float(Qc) < eps):
            warnings.warn("""sde_RBF kernel: the noise variance Qc is either very large or very small. 
                                It influece conditioning of P_inf: {0:e}""".format(float(Qc)) )

        pp1 = np.zeros((2*N+1,)) # array of polynomial coefficients from higher power to lower

        for n in range(0, N+1): # (2N+1) - number of polynomial coefficients
            pp1[2*(N-n)] = fn*(4.0*kappa)**(N-n)/np.math.factorial(n)*(-1)**n
            
        pp = sp.poly1d(pp1)
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
        #import pdb; pdb.set_trace()
        Pinf = sp.linalg.solve_lyapunov(F, -np.dot(L,np.dot( Qc[0,0],L.T)))
        Pinf = 0.5*(Pinf + Pinf.T)
        # Allocating space for derivatives
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2])
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2])

        # Derivatives:
        dFvariance = np.zeros(F.shape)
        dFlengthscale = np.zeros(F.shape)
        dFlengthscale[-1,:] = -aa[-1:0:-1]/p_lengthscale * np.arange(-N,0,1)

        dQcvariance = Qc/p_variance
        dQclengthscale = np.array(( (p_variance*np.sqrt(2*np.pi)*fn*2**N*p_lengthscale**(-2*N)*(1-2*N),),))
        
        dPinf_variance = Pinf/p_variance

        lp = Pinf.shape[0]
        coeff = np.arange(1,lp+1).reshape(lp,1) + np.arange(1,lp+1).reshape(1,lp) - 2
        coeff[np.mod(coeff,2) != 0] = 0
        dPinf_lengthscale = -1/p_lengthscale*Pinf*coeff

        dF[:,:,0]    = dFvariance
        dF[:,:,1]    = dFlengthscale
        dQc[:,:,0]   = dQcvariance
        dQc[:,:,1]   = dQclengthscale
        dPinf[:,:,0] = dPinf_variance
        dPinf[:,:,1] = dPinf_lengthscale

        P0 = Pinf.copy()
        dP0 = dPinf.copy()

        if self.balance:
            # Benefits of this are not very sound. Helps only in one case:
            # SVD Kalman + RBF kernel
            import GPy.models.state_space_main as ssm
            (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf,dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )

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
