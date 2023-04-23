# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy, Arno Solin
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .standard_periodic import StdPeriodic

import numpy as np
import scipy as sp
import warnings

from scipy import special as special

class sde_StdPeriodic(StdPeriodic):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Standard Periodic kernel:

    .. math::

       k(x,y) = \theta_1 \exp \left[  - \frac{1}{2} {}\sum_{i=1}^{input\_dim}  
       \left( \frac{\sin(\frac{\pi}{\lambda_i} (x_i - y_i) )}{l_i} \right)^2 \right] }

    """
    # TODO: write comment to the constructor arguments
    def __init__(self, *args, **kwargs):
        """
        Init constructior.
        
        Two optinal extra parameters are added in addition to the ones in 
        StdPeriodic kernel.
        
        :param approx_order: approximation order for the RBF covariance. (Default 7)
        :type approx_order: int
        
        :param balance: Whether to balance this kernel separately. (Defaulf False). Model has a separate parameter for balancing.
        :type balance: bool
        """
        
        #import pdb; pdb.set_trace()
        
        if 'approx_order' in kwargs:
            self.approx_order = kwargs.get('approx_order')
            del kwargs['approx_order']
        else:
            self.approx_order = 7
        
        
        if 'balance' in kwargs:
            self.balance = bool( kwargs.get('balance') )
            del kwargs['balance']
        else:
            self.balance = False
        
        super(sde_StdPeriodic, self).__init__(*args, **kwargs)
        
    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """
    
        self.variance.gradient = gradients[0]
        self.period.gradient = gradients[1]
        self.lengthscale.gradient = gradients[2]
        
    def sde(self): 
        """ 
        Return the state space representation of the standard periodic covariance.
        
        
        ! Note: one must constrain lengthscale not to drop below 0.2. (independently of approximation order)
        After this Bessel functions of the first becomes NaN. Rescaling
        time variable might help.
        
        ! Note: one must keep period also not very low. Because then
        the gradients wrt wavelength become ustable. 
        However this might depend on the data. For test example with
        300 data points the low limit is 0.15. 
        """ 
        
        #import pdb; pdb.set_trace()
        # Params to use: (in that order)
        #self.variance
        #self.period
        #self.lengthscale
        if self.approx_order is not None:
            N = int(self.approx_order)
        else:
            N = 7 # approximation order        
        
        p_period = float(self.period)        
        p_lengthscale = 2*float(self.lengthscale)
        p_variance = float(self.variance)        
        
        w0 = 2*np.pi/p_period # frequency
        # lengthscale is multiplied by 2 because of different definition of lengthscale
        
        [q2,dq2l] = seriescoeff(N, p_lengthscale, p_variance)        
        
        dq2l = 2*dq2l  # This is because the lengthscale if multiplied by 2.
        
        eps = 1e-12
        if np.any( np.isfinite(q2) == False) or np.any( np.abs(q2) > 1.0/eps) or np.any( np.abs(q2) < eps):
            warnings.warn("sde_Periodic:  Infinite, too small, or too large (eps={0:e}) values in q2 :".format(eps) + q2.__format__("") )
                                
        if np.any( np.isfinite(dq2l) == False) or np.any( np.abs(dq2l) > 1.0/eps) or np.any( np.abs(dq2l) < eps):
            warnings.warn("sde_Periodic:  Infinite, too small, or too large (eps={0:e}) values in dq2l :".format(eps) + q2.__format__("") )
                 
                 
        F    = np.kron(np.diag(range(0,N+1)),np.array( ((0, -w0), (w0, 0)) ) )
        L    = np.eye(2*(N+1))
        Qc   = np.zeros((2*(N+1), 2*(N+1)))
        P_inf = np.kron(np.diag(q2),np.eye(2))
        H    = np.kron(np.ones((1,N+1)),np.array((1,0)) )
        P0 = P_inf.copy()
        
        # Derivatives
        dF = np.empty((F.shape[0], F.shape[1], 3))
        dQc = np.empty((Qc.shape[0], Qc.shape[1], 3))
        dP_inf = np.empty((P_inf.shape[0], P_inf.shape[1], 3))         
        
        # Derivatives wrt self.variance
        dF[:,:,0] = np.zeros(F.shape)
        dQc[:,:,0] = np.zeros(Qc.shape)
        dP_inf[:,:,0] = P_inf / p_variance

        # Derivatives self.period
        dF[:,:,1] = np.kron(np.diag(range(0,N+1)),np.array( ((0,  w0), (-w0, 0)) ) / p_period );
        dQc[:,:,1] = np.zeros(Qc.shape)
        dP_inf[:,:,1] = np.zeros(P_inf.shape)      
        
        # Derivatives self.lengthscales        
        dF[:,:,2] = np.zeros(F.shape)
        dQc[:,:,2] = np.zeros(Qc.shape)
        dP_inf[:,:,2] = np.kron(np.diag(dq2l),np.eye(2))
        dP0 = dP_inf.copy()
        
        if self.balance:
            # Benefits of this are not very sound.
            import GPy.models.state_space_main as ssm
            (F, L, Qc, H, P_inf, P0, dF, dQc, dP_inf,dP0) = ssm.balance_ss_model(F, L, Qc, H, P_inf, P0, dF, dQc, dP_inf, dP0 )
            
        return (F, L, Qc, H, P_inf, P0, dF, dQc, dP_inf, dP0)
        
        
        
        
def seriescoeff(m=6,lengthScale=1.0,magnSigma2=1.0, true_covariance=False):
    """
    Calculate the coefficients q_j^2 for the covariance function 
    approximation:
    
        k(\tau) =  \sum_{j=0}^{+\infty} q_j^2 \cos(j\omega_0 \tau)
    
    Reference is:

    [1] Arno Solin and Simo Särkkä (2014). Explicit link between periodic 
        covariance functions and state space models. In Proceedings of the 
        Seventeenth International Conference on Artifcial Intelligence and 
        Statistics (AISTATS 2014). JMLR: W&CP, volume 33.    
    
    Note! Only the infinite approximation (through Bessel function) 
          is currently implemented.

    Input:
    ----------------
    
    m: int
        Degree of approximation. Default 6.
    lengthScale: float
        Length scale parameter in the kerenl
    magnSigma2:float
        Multiplier in front of the kernel.
        
    
    Output:
    -----------------
    
    coeffs: array(m+1)
        Covariance series coefficients
    
    coeffs_dl: array(m+1)
        Derivatives of the coefficients with respect to lengthscale.
    
    """
    
    if true_covariance:
        
        bb = lambda j,m: (1.0 + np.array((j != 0), dtype=float) ) / (2**(j)) *\
            sp.special.binom(j, sp.floor( (j-m)/2.0 * np.array(m<=j, dtype=float) ))*\
            np.array(m<=j, dtype=float) *np.array(sp.mod(j-m,2)==0, dtype=float)
                
        M,J = np.meshgrid(range(0,m+1),range(0,m+1))
        
        coeffs = bb(J,M) / sp.misc.factorial(J) * sp.exp( -lengthScale**(-2) ) *\
             (lengthScale**(-2))**J  *magnSigma2
        
        coeffs_dl = np.sum( coeffs*lengthScale**(-3)*(2.0-2.0*J*lengthScale**2),0)         
        
        coeffs = np.sum(coeffs,0)
        
    else:
        coeffs = 2*magnSigma2*sp.exp( -lengthScale**(-2) ) * special.iv(range(0,m+1),1.0/lengthScale**(2))
        if np.any( np.isfinite(coeffs) == False):
            raise ValueError("sde_standard_periodic: Coefficients are not finite!")
        #import pdb; pdb.set_trace()
        coeffs[0] = 0.5*coeffs[0]
        #print(coeffs)
        # Derivatives wrt (lengthScale)
        coeffs_dl = np.zeros(m+1)
        coeffs_dl[1:] = magnSigma2*lengthScale**(-3) * sp.exp(-lengthScale**(-2))*\
        (-4*special.iv(range(0,m),lengthScale**(-2)) + 4*(1+np.arange(1,m+1)*lengthScale**(2))*special.iv(range(1,m+1),lengthScale**(-2)) )    
            
        # The first element
        coeffs_dl[0] = magnSigma2*lengthScale**(-3) * np.exp(-lengthScale**(-2))*\
            (2*special.iv(0,lengthScale**(-2)) - 2*special.iv(1,lengthScale**(-2)) )     
        

    return coeffs.squeeze(), coeffs_dl.squeeze()
