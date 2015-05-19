# -*- coding: utf-8 -*-
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .stationary import Matern32
from .stationary import Matern52
import numpy as np

class sde_Matern32(Matern32):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Matern 3/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

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
        lengthscale = float(self.lengthscale.values)
        
        foo  = np.sqrt(3.)/lengthscale 
        F    = np.array([[0, 1], [-foo**2, -2*foo]]) 
        L    = np.array([[0], [1]]) 
        Qc   = np.array([[12.*np.sqrt(3) / lengthscale**3 * variance]]) 
        H    = np.array([[1, 0]]) 
        Pinf = np.array([[variance, 0],  
        [0,              3.*variance/(lengthscale**2)]]) 
        # Allocate space for the derivatives 
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2]) 
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2]) 
        # The partial derivatives 
        dFvariance       = np.zeros([2,2]) 
        dFlengthscale    = np.array([[0,0], 
        [6./lengthscale**3,2*np.sqrt(3)/lengthscale**2]]) 
        dQcvariance      = np.array([12.*np.sqrt(3)/lengthscale**3]) 
        dQclengthscale   = np.array([-3*12*np.sqrt(3)/lengthscale**4*variance]) 
        dPinfvariance    = np.array([[1,0],[0,3./lengthscale**2]]) 
        dPinflengthscale = np.array([[0,0], 
        [0,-6*variance/lengthscale**3]]) 
        # Combine the derivatives 
        dF[:,:,0]    = dFvariance 
        dF[:,:,1]    = dFlengthscale 
        dQc[:,:,0]   = dQcvariance 
        dQc[:,:,1]   = dQclengthscale 
        dPinf[:,:,0] = dPinfvariance 
        dPinf[:,:,1] = dPinflengthscale 

        return (F, L, Qc, H, Pinf, dF, dQc, dPinf)

class sde_Matern52(Matern52):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Matern 5/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{5} r + \frac{5}{3}r^2) \exp(- \sqrt{5} r) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

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
        lengthscale = float(self.lengthscale.values)

        lamda = np.sqrt(5.0)/lengthscale
        kappa = 5.0/3.0*variance/lengthscale**2        
        
        F = np.array(((0, 1,0), (0, 0, 1), (-lamda**3, -3.0*lamda**2, -3*lamda)))
        L = np.array(((0,),(0,),(1,)))
        Qc = np.array((((variance*400.0*np.sqrt(5.0)/3.0/lengthscale**5),),))
        H = np.array(((1,0,0),))        
        
        Pinf = np.array(((variance,0,-kappa), (0, kappa, 0), (-kappa, 0, 25.0*variance/lengthscale**4)))
        
        # Allocate space for the derivatives         
        dF = np.empty((3,3,2))        
        dQc = np.empty((1,1,2))        
        dPinf = np.empty((3,3,2))
        
         # The partial derivatives 
        dFvariance = np.zeros((3,3))
        dFlengthscale = np.array(((0,0,0),(0,0,0),(15.0*np.sqrt(5.0)/lengthscale**4, 
                                   30.0/lengthscale**3, 3*np.sqrt(5.0)/lengthscale**2)))
        dQcvariance = np.array((((400*np.sqrt(5)/3/lengthscale**5,),)))
        dQclengthscale = np.array((((-variance*2000*np.sqrt(5)/3/lengthscale**6,),)))        
        
        dPinf_variance = Pinf/variance
        kappa2 = -2.0*kappa/lengthscale
        dPinf_lengthscale = np.array(((0,0,-kappa2),(0,kappa2,0),(-kappa2, 
                                    0,-100*variance/lengthscale**5)))        
        # Combine the derivatives 
        dF[:,:,0] = dFvariance
        dF[:,:,1] = dFlengthscale        
        dQc[:,:,0] = dQcvariance         
        dQc[:,:,1] = dQclengthscale        
        dPinf[:,:,0] = dPinf_variance
        dPinf[:,:,1] = dPinf_lengthscale
        
#        % Derivative of F w.r.t. parameter magnSigma2
#    dFmagnSigma2    =  [0,  0,  0;
#                        0,  0,  0;
#                        0,  0,  0];
#    
#    % Derivative of F w.r.t parameter lengthScale
#    dFlengthScale   =  [0,                          0,                  0;
#                        0,                          0,                  0;
#                        15*sqrt(5)/lengthScale^4,    30/lengthScale^3,   3*sqrt(5)/lengthScale^2];
#    
#    % Derivative of Qc w.r.t. parameter magnSigma2
#    dQcmagnSigma2   =   400*sqrt(5)/3/lengthScale^5;
#    
#    % Derivative of Qc w.r.t. parameter lengthScale
#    dQclengthScale  =   -magnSigma2*2000*sqrt(5)/3/lengthScale^6;
#    
#    % Derivative of Pinf w.r.t. parameter magnSigma2    
#    dPinfmagnSigma2 = Pinf/magnSigma2;
#    
#    % Derivative of Pinf w.r.t. parameter lengthScale
#    kappa2 = -2*kappa/lengthScale;
#    dPinflengthScale = [0,          0,       -kappa2;
#                        0,          kappa2,  0;
#                        -kappa2,    0,       -100*magnSigma2/lengthScale^5];
#  
#    % Stack all derivatives
#    dF = zeros(3,3,2);  
#    dQc = zeros(1,1,2); 
#    dPinf = zeros(3,3,2);
#  
#    dF(:,:,1) = dFmagnSigma2;
#    dF(:,:,2) = dFlengthScale;
#    dQc(:,:,1) = dQcmagnSigma2;
#    dQc(:,:,2) = dQclengthScale;
#    dPinf(:,:,1) = dPinfmagnSigma2;
#    dPinf(:,:,2) = dPinflengthScale; 
  
#        % Derived constants
#          lambda = sqrt(5)/lengthScale;
#        
#          % Feedback matrix
#          F = [ 0,          1,          0;
#                0,          0,          1;
#               -lambda^3, -3*lambda^2, -3*lambda];
#        
#          % Noise effect matrix
#          L = [0; 0; 1];
#        
#          % Spectral density
#          Qc = magnSigma2*400*sqrt(5)/3/lengthScale^5;
#        
#          % Observation model
#          H = [1, 0, 0];
  
  
#        %% Stationary covariance
#          
#          % Calculate Pinf only if requested
#          if nargout > 4,
#              
#            % Derived constant
#            kappa = 5/3*magnSigma2/lengthScale^2;
#            
#            % Stationary covariance
#            Pinf = [magnSigma2, 0,      -kappa;
#                    0,          kappa,  0;
#                    -kappa,     0,      25*magnSigma2/lengthScale^4];
#                
#          end
        
        return (F, L, Qc, H, Pinf, dF, dQc, dPinf)  