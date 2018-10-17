# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np

class ODE_1(Kernpart):
    """
    kernel resultiong from a first order ODE with OU driving GP

    :param input_dim: the number of input dimension, has to be equal to one
    :type input_dim: int
    :param varianceU: variance of the driving GP
    :type varianceU: float
    :param lengthscaleU: lengthscale of the driving GP  (sqrt(3)/lengthscaleU)
    :type lengthscaleU: float
    :param varianceY: 'variance' of the transfer function
    :type varianceY: float
    :param lengthscaleY: 'lengthscale' of the transfer function (1/lengthscaleY)
    :type lengthscaleY: float
    :rtype: kernel object

    """
    def __init__(self, input_dim=1, varianceU=1., varianceY=1., lengthscaleU=None, lengthscaleY=None):
        assert input_dim==1, "Only defined for input_dim = 1"
        self.input_dim = input_dim
        self.num_params = 4
        self.name = 'ODE_1'
        if lengthscaleU is not None:
            lengthscaleU = np.asarray(lengthscaleU)
            assert lengthscaleU.size == 1, "lengthscaleU should be one dimensional"
        else:
            lengthscaleU = np.ones(1)
        if lengthscaleY is not None:
            lengthscaleY = np.asarray(lengthscaleY)
            assert lengthscaleY.size == 1, "lengthscaleY should be one dimensional"
        else:
            lengthscaleY = np.ones(1)
            #lengthscaleY = 0.5
        self._set_params(np.hstack((varianceU, varianceY, lengthscaleU,lengthscaleY)))

    def _get_params(self):
        """return the value of the parameters."""
        return np.hstack((self.varianceU,self.varianceY, self.lengthscaleU,self.lengthscaleY))

    def _set_params(self, x):
        """set the value of the parameters."""
        assert x.size == self.num_params
        self.varianceU = x[0]
        self.varianceY = x[1]
        self.lengthscaleU = x[2]
        self.lengthscaleY = x[3]

    def _get_param_names(self):
        """return parameter names."""
        return ['varianceU','varianceY', 'lengthscaleU', 'lengthscaleY']


    def K(self, X, X2, target):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: X2 = X
       # i1 = X[:,1]
       # i2 = X2[:,1]
       # X = X[:,0].reshape(-1,1)
       # X2 = X2[:,0].reshape(-1,1)
        dist = np.abs(X - X2.T)
        
        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU
        #ly=self.lengthscaleY
        #lu=self.lengthscaleU

        k1 = np.exp(-ly*dist)*(2*lu+ly)/(lu+ly)**2
        k2 = (np.exp(-lu*dist)*(ly-2*lu+lu*ly*dist-lu**2*dist) + np.exp(-ly*dist)*(2*lu-ly) ) / (ly-lu)**2 
        k3 = np.exp(-lu*dist) * ( (1+lu*dist)/(lu+ly) + (lu)/(lu+ly)**2 )

        np.add(self.varianceU*self.varianceY*(k1+k2+k3), target, target)

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix associated to X."""
        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU
        #ly=self.lengthscaleY
        #lu=self.lengthscaleU
        
        k1 = (2*lu+ly)/(lu+ly)**2
        k2 = (ly-2*lu + 2*lu-ly ) / (ly-lu)**2 
        k3 = 1/(lu+ly) + (lu)/(lu+ly)**2 

        np.add(self.varianceU*self.varianceY*(k1+k2+k3), target, target)

    def _param_grad_helper(self, dL_dK, X, X2, target):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None: X2 = X
        dist = np.abs(X - X2.T)

        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU
        #ly=self.lengthscaleY
        #lu=self.lengthscaleU

        dk1theta1 = np.exp(-ly*dist)*2*(-lu)/(lu+ly)**3
        #c=np.sqrt(3)
        #t1=c/lu
        #t2=1/ly
        #dk1theta1=np.exp(-dist*ly)*t2*( (2*c*t2+2*t1)/(c*t2+t1)**2 -2*(2*c*t2*t1+t1**2)/(c*t2+t1)**3   )
        
        dk2theta1 = 1*( 
            np.exp(-lu*dist)*dist*(-ly+2*lu-lu*ly*dist+dist*lu**2)*(ly-lu)**(-2) + np.exp(-lu*dist)*(-2+ly*dist-2*dist*lu)*(ly-lu)**(-2) 
            +np.exp(-dist*lu)*(ly-2*lu+ly*lu*dist-dist*lu**2)*2*(ly-lu)**(-3) 
            +np.exp(-dist*ly)*2*(ly-lu)**(-2)
            +np.exp(-dist*ly)*2*(2*lu-ly)*(ly-lu)**(-3)
            )
      
        dk3theta1 = np.exp(-dist*lu)*(lu+ly)**(-2)*((2*lu+ly+dist*lu**2+lu*ly*dist)*(-dist-2/(lu+ly))+2+2*lu*dist+ly*dist)

        dktheta1 = self.varianceU*self.varianceY*(dk1theta1+dk2theta1+dk3theta1)




        dk1theta2 = np.exp(-ly*dist) * ((lu+ly)**(-2)) * (  (-dist)*(2*lu+ly)  +  1  +  (-2)*(2*lu+ly)/(lu+ly)  )

        dk2theta2 = 1*(
            np.exp(-dist*lu)*(ly-lu)**(-2) * ( 1+lu*dist+(-2)*(ly-2*lu+lu*ly*dist-dist*lu**2)*(ly-lu)**(-1) )
            +np.exp(-dist*ly)*(ly-lu)**(-2) * ( (-dist)*(2*lu-ly) -1+(2*lu-ly)*(-2)*(ly-lu)**(-1) )
            )

        dk3theta2 = np.exp(-dist*lu) * (-3*lu-ly-dist*lu**2-lu*ly*dist)/(lu+ly)**3

        dktheta2 = self.varianceU*self.varianceY*(dk1theta2 + dk2theta2 +dk3theta2)



        k1 = np.exp(-ly*dist)*(2*lu+ly)/(lu+ly)**2
        k2 = (np.exp(-lu*dist)*(ly-2*lu+lu*ly*dist-lu**2*dist) + np.exp(-ly*dist)*(2*lu-ly) ) / (ly-lu)**2 
        k3 = np.exp(-lu*dist) * ( (1+lu*dist)/(lu+ly) + (lu)/(lu+ly)**2 )
        dkdvar = k1+k2+k3

        target[0] += np.sum(self.varianceY*dkdvar * dL_dK)
        target[1] += np.sum(self.varianceU*dkdvar * dL_dK)
        target[2] += np.sum(dktheta1*(-np.sqrt(3)*self.lengthscaleU**(-2)) * dL_dK)
        target[3] += np.sum(dktheta2*(-self.lengthscaleY**(-2)) * dL_dK)


    # def dKdiag_dtheta(self, dL_dKdiag, X, target):
    #     """derivative of the diagonal of the covariance matrix with respect to the parameters."""
    #     # NB: derivative of diagonal elements wrt lengthscale is 0
    #     target[0] += np.sum(dL_dKdiag)

    # def dK_dX(self, dL_dK, X, X2, target):
    #     """derivative of the covariance matrix with respect to X."""
    #     if X2 is None: X2 = X
    #     dist = np.sqrt(np.sum(np.square((X[:, None, :] - X2[None, :, :]) / self.lengthscale), -1))[:, :, None]
    #     ddist_dX = (X[:, None, :] - X2[None, :, :]) / self.lengthscale ** 2 / np.where(dist != 0., dist, np.inf)
    #     dK_dX = -np.transpose(self.variance * np.exp(-dist) * ddist_dX, (1, 0, 2))
    #     target += np.sum(dK_dX * dL_dK.T[:, :, None], 0)

    # def dKdiag_dX(self, dL_dKdiag, X, target):
    #     pass
