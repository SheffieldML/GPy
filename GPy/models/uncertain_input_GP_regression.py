# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from ..inference.likelihoods import likelihood
from sparse_GP_regression import sparse_GP_regression

class uncertain_input_GP_regression(sparse_GP_regression):
    """
    Variational sparse GP model (Regression) with uncertainty on the inputs

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param X_uncertainty: uncertainty on X (Gaussian variances, assumed isotrpic)
    :type X_uncertainty: np.ndarray (N x Q)
    :param Y: observed data
    :type Y: np.ndarray of observations (N x D)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param beta: noise precision. TODO> ignore beta if doing EP
    :type beta: float
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self,X,Y,X_uncertainty,kernel=None, beta=100., Z=None,Zslices=None,M=10,normalize_X=False,normalize_Y=False):
        self.X_uncertainty = X_uncertainty
        sparse_GP_regression.__init__(self, X, Y, kernel = kernel, beta = beta, normalize_X = normalize_X, normalize_Y = normalize_Y)
        self.trYYT = np.sum(np.square(self.Y))

    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        #TODO: slices for psi statistics (easy enough)
        self.Kmm = self.kern.K(self.Z)
        self.psi0 = self.kern.psi0(self.Z,self.X, self.X_uncertainty).sum()
        self.psi1 = self.kern.psi1(self.Z,self.X, self.X_uncertainty).T
        self.psi2 = self.kern.psi2(self.Z,self.X, self.X_uncertainty)

    def dL_dtheta(self):
        #re-cast computations in psi2 back to psi1:
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z,self.X,self.X_uncertainty)
        dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
        dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty) # for multiple_beta, dL_dpsi2 will be a different shape
        return dL_dtheta

    def dL_dZ(self):
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm,self.Z,)#factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
        dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty)
        return dL_dZ

    def plot(self,*args,**kwargs):
        """
        Plot the fitted model: just call the sparse GP_regression plot function and then add
        markers to represent uncertainty on the inputs
        """
        sparse_GP_regression.plot(self,*args,**kwargs)
        if self.Q==1:
            pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_uncertainty.flatten()))

