# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
from GPy.core.model import Model

class Kern_check_model(Model):
    """This is a dummy model class used as a base class for checking that the gradients of a given kernel are implemented correctly. It enables checkgradient() to be called independently on a kernel."""
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        num_samples = 20
        num_samples2 = 10
        if kernel==None:
            kernel = GPy.kern.rbf(1)
        if X==None:
            X = np.random.randn(num_samples, kernel.input_dim)
        if X2==None:
            X2 = np.random.randn(num_samples2, kernel.input_dim)
        if dL_dK==None:
            dL_dK = np.ones((X.shape[0], X2.shape[0]))
        
        self.kernel=kernel
        self.X = X
        self.X2 = X2
        self.dL_dK = dL_dK
        #self.constrained_indices=[]
        #self.constraints=[]
        Model.__init__(self)

    def is_positive_definite(self):
        v = np.linalg.eig(self.kernel.K(self.X))[0]
        if any(v<0):
            return False
        else:
            return True
        
    def _get_params(self):
        return self.kernel._get_params()

    def _get_param_names(self):
        return self.kernel._get_param_names()

    def _set_params(self, x):
        self.kernel._set_params(x)

    def log_likelihood(self):
        return (self.dL_dK*self.kernel.K(self.X, self.X2)).sum()

    def _log_likelihood_gradients(self):
        raise NotImplementedError, "This needs to be implemented to use the kern_check_model class."
    
class Kern_check_dK_dtheta(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel with respect to parameters. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)

    def _log_likelihood_gradients(self):
        return self.kernel.dK_dtheta(self.dL_dK, self.X, self.X2)

class Kern_check_dKdiag_dtheta(Kern_check_model):
    """This class allows gradient checks of the gradient of the diagonal of a kernel with respect to the parameters."""
    def __init__(self, kernel=None, dL_dK=None, X=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)
        if dL_dK==None:
            self.dL_dK = np.ones((self.X.shape[0]))
        
    def log_likelihood(self):
        return (self.dL_dK*self.kernel.Kdiag(self.X)).sum()

    def _log_likelihood_gradients(self):
        return self.kernel.dKdiag_dtheta(self.dL_dK, self.X)

class Kern_check_dK_dX(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)

    def _log_likelihood_gradients(self):
        return self.kernel.dK_dX(self.dL_dK, self.X, self.X2).flatten()

    def _get_param_names(self):
        names = []
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                names.append('X_' +str(i) + ','+str(j))
        return names
                
    def _get_params(self):
        return self.X.flatten()

    def _set_params(self, x):
        self.X=x.reshape(self.X.shape)
    


class KernelTests(unittest.TestCase):
    def test_kerneltie(self):
        K = GPy.kern.rbf(5, ARD=True)
        K.tie_params('.*[01]')
        K.constrain_fixed('2')
        
        X = np.random.rand(5,5)
        Y = np.ones((5,1))
        m = GPy.models.GPRegression(X,Y,K)
        self.assertTrue(m.checkgrad())

    def test_rbfkernel(self):
        verbose = False
        kern = GPy.kern.rbf(5)
        self.assertTrue(Kern_check_model(kern).is_positive_definite())
        self.assertTrue(Kern_check_dK_dtheta(kern).checkgrad(verbose=verbose))
        self.assertTrue(Kern_check_dKdiag_dtheta(kern).checkgrad(verbose=verbose))
        self.assertTrue(Kern_check_dK_dX(kern).checkgrad(verbose=verbose))

    def test_fixedkernel(self):
        """
        Fixed effect kernel test
        """
        X = np.random.rand(30, 4)
        K = np.dot(X, X.T)
        kernel = GPy.kern.fixed(4, K)
        Y = np.ones((30,1))
        m = GPy.models.GPRegression(X,Y,kernel=kernel)
        self.assertTrue(m.checkgrad())

    def test_coregionalisation(self):
        X1 = np.random.rand(50,1)*8
        X2 = np.random.rand(30,1)*5
        index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
        X = np.hstack((np.vstack((X1,X2)),index))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
        Y2 = np.sin(X2) + np.random.randn(*X2.shape)*0.05 + 2.
        Y = np.vstack((Y1,Y2))

        k1 = GPy.kern.rbf(1) + GPy.kern.bias(1)
        k2 = GPy.kern.coregionalise(2,1)
        k = k1.prod(k2,tensor=True)
        m = GPy.models.GPRegression(X,Y,kernel=k)
        self.assertTrue(m.checkgrad())


       
if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
