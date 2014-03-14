# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import sys

verbose = 0



class Kern_check_model(GPy.core.Model):
    """
    This is a dummy model class used as a base class for checking that the
    gradients of a given kernel are implemented correctly. It enables
    checkgrad() to be called independently on a kernel.
    """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        GPy.core.Model.__init__(self, 'kernel_test_model')
        if kernel==None:
            kernel = GPy.kern.RBF(1)
        if X is None:
            X = np.random.randn(20, kernel.input_dim)
        if dL_dK is None:
            if X2 is None:
                dL_dK = np.ones((X.shape[0], X.shape[0]))
            else:
                dL_dK = np.ones((X.shape[0], X2.shape[0]))

        self.kernel = kernel
        self.X = GPy.core.parameterization.Param('X',X)
        self.X2 = X2
        self.dL_dK = dL_dK

    def is_positive_semi_definite(self):
        v = np.linalg.eig(self.kernel.K(self.X))[0]
        if any(v.real<=-1e-10):
            print v.real.min()
            return False
        else:
            return True

    def log_likelihood(self):
        return np.sum(self.dL_dK*self.kernel.K(self.X, self.X2))

class Kern_check_dK_dtheta(Kern_check_model):
    """
    This class allows gradient checks for the gradient of a kernel with
    respect to parameters.
    """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)
        self.add_parameter(self.kernel)

    def parameters_changed(self):
        return self.kernel.update_gradients_full(self.dL_dK, self.X, self.X2)


class Kern_check_dKdiag_dtheta(Kern_check_model):
    """
    This class allows gradient checks of the gradient of the diagonal of a
    kernel with respect to the parameters.
    """
    def __init__(self, kernel=None, dL_dK=None, X=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)
        self.add_parameter(self.kernel)

    def log_likelihood(self):
        return (np.diag(self.dL_dK)*self.kernel.Kdiag(self.X)).sum()

    def parameters_changed(self):
        self.kernel.update_gradients_diag(np.diag(self.dL_dK), self.X)

class Kern_check_dK_dX(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)
        self.add_parameter(self.X)

    def parameters_changed(self):
        self.X.gradient =  self.kernel.gradients_X(self.dL_dK, self.X, self.X2)

class Kern_check_dKdiag_dX(Kern_check_dK_dX):
    """This class allows gradient checks for the gradient of a kernel diagonal with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_dK_dX.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)

    def log_likelihood(self):
        return (np.diag(self.dL_dK)*self.kernel.Kdiag(self.X)).sum()

    def parameters_changed(self):
        self.X.gradient =  self.kernel.gradients_X_diag(self.dL_dK.diagonal(), self.X)



def check_kernel_gradient_functions(kern, X=None, X2=None, output_ind=None, verbose=False):
    """
    This function runs on kernels to check the correctness of their
    implementation. It checks that the covariance function is positive definite
    for a randomly generated data set.

    :param kern: the kernel to be tested.
    :type kern: GPy.kern.Kernpart
    :param X: X input values to test the covariance function.
    :type X: ndarray
    :param X2: X2 input values to test the covariance function.
    :type X2: ndarray

    """
    pass_checks = True
    if X==None:
        X = np.random.randn(10, kern.input_dim)
        if output_ind is not None:
            X[:, output_ind] = np.random.randint(kern.output_dim, X.shape[0])
    if X2==None:
        X2 = np.random.randn(20, kern.input_dim)
        if output_ind is not None:
            X2[:, output_ind] = np.random.randint(kern.output_dim, X2.shape[0])

    if verbose:
        print("Checking covariance function is positive definite.")
    result = Kern_check_model(kern, X=X).is_positive_semi_definite()
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Positive definite check failed for " + kern.name + " covariance function.")
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of K(X, X) wrt theta.")
    result = Kern_check_dK_dtheta(kern, X=X, X2=None).checkgrad(verbose=verbose)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of K(X, X) wrt theta failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dK_dtheta(kern, X=X, X2=None).checkgrad(verbose=True)
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of K(X, X2) wrt theta.")
    result = Kern_check_dK_dtheta(kern, X=X, X2=X2).checkgrad(verbose=verbose)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of K(X, X) wrt theta failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dK_dtheta(kern, X=X, X2=X2).checkgrad(verbose=True)
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of Kdiag(X) wrt theta.")
    result = Kern_check_dKdiag_dtheta(kern, X=X).checkgrad(verbose=verbose)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of Kdiag(X) wrt theta failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dKdiag_dtheta(kern, X=X).checkgrad(verbose=True)
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of K(X, X) wrt X.")
    try:
        result = Kern_check_dK_dX(kern, X=X, X2=None).checkgrad(verbose=verbose)
    except NotImplementedError:
        result=True
        if verbose:
            print("gradients_X not implemented for " + kern.name)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of K(X, X) wrt X failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dK_dX(kern, X=X, X2=None).checkgrad(verbose=True)
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of K(X, X2) wrt X.")
    try:
        result = Kern_check_dK_dX(kern, X=X, X2=X2).checkgrad(verbose=verbose)
    except NotImplementedError:
        result=True
        if verbose:
            print("gradients_X not implemented for " + kern.name)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of K(X, X) wrt X failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dK_dX(kern, X=X, X2=X2).checkgrad(verbose=True)
        pass_checks = False
        return False

    if verbose:
        print("Checking gradients of Kdiag(X) wrt X.")
    try:
        result = Kern_check_dKdiag_dX(kern, X=X).checkgrad(verbose=verbose)
    except NotImplementedError:
        result=True
        if verbose:
            print("gradients_X not implemented for " + kern.name)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of Kdiag(X) wrt X failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dKdiag_dX(kern, X=X).checkgrad(verbose=True)
        pass_checks = False
        return False

    return pass_checks



class KernelGradientTestsContinuous(unittest.TestCase):
    def setUp(self):
        self.N, self.D = 100, 5
        self.X = np.random.randn(self.N,self.D)
        self.X2 = np.random.randn(self.N+10,self.D)

        continuous_kerns = ['RBF', 'Linear']
        self.kernclasses = [getattr(GPy.kern, s) for s in continuous_kerns]

    def test_Matern32(self):
        k = GPy.kern.Matern32(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

    def test_Prod(self):
        k = GPy.kern.Matern32([2,3]) * GPy.kern.RBF([0,4]) + GPy.kern.Linear(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

    def test_Add(self):
        k = GPy.kern.Matern32([2,3]) + GPy.kern.RBF([0,4]) + GPy.kern.Linear(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

    def test_Matern52(self):
        k = GPy.kern.Matern52(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

    def test_RBF(self):
        k = GPy.kern.RBF(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

    def test_Linear(self):
        k = GPy.kern.Linear(self.D)
        k.randomize()
        self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))

#TODO: turn off grad checkingwrt X for indexed kernels like coregionalize
# class KernelGradientTestsContinuous1D(unittest.TestCase):
#     def setUp(self):
#         self.N, self.D = 100, 1
#         self.X = np.random.randn(self.N,self.D)
#         self.X2 = np.random.randn(self.N+10,self.D)
#
#         continuous_kerns = ['RBF', 'Linear']
#         self.kernclasses = [getattr(GPy.kern, s) for s in continuous_kerns]
#
#     def test_PeriodicExponential(self):
#         k = GPy.kern.PeriodicExponential(self.D)
#         k.randomize()
#         self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))
#
#     def test_PeriodicMatern32(self):
#         k = GPy.kern.PeriodicMatern32(self.D)
#         k.randomize()
#         self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))
#
#     def test_PeriodicMatern52(self):
#         k = GPy.kern.PeriodicMatern52(self.D)
#         k.randomize()
#         self.assertTrue(check_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose))


class KernelTestsMiscellaneous(unittest.TestCase):
    def setUp(self):
        N, D = 100, 10
        self.X = np.linspace(-np.pi, +np.pi, N)[:,None] * np.ones(D)
        self.rbf = GPy.kern.RBF(range(2))
        self.linear = GPy.kern.Linear((3,6))
        self.matern = GPy.kern.Matern32(np.array([2,4,7]))
        self.sumkern = self.rbf + self.linear
        self.sumkern += self.matern
        self.sumkern.randomize()

    def test_active_dims(self):
        self.assertListEqual(self.sumkern.active_dims.tolist(), range(8))

    def test_which_parts(self):
        self.assertTrue(np.allclose(self.sumkern.K(self.X, which_parts=[self.linear, self.matern]), self.linear.K(self.X)+self.matern.K(self.X)))
        self.assertTrue(np.allclose(self.sumkern.K(self.X, which_parts=[self.linear, self.rbf]), self.linear.K(self.X)+self.rbf.K(self.X)))
        self.assertTrue(np.allclose(self.sumkern.K(self.X, which_parts=self.sumkern.parts[0]), self.rbf.K(self.X)))

class KernelTestsNonContinuous(unittest.TestCase):
    def setUp(self):
        N = 100
        N1 = 110
        self.D = 2
        D = self.D
        self.X = np.random.randn(N,D)
        self.X2 = np.random.randn(N1,D)
        self.X_block = np.zeros((N+N1, D+D+1))
        self.X_block[0:N, 0:D] = self.X
        self.X_block[N:N+N1, D:D+D] = self.X2
        self.X_block[0:N, -1] = 1
        self.X_block[N:N+1, -1] = 2

    def test_IndependantOutputs(self):
        k = GPy.kern.RBF(self.D)
        kern = GPy.kern.IndependentOutputs(self.D+self.D,k)
        self.assertTrue(check_kernel_gradient_functions(kern, X=self.X, X2=self.X2, verbose=verbose))

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
