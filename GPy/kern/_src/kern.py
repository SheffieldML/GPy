# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
import itertools
from ..core.parameterization import Parameterized
from GPy.core.parameterization.param import Param


class Kern(Parameterized):
    def __init__(self,input_dim,name):
        """
        The base class for a kernel: a positive definite function
        which forms of a covariance function (kernel).

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        super(Kern, self).__init__(name)
        self.input_dim = input_dim

    def K(self,X,X2,target):
        raise NotImplementedError
    def Kdiag(self,X,target):
        raise NotImplementedError
    def _param_grad_helper(self,dL_dK,X,X2,target):
        raise NotImplementedError
    def dKdiag_dtheta(self,dL_dKdiag,X,target): # TODO: Max??
        # In the base case compute this by calling _param_grad_helper. Need to
        # override for stationary covariances (for example) to save
        # time.
        for i in range(X.shape[0]):
            self._param_grad_helper(dL_dKdiag[i], X[i, :][None, :], X2=None, target=target)
    def psi0(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,dL_dpsi0,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,dL_dpsi1,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,dL_dpsi1,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,dL_dpsi2,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2, target):
        raise NotImplementedError
    def dKdiag_dX(self, dL_dK, X, target):
        raise NotImplementedError
    def update_gradients_full(self, dL_dK, X):
        """Set the gradients of all parameters when doing full (N) inference."""
        raise NotImplementedError
    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        """Set the gradients of all parameters when doing sparse (M) inference."""
        raise NotImplementedError
    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        """Set the gradients of all parameters when doing variational (M) inference with uncertain inputs."""
        raise NotImplementedError

    def plot_ARD(self, *args):
        """If an ARD kernel is present, plot a bar representation using matplotlib

        See GPy.plotting.matplot_dep.plot_ARD
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import kernel_plots
        return kernel_plots.plot_ARD(self,*args)


    def __add__(self, other):
        """ Overloading of the '+' operator. for more control, see self.add """
        return self.add(other)

    def add(self, other, tensor=False):
        """
        Add another kernel to this one.

        If Tensor is False, both kernels are defined on the same _space_. then
        the created kernel will have the same number of inputs as self and
        other (which must be the same).

        If Tensor is True, then the dimensions are stacked 'horizontally', so
        that the resulting kernel has self.input_dim + other.input_dim

        :param other: the other kernel to be added
        :type other: GPy.kern

        """
        assert isinstance(other, Kern), "only kernels can be added to kernels..."
        from add import Add
        return Add([self, other], tensor)

    def __call__(self, X, X2=None):
        return self.K(X, X2)

    def __mul__(self, other):
        """ Here we overload the '*' operator. See self.prod for more information"""
        return self.prod(other)

    def __pow__(self, other, tensor=False):
        """
        Shortcut for tensor `prod`.
        """
        return self.prod(other, tensor=True)

    def prod(self, other, tensor=False):
        """
        Multiply two kernels (either on the same space, or on the tensor product of the input space).

        :param other: the other kernel to be added
        :type other: GPy.kern
        :param tensor: whether or not to use the tensor space (default is false).
        :type tensor: bool

        """
        assert isinstance(other, Kern), "only kernels can be added to kernels..."
        from prod import Prod
        return Prod(self, other, tensor)


from GPy.core.model import Model

class Kern_check_model(Model):
    """This is a dummy model class used as a base class for checking that the gradients of a given kernel are implemented correctly. It enables checkgradient() to be called independently on a kernel."""
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Model.__init__(self, 'kernel_test_model')
        num_samples = 20
        num_samples2 = 10
        if kernel==None:
            kernel = GPy.kern.rbf(1)
        if X==None:
            X = np.random.randn(num_samples, kernel.input_dim)
        if dL_dK==None:
            if X2==None:
                dL_dK = np.ones((X.shape[0], X.shape[0]))
            else:
                dL_dK = np.ones((X.shape[0], X2.shape[0]))
        
        self.kernel=kernel
        self.add_parameter(kernel)
        self.X = X
        self.X2 = X2
        self.dL_dK = dL_dK

    def is_positive_definite(self):
        v = np.linalg.eig(self.kernel.K(self.X))[0]
        if any(v<-10*sys.float_info.epsilon):
            return False
        else:
            return True

    def log_likelihood(self):
        return (self.dL_dK*self.kernel.K(self.X, self.X2)).sum()

    def _log_likelihood_gradients(self):
        raise NotImplementedError, "This needs to be implemented to use the kern_check_model class."

class Kern_check_dK_dtheta(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel with respect to parameters. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)

    def _log_likelihood_gradients(self):
        return self.kernel._param_grad_helper(self.dL_dK, self.X, self.X2)





class Kern_check_dKdiag_dtheta(Kern_check_model):
    """This class allows gradient checks of the gradient of the diagonal of a kernel with respect to the parameters."""
    def __init__(self, kernel=None, dL_dK=None, X=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)
        if dL_dK==None:
            self.dL_dK = np.ones((self.X.shape[0]))
    def parameters_changed(self):
        self.kernel.update_gradients_full(self.dL_dK, self.X)        

    def log_likelihood(self):
        return (self.dL_dK*self.kernel.Kdiag(self.X)).sum()

    def _log_likelihood_gradients(self):
        return self.kernel.dKdiag_dtheta(self.dL_dK, self.X)

class Kern_check_dK_dX(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=X2)
        self.remove_parameter(kernel)
        self.X = Param('X', self.X)
        self.add_parameter(self.X)
    def _log_likelihood_gradients(self):
        return self.kernel.gradients_X(self.dL_dK, self.X, self.X2).flatten()

class Kern_check_dKdiag_dX(Kern_check_dK_dX):
    """This class allows gradient checks for the gradient of a kernel diagonal with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_dK_dX.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)
        if dL_dK==None:
            self.dL_dK = np.ones((self.X.shape[0]))
        
    def log_likelihood(self):
        return (self.dL_dK*self.kernel.Kdiag(self.X)).sum()

    def _log_likelihood_gradients(self):
        return self.kernel.dKdiag_dX(self.dL_dK, self.X).flatten()

def kern_test(kern, X=None, X2=None, output_ind=None, verbose=False):
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
    result = Kern_check_model(kern, X=X).is_positive_definite()
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
