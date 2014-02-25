# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
import itertools
from ...core.parameterization import Parameterized
from ...core.parameterization.param import Param


class Kern(Parameterized):
    def __init__(self, input_dim, name, *a, **kw):
        """
        The base class for a kernel: a positive definite function
        which forms of a covariance function (kernel).

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        super(Kern, self).__init__(name=name, *a, **kw)
        self.input_dim = input_dim

    def K(self, X, X2):
        raise NotImplementedError
    def Kdiag(self, Xa):
        raise NotImplementedError
    def psi0(self,Z,variational_posterior):
        raise NotImplementedError
    def psi1(self,Z,variational_posterior):
        raise NotImplementedError
    def psi2(self,Z,variational_posterior):
        raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2):
        raise NotImplementedError
    def gradients_X_diag(self, dL_dK, X):
        raise NotImplementedError
    def update_gradients_full(self, dL_dK, X, X2):
        """Set the gradients of all parameters when doing full (N) inference."""
        raise NotImplementedError
    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        target = np.zeros(self.size)
        self.update_gradients_diag(dL_dKdiag, X)
        self._collect_gradient(target)
        self.update_gradients_full(dL_dKnm, X, Z)
        self._collect_gradient(target)
        self.update_gradients_full(dL_dKmm, Z, None)
        self._collect_gradient(target)
        self._set_gradient(target)

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        """Set the gradients of all parameters when doing variational (M) inference with uncertain inputs."""
        raise NotImplementedError
    def gradients_Z_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        grad = self.gradients_X(dL_dKmm, Z)
        grad += self.gradients_X(dL_dKnm.T, Z, X)
        return grad
    def gradients_Z_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        raise NotImplementedError
    def gradients_q_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        raise NotImplementedError
    
    def plot_ARD(self, *args, **kw):
        if "matplotlib" in sys.modules:
            from ...plotting.matplot_dep import kernel_plots
            self.plot_ARD.__doc__ += kernel_plots.plot_ARD.__doc__
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import kernel_plots
        return kernel_plots.plot_ARD(self,*args,**kw)
    
    def input_sensitivity(self):
        """
        Returns the sensitivity for each dimension of this kernel.
        """
        return np.zeros(self.input_dim)
    
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

    def __pow__(self, other):
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
