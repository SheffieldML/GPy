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
    def psi0(self, Z, variational_posterior):
        raise NotImplementedError
    def psi1(self, Z, variational_posterior):
        raise NotImplementedError
    def psi2(self, Z, variational_posterior):
        raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2):
        raise NotImplementedError
    def gradients_X_diag(self, dL_dK, X):
        raise NotImplementedError

    def update_gradients_full(self, dL_dK, X, X2):
        """Set the gradients of all parameters when doing full (N) inference."""
        raise NotImplementedError

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        """
        Set the gradients of all parameters when doing inference with
        uncertain inputs, using expectations of the kernel.

        The esential maths is

        dL_d{theta_i} = dL_dpsi0 * dpsi0_d{theta_i} +
                        dL_dpsi1 * dpsi1_d{theta_i} +
                        dL_dpsi2 * dpsi2_d{theta_i}
        """
        raise NotImplementedError

    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        """
        Returns the derivative of the objective wrt Z, using the chain rule
        through the expectation variables.
        """
        raise NotImplementedError

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        """
        Compute the gradients wrt the parameters of the variational
        distruibution q(X), chain-ruling via the expectations of the kernel
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        """
        See GPy.plotting.matplot_dep.plot
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import kernel_plots
        kernel_plots.plot(self,*args)

    def plot_ARD(self, *args, **kw):
        """
        See :class:`~GPy.plotting.matplot_dep.kernel_plots`
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import kernel_plots
        return kernel_plots.plot_ARD(self,*args,**kw)

    def input_sensitivity(self):
        """
        Returns the sensitivity for each dimension of this kernel.
        """
        return self.kern.input_sensitivity()

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
        kernels = []
        if not tensor and isinstance(self, Add): kernels.extend(self._parameters_)
        else: kernels.append(self)
        if not tensor and isinstance(other, Add): kernels.extend(other._parameters_)
        else: kernels.append(other)
        return Add(kernels, tensor)

    def __mul__(self, other):
        """ Here we overload the '*' operator. See self.prod for more information"""
        return self.prod(other)

    def __pow__(self, other):
        """
        Shortcut for tensor `prod`.
        """
        return self.prod(other, tensor=True)

    def prod(self, other, tensor=False, name=None):
        """
        Multiply two kernels (either on the same space, or on the tensor
        product of the input space).

        :param other: the other kernel to be added
        :type other: GPy.kern
        :param tensor: whether or not to use the tensor space (default is false).
        :type tensor: bool

        """
        assert isinstance(other, Kern), "only kernels can be added to kernels..."
        from prod import Prod
        return Prod(self, other, tensor, name)
