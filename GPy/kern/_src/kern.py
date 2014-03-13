# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
from ...core.parameterization.parameterized import Parameterized
from kernel_slice_operations import KernCallsViaSlicerMeta
from ...util.caching import Cache_this



class Kern(Parameterized):
    #===========================================================================
    # This adds input slice support. The rather ugly code for slicing can be 
    # found in kernel_slice_operations
    __metaclass__ = KernCallsViaSlicerMeta
    #===========================================================================
    _debug=False
    def __init__(self, input_dim, name, *a, **kw):
        """
        The base class for a kernel: a positive definite function
        which forms of a covariance function (kernel).

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        super(Kern, self).__init__(name=name, *a, **kw)
        if isinstance(input_dim, int):
            self.active_dims = np.r_[0:input_dim]
            self.input_dim = input_dim
        else:
            self.active_dims = np.r_[input_dim]
            self.input_dim = len(self.active_dims)
        self._sliced_X = 0

    @Cache_this(limit=10)#, ignore_args = (0,))
    def _slice_X(self, X):
        return X[:, self.active_dims]

    def K(self, X, X2):
        """
        Compute the kernel function.

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handLes this as X2 == X.
        """
        raise NotImplementedError
    def Kdiag(self, X):
        raise NotImplementedError
    def psi0(self, Z, variational_posterior):
        raise NotImplementedError
    def psi1(self, Z, variational_posterior):
        raise NotImplementedError
    def psi2(self, Z, variational_posterior):
        raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2):
        raise NotImplementedError
    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def update_gradients_diag(self, dL_dKdiag, X):
        """ update the gradients of all parameters when using only the diagonal elements of the covariance matrix"""
        raise NotImplementedError

    def update_gradients_full(self, dL_dK, X, X2):
        """Set the gradients of all parameters when doing full (N) inference."""
        raise NotImplementedError
    def update_gradients_diag(self, dL_dKdiag, X):
        """Set the gradients for all parameters for the derivative of the diagonal of the covariance w.r.t the kernel parameters."""
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
        return np.zeros(self.input_dim)

    def __add__(self, other):
        """ Overloading of the '+' operator. for more control, see self.add """
        return self.add(other)

    def add(self, other, name='add'):
        """
        Add another kernel to this one.

        :param other: the other kernel to be added
        :type other: GPy.kern

        """
        assert isinstance(other, Kern), "only kernels can be added to kernels..."
        from add import Add
        kernels = []
        if isinstance(self, Add): kernels.extend(self._parameters_)
        else: kernels.append(self)
        if isinstance(other, Add): kernels.extend(other._parameters_)
        else: kernels.append(other)
        return Add(kernels, name=name)

    def __mul__(self, other):
        """ Here we overload the '*' operator. See self.prod for more information"""
        return self.prod(other)

    def __pow__(self, other):
        """
        Shortcut for tensor `prod`.
        """
        assert self.active_dims == range(self.input_dim), "Can only use kernels, which have their input_dims defined from 0"
        assert other.active_dims == range(other.input_dim), "Can only use kernels, which have their input_dims defined from 0"
        other.active_dims += self.input_dim
        return self.prod(other)

    def prod(self, other, name=None):
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
        kernels = []
        if isinstance(self, Prod): kernels.extend(self._parameters_)
        else: kernels.append(self)
        if isinstance(other, Prod): kernels.extend(other._parameters_)
        else: kernels.append(other)
        return Prod(self, other, name)

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return super(Kern, self)._getstate() + [
                self.active_dims,
                self.input_dim,
                self._sliced_X]

    def _setstate(self, state):
        self._sliced_X = state.pop()
        self.input_dim = state.pop()
        self.active_dims = state.pop()
        super(Kern, self)._setstate(state)

class CombinationKernel(Kern):
    def __init__(self, kernels, name):
        assert all([isinstance(k, Kern) for k in kernels])
        ma = reduce(lambda a,b: max(a, max(b)), (x.active_dims for x in kernels), 0)
        input_dim = np.r_[0:ma+1]
        super(CombinationKernel, self).__init__(input_dim, name)
        self.add_parameters(*kernels)

    @property
    def parts(self):
        return self._parameters_

    def input_sensitivity(self):
        in_sen = np.zeros((self.num_params, self.input_dim))
        for i, p in enumerate(self.parts):
            in_sen[i, p.active_dims] = p.input_sensitivity()
        return in_sen
