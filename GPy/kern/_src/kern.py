# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
from ...core.parameterization.parameterized import ParametersChangedMeta, Parameterized
from ...util.caching import Cache_this

class KernCallsViaSlicerMeta(ParametersChangedMeta):
    def __call__(self, *args, **kw):
        instance = super(KernCallsViaSlicerMeta, self).__call__(*args, **kw)
        instance.K = instance._slice_wrapper(instance.K)
        instance.Kdiag = instance._slice_wrapper(instance.Kdiag, True)
        instance.update_gradients_full = instance._slice_wrapper(instance.update_gradients_full, False, True)
        instance.update_gradients_diag = instance._slice_wrapper(instance.update_gradients_diag, True, True)
        instance.gradients_X = instance._slice_wrapper(instance.gradients_X, False, True)
        instance.gradients_X_diag = instance._slice_wrapper(instance.gradients_X_diag, True, True)
        instance.psi0 = instance._slice_wrapper(instance.psi0, False, False)
        instance.psi1 = instance._slice_wrapper(instance.psi1, False, False)
        instance.psi2 = instance._slice_wrapper(instance.psi2, False, False)
        return instance

class Kern(Parameterized):
    __metaclass__ = KernCallsViaSlicerMeta
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
            self.active_dims = slice(0, input_dim)
            self.input_dim = input_dim
        else:
            self.active_dims = input_dim
            self.input_dim = len(self.active_dims)
        self._sliced_X = False
        self._sliced_X2 = False

    @Cache_this(limit=10)#, ignore_args = (0,))
    def _slice_X(self, X):
        return X[:, self.active_dims]

    def _slice_wrapper(self, operation, diag=False, derivative=False):
        """
        This method wraps the functions in kernel to make sure all kernels allways see their respective input dimension.
        The different switches are:
            diag: if X2 exists
            derivative: if firest arg is dL_dK
        """
        if derivative:
            if diag:
                def x_slice_wrapper(dL_dK, X, *args, **kw):
                    X = self._slice_X(X) if not self._sliced_X else X
                    self._sliced_X = True
                    try:
                        ret = operation(dL_dK, X, *args, **kw)
                    except:
                        raise
                    finally:
                        self._sliced_X = False
                    return ret
            else: 
                def x_slice_wrapper(dL_dK, X, X2=None, *args, **kw):
                    X, X2 = self._slice_X(X) if not self._sliced_X else X, self._slice_X(X2) if X2 is not None and not self._sliced_X2 else X2
                    self._sliced_X = True
                    self._sliced_X2 = True
                    try:
                        ret = operation(dL_dK, X, X2, *args, **kw)
                    except:
                        raise
                    finally:
                        self._sliced_X = False
                        self._sliced_X2 = False
                    return ret
        else:
            if diag:
                def x_slice_wrapper(X, *args, **kw):
                    X = self._slice_X(X) if not self._sliced_X else X
                    self._sliced_X = True
                    try:
                        ret = operation(X, *args, **kw)
                    except:
                        raise
                    finally:
                        self._sliced_X = False
                    return ret
            else: 
                def x_slice_wrapper(X, X2=None, *args, **kw):
                    X, X2 = self._slice_X(X) if not self._sliced_X else X, self._slice_X(X2) if X2 is not None and not self._sliced_X2 else X2
                    self._sliced_X = True
                    self._sliced_X2 = True
                    try:
                        ret = operation(X, X2, *args, **kw)
                    except: raise
                    finally:
                        self._sliced_X = False
                        self._sliced_X2 = False
                    return ret
        x_slice_wrapper._operation = operation
        x_slice_wrapper.__name__ = ("slicer("+operation.__name__
                                    +(","+str(bool(diag)) if diag else'')
                                    +(','+str(bool(derivative)) if derivative else '')
                                    +')')
        x_slice_wrapper.__doc__ = "**sliced**\n" + (operation.__doc__ or "")
        return x_slice_wrapper
    
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

    #def __pow__(self, other):
    #    """
    #    Shortcut for tensor `prod`.
    #    """
    #    return self.prod(other, tensor=True)

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


class CombinationKernel(Kern):
    def __init__(self, kernels, name):
        assert all([isinstance(k, Kern) for k in kernels])
        input_dim = reduce(np.union1d, (np.r_[x.active_dims] for x in kernels))
        super(CombinationKernel, self).__init__(input_dim, name)
        self.add_parameters(*kernels)

    @property
    def parts(self):
        return self._parameters_

    def update_gradients_full(self, dL_dK, X, X2=None):
        [p.update_gradients_full(dL_dK, X, X2) for p in self.parts]

    def update_gradients_diag(self, dL_dK, X):
        [p.update_gradients_diag(dL_dK, X) for p in self.parts]
