# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
from ...core.parameterization.parameterized import Parameterized
from .kernel_slice_operations import KernCallsViaSlicerMeta
from ...util.caching import Cache_this
from GPy.core.parameterization.observable_array import ObsAr
from functools import reduce



class Kern(Parameterized):
    #===========================================================================
    # This adds input slice support. The rather ugly code for slicing can be
    # found in kernel_slice_operations
    __metaclass__ = KernCallsViaSlicerMeta
    #===========================================================================
    _support_GPU=False
    def __init__(self, input_dim, active_dims, name, useGPU=False, *a, **kw):
        """
        The base class for a kernel: a positive definite function
        which forms of a covariance function (kernel).

        input_dim:

            is the number of dimensions to work on. Make sure to give the
            tight dimensionality of inputs.
            You most likely want this to be the integer telling the number of
            input dimensions of the kernel.
            If this is not an integer (!) we will work on the whole input matrix X,
            and not check whether dimensions match or not (!).

        active_dims:

            is the active_dimensions of inputs X we will work on.
            All kernels will get sliced Xes as inputs, if active_dims is not None
            Only positive integers are allowed in active_dims!
            if active_dims is None, slicing is switched off and all X will be passed through as given.

        :param int input_dim: the number of input dimensions to the function
        :param array-like|None active_dims: list of indices on which dimensions this kernel works on, or none if no slicing

        Do not instantiate.
        """
        super(Kern, self).__init__(name=name, *a, **kw)
        self.input_dim = int(input_dim)

        if active_dims is None:
            active_dims = np.arange(input_dim)

        self.active_dims = np.atleast_1d(active_dims).astype(int)

        assert self.active_dims.size == self.input_dim, "input_dim={} does not match len(active_dim)={}, active_dims={}".format(self.input_dim, self.active_dims.size, self.active_dims)

        self._sliced_X = 0
        self.useGPU = self._support_GPU and useGPU
        self._return_psi2_n_flag = ObsAr(np.zeros(1)).astype(bool)

    @property
    def return_psi2_n(self):
        """
        Flag whether to pass back psi2 as NxMxM or MxM, by summing out N.
        """
        return self._return_psi2_n_flag[0]
    @return_psi2_n.setter
    def return_psi2_n(self, val):
        def visit(self):
            if isinstance(self, Kern):
                self._return_psi2_n_flag[0]=val
        self.traverse(visit)

    @Cache_this(limit=20)
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

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
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

    def plot(self, x=None, fignum=None, ax=None, title=None, plot_limits=None, resolution=None, **mpl_kwargs):
        """
        plot this kernel.
        :param x: the value to use for the other kernel argument (kernels are a function of two variables!)
        :param fignum: figure number of the plot
        :param ax: matplotlib axis to plot on
        :param title: the matplotlib title
        :param plot_limits: the range over which to plot the kernel
        :resolution: the resolution of the lines used in plotting
        :mpl_kwargs avalid keyword arguments to pass through to matplotlib (e.g. lw=7)
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import kernel_plots
        kernel_plots.plot(self, x, fignum, ax, title, plot_limits, resolution, **mpl_kwargs)

    def plot_ARD(self, *args, **kw):
        """
        See :class:`~GPy.plotting.matplot_dep.kernel_plots`
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import kernel_plots
        return kernel_plots.plot_ARD(self,*args,**kw)

    def input_sensitivity(self, summarize=True):
        """
        Returns the sensitivity for each dimension of this kernel.
        """
        return np.zeros(self.input_dim)

    def __add__(self, other):
        """ Overloading of the '+' operator. for more control, see self.add """
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)

    def add(self, other, name='add'):
        """
        Add another kernel to this one.

        :param other: the other kernel to be added
        :type other: GPy.kern

        """
        assert isinstance(other, Kern), "only kernels can be added to kernels..."
        from .add import Add
        return Add([self, other], name=name)

    def __mul__(self, other):
        """ Here we overload the '*' operator. See self.prod for more information"""
        return self.prod(other)

    def __imul__(self, other):
        """ Here we overload the '*' operator. See self.prod for more information"""
        return self.prod(other)

    def __pow__(self, other):
        """
        Shortcut for tensor `prod`.
        """
        assert np.all(self.active_dims == range(self.input_dim)), "Can only use kernels, which have their input_dims defined from 0"
        assert np.all(other.active_dims == range(other.input_dim)), "Can only use kernels, which have their input_dims defined from 0"
        other.active_dims += self.input_dim
        return self.prod(other)

    def prod(self, other, name='mul'):
        """
        Multiply two kernels (either on the same space, or on the tensor
        product of the input space).

        :param other: the other kernel to be added
        :type other: GPy.kern
        :param tensor: whether or not to use the tensor space (default is false).
        :type tensor: bool

        """
        assert isinstance(other, Kern), "only kernels can be multiplied to kernels..."
        from .prod import Prod
        #kernels = []
        #if isinstance(self, Prod): kernels.extend(self.parameters)
        #else: kernels.append(self)
        #if isinstance(other, Prod): kernels.extend(other.parameters)
        #else: kernels.append(other)
        return Prod([self, other], name)

    def _check_input_dim(self, X):
        assert X.shape[1] == self.input_dim, "{} did not specify active_dims and X has wrong shape: X_dim={}, whereas input_dim={}".format(self.name, X.shape[1], self.input_dim)

    def _check_active_dims(self, X):
        assert X.shape[1] >= len(self.active_dims), "At least {} dimensional X needed, X.shape={!s}".format(len(self.active_dims), X.shape)


class CombinationKernel(Kern):
    """
    Abstract super class for combination kernels.
    A combination kernel combines (a list of) kernels and works on those.
    Examples are the HierarchicalKernel or Add and Prod kernels.
    """
    def __init__(self, kernels, name, extra_dims=[]):
        """
        Abstract super class for combination kernels.
        A combination kernel combines (a list of) kernels and works on those.
        Examples are the HierarchicalKernel or Add and Prod kernels.

        :param list kernels: List of kernels to combine (can be only one element)
        :param str name: name of the combination kernel
        :param array-like extra_dims: if needed extra dimensions for the combination kernel to work on
        """
        assert all([isinstance(k, Kern) for k in kernels])
        extra_dims = np.array(extra_dims, dtype=int)
        input_dim, active_dims = self.get_input_dim_active_dims(kernels, extra_dims)
        # initialize the kernel with the full input_dim
        super(CombinationKernel, self).__init__(input_dim, active_dims, name)
        self.extra_dims = extra_dims
        self.link_parameters(*kernels)

    @property
    def parts(self):
        return self.parameters

    def get_input_dim_active_dims(self, kernels, extra_dims = None):
        #active_dims = reduce(np.union1d, (np.r_[x.active_dims] for x in kernels), np.array([], dtype=int))
        #active_dims = np.array(np.concatenate((active_dims, extra_dims if extra_dims is not None else [])), dtype=int)
        input_dim = reduce(max, (k.active_dims.max() for k in kernels)) + 1

        if extra_dims is not None:
            input_dim += extra_dims.size

        active_dims = np.arange(input_dim)
        return input_dim, active_dims

    def input_sensitivity(self, summarize=True):
        """
        If summize is true, we want to get the summerized view of the sensitivities,
        otherwise put everything into an array with shape (#kernels, input_dim)
        in the order of appearance of the kernels in the parameterized object.
        """
        raise NotImplementedError("Choose the kernel you want to get the sensitivity for. You need to override the default behaviour for getting the input sensitivity to be able to get the input sensitivity. For sum kernel it is the sum of all sensitivities, TODO: product kernel? Other kernels?, also TODO: shall we return all the sensitivities here in the combination kernel? So we can combine them however we want? This could lead to just plot all the sensitivities here...")

    def _check_active_dims(self, X):
        return

    def _check_input_dim(self, X):
        # As combination kernels cannot always know, what their inner kernels have as input dims, the check will be done inside them, respectively
        return
