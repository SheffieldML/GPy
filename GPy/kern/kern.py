# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
import itertools
from parts.prod import Prod as prod
from parts.linear import Linear
from parts.kernpart import Kernpart
from ..core.parameterization import Parameterized
from GPy.core.parameterization.param import Param

class kern(Parameterized):
    def __init__(self, input_dim, parts=[], input_slices=None):
        """
        This is the main kernel class for GPy. It handles multiple
        (additive) kernel functions, and keeps track of various things
        like which parameters live where.

        The technical code for kernels is divided into _parts_ (see
        e.g. rbf.py). This object contains a list of parts, which are
        computed additively. For multiplication, special _prod_ parts
        are used.

        :param input_dim: The dimensionality of the kernel's input space
        :type input_dim: int
        :param parts: the 'parts' (PD functions) of the kernel
        :type parts: list of Kernpart objects
        :param input_slices: the slices on the inputs which apply to each kernel
        :type input_slices: list of slice objects, or list of bools

        """
        super(kern, self).__init__('kern')
        self.add_parameters(*parts)
        self.input_dim = input_dim

        if input_slices is None:
            self.input_slices = [slice(None) for p in self._parameters_]
        else:
            assert len(input_slices) == len(self._parameters_)
            self.input_slices = [sl if type(sl) is slice else slice(None) for sl in input_slices]

        for p in self._parameters_:
            assert isinstance(p, Kernpart), "bad kernel part"

    def parameters_changed(self):
        [p.parameters_changed() for p in self._parameters_]

    def connect_input(self, Xparam):
        [p.connect_input(Xparam) for p in self._parameters_]

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return Parameterized._getstate(self) + [#self._parameters_,
                #self.num_params,
                self.input_dim,
                self.input_slices,
                self._param_slices_
                ]

    def _setstate(self, state):
        self._param_slices_ = state.pop()
        self.input_slices = state.pop()
        self.input_dim = state.pop()
        #self.num_params = state.pop()
        #self._parameters_ = state.pop()
        Parameterized._setstate(self, state)


    def plot_ARD(self, *args):
        """If an ARD kernel is present, plot a bar representation using matplotlib

        See GPy.plotting.matplot_dep.plot_ARD
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import kernel_plots
        return kernel_plots.plot_ARD(self,*args)

#     def _transform_gradients(self, g):
#         """
#         Apply the transformations of the kernel so that the returned vector
#         represents the gradient in the transformed space (i.e. that given by
#         get_params_transformed())
#
#         :param g: the gradient vector for the current model, usually created by _param_grad_helper
#         """
#         x = self._get_params()
#         [np.place(g, index, g[index] * constraint.gradfactor(x[index]))
#          for constraint, index in self.constraints.iteritems() if constraint is not __fixed__]
# #         for constraint, index in self.constraints.iteritems():
# #             if constraint != __fixed__:
# #                 g[index] = g[index] * constraint.gradfactor(x[index])
#         #[np.put(g, i, v) for i, v in [(t[0], np.sum(g[t])) for t in self.tied_indices]]
#         [np.put(g, i, v) for i, v in [[i, t.sum()] for p in self._parameters_ for t,i in p._tied_to_me_.iteritems()]]
# #         if len(self.tied_indices) or len(self.fixed_indices):
# #             to_remove = np.hstack((self.fixed_indices + [t[1:] for t in self.tied_indices]))
# #             return np.delete(g, to_remove)
# #         else:
#         if self._fixes_ is not None: return g[self._fixes_]
#         return g
#         x = self._get_params()
#         [np.put(x, i, x * t.gradfactor(x[i])) for i, t in zip(self.constrained_indices, self.constraints)]
#         [np.put(g, i, v) for i, v in [(t[0], np.sum(g[t])) for t in self.tied_indices]]
#         if len(self.tied_indices) or len(self.fixed_indices):
#             to_remove = np.hstack((self.fixed_indices + [t[1:] for t in self.tied_indices]))
#             return np.delete(g, to_remove)
#         else:
#             return g

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
        if tensor:
            D = self.input_dim + other.input_dim
            self_input_slices = [slice(*sl.indices(self.input_dim)) for sl in self.input_slices]
            other_input_indices = [sl.indices(other.input_dim) for sl in other.input_slices]
            other_input_slices = [slice(i[0] + self.input_dim, i[1] + self.input_dim, i[2]) for i in other_input_indices]

            newkern = kern(D, self._parameters_ + other._parameters_, self_input_slices + other_input_slices)

            # transfer constraints:
#             newkern.constrained_indices = self.constrained_indices + [x + self.num_params for x in other.constrained_indices]
#             newkern.constraints = self.constraints + other.constraints
#             newkern.fixed_indices = self.fixed_indices + [self.num_params + x for x in other.fixed_indices]
#             newkern.fixed_values = self.fixed_values + other.fixed_values
#             newkern.constraints = self.constraints + other.constraints
#             newkern.tied_indices = self.tied_indices + [self.num_params + x for x in other.tied_indices]
        else:
            assert self.input_dim == other.input_dim
            newkern = kern(self.input_dim, self._parameters_ + other._parameters_, self.input_slices + other.input_slices)
            # transfer constraints:
#             newkern.constrained_indices = self.constrained_indices + [i + self.num_params  for i in other.constrained_indices]
#             newkern.constraints = self.constraints + other.constraints
#             newkern.fixed_indices = self.fixed_indices + [self.num_params + x for x in other.fixed_indices]
#             newkern.fixed_values = self.fixed_values + other.fixed_values
#             newkern.tied_indices = self.tied_indices + [self.num_params + x for x in other.tied_indices]
        [newkern._add_constrain(param, transform, warning=False)
         for param, transform in itertools.izip(
                *itertools.chain(self.constraints.iteritems(),
                                 other.constraints.iteritems()))]
        newkern._fixes_ = ((self._fixes_ or 0) + (other._fixes_ or 0)) or None

        return newkern

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
        K1 = self
        K2 = other
        #K1 = self.copy()
        #K2 = other.copy()

        slices = []
        for sl1, sl2 in itertools.product(K1.input_slices, K2.input_slices):
            s1, s2 = [False] * K1.input_dim, [False] * K2.input_dim
            s1[sl1], s2[sl2] = [True], [True]
            slices += [s1 + s2]

        newkernparts = [prod(k1, k2, tensor) for k1, k2 in itertools.product(K1._parameters_, K2._parameters_)]

        if tensor:
            newkern = kern(K1.input_dim + K2.input_dim, newkernparts, slices)
        else:
            newkern = kern(K1.input_dim, newkernparts, slices)

        #newkern._follow_constrains(K1, K2)
        return newkern

#     def _follow_constrains(self, K1, K2):
#
#         # Build the array that allows to go from the initial indices of the param to the new ones
#         K1_param = []
#         n = 0
#         for k1 in K1.parts:
#             K1_param += [range(n, n + k1.num_params)]
#             n += k1.num_params
#         n = 0
#         K2_param = []
#         for k2 in K2.parts:
#             K2_param += [range(K1.num_params + n, K1.num_params + n + k2.num_params)]
#             n += k2.num_params
#         index_param = []
#         for p1 in K1_param:
#             for p2 in K2_param:
#                 index_param += p1 + p2
#         index_param = np.array(index_param)
#
#         # Get the ties and constrains of the kernels before the multiplication
#         prev_ties = K1.tied_indices + [arr + K1.num_params for arr in K2.tied_indices]
#
#         prev_constr_ind = [K1.constrained_indices] + [K1.num_params + i for i in K2.constrained_indices]
#         prev_constr = K1.constraints + K2.constraints
#
#         # prev_constr_fix = K1.fixed_indices + [arr + K1.num_params for arr in K2.fixed_indices]
#         # prev_constr_fix_values = K1.fixed_values + K2.fixed_values
#
#         # follow the previous ties
#         for arr in prev_ties:
#             for j in arr:
#                 index_param[np.where(index_param == j)[0]] = arr[0]
#
#         # ties and constrains
#         for i in range(K1.num_params + K2.num_params):
#             index = np.where(index_param == i)[0]
#             if index.size > 1:
#                 self.tie_params(index)
#         for i, t in zip(prev_constr_ind, prev_constr):
#             self.constrain(np.where(index_param == i)[0], t)
#
#     def _get_params(self):
#         return np.hstack(self._parameters_)
#         return np.hstack([p._get_params() for p in self._parameters_])

#     def _set_params(self, x):
#         import ipdb;ipdb.set_trace()
#         [p._set_params(x[s]) for p, s in zip(self._parameters_, self._param_slices_)]

#     def _get_param_names(self):
#         # this is a bit nasty: we want to distinguish between parts with the same name by appending a count
#         part_names = np.array([k.name for k in self._parameters_], dtype=np.str)
#         counts = [np.sum(part_names == ni) for i, ni in enumerate(part_names)]
#         cum_counts = [np.sum(part_names[i:] == ni) for i, ni in enumerate(part_names)]
#         names = [name + '_' + str(cum_count) if count > 1 else name for name, count, cum_count in zip(part_names, counts, cum_counts)]
#
#         return sum([[name + '_' + n for n in k._get_param_names()] for name, k in zip(names, self._parameters_)], [])

    def K(self, X, X2=None, which_parts='all'):
        """
        Compute the kernel function.

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handles this as X2 == X.
        :param which_parts: a list of booleans detailing whether to include
                            each of the part functions. By default, 'all'
                            indicates all parts
        """
        if which_parts == 'all':
            which_parts = [True] * self.size
        assert X.shape[1] == self.input_dim
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            [p.K(X[:, i_s], None, target=target) for p, i_s, part_i_used in zip(self._parameters_, self.input_slices, which_parts) if part_i_used]
        else:
            target = np.zeros((X.shape[0], X2.shape[0]))
            [p.K(X[:, i_s], X2[:, i_s], target=target) for p, i_s, part_i_used in zip(self._parameters_, self.input_slices, which_parts) if part_i_used]
        return target

    def update_gradients_full(self, dL_dK, X):
        [p.update_gradients_full(dL_dK, X) for p in self._parameters_]

    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        [p.update_gradients_sparse(dL_dKmm, dL_dKnm, dL_dKdiag, X, Z) for p in self._parameters_]

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        [p.update_gradients_variational(dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z) for p in self._parameters_]

    def _param_grad_helper(self, dL_dK, X, X2=None):
        """
        Compute the gradient of the covariance function with respect to the parameters.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: Np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)

        returns: dL_dtheta
        """
        assert X.shape[1] == self.input_dim
        target = np.zeros(self.size)
        if X2 is None:
            [p._param_grad_helper(dL_dK, X[:, i_s], None, target[ps]) for p, i_s, ps, in zip(self._parameters_, self.input_slices, self._param_slices_)]
        else:
            [p._param_grad_helper(dL_dK, X[:, i_s], X2[:, i_s], target[ps]) for p, i_s, ps, in zip(self._parameters_, self.input_slices, self._param_slices_)]

        return self._transform_gradients(target)

    def gradients_X(self, dL_dK, X, X2=None):
        """Compute the gradient of the objective function with respect to X.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)"""

        target = np.zeros_like(X)
        if X2 is None:
            [p.gradients_X(dL_dK, X[:, i_s], None, target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        else:
            [p.gradients_X(dL_dK, X[:, i_s], X2[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def Kdiag(self, X, which_parts='all'):
        """Compute the diagonal of the covariance function for inputs X."""
        if which_parts == 'all':
            which_parts = [True] * self.size
        assert X.shape[1] == self.input_dim
        target = np.zeros(X.shape[0])
        [p.Kdiag(X[:, i_s], target=target) for p, i_s, part_on in zip(self._parameters_, self.input_slices, which_parts) if part_on]
        return target

    def dKdiag_dtheta(self, dL_dKdiag, X):
        """Compute the gradient of the diagonal of the covariance function with respect to the parameters."""
        assert X.shape[1] == self.input_dim
        assert dL_dKdiag.size == X.shape[0]
        target = np.zeros(self.size)
        [p.dKdiag_dtheta(dL_dKdiag, X[:, i_s], target[ps]) for p, i_s, ps in zip(self._parameters_, self.input_slices, self._param_slices_)]
        return self._transform_gradients(target)

    def dKdiag_dX(self, dL_dKdiag, X):
        assert X.shape[1] == self.input_dim
        target = np.zeros_like(X)
        [p.dKdiag_dX(dL_dKdiag, X[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def psi0(self, Z, mu, S):
        target = np.zeros(mu.shape[0])
        [p.psi0(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S):
        target = np.zeros(self.size)
        [p.dpsi0_dtheta(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self._parameters_, self._param_slices_, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S):
        target_mu, target_S = np.zeros_like(mu), np.zeros_like(S)
        [p.dpsi0_dmuS(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target_mu, target_S

    def psi1(self, Z, mu, S):
        target = np.zeros((mu.shape[0], Z.shape[0]))
        [p.psi1(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S):
        target = np.zeros((self.size))
        [p.dpsi1_dtheta(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self._parameters_, self._param_slices_, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi1_dZ(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S):
        """return shapes are num_samples,num_inducing,input_dim"""
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi1_dmuS(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target_mu, target_S

    def psi2(self, Z, mu, S):
        """
        Computer the psi2 statistics for the covariance function.

        :param Z: np.ndarray of inducing inputs (num_inducing x input_dim)
        :param mu, S: np.ndarrays of means and variances (each num_samples x input_dim)
        :returns psi2: np.ndarray (num_samples,num_inducing,num_inducing)

        """
        target = np.zeros((mu.shape[0], Z.shape[0], Z.shape[0]))
        [p.psi2(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]

        # compute the "cross" terms
        # TODO: input_slices needed
        crossterms = 0

        for [p1, i_s1], [p2, i_s2] in itertools.combinations(zip(self._parameters_, self.input_slices), 2):
            if i_s1 == i_s2:
                # TODO psi1 this must be faster/better/precached/more nice
                tmp1 = np.zeros((mu.shape[0], Z.shape[0]))
                p1.psi1(Z[:, i_s1], mu[:, i_s1], S[:, i_s1], tmp1)
                tmp2 = np.zeros((mu.shape[0], Z.shape[0]))
                p2.psi1(Z[:, i_s2], mu[:, i_s2], S[:, i_s2], tmp2)

                prod = np.multiply(tmp1, tmp2)
                crossterms += prod[:, :, None] + prod[:, None, :]

        target += crossterms
        return target

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S):
        """Gradient of the psi2 statistics with respect to the parameters."""
        target = np.zeros(self.size)
        [p.dpsi2_dtheta(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, i_s, ps in zip(self._parameters_, self.input_slices, self._param_slices_)]

        # compute the "cross" terms
        # TODO: better looping, input_slices
        for i1, i2 in itertools.permutations(range(len(self._parameters_)), 2):
            p1, p2 = self._parameters_[i1], self._parameters_[i2]
#             ipsl1, ipsl2 = self.input_slices[i1], self.input_slices[i2]
            ps1, ps2 = self._param_slices_[i1], self._param_slices_[i2]

            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dtheta((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target[ps2])

        return self._transform_gradients(target)

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi2_dZ(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        # target *= 2

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self._parameters_, 2):
#             if p1.name == 'linear' and p2.name == 'linear':
#                 raise NotImplementedError("We don't handle linear/linear cross-terms")
            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dZ((tmp[:, None, :] * dL_dpsi2).sum(1), Z, mu, S, target)

        return target * 2

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S):
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi2_dmuS(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self._parameters_, 2):
#             if p1.name == 'linear' and p2.name == 'linear':
#                 raise NotImplementedError("We don't handle linear/linear cross-terms")
            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dmuS((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target_mu, target_S)

        return target_mu, target_S

    def plot(self, *args, **kwargs):
        """
        See GPy.plotting.matplot_dep.plot
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import kernel_plots
        kernel_plots.plot(self,*args)

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
