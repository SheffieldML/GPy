# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from ..core.parameterised import Parameterised
from kernpart import Kernpart
import itertools
from prod import prod

class kern(Parameterised):
    def __init__(self, input_dim, parts=[], input_slices=None):
        """
        This is the main kernel class for GPy. It handles multiple (additive) kernel functions, and keeps track of variaous things like which parameters live where.

        The technical code for kernels is divided into _parts_ (see e.g. rbf.py). This obnject contains a list of parts, which are computed additively. For multiplication, special _prod_ parts are used.

        :param input_dim: The dimensionality of the kernel's input space
        :type input_dim: int
        :param parts: the 'parts' (PD functions) of the kernel
        :type parts: list of Kernpart objects
        :param input_slices: the slices on the inputs which apply to each kernel
        :type input_slices: list of slice objects, or list of bools

        """
        self.parts = parts
        self.Nparts = len(parts)
        self.num_params = sum([p.num_params for p in self.parts])

        self.input_dim = input_dim

        # deal with input_slices
        if input_slices is None:
            self.input_slices = [slice(None) for p in self.parts]
        else:
            assert len(input_slices) == len(self.parts)
            self.input_slices = [sl if type(sl) is slice else slice(None) for sl in input_slices]

        for p in self.parts:
            assert isinstance(p, Kernpart), "bad kernel part"

        self.compute_param_slices()

        Parameterised.__init__(self)


    def plot_ARD(self, fignum=None, ax=None):
        """If an ARD kernel is present, it bar-plots the ARD parameters"""
        if ax is None:
            fig = pb.figure(fignum)
            ax = fig.add_subplot(111)
        for p in self.parts:
            if hasattr(p, 'ARD') and p.ARD:
                ax.set_title('ARD parameters, %s kernel' % p.name)

                if p.name == 'linear':
                    ard_params = p.variances
                else:
                    ard_params = 1. / p.lengthscale

                ax.bar(np.arange(len(ard_params)) - 0.4, ard_params)
                ax.set_xticks(np.arange(len(ard_params)))
                ax.set_xticklabels([r"${}$".format(i) for i in range(len(ard_params))])
        return ax

    def _transform_gradients(self, g):
        x = self._get_params()
        [np.put(x, i, x * t.gradfactor(x[i])) for i, t in zip(self.constrained_indices, self.constraints)]
        [np.put(g, i, v) for i, v in [(t[0], np.sum(g[t])) for t in self.tied_indices]]
        if len(self.tied_indices) or len(self.fixed_indices):
            to_remove = np.hstack((self.fixed_indices + [t[1:] for t in self.tied_indices]))
            return np.delete(g, to_remove)
        else:
            return g

    def compute_param_slices(self):
        """create a set of slices that can index the parameters of each part"""
        self.param_slices = []
        count = 0
        for p in self.parts:
            self.param_slices.append(slice(count, count + p.num_params))
            count += p.num_params

    def __add__(self, other):
        """
        Shortcut for `add`.
        """
        return self.add(other)

    def add(self, other, tensor=False):
        """
        Add another kernel to this one. Both kernels are defined on the same _space_
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        if tensor:
            D = self.input_dim + other.input_dim
            self_input_slices = [slice(*sl.indices(self.input_dim)) for sl in self.input_slices]
            other_input_indices = [sl.indices(other.input_dim) for sl in other.input_slices]
            other_input_slices = [slice(i[0] + self.input_dim, i[1] + self.input_dim, i[2]) for i in other_input_indices]

            newkern = kern(D, self.parts + other.parts, self_input_slices + other_input_slices)

            # transfer constraints:
            newkern.constrained_indices = self.constrained_indices + [x + self.num_params for x in other.constrained_indices]
            newkern.constraints = self.constraints + other.constraints
            newkern.fixed_indices = self.fixed_indices + [self.num_params + x for x in other.fixed_indices]
            newkern.fixed_values = self.fixed_values + other.fixed_values
            newkern.constraints = self.constraints + other.constraints
            newkern.tied_indices = self.tied_indices + [self.num_params + x for x in other.tied_indices]
        else:
            assert self.input_dim == other.input_dim
            newkern = kern(self.input_dim, self.parts + other.parts, self.input_slices + other.input_slices)
            # transfer constraints:
            newkern.constrained_indices = self.constrained_indices + [i + self.num_params  for i in other.constrained_indices]
            newkern.constraints = self.constraints + other.constraints
            newkern.fixed_indices = self.fixed_indices + [self.num_params + x for x in other.fixed_indices]
            newkern.fixed_values = self.fixed_values + other.fixed_values
            newkern.tied_indices = self.tied_indices + [self.num_params + x for x in other.tied_indices]
        return newkern

    def __mul__(self, other):
        """
        Shortcut for `prod`.
        """
        return self.prod(other)

    def prod(self, other, tensor=False):
        """
        multiply two kernels (either on the same space, or on the tensor product of the input space)
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        K1 = self.copy()
        K2 = other.copy()

        slices = []
        for sl1, sl2 in itertools.product(K1.input_slices, K2.input_slices):
            s1, s2 = [False] * K1.input_dim, [False] * K2.input_dim
            s1[sl1], s2[sl2] = [True], [True]
            slices += [s1 + s2]

        newkernparts = [prod(k1, k2, tensor) for k1, k2 in itertools.product(K1.parts, K2.parts)]

        if tensor:
            newkern = kern(K1.input_dim + K2.input_dim, newkernparts, slices)
        else:
            newkern = kern(K1.input_dim, newkernparts, slices)

        newkern._follow_constrains(K1, K2)
        return newkern

    def _follow_constrains(self, K1, K2):

        # Build the array that allows to go from the initial indices of the param to the new ones
        K1_param = []
        n = 0
        for k1 in K1.parts:
            K1_param += [range(n, n + k1.num_params)]
            n += k1.num_params
        n = 0
        K2_param = []
        for k2 in K2.parts:
            K2_param += [range(K1.num_params + n, K1.num_params + n + k2.num_params)]
            n += k2.num_params
        index_param = []
        for p1 in K1_param:
            for p2 in K2_param:
                index_param += p1 + p2
        index_param = np.array(index_param)

        # Get the ties and constrains of the kernels before the multiplication
        prev_ties = K1.tied_indices + [arr + K1.num_params for arr in K2.tied_indices]

        prev_constr_ind = [K1.constrained_indices] + [K1.num_params + i for i in K2.constrained_indices]
        prev_constr = K1.constraints + K2.constraints

        # prev_constr_fix = K1.fixed_indices + [arr + K1.num_params for arr in K2.fixed_indices]
        # prev_constr_fix_values = K1.fixed_values + K2.fixed_values

        # follow the previous ties
        for arr in prev_ties:
            for j in arr:
                index_param[np.where(index_param == j)[0]] = arr[0]

        # ties and constrains
        for i in range(K1.num_params + K2.num_params):
            index = np.where(index_param == i)[0]
            if index.size > 1:
                self.tie_params(index)
        for i, t in zip(prev_constr_ind, prev_constr):
            self.constrain(np.where(index_param == i)[0], t)

    def _get_params(self):
        return np.hstack([p._get_params() for p in self.parts])

    def _set_params(self, x):
        [p._set_params(x[s]) for p, s in zip(self.parts, self.param_slices)]

    def _get_param_names(self):
        # this is a bit nasty: we wat to distinguish between parts with the same name by appending a count
        part_names = np.array([k.name for k in self.parts], dtype=np.str)
        counts = [np.sum(part_names == ni) for i, ni in enumerate(part_names)]
        cum_counts = [np.sum(part_names[i:] == ni) for i, ni in enumerate(part_names)]
        names = [name + '_' + str(cum_count) if count > 1 else name for name, count, cum_count in zip(part_names, counts, cum_counts)]

        return sum([[name + '_' + n for n in k._get_param_names()] for name, k in zip(names, self.parts)], [])

    def K(self, X, X2=None, which_parts='all'):
        if which_parts == 'all':
            which_parts = [True] * self.Nparts
        assert X.shape[1] == self.input_dim
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            [p.K(X[:, i_s], None, target=target) for p, i_s, part_i_used in zip(self.parts, self.input_slices, which_parts) if part_i_used]
        else:
            target = np.zeros((X.shape[0], X2.shape[0]))
            [p.K(X[:, i_s], X2[:, i_s], target=target) for p, i_s, part_i_used in zip(self.parts, self.input_slices, which_parts) if part_i_used]
        return target

    def dK_dtheta(self, dL_dK, X, X2=None):
        """
        :param dL_dK: An array of dL_dK derivaties, dL_dK
        :type dL_dK: Np.ndarray (N x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (N x input_dim)
        :param X2: Observed dara inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)
        """
        assert X.shape[1] == self.input_dim
        target = np.zeros(self.num_params)
        if X2 is None:
            [p.dK_dtheta(dL_dK, X[:, i_s], None, target[ps]) for p, i_s, ps, in zip(self.parts, self.input_slices, self.param_slices)]
        else:
            [p.dK_dtheta(dL_dK, X[:, i_s], X2[:, i_s], target[ps]) for p, i_s, ps, in zip(self.parts, self.input_slices, self.param_slices)]

        return self._transform_gradients(target)

    def dK_dX(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        target = np.zeros_like(X)
        if X2 is None:
            [p.dK_dX(dL_dK, X[:, i_s], None, target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        else:
            [p.dK_dX(dL_dK, X[:, i_s], X2[:, i_s], target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def Kdiag(self, X, which_parts='all'):
        if which_parts == 'all':
            which_parts = [True] * self.Nparts
        assert X.shape[1] == self.input_dim
        target = np.zeros(X.shape[0])
        [p.Kdiag(X[:, i_s], target=target) for p, i_s, part_on in zip(self.parts, self.input_slices, which_parts) if part_on]
        return target

    def dKdiag_dtheta(self, dL_dKdiag, X):
        assert X.shape[1] == self.input_dim
        assert dL_dKdiag.size == X.shape[0]
        target = np.zeros(self.num_params)
        [p.dKdiag_dtheta(dL_dKdiag, X[:, i_s], target[ps]) for p, i_s, ps in zip(self.parts, self.input_slices, self.param_slices)]
        return self._transform_gradients(target)

    def dKdiag_dX(self, dL_dKdiag, X):
        assert X.shape[1] == self.input_dim
        target = np.zeros_like(X)
        [p.dKdiag_dX(dL_dKdiag, X[:, i_s], target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def psi0(self, Z, mu, S):
        target = np.zeros(mu.shape[0])
        [p.psi0(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S):
        target = np.zeros(self.num_params)
        [p.dpsi0_dtheta(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self.parts, self.param_slices, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S):
        target_mu, target_S = np.zeros_like(mu), np.zeros_like(S)
        [p.dpsi0_dmuS(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target_mu, target_S

    def psi1(self, Z, mu, S):
        target = np.zeros((mu.shape[0], Z.shape[0]))
        [p.psi1(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S):
        target = np.zeros((self.num_params))
        [p.dpsi1_dtheta(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self.parts, self.param_slices, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi1_dZ(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S):
        """return shapes are N,num_inducing,input_dim"""
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi1_dmuS(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target_mu, target_S

    def psi2(self, Z, mu, S):
        """
        :param Z: np.ndarray of inducing inputs (num_inducing x input_dim)
        :param mu, S: np.ndarrays of means and variances (each N x input_dim)
        :returns psi2: np.ndarray (N,num_inducing,num_inducing)
        """
        target = np.zeros((mu.shape[0], Z.shape[0], Z.shape[0]))
        [p.psi2(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self.parts, self.input_slices)]

        # compute the "cross" terms
        # TODO: input_slices needed
        crossterms = 0

        for p1, p2 in itertools.combinations(self.parts, 2):

            # TODO psi1 this must be faster/better/precached/more nice
            tmp1 = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp1)
            tmp2 = np.zeros((mu.shape[0], Z.shape[0]))
            p2.psi1(Z, mu, S, tmp2)

            prod = np.multiply(tmp1, tmp2)
            crossterms += prod[:, :, None] + prod[:, None, :]

        target += crossterms
        return target

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S):
        target = np.zeros(self.num_params)
        [p.dpsi2_dtheta(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, i_s, ps in zip(self.parts, self.input_slices, self.param_slices)]

        # compute the "cross" terms
        # TODO: better looping, input_slices
        for i1, i2 in itertools.permutations(range(len(self.parts)), 2):
            p1, p2 = self.parts[i1], self.parts[i2]
#             ipsl1, ipsl2 = self.input_slices[i1], self.input_slices[i2]
            ps1, ps2 = self.param_slices[i1], self.param_slices[i2]

            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dtheta((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target[ps2])

        return self._transform_gradients(target)

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi2_dZ(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        # target *= 2

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self.parts, 2):
            if p1.name == 'linear' and p2.name == 'linear':
                raise NotImplementedError("We don't handle linear/linear cross-terms")
            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dZ((tmp[:, None, :] * dL_dpsi2).sum(1), Z, mu, S, target)

        return target * 2

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S):
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi2_dmuS(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self.parts, 2):
            if p1.name == 'linear' and p2.name == 'linear':
                raise NotImplementedError("We don't handle linear/linear cross-terms")

            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dmuS((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target_mu, target_S)

        return target_mu, target_S

    def plot(self, x=None, plot_limits=None, which_parts='all', resolution=None, *args, **kwargs):
        if which_parts == 'all':
            which_parts = [True] * self.Nparts
        if self.input_dim == 1:
            if x is None:
                x = np.zeros((1, 1))
            else:
                x = np.asarray(x)
                assert x.size == 1, "The size of the fixed variable x is not 1"
                x = x.reshape((1, 1))

            if plot_limits == None:
                xmin, xmax = (x - 5).flatten(), (x + 5).flatten()
            elif len(plot_limits) == 2:
                xmin, xmax = plot_limits
            else:
                raise ValueError, "Bad limits for plotting"

            Xnew = np.linspace(xmin, xmax, resolution or 201)[:, None]
            Kx = self.K(Xnew, x, which_parts)
            pb.plot(Xnew, Kx, *args, **kwargs)
            pb.xlim(xmin, xmax)
            pb.xlabel("x")
            pb.ylabel("k(x,%0.1f)" % x)

        elif self.input_dim == 2:
            if x is None:
                x = np.zeros((1, 2))
            else:
                x = np.asarray(x)
                assert x.size == 2, "The size of the fixed variable x is not 2"
                x = x.reshape((1, 2))

            if plot_limits == None:
                xmin, xmax = (x - 5).flatten(), (x + 5).flatten()
            elif len(plot_limits) == 2:
                xmin, xmax = plot_limits
            else:
                raise ValueError, "Bad limits for plotting"

            resolution = resolution or 51
            xx, yy = np.mgrid[xmin[0]:xmax[0]:1j * resolution, xmin[1]:xmax[1]:1j * resolution]
            xg = np.linspace(xmin[0], xmax[0], resolution)
            yg = np.linspace(xmin[1], xmax[1], resolution)
            Xnew = np.vstack((xx.flatten(), yy.flatten())).T
            Kx = self.K(Xnew, x, which_parts)
            Kx = Kx.reshape(resolution, resolution).T
            pb.contour(xg, yg, Kx, vmin=Kx.min(), vmax=Kx.max(), cmap=pb.cm.jet, *args, **kwargs) # @UndefinedVariable
            pb.xlim(xmin[0], xmax[0])
            pb.ylim(xmin[1], xmax[1])
            pb.xlabel("x1")
            pb.ylabel("x2")
            pb.title("k(x1,x2 ; %0.1f,%0.1f)" % (x[0, 0], x[0, 1]))
        else:
            raise NotImplementedError, "Cannot plot a kernel with more than two input dimensions"
