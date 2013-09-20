# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..core.parameterized import Parameterized
from parts.kernpart import Kernpart
import itertools
from parts.prod import Prod as prod
from matplotlib.transforms import offset_copy

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
        self.parts = parts
        self.Nparts = len(parts)
        self.num_params = sum([p.num_params for p in self.parts])

        self.input_dim = input_dim

        part_names = [k.name for k in self.parts]
        self.name=''
        for name in part_names:
            self.name += name + '+'
        self.name = self.name[:-1]
        # deal with input_slices
        if input_slices is None:
            self.input_slices = [slice(None) for p in self.parts]
        else:
            assert len(input_slices) == len(self.parts)
            self.input_slices = [sl if type(sl) is slice else slice(None) for sl in input_slices]

        for p in self.parts:
            assert isinstance(p, Kernpart), "bad kernel part"

        self.compute_param_slices()

        Parameterized.__init__(self)

    def getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return Parameterized.getstate(self) + [self.parts,
                self.Nparts,
                self.num_params,
                self.input_dim,
                self.input_slices,
                self.param_slices
                ]

    def setstate(self, state):
        self.param_slices = state.pop()
        self.input_slices = state.pop()
        self.input_dim = state.pop()
        self.num_params = state.pop()
        self.Nparts = state.pop()
        self.parts = state.pop()
        Parameterized.setstate(self, state)


    def plot_ARD(self, fignum=None, ax=None, title='', legend=False):
        """If an ARD kernel is present, it bar-plots the ARD parameters.

        :param fignum: figure number of the plot
        :param ax: matplotlib axis to plot on
        :param title: 
            title of the plot, 
            pass '' to not print a title
            pass None for a generic title

        """
        if ax is None:
            fig = pb.figure(fignum)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        from GPy.util import Tango
        from matplotlib.textpath import TextPath
        Tango.reset()
        xticklabels = []
        bars = []
        x0 = 0
        for p in self.parts:
            c = Tango.nextMedium()
            if hasattr(p, 'ARD') and p.ARD:
                if title is None:
                    ax.set_title('ARD parameters, %s kernel' % p.name)
                else:
                    ax.set_title(title)
                if p.name == 'linear':
                    ard_params = p.variances
                else:
                    ard_params = 1. / p.lengthscale

                x = np.arange(x0, x0 + len(ard_params))
                bars.append(ax.bar(x, ard_params, align='center', color=c, edgecolor='k', linewidth=1.2, label=p.name))
                xticklabels.extend([r"$\mathrm{{{name}}}\ {x}$".format(name=p.name, x=i) for i in np.arange(len(ard_params))])
                x0 += len(ard_params)
        x = np.arange(x0)
        transOffset = offset_copy(ax.transData, fig=fig,
                                  x=0., y= -2., units='points')
        transOffsetUp = offset_copy(ax.transData, fig=fig,
                                  x=0., y=1., units='points')
        for bar in bars:
            for patch, num in zip(bar.patches, np.arange(len(bar.patches))):
                height = patch.get_height()
                xi = patch.get_x() + patch.get_width() / 2.
                va = 'top'
                c = 'w'
                t = TextPath((0, 0), "${xi}$".format(xi=xi), rotation=0, usetex=True, ha='center')
                transform = transOffset
                if patch.get_extents().height <= t.get_extents().height + 3:
                    va = 'bottom'
                    c = 'k'
                    transform = transOffsetUp
                ax.text(xi, height, "${xi}$".format(xi=int(num)), color=c, rotation=0, ha='center', va=va, transform=transform)
        # for xi, t in zip(x, xticklabels):
        #    ax.text(xi, maxi / 2, t, rotation=90, ha='center', va='center')
        # ax.set_xticklabels(xticklabels, rotation=17)
        ax.set_xticks([])
        ax.set_xlim(-.5, x0 - .5)
        if legend:
            if title is '':
                mode = 'expand'
                if len(bars) > 1:
                    mode = 'expand'
                ax.legend(bbox_to_anchor=(0., 1.02, 1., 1.02), loc=3,
                          ncol=len(bars), mode=mode, borderaxespad=0.)
                fig.tight_layout(rect=(0, 0, 1, .9))
            else:
                ax.legend()
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
        """create a set of slices that can index the parameters of each part."""
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
        # this is a bit nasty: we want to distinguish between parts with the same name by appending a count
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
        Compute the gradient of the covariance function with respect to the parameters.
        
        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: Np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
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
        """Compute the gradient of the covariance function with respect to X.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)"""

        target = np.zeros_like(X)
        if X2 is None: 
            [p.dK_dX(dL_dK, X[:, i_s], None, target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        else:
            [p.dK_dX(dL_dK, X[:, i_s], X2[:, i_s], target[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target

    def Kdiag(self, X, which_parts='all'):
        """Compute the diagonal of the covariance function for inputs X."""
        if which_parts == 'all':
            which_parts = [True] * self.Nparts
        assert X.shape[1] == self.input_dim
        target = np.zeros(X.shape[0])
        [p.Kdiag(X[:, i_s], target=target) for p, i_s, part_on in zip(self.parts, self.input_slices, which_parts) if part_on]
        return target

    def dKdiag_dtheta(self, dL_dKdiag, X):
        """Compute the gradient of the diagonal of the covariance function with respect to the parameters."""
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
        """return shapes are num_samples,num_inducing,input_dim"""
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi1_dmuS(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self.parts, self.input_slices)]
        return target_mu, target_S

    def psi2(self, Z, mu, S):
        """
        Computer the psi2 statistics for the covariance function.
        
        :param Z: np.ndarray of inducing inputs (num_inducing x input_dim)
        :param mu, S: np.ndarrays of means and variances (each num_samples x input_dim)
        :returns psi2: np.ndarray (num_samples,num_inducing,num_inducing)

        """
        target = np.zeros((mu.shape[0], Z.shape[0], Z.shape[0]))
        [p.psi2(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self.parts, self.input_slices)]

        # compute the "cross" terms
        # TODO: input_slices needed
        crossterms = 0

        for [p1, i_s1], [p2, i_s2] in itertools.combinations(zip(self.parts, self.input_slices), 2):
            if i_s1 == i_s2:
                # TODO psi1 this must be faster/better/precached/more nice
                tmp1 = np.zeros((mu.shape[0], Z.shape[0]))
                p1.psi1(Z[:, i_s1], mu[:, i_s1], S[:, i_s1], tmp1)
                tmp2 = np.zeros((mu.shape[0], Z.shape[0]))
                p2.psi1(Z[:, i_s2], mu[:, i_s2], S[:, i_s2], tmp2)
    
                prod = np.multiply(tmp1, tmp2)
                crossterms += prod[:, :, None] + prod[:, None, :]

        # target += crossterms
        return target + crossterms

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S):
        """Gradient of the psi2 statistics with respect to the parameters."""
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
        if dL_dK==None:
            if X2==None:
                dL_dK = np.ones((X.shape[0], X.shape[0]))
            else:
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
        if any(v<-1e-6):
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
        return ['X_'  +str(i) + ','+str(j) for j in range(self.X.shape[1]) for i in range(self.X.shape[0])]
                
    def _get_params(self):
        return self.X.flatten()

    def _set_params(self, x):
        self.X=x.reshape(self.X.shape)

class Kern_check_dKdiag_dX(Kern_check_model):
    """This class allows gradient checks for the gradient of a kernel diagonal with respect to X. """
    def __init__(self, kernel=None, dL_dK=None, X=None, X2=None):
        Kern_check_model.__init__(self,kernel=kernel,dL_dK=dL_dK, X=X, X2=None)
        if dL_dK==None:
            self.dL_dK = np.ones((self.X.shape[0]))

    def log_likelihood(self):
        return (self.dL_dK*self.kernel.Kdiag(self.X)).sum()

    def _log_likelihood_gradients(self):
        return self.kernel.dKdiag_dX(self.dL_dK, self.X).flatten()

    def _get_param_names(self):
        return ['X_'  +str(i) + ','+str(j) for j in range(self.X.shape[1]) for i in range(self.X.shape[0])]
                
    def _get_params(self):
        return self.X.flatten()

    def _set_params(self, x):
        self.X=x.reshape(self.X.shape)

def kern_test(kern, X=None, X2=None, verbose=False):
    """This function runs on kernels to check the correctness of their implementation. It checks that the covariance function is positive definite for a randomly generated data set.

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
    if X2==None:
        X2 = np.random.randn(20, kern.input_dim)
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
            print("dK_dX not implemented for " + kern.name)
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
            print("dK_dX not implemented for " + kern.name)
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
            print("dK_dX not implemented for " + kern.name)
    if result and verbose:
        print("Check passed.")
    if not result:
        print("Gradient of Kdiag(X) wrt X failed for " + kern.name + " covariance function. Gradient values as follows:")
        Kern_check_dKdiag_dX(kern, X=X).checkgrad(verbose=True)
        pass_checks = False
        return False

    return pass_checks
