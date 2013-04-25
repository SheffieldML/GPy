# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys, pdb
from GPLVM import GPLVM
from sparse_GP import sparse_GP
from GPy.util.linalg import pdinv
from ..likelihoods import Gaussian
from .. import kern
from numpy.linalg.linalg import LinAlgError
import itertools

class Bayesian_GPLVM(sparse_GP, GPLVM):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, X=None, X_variance=None, init='PCA', M=10,
                 Z=None, kernel=None, oldpsave=5, _debug=False,
                 **kwargs):
        if X == None:
            X = self.initialise_latent(init, Q, Y)

        if X_variance is None:
            X_variance = np.ones_like(X) * 0.5

        if Z is None:
            Z = np.random.permutation(X.copy())[:M]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(Q) + kern.white(Q)

        self.oldpsave = oldpsave
        self._oldps = []
        self._debug = _debug

        if self._debug:
            self._count = itertools.count()
            self._savedklll = []
            self._savedparams = []

        sparse_GP.__init__(self, X, Gaussian(Y), kernel, Z=Z, X_variance=X_variance, **kwargs)

    @property
    def oldps(self):
        return self._oldps
    @oldps.setter
    def oldps(self, p):
        if len(self._oldps) == (self.oldpsave + 1):
            self._oldps.pop()
        # if len(self._oldps) == 0 or not np.any([np.any(np.abs(p - op) > 1e-5) for op in self._oldps]):
        self._oldps.insert(0, p.copy())

    def _get_param_names(self):
        X_names = sum([['X_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        return (X_names + S_names + sparse_GP._get_param_names(self))

    def _get_params(self):
        """
        Horizontally stacks the parameters in order to present them to the optimizer.
        The resulting 1-D array has this structure:

        ===============================================================
        |       mu       |        S        |    Z    | theta |  beta  |
        ===============================================================

        """
        x = np.hstack((self.X.flatten(), self.X_variance.flatten(), sparse_GP._get_params(self)))
        return x

    def _set_params(self, x, save_old=True, save_count=0):
        try:
            N, Q = self.N, self.Q
            self.X = x[:self.X.size].reshape(N, Q).copy()
            self.X_variance = x[(N * Q):(2 * N * Q)].reshape(N, Q).copy()
            sparse_GP._set_params(self, x[(2 * N * Q):])
            self.oldps = x
        except (LinAlgError, FloatingPointError, ZeroDivisionError):
            print "\rWARNING: Caught LinAlgError, continueing without setting            "
#             if save_count > 10:
#                 raise
#             self._set_params(self.oldps[-1], save_old=False, save_count=save_count + 1)

    def dKL_dmuS(self):
        dKL_dS = (1. - (1. / (self.X_variance))) * 0.5
        dKL_dmu = self.X
        return dKL_dmu, dKL_dS

    def dL_dmuS(self):
        dL_dmu_psi0, dL_dS_psi0 = self.kern.dpsi0_dmuS(self.dL_dpsi0, self.Z, self.X, self.X_variance)
        dL_dmu_psi1, dL_dS_psi1 = self.kern.dpsi1_dmuS(self.dL_dpsi1, self.Z, self.X, self.X_variance)
        dL_dmu_psi2, dL_dS_psi2 = self.kern.dpsi2_dmuS(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        dL_dmu = dL_dmu_psi0 + dL_dmu_psi1 + dL_dmu_psi2
        dL_dS = dL_dS_psi0 + dL_dS_psi1 + dL_dS_psi2

        return dL_dmu, dL_dS

    def KL_divergence(self):
        var_mean = np.square(self.X).sum()
        var_S = np.sum(self.X_variance - np.log(self.X_variance))
        return 0.5 * (var_mean + var_S) - 0.5 * self.Q * self.N

    def log_likelihood(self):
        ll = sparse_GP.log_likelihood(self)
        kl = self.KL_divergence()

#         if ll < -2E4:
#             ll = -2E4 + np.random.randn()
#         if kl > 5E4:
#             kl = 5E4 + np.random.randn()

        if self._debug:
            f_call = self._count.next()
            self._savedklll.append([f_call, ll, kl])
            if f_call % 1 == 0:
                self._savedparams.append([f_call, self._get_params()])


        # print "\nkl:", kl, "ll:", ll
        return ll - kl

    def _log_likelihood_gradients(self):
        dKL_dmu, dKL_dS = self.dKL_dmuS()
        dL_dmu, dL_dS = self.dL_dmuS()
        # TODO: find way to make faster

        d_dmu = (dL_dmu - dKL_dmu).flatten()
        d_dS = (dL_dS - dKL_dS).flatten()
        # TEST KL: ====================
        # d_dmu = (dKL_dmu).flatten()
        # d_dS = (dKL_dS).flatten()
        # ========================
        # TEST L: ====================
#         d_dmu = (dL_dmu).flatten()
#         d_dS = (dL_dS).flatten()
        # ========================
        dbound_dmuS = np.hstack((d_dmu, d_dS))
        return np.hstack((dbound_dmuS.flatten(), sparse_GP._log_likelihood_gradients(self)))

    def plot_latent(self, which_indices=None, *args, **kwargs):

        if which_indices is None:
            try:
                input_1, input_2 = np.argsort(self.input_sensitivity())[:2]
            except:
                raise ValueError, "cannot Atomatically determine which dimensions to plot, please pass 'which_indices'"
        else:
            input_1, input_2 = which_indices
        ax = GPLVM.plot_latent(self, which_indices=[input_1, input_2], *args, **kwargs)
        ax.plot(self.Z[:, input_1], self.Z[:, input_2], '^w')
        return ax

    def plot_X_1d(self, fig_num="MRD X 1d", axes=None, colors=None):
        import pylab

        fig = pylab.figure(num=fig_num, figsize=(min(8, (3 * len(self.bgplvms))), min(12, (2 * self.X.shape[1]))))
        if colors is None:
            colors = pylab.gca()._get_lines.color_cycle
            pylab.clf()
        plots = []
        for i in range(self.X.shape[1]):
            if axes is None:
                ax = fig.add_subplot(self.X.shape[1], 1, i + 1)
            else:
                ax = axes[i]
            ax.plot(self.X, c='k', alpha=.3)
            plots.extend(ax.plot(self.X.T[i], c=colors.next(), label=r"$\mathbf{{X_{}}}$".format(i)))
            ax.fill_between(np.arange(self.X.shape[0]),
                            self.X.T[i] - 2 * np.sqrt(self.X_variance.T[i]),
                            self.X.T[i] + 2 * np.sqrt(self.X_variance.T[i]),
                            facecolor=plots[-1].get_color(),
                            alpha=.3)
            ax.legend(borderaxespad=0.)
            if i < self.X.shape[1] - 1:
                ax.set_xticklabels('')
        pylab.draw()
        fig.tight_layout(h_pad=.01)  # , rect=(0, 0, 1, .95))
        return fig

    def _debug_filter_params(self, x):
        start, end = 0, self.X.size,
        X = x[start:end].reshape(self.N, self.Q)
        start, end = end, end + self.X_variance.size
        X_v = x[start:end].reshape(self.N, self.Q)
        start, end = end, end + (self.M * self.Q)
        Z = x[start:end].reshape(self.M, self.Q)
        start, end = end, end + self.Q
        theta = x[start:]
        return X, X_v, Z, theta

    def _debug_plot(self):
        assert self._debug, "must enable _debug, to debug-plot"
        import pylab
        from mpl_toolkits.mplot3d import Axes3D
        fig = pylab.figure('BGPLVM DEBUG', figsize=(12, 10))
        fig.clf()

        # log like
        splotshape = (6, 4)
        ax1 = pylab.subplot2grid(splotshape, (0, 0), 1, 4)
        ax1.text(.5, .5, "Optimization", alpha=.3, transform=ax1.transAxes,
                 ha='center', va='center')
        kllls = np.array(self._savedklll)
        LL, = ax1.plot(kllls[:, 0], kllls[:, 1] - kllls[:, 2], label=r'$\log p(\mathbf{Y})$', mew=1.5)
        KL, = ax1.plot(kllls[:, 0], kllls[:, 2], label=r'$\mathcal{KL}(p||q)$', mew=1.5)
        L, = ax1.plot(kllls[:, 0], kllls[:, 1], label=r'$L$', mew=1.5)  # \mathds{E}_{q(\mathbf{X})}[p(\mathbf{Y|X})\frac{p(\mathbf{X})}{q(\mathbf{X})}]

        drawn = dict(self._savedparams)
        iters = np.array(drawn.keys())
        self.showing = 0

        ax2 = pylab.subplot2grid(splotshape, (1, 0), 2, 4)
        ax2.text(.5, .5, r"$\mathbf{X}$", alpha=.5, transform=ax2.transAxes,
                 ha='center', va='center')
        ax3 = pylab.subplot2grid(splotshape, (3, 0), 2, 4, sharex=ax2)
        ax3.text(.5, .5, r"$\mathbf{S}$", alpha=.5, transform=ax3.transAxes,
                 ha='center', va='center')
        ax4 = pylab.subplot2grid(splotshape, (5, 0), 2, 2)
        ax4.text(.5, .5, r"$\mathbf{Z}$", alpha=.5, transform=ax4.transAxes,
                 ha='center', va='center')
        ax5 = pylab.subplot2grid(splotshape, (5, 2), 2, 2)
        ax5.text(.5, .5, r"${\theta}$", alpha=.5, transform=ax5.transAxes,
                 ha='center', va='center')

        X, S, Z, theta = self._debug_filter_params(drawn[self.showing])
        Xlatentplts = ax2.plot(X, ls="-", marker="x")
        Slatentplts = ax3.plot(S, ls="-", marker="x")
        Zplts = ax4.plot(Z, ls="-", marker="x")
        thetaplts = ax5.bar(np.arange(len(theta)) - .4, theta)
        ax5.set_xticks(np.arange(len(theta)))
        ax5.set_xticklabels(self._get_param_names()[-len(theta):], rotation=17)

        Qleg = ax1.legend(Xlatentplts, [r"$Q_{}$".format(i + 1) for i in range(self.Q)],
                   loc=3, ncol=self.Q, bbox_to_anchor=(0, 1.15, 1, 1.15),
                   borderaxespad=0, mode="expand")
        Lleg = ax1.legend()
        Lleg.draggable()
        ax1.add_artist(Qleg)

        indicatorKL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 2], 'o', c=KL.get_color())
        indicatorLL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 1] - kllls[self.showing, 2], 'o', c=LL.get_color())
        indicatorL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 1], 'o', c=L.get_color())

        try:
            pylab.draw()
            pylab.tight_layout(box=(0, .1, 1, .9))
        except:
            pass

        # parameter changes
        # ax2 = pylab.subplot2grid((4, 1), (1, 0), 3, 1, projection='3d')
        def onclick(event):
            if event.inaxes is ax1 and event.button == 1:
#               event.button, event.x, event.y, event.xdata, event.ydata)
                tmp = np.abs(iters - event.xdata)
                closest_hit = iters[tmp == tmp.min()][0]

                if closest_hit != self.showing:
                    self.showing = closest_hit
                    # print closest_hit, iters, event.xdata

                    indicatorLL.set_data(self.showing, kllls[self.showing, 1] - kllls[self.showing, 2])
                    indicatorKL.set_data(self.showing, kllls[self.showing, 2])
                    indicatorL.set_data(self.showing, kllls[self.showing, 1])

                    X, S, Z, theta = self._debug_filter_params(drawn[self.showing])
                    for i, Xlatent in enumerate(Xlatentplts):
                        Xlatent.set_ydata(X[:, i])
                    for i, Slatent in enumerate(Slatentplts):
                        Slatent.set_ydata(S[:, i])
                    for i, Zlatent in enumerate(Zplts):
                        Zlatent.set_ydata(Z[:, i])
                    for p, t in zip(thetaplts, theta):
                        p.set_height(t)

                    ax2.relim()
                    ax3.relim()
                    ax4.relim()
                    ax5.relim()
                    ax2.autoscale()
                    ax3.autoscale()
                    ax4.autoscale()
                    ax5.autoscale()
                    fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        return ax1, ax2, ax3, ax4, ax5
