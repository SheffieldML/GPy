# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import SparseGP
from ..likelihoods import Gaussian
from .. import kern
import itertools
from matplotlib.colors import colorConverter
from GPy.inference.optimization import SCG
from GPy.util import plot_latent
from GPy.models.gplvm import GPLVM

class BayesianGPLVM(SparseGP, GPLVM):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, likelihood_or_Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, oldpsave=10, _debug=False,
                 **kwargs):
        if type(likelihood_or_Y) is np.ndarray:
            likelihood = Gaussian(likelihood_or_Y)
        else:
            likelihood = likelihood_or_Y

        if X == None:
            X = self.initialise_latent(init, input_dim, likelihood.Y)
        self.init = init

        if X_variance is None:
            X_variance = np.clip((np.ones_like(X) * 0.5) + .01 * np.random.randn(*X.shape), 0.001, 1)

        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(input_dim) + kern.white(input_dim)

        self.oldpsave = oldpsave
        self._oldps = []
        self._debug = _debug

        if self._debug:
            self.f_call = 0
            self._count = itertools.count()
            self._savedklll = []
            self._savedparams = []
            self._savedgradients = []
            self._savederrors = []
            self._savedpsiKmm = []
            self._savedABCD = []

        SparseGP.__init__(self, X, likelihood, kernel, Z=Z, X_variance=X_variance, **kwargs)
        self._set_params(self._get_params())

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
        X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        return (X_names + S_names + SparseGP._get_param_names(self))

    def _get_params(self):
        """
        Horizontally stacks the parameters in order to present them to the optimizer.
        The resulting 1-input_dim array has this structure:

        ===============================================================
        |       mu       |        S        |    Z    | theta |  beta  |
        ===============================================================

        """
        x = np.hstack((self.X.flatten(), self.X_variance.flatten(), SparseGP._get_params(self)))
        return x

    def _clipped(self, x):
        return x # np.clip(x, -1e300, 1e300)

    def _set_params(self, x, save_old=True, save_count=0):
#         try:
            x = self._clipped(x)
            N, input_dim = self.num_data, self.input_dim
            self.X = x[:self.X.size].reshape(N, input_dim).copy()
            self.X_variance = x[(N * input_dim):(2 * N * input_dim)].reshape(N, input_dim).copy()
            SparseGP._set_params(self, x[(2 * N * input_dim):])
#             self.oldps = x
#         except (LinAlgError, FloatingPointError, ZeroDivisionError):
#             print "\rWARNING: Caught LinAlgError, continueing without setting            "
#             if self._debug:
#                 self._savederrors.append(self.f_call)
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
        return 0.5 * (var_mean + var_S) - 0.5 * self.input_dim * self.num_data

    def log_likelihood(self):
        ll = SparseGP.log_likelihood(self)
        kl = self.KL_divergence()

#         if ll < -2E4:
#             ll = -2E4 + np.random.randn()
#         if kl > 5E4:
#             kl = 5E4 + np.random.randn()

        if self._debug:
            self.f_call = self._count.next()
            if self.f_call % 1 == 0:
                self._savedklll.append([self.f_call, ll, kl])
                self._savedparams.append([self.f_call, self._get_params()])
                self._savedgradients.append([self.f_call, self._log_likelihood_gradients()])
                self._savedpsiKmm.append([self.f_call, [self.Kmm, self.dL_dKmm]])
#                 sf2 = self.scale_factor ** 2
                if self.likelihood.is_heteroscedastic:
                    A = -0.5 * self.num_data * self.input_dim * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.likelihood.precision)) - 0.5 * np.sum(self.V * self.likelihood.Y)
#                     B = -0.5 * self.input_dim * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A) * sf2)
                    B = -0.5 * self.input_dim * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A))
                else:
                    A = -0.5 * self.num_data * self.input_dim * (np.log(2.*np.pi) + np.log(self.likelihood._variance)) - 0.5 * self.likelihood.precision * self.likelihood.trYYT
#                     B = -0.5 * self.input_dim * (np.sum(self.likelihood.precision * self.psi0) - np.trace(self.A) * sf2)
                    B = -0.5 * self.input_dim * (np.sum(self.likelihood.precision * self.psi0) - np.trace(self.A))
                C = -self.input_dim * (np.sum(np.log(np.diag(self.LB)))) # + 0.5 * self.num_inducing * np.log(sf2))
                D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
                self._savedABCD.append([self.f_call, A, B, C, D])

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
        self.dbound_dmuS = np.hstack((d_dmu, d_dS))
        self.dbound_dZtheta = SparseGP._log_likelihood_gradients(self)
        return self._clipped(np.hstack((self.dbound_dmuS.flatten(), self.dbound_dZtheta)))

    def plot_latent(self, *args, **kwargs):
        return plot_latent.plot_latent_indices(self, *args, **kwargs)

    def do_test_latents(self, Y):
        """
        Compute the latent representation for a set of new points Y

        Notes:
        This will only work with a univariate Gaussian likelihood (for now)
        """
        assert not self.likelihood.is_heteroscedastic
        N_test = Y.shape[0]
        input_dim = self.Z.shape[1]
        means = np.zeros((N_test, input_dim))
        covars = np.zeros((N_test, input_dim))

        dpsi0 = -0.5 * self.input_dim * self.likelihood.precision
        dpsi2 = self.dL_dpsi2[0][None, :, :] # TODO: this may change if we ignore het. likelihoods
        V = self.likelihood.precision * Y
        dpsi1 = np.dot(self.Cpsi1V, V.T)

        start = np.zeros(self.input_dim * 2)

        for n, dpsi1_n in enumerate(dpsi1.T[:, :, None]):
            args = (self.kern, self.Z, dpsi0, dpsi1_n, dpsi2)
            xopt, fopt, neval, status = SCG(f=latent_cost, gradf=latent_grad, x=start, optargs=args, display=False)

            mu, log_S = xopt.reshape(2, 1, -1)
            means[n] = mu[0].copy()
            covars[n] = np.exp(log_S[0]).copy()

        return means, covars


    def plot_X_1d(self, fignum=None, ax=None, colors=None):
        """
        Plot latent space X in 1D:

            -if fig is given, create input_dim subplots in fig and plot in these
            -if ax is given plot input_dim 1D latent space plots of X into each `axis`
            -if neither fig nor ax is given create a figure with fignum and plot in there

        colors:
            colors of different latent space dimensions input_dim
        """
        import pylab
        if ax is None:
            fig = pylab.figure(num=fignum, figsize=(8, min(12, (2 * self.X.shape[1]))))
        if colors is None:
            colors = pylab.gca()._get_lines.color_cycle
            pylab.clf()
        else:
            colors = iter(colors)
        plots = []
        x = np.arange(self.X.shape[0])
        for i in range(self.X.shape[1]):
            if ax is None:
                a = fig.add_subplot(self.X.shape[1], 1, i + 1)
            elif isinstance(ax, (tuple, list)):
                a = ax[i]
            else:
                raise ValueError("Need one ax per latent dimnesion input_dim")
            a.plot(self.X, c='k', alpha=.3)
            plots.extend(a.plot(x, self.X.T[i], c=colors.next(), label=r"$\mathbf{{X_{{{}}}}}$".format(i)))
            a.fill_between(x,
                            self.X.T[i] - 2 * np.sqrt(self.X_variance.T[i]),
                            self.X.T[i] + 2 * np.sqrt(self.X_variance.T[i]),
                            facecolor=plots[-1].get_color(),
                            alpha=.3)
            a.legend(borderaxespad=0.)
            a.set_xlim(x.min(), x.max())
            if i < self.X.shape[1] - 1:
                a.set_xticklabels('')
        pylab.draw()
        fig.tight_layout(h_pad=.01) # , rect=(0, 0, 1, .95))
        return fig

    def __getstate__(self):
        return (self.likelihood, self.input_dim, self.X, self.X_variance,
                self.init, self.num_inducing, self.Z, self.kern,
                self.oldpsave, self._debug)

    def __setstate__(self, state):
        self.__init__(*state)

    def _debug_filter_params(self, x):
        start, end = 0, self.X.size,
        X = x[start:end].reshape(self.num_data, self.input_dim)
        start, end = end, end + self.X_variance.size
        X_v = x[start:end].reshape(self.num_data, self.input_dim)
        start, end = end, end + (self.num_inducing * self.input_dim)
        Z = x[start:end].reshape(self.num_inducing, self.input_dim)
        start, end = end, end + self.input_dim
        theta = x[start:]
        return X, X_v, Z, theta


    def _debug_get_axis(self, figs):
        if figs[-1].axes:
            ax1 = figs[-1].axes[0]
            ax1.cla()
        else:
            ax1 = figs[-1].add_subplot(111)
        return ax1

    def _debug_plot(self):
        assert self._debug, "must enable _debug, to debug-plot"
        import pylab
#         from mpl_toolkits.mplot3d import Axes3D
        figs = [pylab.figure('BGPLVM DEBUG', figsize=(12, 4))]
#         fig.clf()

        # log like
#         splotshape = (6, 4)
#         ax1 = pylab.subplot2grid(splotshape, (0, 0), 1, 4)
        ax1 = self._debug_get_axis(figs)
        ax1.text(.5, .5, "Optimization", alpha=.3, transform=ax1.transAxes,
                 ha='center', va='center')
        kllls = np.array(self._savedklll)
        LL, = ax1.plot(kllls[:, 0], kllls[:, 1] - kllls[:, 2], '-', label=r'$\log p(\mathbf{Y})$', mew=1.5)
        KL, = ax1.plot(kllls[:, 0], kllls[:, 2], '-', label=r'$\mathcal{KL}(p||q)$', mew=1.5)
        L, = ax1.plot(kllls[:, 0], kllls[:, 1], '-', label=r'$L$', mew=1.5) # \mathds{E}_{q(\mathbf{X})}[p(\mathbf{Y|X})\frac{p(\mathbf{X})}{q(\mathbf{X})}]

        param_dict = dict(self._savedparams)
        gradient_dict = dict(self._savedgradients)
#         kmm_dict = dict(self._savedpsiKmm)
        iters = np.array(param_dict.keys())
        ABCD_dict = np.array(self._savedABCD)
        self.showing = 0

#         ax2 = pylab.subplot2grid(splotshape, (1, 0), 2, 4)
        figs.append(pylab.figure("BGPLVM DEBUG X", figsize=(12, 4)))
        ax2 = self._debug_get_axis(figs)
        ax2.text(.5, .5, r"$\mathbf{X}$", alpha=.5, transform=ax2.transAxes,
                 ha='center', va='center')
        figs[-1].canvas.draw()
        figs[-1].tight_layout(rect=(0, 0, 1, .86))
#         ax3 = pylab.subplot2grid(splotshape, (3, 0), 2, 4, sharex=ax2)
        figs.append(pylab.figure("BGPLVM DEBUG S", figsize=(12, 4)))
        ax3 = self._debug_get_axis(figs)
        ax3.text(.5, .5, r"$\mathbf{S}$", alpha=.5, transform=ax3.transAxes,
                 ha='center', va='center')
        figs[-1].canvas.draw()
        figs[-1].tight_layout(rect=(0, 0, 1, .86))
#         ax4 = pylab.subplot2grid(splotshape, (5, 0), 2, 2)
        figs.append(pylab.figure("BGPLVM DEBUG Z", figsize=(6, 4)))
        ax4 = self._debug_get_axis(figs)
        ax4.text(.5, .5, r"$\mathbf{Z}$", alpha=.5, transform=ax4.transAxes,
                 ha='center', va='center')
        figs[-1].canvas.draw()
        figs[-1].tight_layout(rect=(0, 0, 1, .86))
#         ax5 = pylab.subplot2grid(splotshape, (5, 2), 2, 2)
        figs.append(pylab.figure("BGPLVM DEBUG theta", figsize=(6, 4)))
        ax5 = self._debug_get_axis(figs)
        ax5.text(.5, .5, r"${\theta}$", alpha=.5, transform=ax5.transAxes,
                 ha='center', va='center')
        figs[-1].canvas.draw()
        figs[-1].tight_layout(rect=(.15, 0, 1, .86))
#         figs.append(pylab.figure("BGPLVM DEBUG Kmm", figsize=(12, 6)))
#         fig = figs[-1]
#         ax6 = fig.add_subplot(121)
#         ax6.text(.5, .5, r"${\mathbf{K}_{mm}}$", color='magenta', alpha=.5, transform=ax6.transAxes,
#                  ha='center', va='center')
#         ax7 = fig.add_subplot(122)
#         ax7.text(.5, .5, r"${\frac{dL}{dK_{mm}}}$", color='magenta', alpha=.5, transform=ax7.transAxes,
#                  ha='center', va='center')
        figs.append(pylab.figure("BGPLVM DEBUG Kmm", figsize=(12, 6)))
        fig = figs[-1]
        ax8 = fig.add_subplot(121)
        ax8.text(.5, .5, r"${\mathbf{A,B,C,input_dim}}$", color='k', alpha=.5, transform=ax8.transAxes,
                 ha='center', va='center')
        ax8.plot(ABCD_dict[:, 0], ABCD_dict[:, 1], label='A')
        ax8.plot(ABCD_dict[:, 0], ABCD_dict[:, 2], label='B')
        ax8.plot(ABCD_dict[:, 0], ABCD_dict[:, 3], label='C')
        ax8.plot(ABCD_dict[:, 0], ABCD_dict[:, 4], label='input_dim')
        ax8.legend()
        figs[-1].canvas.draw()
        figs[-1].tight_layout(rect=(.15, 0, 1, .86))

        X, S, Z, theta = self._debug_filter_params(param_dict[self.showing])
        Xg, Sg, Zg, thetag = self._debug_filter_params(gradient_dict[self.showing])
#         Xg, Sg, Zg, thetag = -Xg, -Sg, -Zg, -thetag

        quiver_units = 'xy'
        quiver_scale = 1
        quiver_scale_units = 'xy'
        Xlatentplts = ax2.plot(X, ls="-", marker="x")
        colors = colorConverter.to_rgba_array([p.get_color() for p in Xlatentplts], .4)
        Ulatent = np.zeros_like(X)
        xlatent = np.tile(np.arange(0, X.shape[0])[:, None], X.shape[1])
        Xlatentgrads = ax2.quiver(xlatent, X, Ulatent, Xg, color=colors,
                                  units=quiver_units, scale_units=quiver_scale_units,
                                  scale=quiver_scale)

        Slatentplts = ax3.plot(S, ls="-", marker="x")
        Slatentgrads = ax3.quiver(xlatent, S, Ulatent, Sg, color=colors,
                                  units=quiver_units, scale_units=quiver_scale_units,
                                  scale=quiver_scale)
        ax3.set_ylim(0, 1.)

        xZ = np.tile(np.arange(0, Z.shape[0])[:, None], Z.shape[1])
        UZ = np.zeros_like(Z)
        Zplts = ax4.plot(Z, ls="-", marker="x")
        Zgrads = ax4.quiver(xZ, Z, UZ, Zg, color=colors,
                                  units=quiver_units, scale_units=quiver_scale_units,
                                  scale=quiver_scale)

        xtheta = np.arange(len(theta))
        Utheta = np.zeros_like(theta)
        thetaplts = ax5.bar(xtheta - .4, theta, color=colors)
        thetagrads = ax5.quiver(xtheta, theta, Utheta, thetag, color=colors,
                                  units=quiver_units, scale_units=quiver_scale_units,
                                  scale=quiver_scale,
                                  edgecolors=('k',), linewidths=[1])
        pylab.setp(thetaplts, zorder=0)
        pylab.setp(thetagrads, zorder=10)
        ax5.set_xticks(np.arange(len(theta)))
        ax5.set_xticklabels(self._get_param_names()[-len(theta):], rotation=17)

#         imkmm = ax6.imshow(kmm_dict[self.showing][0])
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         divider = make_axes_locatable(ax6)
#         caxkmm = divider.append_axes("right", "5%", pad="1%")
#         cbarkmm = pylab.colorbar(imkmm, cax=caxkmm)
#
#         imkmmdl = ax7.imshow(kmm_dict[self.showing][1])
#         divider = make_axes_locatable(ax7)
#         caxkmmdl = divider.append_axes("right", "5%", pad="1%")
#         cbarkmmdl = pylab.colorbar(imkmmdl, cax=caxkmmdl)

#         input_dimleg = ax1.legend(Xlatentplts, [r"$input_dim_{}$".format(i + 1) for i in range(self.input_dim)],
#                    loc=3, ncol=self.input_dim, bbox_to_anchor=(0, 1.15, 1, 1.15),
#                    borderaxespad=0, mode="expand")
        ax2.legend(Xlatentplts, [r"$input_dim_{}$".format(i + 1) for i in range(self.input_dim)],
                   loc=3, ncol=self.input_dim, bbox_to_anchor=(0, 1.1, 1, 1.1),
                   borderaxespad=0, mode="expand")
        ax3.legend(Xlatentplts, [r"$input_dim_{}$".format(i + 1) for i in range(self.input_dim)],
                   loc=3, ncol=self.input_dim, bbox_to_anchor=(0, 1.1, 1, 1.1),
                   borderaxespad=0, mode="expand")
        ax4.legend(Xlatentplts, [r"$input_dim_{}$".format(i + 1) for i in range(self.input_dim)],
                   loc=3, ncol=self.input_dim, bbox_to_anchor=(0, 1.1, 1, 1.1),
                   borderaxespad=0, mode="expand")
        ax5.legend(Xlatentplts, [r"$input_dim_{}$".format(i + 1) for i in range(self.input_dim)],
                   loc=3, ncol=self.input_dim, bbox_to_anchor=(0, 1.1, 1, 1.1),
                   borderaxespad=0, mode="expand")
        Lleg = ax1.legend()
        Lleg.draggable()
#         ax1.add_artist(input_dimleg)

        indicatorKL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 2], 'o', c=KL.get_color())
        indicatorLL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 1] - kllls[self.showing, 2], 'o', c=LL.get_color())
        indicatorL, = ax1.plot(kllls[self.showing, 0], kllls[self.showing, 1], 'o', c=L.get_color())
#         for err in self._savederrors:
#             if err < kllls.shape[0]:
#                 ax1.scatter(kllls[err, 0], kllls[err, 2], s=50, marker=(5, 2), c=KL.get_color())
#                 ax1.scatter(kllls[err, 0], kllls[err, 1] - kllls[err, 2], s=50, marker=(5, 2), c=LL.get_color())
#                 ax1.scatter(kllls[err, 0], kllls[err, 1], s=50, marker=(5, 2), c=L.get_color())

#         try:
#             for f in figs:
#                 f.canvas.draw()
#                 f.tight_layout(box=(0, .15, 1, .9))
# #             pylab.draw()
# #             pylab.tight_layout(box=(0, .1, 1, .9))
#         except:
#             pass

        # parameter changes
        # ax2 = pylab.subplot2grid((4, 1), (1, 0), 3, 1, projection='3d')
        button_options = [0, 0] # [0]: clicked -- [1]: dragged

        def update_plots(event):
            if button_options[0] and not button_options[1]:
#               event.button, event.x, event.y, event.xdata, event.ydata)
                tmp = np.abs(iters - event.xdata)
                closest_hit = iters[tmp == tmp.min()][0]

                if closest_hit != self.showing:
                    self.showing = closest_hit
                    # print closest_hit, iters, event.xdata

                    indicatorLL.set_data(self.showing, kllls[self.showing, 1] - kllls[self.showing, 2])
                    indicatorKL.set_data(self.showing, kllls[self.showing, 2])
                    indicatorL.set_data(self.showing, kllls[self.showing, 1])

                    X, S, Z, theta = self._debug_filter_params(param_dict[self.showing])
                    Xg, Sg, Zg, thetag = self._debug_filter_params(gradient_dict[self.showing])
#                     Xg, Sg, Zg, thetag = -Xg, -Sg, -Zg, -thetag

                    for i, Xlatent in enumerate(Xlatentplts):
                        Xlatent.set_ydata(X[:, i])
                    Xlatentgrads.set_offsets(np.array([xlatent.ravel(), X.ravel()]).T)
                    Xlatentgrads.set_UVC(Ulatent, Xg)

                    for i, Slatent in enumerate(Slatentplts):
                        Slatent.set_ydata(S[:, i])
                    Slatentgrads.set_offsets(np.array([xlatent.ravel(), S.ravel()]).T)
                    Slatentgrads.set_UVC(Ulatent, Sg)

                    for i, Zlatent in enumerate(Zplts):
                        Zlatent.set_ydata(Z[:, i])
                    Zgrads.set_offsets(np.array([xZ.ravel(), Z.ravel()]).T)
                    Zgrads.set_UVC(UZ, Zg)

                    for p, t in zip(thetaplts, theta):
                        p.set_height(t)
                    thetagrads.set_offsets(np.array([xtheta.ravel(), theta.ravel()]).T)
                    thetagrads.set_UVC(Utheta, thetag)

#                     imkmm.set_data(kmm_dict[self.showing][0])
#                     imkmm.autoscale()
#                     cbarkmm.update_normal(imkmm)
#
#                     imkmmdl.set_data(kmm_dict[self.showing][1])
#                     imkmmdl.autoscale()
#                     cbarkmmdl.update_normal(imkmmdl)

                    ax2.relim()
                    # ax3.relim()
                    ax4.relim()
                    ax5.relim()
                    ax2.autoscale()
                    # ax3.autoscale()
                    ax4.autoscale()
                    ax5.autoscale()

                    [fig.canvas.draw() for fig in figs]
            button_options[0] = 0
            button_options[1] = 0

        def onclick(event):
            if event.inaxes is ax1 and event.button == 1:
                button_options[0] = 1
        def motion(event):
            if button_options[0]:
                button_options[1] = 1

        cidr = figs[0].canvas.mpl_connect('button_release_event', update_plots)
        cidp = figs[0].canvas.mpl_connect('button_press_event', onclick)
        cidd = figs[0].canvas.mpl_connect('motion_notify_event', motion)

        return ax1, ax2, ax3, ax4, ax5 # , ax6, ax7




def latent_cost_and_grad(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    objective function for fitting the latent variables for test points
    (negative log-likelihood: should be minimised!)
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    psi0 = kern.psi0(Z, mu, S)
    psi1 = kern.psi1(Z, mu, S)
    psi2 = kern.psi2(Z, mu, S)

    lik = dL_dpsi0 * psi0 + np.dot(dL_dpsi1.flatten(), psi1.flatten()) + np.dot(dL_dpsi2.flatten(), psi2.flatten()) - 0.5 * np.sum(np.square(mu) + S) + 0.5 * np.sum(log_S)

    mu0, S0 = kern.dpsi0_dmuS(dL_dpsi0, Z, mu, S)
    mu1, S1 = kern.dpsi1_dmuS(dL_dpsi1, Z, mu, S)
    mu2, S2 = kern.dpsi2_dmuS(dL_dpsi2, Z, mu, S)

    dmu = mu0 + mu1 + mu2 - mu
    # dS = S0 + S1 + S2 -0.5 + .5/S
    dlnS = S * (S0 + S1 + S2 - 0.5) + .5
    return -lik, -np.hstack((dmu.flatten(), dlnS.flatten()))

def latent_cost(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    objective function for fitting the latent variables (negative log-likelihood: should be minimised!)
    This is the same as latent_cost_and_grad but only for the objective
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    psi0 = kern.psi0(Z, mu, S)
    psi1 = kern.psi1(Z, mu, S)
    psi2 = kern.psi2(Z, mu, S)

    lik = dL_dpsi0 * psi0 + np.dot(dL_dpsi1.flatten(), psi1.flatten()) + np.dot(dL_dpsi2.flatten(), psi2.flatten()) - 0.5 * np.sum(np.square(mu) + S) + 0.5 * np.sum(log_S)
    return -float(lik)

def latent_grad(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    This is the same as latent_cost_and_grad but only for the grad
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    mu0, S0 = kern.dpsi0_dmuS(dL_dpsi0, Z, mu, S)
    mu1, S1 = kern.dpsi1_dmuS(dL_dpsi1, Z, mu, S)
    mu2, S2 = kern.dpsi2_dmuS(dL_dpsi2, Z, mu, S)

    dmu = mu0 + mu1 + mu2 - mu
    # dS = S0 + S1 + S2 -0.5 + .5/S
    dlnS = S * (S0 + S1 + S2 - 0.5) + .5

    return -np.hstack((dmu.flatten(), dlnS.flatten()))


