# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from gp import GP
from parameterization.param import Param
from ..inference.latent_function_inference import varDTC
from posterior import Posterior

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x input_dim)
    :param num_inducing: Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int

    """

    def __init__(self, X, Y, Z, kernel, likelihood, inference_method=None, X_variance=None, name='sparse gp'):

        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = varDTC.Gaussian_inference()
        else:
            #inference_method = ??
            raise NotImplementedError, "what to do what to do?"
            print "defaulting to ", inference_method, "for latent function inference"

        GP.__init__(self, X, Y, likelihood, inference_method, kernel, name)

        self.Z = Z
        self.num_inducing = Z.shape[0]

        if X_variance is None:
            self.has_uncertain_inputs = False
            self.X_variance = None
        else:
            assert X_variance.shape == X.shape
            self.has_uncertain_inputs = True
            self.X_variance = X_variance

        self.Z = Param('inducing inputs', self.Z)
        self.add_parameter(self.Z, gradient=self.dL_dZ, index=0)
        self.add_parameter(self.kern, gradient=self.dL_dtheta)
        self.add_parameter(self.likelihood, gradient=lambda:self.likelihood._gradients(partial=self.partial_for_likelihood))

    def parameters_changed(self):
        self.posterior = self.inference_method.inference(self.kern, self.X, self.X_variance, self.Z, self.likelihood, self.Y)

                #The derivative of the bound wrt the inducing inputs Z
        self.Z.gradient = self.kern.dK_dX(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            self.Z.gradient += self.kern.dpsi1_dZ(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            self.Z.gradient += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            self.Z.gradient += self.kern.dK_dX(self.dL_dpsi1.T, self.Z, self.X)

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        """
        Make a prediction for the latent function values
        """
        #TODO!!!


    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False, fignum=None, ax=None):
        """
        Plot the belief in the latent function, the "GP's view of the world"
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - Not implemented in higher dimensions

        :param samples: the number of a posteriori samples to plot
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param full_cov:
        :type full_cov: bool
                :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle

        :param output: which output to plot (for multiple output models only)
        :type output: integer (first output is 0)
        """
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data is 'all':
            which_data = slice(None)

        GP.plot_f(self, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, full_cov=full_cov, fignum=fignum, ax=ax)

        if self.X.shape[1] == 1:
            if self.has_uncertain_inputs:
                ax.errorbar(self.X[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu, np.zeros_like(Zu) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, fignum=None, ax=None):
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data is 'all':
            which_data = slice(None)

        GP.plot(self, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, levels=20, fignum=fignum, ax=ax)

        if self.X.shape[1] == 1:
            if self.has_uncertain_inputs:
                ax.errorbar(self.X[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu, np.zeros_like(Zu) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')


        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return GP._getstate(self) + [self.Z,
                self.num_inducing,
                self.has_uncertain_inputs,
                self.X_variance]

    def _setstate(self, state):
        self.X_variance = state.pop()
        self.has_uncertain_inputs = state.pop()
        self.num_inducing = state.pop()
        self.Z = state.pop()
        GP._setstate(self, state)

