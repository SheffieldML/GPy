# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import linalg
import pylab as pb
from .. import kern
from ..core import model
from ..util.linalg import pdinv, mdot, tdot
from ..util.plot import gpplot, x_frame1D, x_frame2D, Tango
from ..likelihoods import EP, Laplace

class GP(model):
    """
    Gaussian Process model for regression and EP

    :param X: input observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :parm likelihood: a GPy likelihood
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :rtype: model object
    :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 0.1
    :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.]
    :type powerep: list

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    def __init__(self, X, likelihood, kernel, normalize_X=False):
        self.has_uncertain_inputs=False

        # parse arguments
        self.X = X
        assert len(self.X.shape) == 2
        self.N, self.Q = self.X.shape
        assert isinstance(kernel, kern.kern)
        self.kern = kernel
        self.likelihood = likelihood
        assert self.X.shape[0] == self.likelihood.data.shape[0]
        self.N, self.D = self.likelihood.data.shape

        # here's some simple normalization for the inputs
        if normalize_X:
            self._Xmean = X.mean(0)[None, :]
            self._Xstd = X.std(0)[None, :]
            self.X = (X.copy() - self._Xmean) / self._Xstd
            if hasattr(self, 'Z'):
                self.Z = (self.Z - self._Xmean) / self._Xstd
        else:
            self._Xmean = np.zeros((1, self.X.shape[1]))
            self._Xstd = np.ones((1, self.X.shape[1]))

        if not hasattr(self,'has_uncertain_inputs'):
            self.has_uncertain_inputs = False
        model.__init__(self)

    def dL_dZ(self):
        """
        TODO: one day we might like to learn Z by gradient methods?
        """
        #FIXME: this doesn;t live here.
        return np.zeros_like(self.Z)

    def _set_params(self, p):
        self.kern._set_params_transformed(p[:self.kern.Nparam_transformed()])
        # self.likelihood._set_params(p[self.kern.Nparam:])               # test by Nicolas
        self.likelihood._set_params(p[self.kern.Nparam_transformed():])  # test by Nicolas

        if isinstance(self.likelihood, Laplace):
            self.likelihood.fit_full(self.kern.K(self.X))
            self.likelihood._set_params(self.likelihood._get_params())

        self.K = self.kern.K(self.X)
        self.K += self.likelihood.covariance_matrix

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        # the gradient of the likelihood wrt the covariance matrix
        if self.likelihood.YYT is None:
            #alpha = np.dot(self.Ki, self.likelihood.Y)
            alpha,_ = linalg.lapack.flapack.dpotrs(self.L, self.likelihood.Y,lower=1)

            self.dL_dK = 0.5 * (tdot(alpha) - self.D * self.Ki)
        else:
            #tmp = mdot(self.Ki, self.likelihood.YYT, self.Ki)
            tmp, _ = linalg.lapack.flapack.dpotrs(self.L, np.asfortranarray(self.likelihood.YYT), lower=1)
            tmp, _ = linalg.lapack.flapack.dpotrs(self.L, np.asfortranarray(tmp.T), lower=1)
            self.dL_dK = 0.5 * (tmp - self.D * self.Ki)

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed(), self.likelihood._get_params()))

    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def _update_params_callback(self, p):
        #parameters will be in transformed space
        self.kern._set_params_transformed(p[:self.kern.Nparam_transformed()])
        #set_params_transformed for likelihood doesn't exist?
        self.likelihood._set_params(p[self.kern.Nparam_transformed():])
        #update the likelihood approximation within the optimisation with the current parameters
        self.update_likelihood_approximation()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        self.likelihood.fit_full(self.kern.K(self.X))
        self._set_params(self._get_params())  # update the GP

    def _model_fit_term(self):
        """
        Computes the model fit using YYT if it's available
        """
        if self.likelihood.YYT is None:
            tmp, _ = linalg.lapack.flapack.dtrtrs(self.L, np.asfortranarray(self.likelihood.Y), lower=1)
            return -0.5 * np.sum(np.square(tmp))
            #return -0.5 * np.sum(np.square(np.dot(self.Li, self.likelihood.Y)))
        else:
            return -0.5 * np.sum(np.multiply(self.Ki, self.likelihood.YYT))

    def log_likelihood(self):
        """
        The log marginal likelihood of the GP.

        For an EP model,  can be written as the log likelihood of a regression
        model for a new variable Y* = v_tilde/tau_tilde, with a covariance
        matrix K* = K + diag(1./tau_tilde) plus a normalization term.
        """
        l = -0.5 * self.D * self.K_logdet + self._model_fit_term() + self.likelihood.Z
        return l

    def _log_likelihood_gradients(self):
        """
        The gradient of all parameters.

        Note, we use the chain rule: dL_dtheta = dL_dK * d_K_dtheta
        """
        dL_dthetaK = self.kern.dK_dtheta(dL_dK=self.dL_dK, X=self.X)
        print "dL_dthetaK should be: ", dL_dthetaK
        if isinstance(self.likelihood, Laplace):
            self.likelihood.fit_full(self.kern.K(self.X))
            self.likelihood._set_params(self.likelihood._get_params())
            dK_dthetaK = self.kern.dK_dtheta
            dL_dthetaK = self.likelihood._Kgradients(dK_dthetaK, self.X)
            dL_dthetaL = self.likelihood._gradients(partial=np.diag(self.dL_dK))
            #print "dL_dthetaK after: ",dL_dthetaK
            #print "Stacked dL_dthetaK, dL_dthetaL: ", np.hstack((dL_dthetaK, dL_dthetaL))
        else:
            dL_dthetaL = self.likelihood._gradients(partial=np.diag(self.dL_dK))
            #print "Stacked dL_dthetaK, dL_dthetaL: ", np.hstack((dL_dthetaK, dL_dthetaL))
        print "dL_dthetaK is: ", dL_dthetaK

        return np.hstack((dL_dthetaK, dL_dthetaL))
        #return np.hstack((self.kern.dK_dtheta(dL_dK=self.dL_dK, X=self.X), self.likelihood._gradients(partial=np.diag(self.dL_dK))))

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False,stop=False):
        """
        Internal helper function for making predictions, does not account
        for normalization or likelihood
        """
        Kx = self.kern.K(_Xnew,self.X,which_parts=which_parts).T
        #KiKx = np.dot(self.Ki, Kx)
        KiKx, _ = linalg.lapack.flapack.dpotrs(self.L, np.asfortranarray(Kx), lower=1)
        mu = np.dot(KiKx.T, self.likelihood.Y)
        if full_cov:
            Kxx = self.kern.K(_Xnew, which_parts=which_parts)
            var = Kxx - np.dot(KiKx.T, Kx)
        else:
            Kxx = self.kern.Kdiag(_Xnew, which_parts=which_parts)
            var = Kxx - np.sum(np.multiply(KiKx, Kx), 0)
            var = var[:, None]
        if stop:
            debug_this
        return mu, var


    def predict(self, Xnew, which_parts='all', full_cov=False):
        """
        Predict the function(s) at the new point(s) Xnew.

        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.Q
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the folll covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.D
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.D


           If full_cov and self.D > 1, the return shape of var is Nnew x Nnew x self.D. If self.D == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        # normalize X values
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        mu, var = self._raw_predict(Xnew, which_parts, full_cov)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov)

        return mean, var, _025pm, _975pm


    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False):
        """
        Plot the GP's view of the world, where the data is normalized and the
        likelihood is Gaussian.

        :param samples: the number of a posteriori samples to plot
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions
        using which_data and which_functions
        """
        if which_data == 'all':
            which_data = slice(None)

        if self.X.shape[1] == 1:
            Xnew, xmin, xmax = x_frame1D(self.X, plot_limits=plot_limits)
            if samples == 0:
                m, v = self._raw_predict(Xnew, which_parts=which_parts)
                gpplot(Xnew, m, m - 2 * np.sqrt(v), m + 2 * np.sqrt(v))
                pb.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            else:
                m, v = self._raw_predict(Xnew, which_parts=which_parts, full_cov=True)
                Ysim = np.random.multivariate_normal(m.flatten(), v, samples)
                gpplot(Xnew, m, m - 2 * np.sqrt(np.diag(v)[:, None]), m + 2 * np.sqrt(np.diag(v))[:, None])
                for i in range(samples):
                    pb.plot(Xnew, Ysim[i, :], Tango.colorsHex['darkBlue'], linewidth=0.25)
            pb.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            pb.xlim(xmin, xmax)
            ymin, ymax = min(np.append(self.likelihood.Y, m - 2 * np.sqrt(np.diag(v)[:, None]))), max(np.append(self.likelihood.Y, m + 2 * np.sqrt(np.diag(v)[:, None])))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            pb.ylim(ymin, ymax)
            if hasattr(self, 'Z'):
                pb.plot(self.Z, self.Z * 0 + pb.ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            resolution = resolution or 50
            Xnew, xmin, xmax, xx, yy = x_frame2D(self.X, plot_limits, resolution)
            m, v = self._raw_predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            pb.contour(xx, yy, m, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet)
            pb.scatter(Xorig[:, 0], Xorig[:, 1], 40, Yorig, linewidth=0, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max())
            pb.xlim(xmin[0], xmax[0])
            pb.ylim(xmin[1], xmax[1])
        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20):
        """
        TODO: Docstrings!
        :param levels: for 2D plotting, the number of contour levels to use

        """
        # TODO include samples
        if which_data == 'all':
            which_data = slice(None)

        if self.X.shape[1] == 1:

            Xu = self.X * self._Xstd + self._Xmean  # NOTE self.X are the normalized values now

            Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
            m, var, lower, upper = self.predict(Xnew, which_parts=which_parts)
            gpplot(Xnew, m, lower, upper)
            pb.plot(Xu[which_data], self.likelihood.data[which_data], 'kx', mew=1.5)
            if self.has_uncertain_inputs:
                pb.errorbar(Xu[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)

            ymin, ymax = min(np.append(self.likelihood.data, lower)), max(np.append(self.likelihood.data, upper))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            pb.xlim(xmin, xmax)
            pb.ylim(ymin, ymax)
            if hasattr(self, 'Z'):
                Zu = self.Z * self._Xstd + self._Xmean
                pb.plot(Zu, Zu * 0 + pb.ylim()[0], 'r|', mew=1.5, markersize=12)
                    # pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_variance.flatten()))

        elif self.X.shape[1] == 2:  # FIXME
            resolution = resolution or 50
            Xnew, xx, yy, xmin, xmax = x_frame2D(self.X, plot_limits, resolution)
            x, y = np.linspace(xmin[0], xmax[0], resolution), np.linspace(xmin[1], xmax[1], resolution)
            m, var, lower, upper = self.predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            pb.contour(x, y, m, levels, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet)
            Yf = self.likelihood.Y.flatten()
            pb.scatter(self.X[:, 0], self.X[:, 1], 40, Yf, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max(), linewidth=0.)
            pb.xlim(xmin[0], xmax[0])
            pb.ylim(xmin[1], xmax[1])
            if hasattr(self, 'Z'):
                pb.plot(self.Z[:, 0], self.Z[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"
