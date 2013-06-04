import numpy as np
import model
from .. import kern
from ..util.plot import gpplot, Tango, x_frame1D, x_frame2D
import pylab as pb

class GPBase(model.model):
    """
    Gaussian Process model for holding shared behaviour between
    sprase_GP and GP models
    """

    def __init__(self, X, likelihood, kernel, normalize_X=False):
        self.X = X
        assert len(self.X.shape) == 2
        self.N, self.Q = self.X.shape
        assert isinstance(kernel, kern.kern)
        self.kern = kernel
        self.likelihood = likelihood
        assert self.X.shape[0] == self.likelihood.data.shape[0]
        self.N, self.D = self.likelihood.data.shape

        if normalize_X:
            self._Xmean = X.mean(0)[None, :]
            self._Xstd = X.std(0)[None, :]
            self.X = (X.copy() - self._Xmean) / self._Xstd

        super(GPBase, self).__init__()

        # All leaf nodes should call self._set_params(self._get_params()) at
        # the end

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed(), self.likelihood._get_params()))

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
            pb.scatter(self.X[:, 0], self.X[:, 1], 40, self.likelihood.Y, linewidth=0, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max())
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
            for d in range(m.shape[1]):
                gpplot(Xnew, m[:,d], lower[:,d], upper[:,d])
                pb.plot(Xu[which_data], self.likelihood.data[which_data,d], 'kx', mew=1.5)
            ymin, ymax = min(np.append(self.likelihood.data, lower)), max(np.append(self.likelihood.data, upper))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            pb.xlim(xmin, xmax)
            pb.ylim(ymin, ymax)

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

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"
