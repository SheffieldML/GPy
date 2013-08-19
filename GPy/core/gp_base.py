import numpy as np
from .. import kern
from ..util.plot import gpplot, Tango, x_frame1D, x_frame2D
import pylab as pb
from GPy.core.model import Model

class GPBase(Model):
    """
    Gaussian process base model for holding shared behaviour between
    sparse_GP and GP models.
    """

    def __init__(self, X, likelihood, kernel, normalize_X=False):
        self.X = X
        assert len(self.X.shape) == 2
        self.num_data, self.input_dim = self.X.shape
        assert isinstance(kernel, kern.kern)
        self.kern = kernel
        self.likelihood = likelihood
        assert self.X.shape[0] == self.likelihood.data.shape[0]
        self.num_data, self.output_dim = self.likelihood.data.shape

        if normalize_X:
            self._Xoffset = X.mean(0)[None, :]
            self._Xscale = X.std(0)[None, :]
            self.X = (X.copy() - self._Xoffset) / self._Xscale
        else:
            self._Xoffset = np.zeros((1, self.input_dim))
            self._Xscale = np.ones((1, self.input_dim))

        super(GPBase, self).__init__()
        # Model.__init__(self)
        # All leaf nodes should call self._set_params(self._get_params()) at
        # the end

    def getstate(self):
        """
        Get the current state of the class, here we return everything that is needed to recompute the model.
        """
        return Model.getstate(self) + [self.X,
                self.num_data,
                self.input_dim,
                self.kern,
                self.likelihood,
                self.output_dim,
                self._Xoffset,
                self._Xscale]

    def setstate(self, state):
        self._Xscale = state.pop()
        self._Xoffset = state.pop()
        self.output_dim = state.pop()
        self.likelihood = state.pop()
        self.kern = state.pop()
        self.input_dim = state.pop()
        self.num_data = state.pop()
        self.X = state.pop()
        Model.setstate(self, state)

    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False, fignum=None, ax=None):
        """
        Plot the GP's view of the world, where the data is normalized and the
        likelihood is Gaussian.

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions
        using which_data and which_functions

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

        """
        if which_data == 'all':
            which_data = slice(None)

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        if self.X.shape[1] == 1:
            Xnew, xmin, xmax = x_frame1D(self.X, plot_limits=plot_limits)
            if samples == 0:
                m, v = self._raw_predict(Xnew, which_parts=which_parts)
                gpplot(Xnew, m, m - 2 * np.sqrt(v), m + 2 * np.sqrt(v), axes=ax)
                ax.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            else:
                m, v = self._raw_predict(Xnew, which_parts=which_parts, full_cov=True)
                Ysim = np.random.multivariate_normal(m.flatten(), v, samples)
                gpplot(Xnew, m, m - 2 * np.sqrt(np.diag(v)[:, None]), m + 2 * np.sqrt(np.diag(v))[:, None, ], axes=ax)
                for i in range(samples):
                    ax.plot(Xnew, Ysim[i, :], Tango.colorsHex['darkBlue'], linewidth=0.25)
            ax.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            ax.set_xlim(xmin, xmax)
            ymin, ymax = min(np.append(self.likelihood.Y, m - 2 * np.sqrt(np.diag(v)[:, None]))), max(np.append(self.likelihood.Y, m + 2 * np.sqrt(np.diag(v)[:, None])))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            ax.set_ylim(ymin, ymax)

        elif self.X.shape[1] == 2:
            resolution = resolution or 50
            Xnew, xmin, xmax, xx, yy = x_frame2D(self.X, plot_limits, resolution)
            m, v = self._raw_predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            ax.contour(xx, yy, m, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet) # @UndefinedVariable
            ax.scatter(self.X[:, 0], self.X[:, 1], 40, self.likelihood.Y, linewidth=0, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max()) # @UndefinedVariable
            ax.set_xlim(xmin[0], xmax[0])
            ax.set_ylim(xmin[1], xmax[1])
        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, samples=0, fignum=None, ax=None, fixed_inputs=[], linecol=Tango.colorsHex['darkBlue'],fillcol=Tango.colorsHex['lightBlue']):
        """
        Plot the GP with noise where the likelihood is Gaussian.

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions
        using which_data and which_functions

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param linecol: color of line to plot.
        :type linecol:
        :param fillcol: color of fill
        :type fillcol: 
        :param levels: for 2D plotting, the number of contour levels to use
        is ax is None, create a new figure

        """
        # TODO include samples
        if which_data == 'all':
            which_data = slice(None)

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        plotdims = self.input_dim - len(fixed_inputs)

        if plotdims == 1:

            Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now

            fixed_dims = np.array([i for i,v in fixed_inputs])
            freedim = np.setdiff1d(np.arange(self.input_dim),fixed_dims)

            Xnew, xmin, xmax = x_frame1D(Xu[:,freedim], plot_limits=plot_limits)
            Xgrid = np.empty((Xnew.shape[0],self.input_dim))
            Xgrid[:,freedim] = Xnew
            for i,v in fixed_inputs:
                Xgrid[:,i] = v

            m, _, lower, upper = self.predict(Xgrid, which_parts=which_parts)
            for d in range(m.shape[1]):
                gpplot(Xnew, m[:, d], lower[:, d], upper[:, d], axes=ax, edgecol=linecol, fillcol=fillcol)
                ax.plot(Xu[which_data,freedim], self.likelihood.data[which_data, d], 'kx', mew=1.5)
            ymin, ymax = min(np.append(self.likelihood.data, lower)), max(np.append(self.likelihood.data, upper))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        elif self.X.shape[1] == 2: # FIXME
            resolution = resolution or 50
            Xnew, _, _, xmin, xmax = x_frame2D(self.X, plot_limits, resolution)
            x, y = np.linspace(xmin[0], xmax[0], resolution), np.linspace(xmin[1], xmax[1], resolution)
            m, _, lower, upper = self.predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            ax.contour(x, y, m, levels, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet) # @UndefinedVariable
            Yf = self.likelihood.Y.flatten()
            ax.scatter(self.X[:, 0], self.X[:, 1], 40, Yf, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max(), linewidth=0.) # @UndefinedVariable
            ax.set_xlim(xmin[0], xmax[0])
            ax.set_ylim(xmin[1], xmax[1])

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"
