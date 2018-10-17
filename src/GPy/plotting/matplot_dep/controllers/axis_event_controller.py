'''
Created on 24 Jul 2013

@author: maxz
'''
import numpy

class AxisEventController(object):
    def __init__(self, ax):
        self.ax = ax
        self.activate()
    def __del__(self):
        self.deactivate()
        return self
    def deactivate(self):
        for cb_class in self.ax.callbacks.callbacks.values():
            for cb_num in dict(cb_class).keys():
                self.ax.callbacks.disconnect(cb_num)
    def activate(self):
        self.ax.callbacks.connect('xlim_changed', self.xlim_changed)
        self.ax.callbacks.connect('ylim_changed', self.ylim_changed)
    def xlim_changed(self, ax):
        pass
    def ylim_changed(self, ax):
        pass


class AxisChangedController(AxisEventController):
    '''
    Buffered control of axis limit changes
    '''
    _changing = False

    def __init__(self, ax, update_lim=None):
        '''
        Constructor
        '''
        super(AxisChangedController, self).__init__(ax)
        self._lim_ratio_threshold = update_lim or .95
        self._x_lim = self.ax.get_xlim()
        self._y_lim = self.ax.get_ylim()

    def update(self, ax):
        pass

    def xlim_changed(self, ax):
        super(AxisChangedController, self).xlim_changed(ax)
        if not self._changing and self.lim_changed(ax.get_xlim(), self._x_lim):
            self._changing = True
            self._x_lim = ax.get_xlim()
            self.update(ax)
            self._changing = False

    def ylim_changed(self, ax):
        super(AxisChangedController, self).ylim_changed(ax)
        if not self._changing and self.lim_changed(ax.get_ylim(), self._y_lim):
            self._changing = True
            self._y_lim = ax.get_ylim()
            self.update(ax)
            self._changing = False

    def extent(self, lim):
        return numpy.subtract(*lim)

    def lim_changed(self, axlim, savedlim):
        axextent = self.extent(axlim)
        extent = self.extent(savedlim)
        lim_changed = ((axextent / extent) < self._lim_ratio_threshold ** 2
                       or (extent / axextent) < self._lim_ratio_threshold ** 2
                       or ((1 - (self.extent((axlim[0], savedlim[0])) / self.extent((savedlim[0], axlim[1]))))
                           < self._lim_ratio_threshold)
                       or ((1 - (self.extent((savedlim[0], axlim[0])) / self.extent((axlim[0], savedlim[1]))))
                           < self._lim_ratio_threshold)
                       )
        return lim_changed

    def _buffer_lim(self, lim):
        # buffer_size = 1 - self._lim_ratio_threshold
        # extent = self.extent(lim)
        return lim


class BufferedAxisChangedController(AxisChangedController):
    def __init__(self, ax, plot_function, plot_limits, resolution=50, update_lim=None, **kwargs):
        """
        Buffered axis changed controller. Controls the buffer and handles update events for when the axes changed.

        Updated plotting will be after first reload (first time will be within plot limits, after that the limits will be buffered)

        :param plot_function:
            function to use for creating image for plotting (return ndarray-like)
            plot_function gets called with (2D!) Xtest grid if replotting required
        :type plot_function: function
        :param plot_limits:
            beginning plot limits [xmin, ymin, xmax, ymax]

        :param kwargs: additional kwargs are for pyplot.imshow(**kwargs)
        """
        super(BufferedAxisChangedController, self).__init__(ax, update_lim=update_lim)
        self.resolution = resolution
        self.plot_function = plot_function
        xmin, xmax, ymin, ymax = plot_limits#self._x_lim # self._compute_buffered(*self._x_lim)
        # imshow acts on the limits of the plot, this is why we need to override the limits here, to make sure the right plot limits are used:
        self._x_lim = xmin, xmax
        self._y_lim = ymin, ymax
        self.view = self._init_view(self.ax, self.recompute_X(buffered=False), xmin, xmax, ymin, ymax, **kwargs)

    def update(self, ax):
        super(BufferedAxisChangedController, self).update(ax)
        xmin, xmax = self._compute_buffered(*self._x_lim)
        ymin, ymax = self._compute_buffered(*self._y_lim)
        self.update_view(self.view, self.recompute_X(), xmin, xmax, ymin, ymax)

    def _init_view(self, ax, X, xmin, xmax, ymin, ymax):
        raise NotImplementedError('return view for this controller')

    def update_view(self, view, X, xmin, xmax, ymin, ymax):
        raise NotImplementedError('update view given in here')

    def get_grid(self, buffered=True):
        if buffered: comp = self._compute_buffered
        else: comp = lambda a,b: (a,b)
        xmin, xmax = comp(*self._x_lim)
        ymin, ymax = comp(*self._y_lim)
        x, y = numpy.mgrid[xmin:xmax:1j * self.resolution, ymin:ymax:1j * self.resolution]
        return numpy.hstack((x.flatten()[:, None], y.flatten()[:, None]))

    def recompute_X(self, buffered=True):
        return self.plot_function(self.get_grid(buffered))

    def _compute_buffered(self, mi, ma):
        buffersize = self._buffersize()
        size = ma - mi
        return mi - (buffersize * size), ma + (buffersize * size)

    def _buffersize(self):
        try:
            buffersize = 1. - self._lim_ratio_threshold
        except:
            buffersize = .4
        return buffersize
