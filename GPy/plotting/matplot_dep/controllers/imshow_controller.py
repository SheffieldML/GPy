'''
Created on 24 Jul 2013

@author: maxz
'''
from .axis_event_controller import BufferedAxisChangedController
import itertools
import numpy


class ImshowController(BufferedAxisChangedController):
    def __init__(self, ax, plot_function, plot_limits, resolution=50, update_lim=.9, **kwargs):
        """
        :param plot_function:
            function to use for creating image for plotting (return ndarray-like)
            plot_function gets called with (2D!) Xtest grid if replotting required
        :type plot_function: function
        :param plot_limits:
            beginning plot limits [xmin, ymin, xmax, ymax]

        :param kwargs: additional kwargs are for pyplot.imshow(**kwargs)
        """
        super(ImshowController, self).__init__(ax, plot_function, plot_limits, resolution, update_lim, **kwargs)

    def _init_view(self, canvas, X, xmin, xmax, ymin, ymax, vmin=None, vmax=None, **kwargs):
        #xoffset, yoffset = 0, 0#self._offsets(xmin, xmax, ymin, ymax)
        return canvas.imshow(X, extent=(xmin, xmax, 
                                        ymin, ymax),
                             vmin=vmin, vmax=vmax,
                             **kwargs)

    def update_view(self, view, X, xmin, xmax, ymin, ymax):
        view.set_data(X)
        xoffset, yoffset = 0, 0#self._offsets(xmin, xmax, ymin, ymax)
        view.set_extent((xmin-xoffset, xmax+xoffset, 
                         ymin-yoffset, ymax+yoffset))

    def _offsets(self, xmin, xmax, ymin, ymax):
        return float(xmax - xmin) / (2 * self.resolution), float(ymax - ymin) / (2 * self.resolution)


class ImAnnotateController(ImshowController):
    def __init__(self, ax, plot_function, plot_limits, resolution=20, update_lim=.99, imshow_kwargs=None, **kwargs):
        """
        :param plot_function:
            function to use for creating image for plotting (return ndarray-like)
            plot_function gets called with (2D!) Xtest grid if replotting required
        :type plot_function: function
        :param plot_limits:
            beginning plot limits [xmin, ymin, xmax, ymax]
        :param text_props: kwargs for pyplot.text(**text_props)
        :param kwargs: additional kwargs are for pyplot.imshow(**kwargs)
        """
        self.imshow_kwargs = imshow_kwargs or {}
        super(ImAnnotateController, self).__init__(ax, plot_function, plot_limits, resolution, update_lim, **kwargs)

    def _init_view(self, ax, X, xmin, xmax, ymin, ymax, **kwargs):
        view = [super(ImAnnotateController, self)._init_view(ax, X[0], xmin, xmax, ymin, ymax, **self.imshow_kwargs)]
        xoffset, yoffset = self._offsets(xmin, xmax, ymin, ymax)
        xlin = numpy.linspace(xmin, xmax, self.resolution, endpoint=False)
        ylin = numpy.linspace(ymin, ymax, self.resolution, endpoint=False)
        for [i, x], [j, y] in itertools.product(enumerate(xlin), enumerate(ylin)):
            view.append(ax.text(x+xoffset, y+yoffset, "{}".format(X[1][j, i]), ha='center', va='center', **kwargs))
        return view

    def update_view(self, view, X, xmin, xmax, ymin, ymax):
        super(ImAnnotateController, self).update_view(view[0], X[0], xmin, xmax, ymin, ymax)
        xoffset, yoffset = self._offsets(xmin, xmax, ymin, ymax)
        xlin = numpy.linspace(xmin, xmax, self.resolution, endpoint=False)
        ylin = numpy.linspace(ymin, ymax, self.resolution, endpoint=False)
        for [[i, x], [j, y]], text in zip(itertools.product(enumerate(xlin), enumerate(ylin)), view[1:]):
            text.set_x(x+xoffset)
            text.set_y(y+yoffset)
            text.set_text("{}".format(X[1][j, i]))
        return view
