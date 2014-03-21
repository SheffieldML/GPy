'''
Created on 24 Jul 2013

@author: maxz
'''
from axis_event_controller import BufferedAxisChangedController
import itertools
import numpy


class ImshowController(BufferedAxisChangedController):
    def __init__(self, ax, plot_function, plot_limits, resolution=50, update_lim=.8, **kwargs):
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

    def _init_view(self, ax, X, xmin, xmax, ymin, ymax, **kwargs):
        return ax.imshow(X, extent=(xmin, xmax,
                                    ymin, ymax),
                         vmin=X.min(),
                         vmax=X.max(),
                         **kwargs)

    def update_view(self, view, X, xmin, xmax, ymin, ymax):
        view.set_data(X)
        view.set_extent((xmin, xmax, ymin, ymax))

class ImAnnotateController(ImshowController):
    def __init__(self, ax, plot_function, plot_limits, resolution=20, update_lim=.99, **kwargs):
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
        super(ImAnnotateController, self).__init__(ax, plot_function, plot_limits, resolution, update_lim, **kwargs)

    def _init_view(self, ax, X, xmin, xmax, ymin, ymax, text_props={}, **kwargs):
        view = [super(ImAnnotateController, self)._init_view(ax, X[0], xmin, xmax, ymin, ymax, **kwargs)]
        xoffset, yoffset = self._offsets(xmin, xmax, ymin, ymax)
        xlin = numpy.linspace(xmin, xmax, self.resolution, endpoint=False)
        ylin = numpy.linspace(ymin, ymax, self.resolution, endpoint=False)
        for [i, x], [j, y] in itertools.product(enumerate(xlin), enumerate(ylin[::-1])):
            view.append(ax.text(x + xoffset, y + yoffset, "{}".format(X[1][j, i]), ha='center', va='center', **text_props))
        return view

    def update_view(self, view, X, xmin, xmax, ymin, ymax):
        super(ImAnnotateController, self).update_view(view[0], X[0], xmin, xmax, ymin, ymax)
        xoffset, yoffset = self._offsets(xmin, xmax, ymin, ymax)
        xlin = numpy.linspace(xmin, xmax, self.resolution, endpoint=False)
        ylin = numpy.linspace(ymin, ymax, self.resolution, endpoint=False)
        for [[i, x], [j, y]], text in itertools.izip(itertools.product(enumerate(xlin), enumerate(ylin[::-1])), view[1:]):
            text.set_x(x + xoffset)
            text.set_y(y + yoffset)
            text.set_text("{}".format(X[1][j, i]))
        return view

    def _offsets(self, xmin, xmax, ymin, ymax):
        return (xmax - xmin) / (2 * self.resolution), (ymax - ymin) / (2 * self.resolution)
