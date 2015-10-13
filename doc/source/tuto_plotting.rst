********************
Defining a new plotting function in GPy
********************

GPy has a wrapper for different plotting backends.
There are some functions you can use for standard plotting.
Anything going beyond the scope of the
:py:class:`~GPy.plotting.abstract_plotting_library.AbstractPlottingLibrary` classes plot definitions
should be considered carefully and maybe is a special case for your plotting library only.

All plotting related code lives in :py:mod:`GPy.plotting` and beneath. No plotting related code needs to be
anywhere else in GPy.

As examples are always the easiest way to learn how to, we 
will implement an example of a plotting function, which plots the covariance of a kernel.

Write your plotting function into a module under :py:mod:`GPy.plotting.gpy_plot` ``.<module_name>``
using the plotting routines provided in :py:func:`GPy.plotting.plotting_library`.
I like to ``from . import plotting_library as pl`` and the allways use ``pl().`` to access functionality of
the plotting library.

For the covariance plot we define the function in :py:mod:`GPy.plotting.kernel_plots`.

The first thing is to define the function parameters *and write the documentation for them*!
The first argument of the plotting function is always ``self`` for the class this plotting function
will be attached to (we will get to attaching the function to a class that in detail later on)::

 def plot_covariance(kernel, x=None, label=None,
              plot_limits=None, visible_dims=None, resolution=None,
              projection=None, levels=20, **kwargs):
     """
     Plot a kernel covariance w.r.t. another x.
 
     :param array-like x: the value to use for the other kernel argument (kernels are a function of two variables!)
     :param plot_limits: the range over which to plot the kernel
     :type plot_limits: Either (xmin, xmax) for 1D or (xmin, xmax, ymin, ymax) / ((xmin, xmax), (ymin, ymax)) for 2D
     :param array-like visible_dims: input dimensions (!) to use for x. Make sure to select 2 or less dimensions to plot.
     :resolution: the resolution of the lines used in plotting. for 2D this defines the grid for kernel evaluation.
     :param {2d|3d} projection: What projection shall we use to plot the kernel?
     :param int levels: for 2D projection, how many levels for the contour plot to use?
     :param kwargs:  valid kwargs for your specific plotting library
     """

Having defined the outline of the function we can start implementing
the real plotting.

First, we will write the necessary logic behind getting the covariance function.
This involves getting an Xgrid to plot with and the second x to compare the covariance to::

    from .plot_util import helper_for_plot_data
    X = np.ones((2, kernel.input_dim)) * [-4, 4]
    _, free_dims, Xgrid, xx, yy, _, _, resolution = helper_for_plot_data(kernel, X, plot_limits, visible_dims, None, resolution)
    from numbers import Number
    if x is None:
        x = np.zeros((1, kernel.input_dim))
    elif isinstance(x, Number):
        x = np.ones((1, kernel.input_dim))*x
    K = kernel.K(Xgrid, x)

``free_dims`` holds the free dimensions after selecting
from the visible_dims, ``Xgrid`` is the grid for the covariance,
``xx, yy`` are the grid positions for 2D plotting and ``x`` is the
``X2`` for the kernel and ``K`` holds the kernel covariance for
all positions between ``Xgrid`` and ``x``.

Then we need a canvas to plot on. Always push the keyword arguments
of the specifig library through :py:func:`GPy.plotting.abstract_plotting_library.AbstractPlottingLibrary.new_canvas`::

    if projection == '3d':
        zlabel = "k(X, {!s})" % (np.asanyarray(x).tolist())
        xlabel = 'X[:,0]'
        ylabel = 'X[:,1]'
    else:
        xlabel = 'X'
        ylabel = "k(X, {!s})" % (np.asanyarray(x).tolist())

    canvas, kwargs = pl().new_canvas(projection=projection, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

Also very important is to use the defaults, which are defined for all plotting libraries implemented.
This is done by updating the ``kwargs`` from the defaults. There is a helper function
which takes care for existing keyword arguments. In this case we will just use the default for
plotting a mean function for the covariance plot as well. If you want to define your own defaults
add them to the defaults for each library and add it in here. See for example the defaults for
matplotlib in :py:mod:`GPy.plotting.matplot_dep.defaults`. There is also the default for the
meanplot_1d, which we are for the 1d plot::

    from .plot_util import update_not_existing_kwargs
    update_not_existing_kwargs(kwargs, pl().defaults.meanplot_1d)  # @UndefinedVariable

The full definition of the plotting then looks like this::

    if len(free_dims)<=2:
        if len(free_dims)==1:
            # 1D plotting:
            update_not_existing_kwargs(kwargs, pl().defaults.meanplot_1d)  # @UndefinedVariable
            plots = dict(covariance=[pl().plot(canvas, Xgrid[:, free_dims], K, label=label, **kwargs)])
        else:
            if projection == '2d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_2d)  # @UndefinedVariable
                plots = dict(covariance=[pl().contour(canvas, xx[:, 0], yy[0, :],
                                               K.reshape(resolution, resolution),
                                               levels=levels, label=label, **kwargs)])
            elif projection == '3d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_3d)  # @UndefinedVariable
                plots = dict(covariance=[pl().surface(canvas, xx, yy,
                                               K.reshape(resolution, resolution),
                                               label=label,
                                               **kwargs)])
        return pl().add_to_canvas(canvas, plots)

    else:
        raise NotImplementedError("Cannot plot a kernel with more than two input dimensions")

Where we return whatever is returned by :py:func:`GPy.plotting.abstract_plotting_library.AbstractPlottingLibrary.add_to_canvas`,
so that the plotting library can choose what to do with the plot later, when we want to show it. In order
to show a plot, we can just call :py:func:`GPy.plotting.show` with the output of the plot above.

Now we want to add the plot to the :py:class:`GPy.kern.src.kern.Kern`. In order to do that, we inject the plotting function into the
class in the :py:mod:`GPy.plotting.__init__`, which will make sure that the on the fly change of the backend
works smoothly. Thus, in :py:mod:`GPy.plotting.__init__` we add the line::

    from ..kern import Kern
    Kern.plot_covariance = gpy_plot.kernel_plots.plot_covariance

And that's it. The plot can be shown in plotly by calling::

    GPy.plotting.change_plotting_library('plotly')

    k = GPy.kern.RBF(1) + GPy.kern.Matern32(1)
    k.randomize()
    fig = k.plot()
    GPy.plotting.show(fig, <plot_library specific **kwargs>)

    k = GPy.kern.RBF(2) + GPy.kern.Matern32(2)
    k.randomize()
    fig = k.plot()
    GPy.plotting.show(fig, <plot_library specific **kwargs>)

    k = GPy.kern.RBF(1) + GPy.kern.Matern32(2)
    k.randomize()
    fig = k.plot(projection='3d')
    GPy.plotting.show(fig, <plot_library specific **kwargs>)

This explains the next thing. Changing the backend works *on-the-fly*. To show the above example in matplotlib, we just
exchange the first line by ``GPy.plotting.change_plotting_library('matplotlib')``.
