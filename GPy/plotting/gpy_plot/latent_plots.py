#===============================================================================
# Copyright (c) 2015, Max Zwiessele
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of GPy.plotting.gpy_plot.latent_plots nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================
import numpy as np
from . import pl
from .plot_util import get_x_y_var, get_free_dims, get_which_data_ycols,\
    get_which_data_rows, update_not_existing_kwargs, helper_predict_with_model,\
    helper_for_plot_data
import itertools

def plot_prediction_fit(self, plot_limits=None,
        which_data_rows='all', which_data_ycols='all', 
        fixed_inputs=None, resolution=None,
        plot_raw=False, apply_link=False, visible_dims=None,
        predict_kw=None, scatter_kwargs=None, **plot_kwargs):
    """
    Plot the fit of the (Bayesian)GPLVM latent space prediction to the outputs.
    This scatters two output dimensions against each other and a line
    from the prediction in two dimensions between them.
    
    Give the Y_metadata in the predict_kw if you need it.
    
    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param array-like which_data_ycols: which columns of y to plot (array-like or list of ints)
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input dimension i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param int resolution: The resolution of the prediction [defaults are 1D:200, 2D:50]
    :param bool plot_raw: plot the latent function (usually denoted f) only?
    :param bool apply_link: whether to apply the link function of the GP to the raw prediction.
    :param array-like visible_dims: which columns of the input X (!) to plot (array-like or list of ints)
    :param dict predict_kw: the keyword arguments for the prediction. If you want to plot a specific kernel give dict(kern=<specific kernel>) in here
    :param dict sactter_kwargs: kwargs for the scatter plot, specific for the plotting library you are using
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using
    """
    canvas, kwargs = pl.get_new_canvas(plot_kwargs)
    plots = _plot_prediction_fit(self, canvas, plot_limits, which_data_rows, which_data_ycols, 
                                 fixed_inputs, resolution, plot_raw, 
                                 apply_link, visible_dims,
                                 predict_kw, scatter_kwargs, **kwargs)
    return pl.show_canvas(canvas, plots)

def _plot_prediction_fit(self, canvas, plot_limits=None,
        which_data_rows='all', which_data_ycols='all', 
        fixed_inputs=None, resolution=None,
        plot_raw=False, apply_link=False, visible_dims=False,
        predict_kw=None, scatter_kwargs=None, **plot_kwargs):
    
    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    if visible_dims is None:
        visible_dims = self.get_most_significant_input_dimensions()[:1]

    X, _, Y, _, free_dims, Xgrid, _, _, _, _, resolution = helper_for_plot_data(self, plot_limits, visible_dims, fixed_inputs, resolution)
    
    plots = {}
    
    if len(free_dims)<2:
        if len(free_dims)==1:
            if scatter_kwargs is None:
                scatter_kwargs = {}
            update_not_existing_kwargs(scatter_kwargs, pl.defaults.data_y_1d)  # @UndefinedVariable
            plots['output'] = pl.scatter(canvas, Y[rows, ycols[0]], Y[rows, ycols[1]],
                                      c=X[rows, free_dims[0]],
                                      **scatter_kwargs)
            if predict_kw is None:
                predict_kw = {}
            mu, _, _ = helper_predict_with_model(self, Xgrid, plot_raw, 
                                              apply_link, None, 
                                              ycols, predict_kw)
            update_not_existing_kwargs(plot_kwargs, pl.defaults.data_y_1d_plot)  # @UndefinedVariable
            plots['output_fit'] = pl.plot(canvas, mu[:, 0], mu[:, 1], **plot_kwargs)
        else:
            pass #Nothing to plot!
    else:
        raise NotImplementedError("Cannot plot in more then one dimension.")
    return plots
    
def plot_magnification(self, labels=None, which_indices=None,
                resolution=60, legend=True,
                plot_limits=None,
                updates=False, 
                mean=True, covariance=True, 
                kern=None, marker='<>^vsd', imshow_kwargs=None, **kwargs):
    """
    Plot the magnification factor of the GP on the inputs. This is the 
    density of the GP as a gray scale.
    
    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param int resolution: the resolution at which we predict the magnification factor
    :param bool legend: whether to plot the legend on the figure
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param bool updates: if possible, make interactive updates using the specific library you are using
    :param bool mean: use the mean of the Wishart embedding for the magnification factor
    :param bool covariance: use the covariance of the Wishart embedding for the magnification factor
    :param :py:class:`~GPy.kern.Kern` kern: the kernel to use for prediction
    :param str marker: markers to use - cycle if more labels then markers are given
    :param imshow_kwargs: the kwargs for the imshow (magnification factor)
    :param kwargs: the kwargs for the scatter plots
    """
    input_1, input_2 = self.get_most_significant_input_dimensions(which_indices)

    #fethch the data points X that we'd like to plot
    X, _, _ = get_x_y_var(self)

    if plot_limits is None:
        xmin, ymin = X[:, [input_1, input_2]].min(0)
        xmax, ymax = X[:, [input_1, input_2]].max(0)
        x_r, y_r = xmax-xmin, ymax-ymin
        xmin -= .1*x_r
        xmax += .1*x_r
        ymin -= .1*y_r
        ymax += .1*y_r
    else:
        try:
            xmin, xmax, ymin, ymax = plot_limits
        except (TypeError, ValueError) as e:
            try:
                xmin, xmax = plot_limits
                ymin, ymax = xmin, xmax
            except (TypeError, ValueError) as e:
                raise e.__class__("Wrong plot limits: {} given -> need (xmin, xmax, ymin, ymax)".format(plot_limits))
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)

    from .. import Tango
    Tango.reset()
    
    if labels is None:
        labels = np.ones(self.num_data)

    if X.shape[0] > 1000:
        print("Warning: subsampling X, as it has more samples then 1000. X.shape={!s}".format(X.shape))
        subsample = np.random.choice(X.shape[0], size=1000, replace=False)
        X = X[subsample]
        labels = labels[subsample]
        #=======================================================================
        #     <<<WORK IN PROGRESS>>>
        #     <<<DO NOT DELETE>>>
        #     plt.close('all')
        #     fig, ax = plt.subplots(1,1)
        #     from GPy.plotting.matplot_dep.dim_reduction_plots import most_significant_input_dimensions
        #     import matplotlib.patches as mpatches
        #     i1, i2 = most_significant_input_dimensions(m, None)
        #     xmin, xmax = 100, -100
        #     ymin, ymax = 100, -100
        #     legend_handles = []
        #
        #     X = m.X.mean[:, [i1, i2]]
        #     X = m.X.variance[:, [i1, i2]]
        #
        #     xmin = X[:,0].min(); xmax = X[:,0].max()
        #     ymin = X[:,1].min(); ymax = X[:,1].max()
        #     range_ = [[xmin, xmax], [ymin, ymax]]
        #     ul = np.unique(labels)
        #
        #     for i, l in enumerate(ul):
        #         #cdict = dict(red  =[(0., colors[i][0], colors[i][0]), (1., colors[i][0], colors[i][0])],
        #         #             green=[(0., colors[i][0], colors[i][1]), (1., colors[i][1], colors[i][1])],
        #         #             blue =[(0., colors[i][0], colors[i][2]), (1., colors[i][2], colors[i][2])],
        #         #             alpha=[(0., 0., .0), (.5, .5, .5), (1., .5, .5)])
        #         #cmap = LinearSegmentedColormap('{}'.format(l), cdict)
        #         cmap = LinearSegmentedColormap.from_list('cmap_{}'.format(str(l)), [colors[i], colors[i]], 255)
        #         cmap._init()
        #         #alphas = .5*(1+scipy.special.erf(np.linspace(-2,2, cmap.N+3)))#np.log(np.linspace(np.exp(0), np.exp(1.), cmap.N+3))
        #         alphas = (scipy.special.erf(np.linspace(0,2.4, cmap.N+3)))#np.log(np.linspace(np.exp(0), np.exp(1.), cmap.N+3))
        #         cmap._lut[:, -1] = alphas
        #         print l
        #         x, y = X[labels==l].T
        #
        #         heatmap, xedges, yedges = np.histogram2d(x, y, bins=300, range=range_)
        #         #heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
        #
        #         im = ax.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, aspect='auto', interpolation='nearest', label=str(l))
        #         legend_handles.append(mpatches.Patch(color=colors[i], label=l))
        #     ax.set_xlim(xmin, xmax)
        #     ax.set_ylim(ymin, ymax)
        #     plt.legend(legend_handles, [l.get_label() for l in legend_handles])
        #     plt.draw()
        #     plt.show()
        #=======================================================================
    
    
    canvas, kwargs = pl.get_new_canvas(xlabel='latent dimension %i' % input_1, ylabel='latent dimension %i' % input_2, **kwargs)

    _, _, _, _, _, Xgrid, _, _, _, _, resolution = helper_for_plot_data(self, ((xmin, ymin), (xmax, ymax)), (input_1, input_2), None, resolution)
    
    def plot_function(x):
        Xtest_full = np.zeros((x.shape[0], X.shape[1]))
        Xtest_full[:, [input_1, input_2]] = x
        mf = self.predict_magnification(Xtest_full, kern=kern, mean=mean, covariance=covariance)
        return mf

    imshow_kwargs = update_not_existing_kwargs(imshow_kwargs, pl.defaults.magnification)
    Y = plot_function(Xgrid[:, [input_1, input_2]]).reshape(resolution, resolution).T[::-1, :]
    view = pl.imshow(canvas, Y, 
                     (xmin, ymin, xmax, ymax), 
                     None, plot_function, resolution,
                     vmin=Y.min(), vmax=Y.max(), 
                     **imshow_kwargs)
    
    # make sure labels are in order of input:
    ulabels = []
    for lab in labels:
        if not lab in ulabels:
            ulabels.append(lab)

    marker = itertools.cycle(list(marker))
    scatters = []

    for ul in ulabels:
        if type(ul) is np.string_:
            this_label = ul
        elif type(ul) is np.int64:
            this_label = 'class %i' % ul
        else:
            this_label = unicode(ul)
        m = marker.next()

        index = np.nonzero(labels == ul)[0]
        if self.input_dim == 1:
            x = X[index, input_1]
            y = np.zeros(index.size)
        else:
            x = X[index, input_1]
            y = X[index, input_2]
        update_not_existing_kwargs(kwargs, pl.defaults.latent_scatter)
        scatters.append(pl.scatter(canvas, x, y, marker=m, color=Tango.nextMedium(), label=this_label, **kwargs))
    
    plots = pl.show_canvas(canvas, dict(scatter=scatters, imshow=view), legend=legend, xlim=xlim, ylim=ylim)
    if updates:
        clear = raw_input('yes or enter to deactivate updates - otherwise still do updates - use plots[imshow].deactivate() to clear')
        if clear.lower() in 'yes' or clear == '':
            view.deactivate()
    else:
        view.deactivate()
    return plots
