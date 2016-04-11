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
from . import plotting_library as pl
from .plot_util import get_x_y_var,\
    update_not_existing_kwargs, \
    helper_for_plot_data, scatter_label_generator, subsample_X,\
    find_best_layout_for_subplots

def _wait_for_updates(view, updates):
    if view is not None:
        try:
            if updates:
                clear = raw_input('yes or enter to deactivate updates - otherwise still do updates - use plots[imshow].deactivate() to clear')
                if clear.lower() in 'yes' or clear == '':
                    view.deactivate()
            else:
                view.deactivate()
        except AttributeError:
            # No updateable view:
            pass
        except TypeError:
            # No updateable view:
            pass

def _new_canvas(self, projection, kwargs, which_indices):
    input_1, input_2, input_3 = sig_dims = self.get_most_significant_input_dimensions(which_indices)

    if input_3 is None:
        zlabel = None
    else:
        zlabel = 'latent dimension %i' % input_3
    canvas, kwargs = pl().new_canvas(projection=projection, xlabel='latent dimension %i' % input_1,
        ylabel='latent dimension %i' % input_2,
        zlabel=zlabel, **kwargs)
    return canvas, projection, kwargs, sig_dims

def _plot_latent_scatter(canvas, X, visible_dims, labels, marker, num_samples, projection='2d', **kwargs):
    from .. import Tango
    Tango.reset()
    X, labels = subsample_X(X, labels, num_samples)
    scatters = []
    generate_colors = 'color' not in kwargs
    for x, y, z, this_label, _, m in scatter_label_generator(labels, X, visible_dims, marker):
        update_not_existing_kwargs(kwargs, pl().defaults.latent_scatter)
        if generate_colors:
            kwargs['color'] = Tango.nextMedium()
        if projection == '3d':
            scatters.append(pl().scatter(canvas, x, y, Z=z, marker=m, label=this_label, **kwargs))
        else: scatters.append(pl().scatter(canvas, x, y, marker=m, label=this_label, **kwargs))
    return scatters

def plot_latent_scatter(self, labels=None,
                        which_indices=None,
                        legend=True,
                        plot_limits=None,
                        marker='<>^vsd',
                        num_samples=1000,
                        projection='2d',
                        **kwargs):
    """
    Plot a scatter plot of the latent space.

    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param bool legend: whether to plot the legend on the figure
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param str marker: markers to use - cycle if more labels then markers are given
    :param kwargs: the kwargs for the scatter plots
    """
    canvas, projection, kwargs, sig_dims = _new_canvas(self, projection, kwargs, which_indices)

    X, _, _ = get_x_y_var(self)
    if labels is None:
        labels = np.ones(self.num_data)
        legend = False
    else:
        legend = find_best_layout_for_subplots(len(np.unique(labels)))[1]
    scatters = _plot_latent_scatter(canvas, X, sig_dims, labels, marker, num_samples, projection=projection, **kwargs)
    return pl().add_to_canvas(canvas, dict(scatter=scatters), legend=legend)


def plot_latent_inducing(self,
                        which_indices=None,
                        legend=False,
                        plot_limits=None,
                        marker='^',
                        num_samples=1000,
                        projection='2d',
                        **kwargs):
    """
    Plot a scatter plot of the inducing inputs.

    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param bool legend: whether to plot the legend on the figure
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param str marker: markers to use - cycle if more labels then markers are given
    :param kwargs: the kwargs for the scatter plots
    """
    canvas, projection, kwargs, sig_dims = _new_canvas(self, projection, kwargs, which_indices)

    Z = self.Z.values
    labels = np.array(['inducing'] * Z.shape[0])
    kwargs['marker'] = marker
    update_not_existing_kwargs(kwargs, pl().defaults.inducing_2d)  # @UndefinedVariable
    scatters = _plot_latent_scatter(canvas, Z, sig_dims, labels, num_samples=num_samples, projection=projection, **kwargs)
    return pl().add_to_canvas(canvas, dict(scatter=scatters), legend=legend)






def _plot_magnification(self, canvas, which_indices, Xgrid,
                        xmin, xmax, resolution, updates,
                        mean=True, covariance=True,
                        kern=None,
                        **imshow_kwargs):
    def plot_function(x):
        Xtest_full = np.zeros((x.shape[0], Xgrid.shape[1]))
        Xtest_full[:, which_indices] = x

        mf = self.predict_magnification(Xtest_full, kern=kern, mean=mean, covariance=covariance)
        return mf.reshape(resolution, resolution).T
    imshow_kwargs = update_not_existing_kwargs(imshow_kwargs, pl().defaults.magnification)
    try:
        if updates:
            return pl().imshow_interact(canvas, plot_function, (xmin[0], xmax[0], xmin[1], xmax[1]), resolution=resolution, **imshow_kwargs)
        else: raise NotImplementedError
    except NotImplementedError:
        return pl().imshow(canvas, plot_function(Xgrid[:, which_indices]), (xmin[0], xmax[0], xmin[1], xmax[1]), **imshow_kwargs)

def plot_magnification(self, labels=None, which_indices=None,
                resolution=60, marker='<>^vsd', legend=True,
                plot_limits=None,
                updates=False,
                mean=True, covariance=True,
                kern=None, num_samples=1000,
                scatter_kwargs=None, plot_scatter=True,
                **imshow_kwargs):
    """
    Plot the magnification factor of the GP on the inputs. This is the
    density of the GP as a gray scale.

    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param int resolution: the resolution at which we predict the magnification factor
    :param str marker: markers to use - cycle if more labels then markers are given
    :param bool legend: whether to plot the legend on the figure
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param bool updates: if possible, make interactive updates using the specific library you are using
    :param bool mean: use the mean of the Wishart embedding for the magnification factor
    :param bool covariance: use the covariance of the Wishart embedding for the magnification factor
    :param :py:class:`~GPy.kern.Kern` kern: the kernel to use for prediction
    :param int num_samples: the number of samples to plot maximally. We do a stratified subsample from the labels, if the number of samples (in X) is higher then num_samples.
    :param imshow_kwargs: the kwargs for the imshow (magnification factor)
    :param kwargs: the kwargs for the scatter plots
    """
    input_1, input_2 = which_indices = self.get_most_significant_input_dimensions(which_indices)[:2]
    X = get_x_y_var(self)[0]
    _, _, Xgrid, _, _, xmin, xmax, resolution = helper_for_plot_data(self, X, plot_limits, which_indices, None, resolution)
    canvas, imshow_kwargs = pl().new_canvas(xlim=(xmin[0], xmax[0]), ylim=(xmin[1], xmax[1]),
                           xlabel='latent dimension %i' % input_1, ylabel='latent dimension %i' % input_2, **imshow_kwargs)
    plots = {}
    if legend and plot_scatter:
        if (labels is not None):
            legend = find_best_layout_for_subplots(len(np.unique(labels)))[1]
        else:
            labels = np.ones(self.num_data)
            legend = False
    if plot_scatter:
        plots['scatters'] = _plot_latent_scatter(canvas, X, which_indices, labels, marker, num_samples, projection='2d', **scatter_kwargs or {})
    plots['view'] = _plot_magnification(self, canvas, which_indices, Xgrid, xmin, xmax, resolution, updates, mean, covariance, kern, **imshow_kwargs)
    retval = pl().add_to_canvas(canvas, plots,
                           legend=legend,
                           )
    _wait_for_updates(plots['view'], updates)
    return retval




def _plot_latent(self, canvas, which_indices, Xgrid,
                        xmin, xmax, resolution, updates,
                        kern=None,
                        **imshow_kwargs):
    def plot_function(x):
        Xtest_full = np.zeros((x.shape[0], Xgrid.shape[1]))
        Xtest_full[:, which_indices] = x
        mf = self.predict(Xtest_full, kern=kern)[1]
        if mf.shape[1]==self.output_dim:
            mf = mf.sum(-1)
        else:
            mf *= self.output_dim
        mf = np.log(mf)
        return mf.reshape(resolution, resolution).T

    imshow_kwargs = update_not_existing_kwargs(imshow_kwargs, pl().defaults.latent)
    try:
        if updates:
            return pl().imshow_interact(canvas, plot_function, (xmin[0], xmax[0], xmin[1], xmax[1]), resolution=resolution, **imshow_kwargs)
        else: raise NotImplementedError
    except NotImplementedError:
        return pl().imshow(canvas, plot_function(Xgrid[:, which_indices]), (xmin[0], xmax[0], xmin[1], xmax[1]), **imshow_kwargs)

def plot_latent(self, labels=None, which_indices=None,
                resolution=60, legend=True,
                plot_limits=None,
                updates=False,
                kern=None, marker='<>^vsd',
                num_samples=1000, projection='2d',
                scatter_kwargs=None, **imshow_kwargs):
    """
    Plot the latent space of the GP on the inputs. This is the
    density of the GP posterior as a grey scale and the
    scatter plot of the input dimemsions selected by which_indices.

    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param int resolution: the resolution at which we predict the magnification factor
    :param bool legend: whether to plot the legend on the figure
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param bool updates: if possible, make interactive updates using the specific library you are using
    :param :py:class:`~GPy.kern.Kern` kern: the kernel to use for prediction
    :param str marker: markers to use - cycle if more labels then markers are given
    :param int num_samples: the number of samples to plot maximally. We do a stratified subsample from the labels, if the number of samples (in X) is higher then num_samples.
    :param imshow_kwargs: the kwargs for the imshow (magnification factor)
    :param scatter_kwargs: the kwargs for the scatter plots
    """
    if projection != '2d':
        raise ValueError('Cannot plot latent in other then 2 dimensions, consider plot_scatter')
    input_1, input_2 = which_indices = self.get_most_significant_input_dimensions(which_indices)[:2]
    X = get_x_y_var(self)[0]
    _, _, Xgrid, _, _, xmin, xmax, resolution = helper_for_plot_data(self, X, plot_limits, which_indices, None, resolution)
    canvas, imshow_kwargs = pl().new_canvas(xlim=(xmin[0], xmax[0]), ylim=(xmin[1], xmax[1]),
                           xlabel='latent dimension %i' % input_1, ylabel='latent dimension %i' % input_2, **imshow_kwargs)
    if legend:
        if (labels is not None):
            legend = find_best_layout_for_subplots(len(np.unique(labels)))[1]
        else:
            labels = np.ones(self.num_data)
            legend = False
    scatters = _plot_latent_scatter(canvas, X, which_indices, labels, marker, num_samples, projection='2d', **scatter_kwargs or {})
    view = _plot_latent(self, canvas, which_indices, Xgrid, xmin, xmax, resolution, updates, kern, **imshow_kwargs)
    retval = pl().add_to_canvas(canvas, dict(scatter=scatters, imshow=view), legend=legend)
    _wait_for_updates(view, updates)
    return retval

def _plot_steepest_gradient_map(self, canvas, which_indices, Xgrid,
                        xmin, xmax, resolution, output_labels, updates,
                        kern=None, annotation_kwargs=None,
                        **imshow_kwargs):
    if output_labels is None:
        output_labels = range(self.output_dim)
    def plot_function(x):
        Xgrid[:, which_indices] = x
        dmu_dX = np.sqrt(((self.predictive_gradients(Xgrid, kern=kern)[0])**2).sum(1))
        #dmu_dX = self.predictive_gradients(Xgrid, kern=kern)[0].sum(1)
        argmax = np.argmax(dmu_dX, 1).astype(int)
        return dmu_dX.max(1).reshape(resolution, resolution).T, np.array(output_labels)[argmax].reshape(resolution, resolution).T
    annotation_kwargs = update_not_existing_kwargs(annotation_kwargs or {}, pl().defaults.annotation)
    imshow_kwargs = update_not_existing_kwargs(imshow_kwargs or {}, pl().defaults.gradient)
    try:
        if updates:
            return dict(annotation=pl().annotation_heatmap_interact(canvas, plot_function, (xmin[0], xmax[0], xmin[1], xmax[1]), resolution=resolution, imshow_kwargs=imshow_kwargs, **annotation_kwargs))
        else:
            raise NotImplementedError
    except NotImplementedError:
        imshow, annotation = pl().annotation_heatmap(canvas, *plot_function(Xgrid[:, which_indices]), extent=(xmin[0], xmax[0], xmin[1], xmax[1]), imshow_kwargs=imshow_kwargs, **annotation_kwargs)
        return dict(heatmap=imshow, annotation=annotation)

def plot_steepest_gradient_map(self, output_labels=None, data_labels=None, which_indices=None,
                resolution=15, legend=True,
                plot_limits=None,
                updates=False,
                kern=None, marker='<>^vsd',
                num_samples=1000,
                annotation_kwargs=None, scatter_kwargs=None, **imshow_kwargs):

    """
    Plot the latent space of the GP on the inputs. This is the
    density of the GP posterior as a grey scale and the
    scatter plot of the input dimemsions selected by which_indices.

    :param array-like labels: a label for each data point (row) of the inputs
    :param (int, int) which_indices: which input dimensions to plot against each other
    :param int resolution: the resolution at which we predict the magnification factor
    :param bool legend: whether to plot the legend on the figure, if int plot legend columns on legend
    :param plot_limits: the plot limits for the plot
    :type plot_limits: (xmin, xmax, ymin, ymax) or ((xmin, xmax), (ymin, ymax))
    :param bool updates: if possible, make interactive updates using the specific library you are using
    :param :py:class:`~GPy.kern.Kern` kern: the kernel to use for prediction
    :param str marker: markers to use - cycle if more labels then markers are given
    :param int num_samples: the number of samples to plot maximally. We do a stratified subsample from the labels, if the number of samples (in X) is higher then num_samples.
    :param imshow_kwargs: the kwargs for the imshow (magnification factor)
    :param annotation_kwargs: the kwargs for the annotation plot
    :param scatter_kwargs: the kwargs for the scatter plots
    """
    input_1, input_2 = which_indices = self.get_most_significant_input_dimensions(which_indices)[:2]
    X = get_x_y_var(self)[0]
    _, _, Xgrid, _, _, xmin, xmax, resolution = helper_for_plot_data(self, X, plot_limits, which_indices, None, resolution)
    canvas, imshow_kwargs = pl().new_canvas(xlim=(xmin[0], xmax[0]), ylim=(xmin[1], xmax[1]),
                           xlabel='latent dimension %i' % input_1, ylabel='latent dimension %i' % input_2, **imshow_kwargs)
    if (data_labels is not None):
        legend = find_best_layout_for_subplots(len(np.unique(data_labels)))[1]
    else:
        data_labels = np.ones(self.num_data)
        legend = False
    plots = dict(scatter=_plot_latent_scatter(canvas, X, which_indices, data_labels, marker, num_samples, **scatter_kwargs or {}))
    plots.update(_plot_steepest_gradient_map(self, canvas, which_indices, Xgrid, xmin, xmax, resolution, output_labels, updates, kern, annotation_kwargs=annotation_kwargs, **imshow_kwargs))
    retval = pl().add_to_canvas(canvas, plots, legend=legend)
    _wait_for_updates(plots['annotation'], updates)
    return retval




