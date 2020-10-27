#===============================================================================
# Copyright (c) 2012-2015, GPy authors (see AUTHORS.txt).
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
# * Neither the name of GPy nor the names of its
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
from scipy import sparse
import itertools
from ...models import WarpedGP

def in_ipynb():
    try:
        cfg = get_ipython().config
        return 'IPKernelApp' in cfg
    except NameError:
        return False

def find_best_layout_for_subplots(num_subplots):
    r, c = 1, 1
    while (r*c) < num_subplots:
        if (c==(r+1)) or (r==c):
            c += 1
        elif c==(r+2):
            r += 1
            c -= 1
    return r, c

def helper_predict_with_model(self, Xgrid, plot_raw, apply_link, percentiles, which_data_ycols, predict_kw, samples=0):
    """
    Make the right decisions for prediction with a model
    based on the standard arguments of plotting.

    This is quite complex and will take a while to understand,
    so do not change anything in here lightly!!!
    """
    # Put some standards into the predict_kw so that prediction is done automatically:
    if predict_kw is None:
        predict_kw = {}
    if 'likelihood' not in predict_kw:
        if plot_raw:
            from ...likelihoods import Gaussian
            from ...likelihoods.link_functions import Identity
            lik = Gaussian(Identity(), 1e-9) # Make the likelihood not add any noise
        else:
            lik = None
        predict_kw['likelihood'] = lik
    if 'Y_metadata' not in predict_kw:
        predict_kw['Y_metadata'] = {}
    if 'output_index' not in predict_kw['Y_metadata']:
        predict_kw['Y_metadata']['output_index'] = Xgrid[:,-1:].astype(np.int)

    mu, _ = self.predict(Xgrid, **predict_kw)

    if percentiles is not None:
        percentiles = self.predict_quantiles(Xgrid, quantiles=percentiles, **predict_kw)
    else: percentiles = []

    if samples > 0:
        fsamples = self.posterior_samples(Xgrid, size=samples, **predict_kw)
        fsamples = fsamples[:, which_data_ycols, :]
    else:
        fsamples = None

    # Filter out the ycolums which we want to plot:
    retmu = mu[:, which_data_ycols]
    percs = [p[:, which_data_ycols] for p in percentiles]

    if plot_raw and apply_link:
        for i in range(len(which_data_ycols)):
            retmu[:, [i]] = self.likelihood.gp_link.transf(mu[:, [i]])
            for perc in percs:
                perc[:, [i]] = self.likelihood.gp_link.transf(perc[:, [i]])
            if fsamples is not None:
                for s in range(fsamples.shape[-1]):
                    fsamples[:, i, s] = self.likelihood.gp_link.transf(fsamples[:, i, s])
    return retmu, percs, fsamples

def helper_for_plot_data(self, X, plot_limits, visible_dims, fixed_inputs, resolution):
    """
    Figure out the data, free_dims and create an Xgrid for
    the prediction.

    This is only implemented for two dimensions for now!
    """
    #work out what the inputs are for plotting (1D or 2D)
    if fixed_inputs is None:
        fixed_inputs = []
    fixed_dims = get_fixed_dims(fixed_inputs)
    free_dims = get_free_dims(self, visible_dims, fixed_dims)

    if len(free_dims) == 1:
        #define the frame on which to plot
        resolution = resolution or 200
        Xnew, xmin, xmax = x_frame1D(X[:,free_dims], plot_limits=plot_limits, resolution=resolution)
        Xgrid = np.zeros((Xnew.shape[0],self.input_dim))
        Xgrid[:,free_dims] = Xnew
        for i,v in fixed_inputs:
            Xgrid[:,i] = v
        x = Xgrid
        y = None
    elif len(free_dims) == 2:
        #define the frame for plotting on
        resolution = resolution or 35
        Xnew, x, y, xmin, xmax = x_frame2D(X[:,free_dims], plot_limits, resolution)
        Xgrid = np.zeros((Xnew.shape[0], self.input_dim))
        Xgrid[:,free_dims] = Xnew
        #xmin = Xgrid.min(0)[free_dims]
        #xmax = Xgrid.max(0)[free_dims]
        for i,v in fixed_inputs:
            Xgrid[:,i] = v
    else:
        raise TypeError("calculated free_dims {} from visible_dims {} and fixed_dims {} is neither 1D nor 2D".format(free_dims, visible_dims, fixed_dims))
    return fixed_dims, free_dims, Xgrid, x, y, xmin, xmax, resolution

def scatter_label_generator(labels, X, visible_dims, marker=None):
    ulabels = []
    for lab in labels:
        if not lab in ulabels:
            ulabels.append(lab)
    if marker is not None:
        marker = itertools.cycle(list(marker))
    else:
        m = None

    try:
        input_1, input_2, input_3 = visible_dims
    except:
        try:
            # tuple or int?
            input_1, input_2 = visible_dims
            input_3 = None
        except:
            input_1 = visible_dims
            input_2 = input_3 = None

    for ul in ulabels:
        from numbers import Number
        if isinstance(ul, str):
            try:
                this_label = unicode(ul)
            except NameError:
                #python3
                this_label = ul
        elif isinstance(ul, Number):
            this_label = 'class {!s}'.format(ul)
        else:
            this_label = ul

        if marker is not None:
            m = next(marker)

        index = np.nonzero(labels == ul)[0]

        if input_2 is None:
            x = X[index, input_1]
            y = np.zeros(index.size)
            z = None
        elif input_3 is None:
            x = X[index, input_1]
            y = X[index, input_2]
            z = None
        else:
            x = X[index, input_1]
            y = X[index, input_2]
            z = X[index, input_3]

        yield x, y, z, this_label, index, m

def subsample_X(X, labels, num_samples=1000):
    """
    Stratified subsampling if labels are given.
    This means due to rounding errors you might get a little differences between the
    num_samples and the returned subsampled X.
    """
    if X.shape[0] > num_samples:
        print("Warning: subsampling X, as it has more samples then {}. X.shape={!s}".format(int(num_samples), X.shape))
        if labels is not None:
            subsample = []
            for _, _, _, _, index, _ in scatter_label_generator(labels, X, (0, None, None)):
                subsample.append(np.random.choice(index, size=max(2, int(index.size*(float(num_samples)/X.shape[0]))), replace=False))
            subsample = np.hstack(subsample)
        else:
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
    return X, labels


def update_not_existing_kwargs(to_update, update_from):
    """
    This function updates the keyword aguments from update_from in
    to_update, only if the keys are not set in to_update.

    This is used for updated kwargs from the default dicts.
    """
    if to_update is None:
        to_update = {}
    to_update.update({k:v for k,v in update_from.items() if k not in to_update})
    return to_update

def get_x_y_var(model):
    """
    Either the the data from a model as
    X the inputs,
    X_variance the variance of the inputs ([default: None])
    and Y the outputs

    If (X, X_variance, Y) is given, this just returns.

    :returns: (X, X_variance, Y)
    """
    # model given
    if hasattr(model, 'has_uncertain_inputs') and model.has_uncertain_inputs():
        X = model.X.mean.values
        X_variance = model.X.variance.values
    else:
        try:
            X = model.X.values
        except AttributeError:
            X = model.X
        X_variance = None
    try:
        Y = model.Y.values
    except AttributeError:
        Y = model.Y

    if isinstance(model, WarpedGP) and not model.predict_in_warped_space:
        Y = model.Y_normalized
    
    if sparse.issparse(Y): Y = Y.todense().view(np.ndarray)
    return X, X_variance, Y

def get_free_dims(model, visible_dims, fixed_dims):
    """
    work out what the inputs are for plotting (1D or 2D)

    The visible dimensions are the dimensions, which are visible.
    the fixed_dims are the fixed dimensions for this.

    The free_dims are then the visible dims without the fixed dims.
    """
    if visible_dims is None:
        visible_dims = np.arange(model.input_dim)
    dims = np.asanyarray(visible_dims)
    if fixed_dims is not None:
        dims = [dim for dim in dims if dim not in fixed_dims]
    return np.asanyarray([dim for dim in dims if dim is not None])


def get_fixed_dims(fixed_inputs):
    """
    Work out the fixed dimensions from the fixed_inputs list of tuples.
    """
    return np.array([i for i,_ in fixed_inputs])

def get_which_data_ycols(model, which_data_ycols):
    """
    Helper to get the data columns to plot.
    """
    if which_data_ycols == 'all' or which_data_ycols is None:
        return np.arange(model.output_dim)
    return which_data_ycols

def get_which_data_rows(model, which_data_rows):
    """
    Helper to get the data rows to plot.
    """
    if which_data_rows == 'all' or which_data_rows is None:
        return slice(None)
    return which_data_rows

def x_frame1D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1] ==1, "x_frame1D is defined for one-dimensional inputs"
    if plot_limits is None:
        from GPy.core.parameterization.variational import VariationalPosterior
        if isinstance(X, VariationalPosterior):
            xmin,xmax = X.mean.min(0),X.mean.max(0)
        else:
            xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.25*(xmax-xmin), xmax+0.25*(xmax-xmin)
    elif len(plot_limits) == 2:
        xmin, xmax = map(np.atleast_1d, plot_limits)
    else:
        raise ValueError("Bad limits for plotting")

    Xnew = np.linspace(float(xmin),float(xmax),int(resolution) or 200)[:,None]
    return Xnew, xmin, xmax

def x_frame2D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1]==2, "x_frame2D is defined for two-dimensional inputs"
    if plot_limits is None:
        xmin, xmax = X.min(0), X.max(0)
        xmin, xmax = xmin-0.075*(xmax-xmin), xmax+0.075*(xmax-xmin)
    elif len(plot_limits) == 2:
        xmin, xmax = plot_limits
        try:
            xmin = xmin[0], xmin[1]
        except:
            # only one limit given, copy over to other lim
            xmin = [plot_limits[0], plot_limits[0]]
            xmax = [plot_limits[1], plot_limits[1]]
    elif len(plot_limits) == 4:
        xmin, xmax = (plot_limits[0], plot_limits[2]), (plot_limits[1], plot_limits[3])
    else:
        raise ValueError("Bad limits for plotting")

    resolution = resolution or 50
    xx, yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
    Xnew = np.c_[xx.flat, yy.flat]
    return Xnew, xx, yy, xmin, xmax
