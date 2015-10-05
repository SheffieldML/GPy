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
        fsamples = self.posterior_samples(Xgrid, full_cov=True, size=samples, **predict_kw)
        fsamples = fsamples[which_data_ycols] if fsamples.ndim == 3 else fsamples
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
            if fsamples is not None and fsamples.ndim == 3:
                for s in range(fsamples.shape[-1]):
                    fsamples[i, :, s] = self.likelihood.gp_link.transf(fsamples[i, :, s])
            elif fsamples is not None:
                for s in range(fsamples.shape[-1]):
                    fsamples[:, s] = self.likelihood.gp_link.transf(fsamples[:, s])
    return retmu, percs, fsamples

def helper_for_plot_data(self, plot_limits, visible_dims, fixed_inputs, resolution):
    """
    Figure out the data, free_dims and create an Xgrid for
    the prediction. 
    """
    X, Xvar, Y = get_x_y_var(self)

    #work out what the inputs are for plotting (1D or 2D)
    if fixed_inputs is None:
        fixed_inputs = []
    fixed_dims = get_fixed_dims(self, fixed_inputs)
    free_dims = get_free_dims(self, visible_dims, fixed_dims)
    
    if len(free_dims) == 1:
        #define the frame on which to plot
        resolution = resolution or 200
        Xnew, xmin, xmax = x_frame1D(X[:,free_dims], plot_limits=plot_limits, resolution=resolution)
        Xgrid = np.empty((Xnew.shape[0],self.input_dim))
        Xgrid[:,free_dims] = Xnew
        for i,v in fixed_inputs:
            Xgrid[:,i] = v
        x = Xgrid
        y = None
    elif len(free_dims) == 2:
        #define the frame for plotting on
        resolution = resolution or 50
        Xnew, x, y, xmin, xmax = x_frame2D(X[:,free_dims], plot_limits, resolution)
        Xgrid = np.empty((Xnew.shape[0],self.input_dim))
        Xgrid[:,free_dims] = Xnew
        for i,v in fixed_inputs:
            Xgrid[:,i] = v    
    return X, Xvar, Y, fixed_dims, free_dims, Xgrid, x, y, xmin, xmax, resolution


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
    The the data from a model as 
    X the inputs, 
    X_variance the variance of the inputs ([default: None])
    and Y the outputs
    
    :returns: (X, X_variance, Y) 
    """
    if hasattr(model, 'has_uncertain_inputs') and model.has_uncertain_inputs():
        X = model.X.mean
        X_variance = model.X.variance
    else:
        X = model.X
        X_variance = None
    Y = model.Y
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
    visible_dims = np.asanyarray(visible_dims)
    if fixed_dims is not None:
        return np.setdiff1d(visible_dims, fixed_dims)
    return visible_dims

def get_fixed_dims(model, fixed_inputs):
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
        from ...core.parameterization.variational import VariationalPosterior
        if isinstance(X, VariationalPosterior):
            xmin,xmax = X.mean.min(0),X.mean.max(0)
        else:
            xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError("Bad limits for plotting")

    Xnew = np.linspace(xmin,xmax,resolution or 200)[:,None]
    return Xnew, xmin, xmax

def x_frame2D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1]==2, "x_frame2D is defined for two-dimensional inputs"
    if plot_limits is None:
        xmin, xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    elif len(plot_limits)==4:
        xmin, xmax = (plot_limits[0], plot_limits[2]), (plot_limits[1], plot_limits[3])
    else:
        raise ValueError("Bad limits for plotting")

    resolution = resolution or 50
    xx, yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
    Xnew = np.vstack((xx.flatten(),yy.flatten())).T
    return Xnew, xx, yy, xmin, xmax