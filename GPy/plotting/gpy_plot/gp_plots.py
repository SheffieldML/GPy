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

from . import pl
from . import update_not_existing_kwargs, defaults
from .util import x_frame1D
from scipy import sparse
import numpy as np

def plot_mean(self, plot_limits=None, fixed_inputs=[],
              resolution=None, plot_raw=False,
              Y_metadata=None, apply_link=False, 
              plot_uncertain_inputs=True, predict_kw=None, 
              **kwargs):
    """
    Plot the mean of a GP.
    
    :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
    :type plot_limits: np.array
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
    :type levels: int
    """
    if hasattr(self, 'has_uncertain_inputs') and self.has_uncertain_inputs():
        X = self.X.mean
        X_variance = self.X.variance
    else:
        X = self.X

    Y = self.Y
    
    if sparse.issparse(Y): Y = Y.todense().view(np.ndarray)

    if predict_kw is None:
        predict_kw = {}

    #work out what the inputs are for plotting (1D or 2D)
    fixed_dims = np.array([i for i,v in fixed_inputs])
    free_dims = np.setdiff1d(np.arange(self.input_dim),fixed_dims)
    
    #define the frame on which to plot
    Xnew, xmin, xmax = x_frame1D(X[:,free_dims], plot_limits=plot_limits, resolution=resolution or 200)
    Xgrid = np.empty((Xnew.shape[0],self.input_dim))
    Xgrid[:,free_dims] = Xnew
    for i,v in fixed_inputs:
        Xgrid[:,i] = v

    if plot_raw:
        mu = self._raw_predict(Xgrid)[0]
        
    update_not_existing_kwargs(kwargs, defaults.meanplot)
    return pl.plot(Xgrid, mu, **kwargs)

def gpplot(x, mu, lower, upper, edgecol='#3300FF', fillcol='#33CCFF', ax=None, fignum=None, **kwargs):
    _, axes = ax_default(fignum, ax)

    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    plots = []

    #here's the mean
    plots.append(meanplot(x, mu, edgecol, axes))

    #here's the box
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    plots.append(axes.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color=fillcol,**kwargs))

    #this is the edge:
    plots.append(meanplot(x, upper,color=edgecol, linewidth=0.2, ax=axes))
    plots.append(meanplot(x, lower,color=edgecol, linewidth=0.2, ax=axes))

    return plots
