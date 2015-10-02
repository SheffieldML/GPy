#===============================================================================
# Copyright (c) 2012-2015 GPy Authors
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
    assert X.shape[1] ==2, "x_frame2D is defined for two-dimensional inputs"
    if plot_limits is None:
        xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError("Bad limits for plotting")

    resolution = resolution or 50
    xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
    Xnew = np.vstack((xx.flatten(),yy.flatten())).T
    return Xnew, xx, yy, xmin, xmax
