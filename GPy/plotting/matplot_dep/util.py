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
# * Neither the name of GPy.plotting.matplot_dep.util nor the names of its
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

from matplotlib import pyplot as plt
import numpy as np

def legend_ontop(ax, mode='expand', ncol=3, fontdict=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    handles, labels = ax.get_legend_handles_labels()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "5%", pad=0)
    lgd = cax.legend(handles, labels, bbox_to_anchor=(0., 0., 1., 1.), loc=3,
            ncol=ncol, mode=mode, borderaxespad=0., prop=fontdict or {})
    cax.set_axis_off()
    #lgd = cax.legend(bbox_to_anchor=(0., 1.02, 1., 1.02), loc=3,
    #        ncol=ncol, mode=mode, borderaxespad=0., prop=fontdict or {})
    return lgd

def removeRightTicks(ax=None):
    ax = ax or plt.gca()
    for i, line in enumerate(ax.get_yticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)

def removeUpperTicks(ax=None):
    ax = ax or plt.gca()
    for i, line in enumerate(ax.get_xticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)

def fewerXticks(ax=None,divideby=2):
    ax = ax or plt.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])
    
def align_subplots(N,M,xlim=None, ylim=None):
    """make all of the subplots have the same limits, turn off unnecessary ticks"""
    #find sensible xlim,ylim
    if xlim is None:
        xlim = [np.inf,-np.inf]
        for i in range(N*M):
            plt.subplot(N,M,i+1)
            xlim[0] = min(xlim[0],plt.xlim()[0])
            xlim[1] = max(xlim[1],plt.xlim()[1])
    if ylim is None:
        ylim = [np.inf,-np.inf]
        for i in range(N*M):
            plt.subplot(N,M,i+1)
            ylim[0] = min(ylim[0],plt.ylim()[0])
            ylim[1] = max(ylim[1],plt.ylim()[1])

    for i in range(N*M):
        plt.subplot(N,M,i+1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if (i)%M:
            plt.yticks([])
        else:
            removeRightTicks()
        if i<(M*(N-1)):
            plt.xticks([])
        else:
            removeUpperTicks()

def align_subplot_array(axes,xlim=None, ylim=None):
    """
    Make all of the axes in the array hae the same limits, turn off unnecessary ticks
    use plt.subplots() to get an array of axes
    """
    #find sensible xlim,ylim
    if xlim is None:
        xlim = [np.inf,-np.inf]
        for ax in axes.flatten():
            xlim[0] = min(xlim[0],ax.get_xlim()[0])
            xlim[1] = max(xlim[1],ax.get_xlim()[1])
    if ylim is None:
        ylim = [np.inf,-np.inf]
        for ax in axes.flatten():
            ylim[0] = min(ylim[0],ax.get_ylim()[0])
            ylim[1] = max(ylim[1],ax.get_ylim()[1])

    N,M = axes.shape
    for i,ax in enumerate(axes.flatten()):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if (i)%M:
            ax.set_yticks([])
        else:
            removeRightTicks(ax)
        if i<(M*(N-1)):
            ax.set_xticks([])
        else:
            removeUpperTicks(ax)

def fixed_inputs(model, non_fixed_inputs, fix_routine='median', as_list=True, X_all=False):
    """
    Convenience function for returning back fixed_inputs where the other inputs
    are fixed using fix_routine
    :param model: model
    :type model: Model
    :param non_fixed_inputs: dimensions of non fixed inputs
    :type non_fixed_inputs: list
    :param fix_routine: fixing routine to use, 'mean', 'median', 'zero'
    :type fix_routine: string
    :param as_list: if true, will return a list of tuples with (dimension, fixed_val) otherwise it will create the corresponding X matrix
    :type as_list: boolean
    """
    f_inputs = []
    if hasattr(model, 'has_uncertain_inputs') and model.has_uncertain_inputs():
        X = model.X.mean.values.copy()
    elif isinstance(model.X, VariationalPosterior):
        X = model.X.values.copy()
    else:
        if X_all:
            X = model.X_all.copy()
        else:
            X = model.X.copy()
    for i in range(X.shape[1]):
        if i not in non_fixed_inputs:
            if fix_routine == 'mean':
                f_inputs.append( (i, np.mean(X[:,i])) )
            if fix_routine == 'median':
                f_inputs.append( (i, np.median(X[:,i])) )
            else: # set to zero zero
                f_inputs.append( (i, 0) )
            if not as_list:
                X[:,i] = f_inputs[-1][1]
    if as_list:
        return f_inputs
    else:
        return X
