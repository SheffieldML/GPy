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
# * Neither the name of GPy.plotting.gpy_plot.kernel_plots nor the names of its
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
from .. import Tango

def plot_ARD(kernel, filtering=None, **kwargs):
    """
    If an ARD kernel is present, plot a bar representation using matplotlib

    :param fignum: figure number of the plot
    :param filtering: list of names, which to use for plotting ARD parameters.
                      Only kernels which match names in the list of names in filtering
                      will be used for plotting.
    :type filtering: list of names to use for ARD plot
    """
    canvas, kwargs = pl.get_new_canvas(kwargs)

    Tango.reset()
    
    bars = []
    ard_params = np.atleast_2d(kernel.input_sensitivity(summarize=False))
    bottom = 0
    last_bottom = bottom

    x = np.arange(kernel.input_dim)

    if filtering is None:
        filtering = kernel.parameter_names(recursive=False)

    for i in range(ard_params.shape[0]):
        if kernel.parameters[i].name in filtering:
            c = Tango.nextMedium()
            bars.append(pl.barplot(canvas, x, ard_params[i,:], color=c, label=kernel.parameters[i].name, bottom=bottom))
            last_bottom = ard_params[i,:]
            bottom += last_bottom
        else:
            print("filtering out {}".format(kernel.parameters[i].name))

    plt.show_canvas()
    ax.set_xlim(-.5, kernel.input_dim - .5)
    add_bar_labels(fig, ax, [bars[-1]], bottom=bottom-last_bottom)

    if legend:
        if title is '':
            mode = 'expand'
            if len(bars) > 1:
                mode = 'expand'
            ax.legend(bbox_to_anchor=(0., 1.02, 1., 1.02), loc=3,
                      ncol=len(bars), mode=mode, borderaxespad=0.)
            fig.tight_layout(rect=(0, 0, 1, .9))
        else:
            ax.legend()

    return dict(barplots=bars)