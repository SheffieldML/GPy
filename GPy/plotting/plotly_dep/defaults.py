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

from .. import Tango
from plotly.graph_objs import Line

'''
This file is for defaults for the gpy plot, specific to the plotting library.

Create a kwargs dictionary with the right name for the plotting function
you are implementing. If you do not provide defaults, the default behaviour of
the plotting library will be used.

In the code, always ise plotting.gpy_plots.defaults to get the defaults, as
it gives back an empty default, when defaults are not defined.
'''

# Data plots:
data_1d = dict(marker_kwargs=dict(), marker='x', color='black')
data_2d = dict(marker='o', cmap='Hot', marker_kwargs=dict(opacity=1., size='5', line=Line(width=.5, color='black')))
inducing_1d = dict(color=Tango.colorsHex['darkRed'])
inducing_2d = dict(marker_kwargs=dict(size='5', opacity=.7, line=Line(width=.5, color='black')), opacity=.7, color='white', marker='star-triangle-up')
inducing_3d = dict(marker_kwargs=dict(symbol='diamond', size='5', opacity=.7, line=Line(width=.1, color='black')), color='#F5F5F5')
xerrorbar = dict(color='black', error_kwargs=dict(thickness=.5), opacity=.5)
yerrorbar = dict(color=Tango.colorsHex['darkRed'], error_kwargs=dict(thickness=.5), opacity=.5)
#
# # GP plots:
meanplot_1d = dict(color=Tango.colorsHex['mediumBlue'], line_kwargs=dict(width=2))
meanplot_2d = dict(colorscale='Hot')
meanplot_3d = dict(colorscale='Hot', opacity=.9)
samples_1d = dict(color=Tango.colorsHex['mediumBlue'], line_kwargs=dict(width=.3))
samples_3d = dict(cmap='Hot', opacity=.5)
confidence_interval = dict(mode='lines', line_kwargs=dict(color=Tango.colorsHex['darkBlue'], width=.4),
                           color=Tango.colorsHex['lightBlue'], opacity=.3)
# density = dict(alpha=.5, color=Tango.colorsHex['lightBlue'])
#
# # GPLVM plots:
# data_y_1d = dict(linewidth=0, cmap='RdBu', s=40)
# data_y_1d_plot = dict(color='k', linewidth=1.5)
#
# # Kernel plots:
ard = dict(linewidth=1.2, barmode='stack')
#
# # Input plots:
latent = dict(colorscale='Greys', reversescale=True, zsmooth='best')
gradient = dict(colorscale='RdBu', opacity=.7)
magnification = dict(colorscale='Greys', zsmooth='best', reversescale=True)
latent_scatter = dict(marker_kwargs=dict(size='5', opacity=.7))
# annotation = dict(fontdict=dict(family='sans-serif', weight='light', fontsize=9), zorder=.3, alpha=.7)
