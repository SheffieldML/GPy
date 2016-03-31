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

from matplotlib import cm
from .. import Tango

'''
This file is for defaults for the gpy plot, specific to the plotting library.

Create a kwargs dictionary with the right name for the plotting function
you are implementing. If you do not provide defaults, the default behaviour of
the plotting library will be used.

In the code, always ise plotting.gpy_plots.defaults to get the defaults, as
it gives back an empty default, when defaults are not defined.
'''

# Data plots:
data_1d = dict(lw=1.5, marker='x', color='k')
data_2d = dict(s=35, edgecolors='none', linewidth=0., cmap=cm.get_cmap('hot'), alpha=.5)
inducing_1d = dict(lw=0, s=500, facecolors=Tango.colorsHex['darkRed'])
inducing_2d = dict(s=14, edgecolors='k', linewidth=.4, facecolors='white', alpha=.5, marker='^')
inducing_3d = dict(lw=.3, s=500, facecolors='white', edgecolors='k')
xerrorbar = dict(color='k', fmt='none', elinewidth=.5, alpha=.5)
yerrorbar = dict(color=Tango.colorsHex['darkRed'], fmt='none', elinewidth=.5, alpha=.5)

# GP plots:
meanplot_1d = dict(color=Tango.colorsHex['mediumBlue'], linewidth=2)
meanplot_2d = dict(cmap='hot', linewidth=.5)
meanplot_3d = dict(linewidth=0, antialiased=True, cstride=1, rstride=1, cmap='hot', alpha=.3)
samples_1d = dict(color=Tango.colorsHex['mediumBlue'], linewidth=.3)
samples_3d = dict(cmap='hot', alpha=.1, antialiased=True, cstride=1, rstride=1, linewidth=0)
confidence_interval = dict(edgecolor=Tango.colorsHex['darkBlue'], linewidth=.5, color=Tango.colorsHex['lightBlue'],alpha=.2)
density = dict(alpha=.5, color=Tango.colorsHex['lightBlue'])

# GPLVM plots:
data_y_1d = dict(linewidth=0, cmap='RdBu', s=40)
data_y_1d_plot = dict(color='k', linewidth=1.5)

# Kernel plots:
ard = dict(edgecolor='k', linewidth=1.2)

# Input plots:
latent = dict(aspect='auto', cmap='Greys', interpolation='bicubic')
gradient = dict(aspect='auto', cmap='RdBu', interpolation='nearest', alpha=.7)
magnification = dict(aspect='auto', cmap='Greys', interpolation='bicubic')
latent_scatter = dict(s=40, linewidth=.2, edgecolor='k', alpha=.9)
annotation = dict(fontdict=dict(family='sans-serif', weight='light', fontsize=9), zorder=.3, alpha=.7)
