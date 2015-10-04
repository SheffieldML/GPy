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
import numpy as np
import GPy, os, sys
from nose import SkipTest

try:
    from matplotlib import cbook
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
except:
    raise SkipTest("Matplotlib not installed, not testing plots")

def _image_directories(func):
    """
    Compute the baseline and result image directories for testing *func*.
    Create the result directory if it doesn't exist.
    """
    module_name = func.__module__

    path = module_name
    
    mods = module_name.split('.')
    subdir = os.path.join(*mods)
    
    basedir = os.path.join(*mods)
    
    result_dir = os.path.join(basedir, 'testresult')
    baseline_dir = os.path.join(basedir, 'baseline')

    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)

    return baseline_dir, result_dir

import matplotlib.testing.decorators
matplotlib.testing.decorators._image_directories = _image_directories
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

@image_comparison(baseline_images=['gp_{}'.format(sub) for sub in ["data", "mean", 'conf', 'density', 'error']], extensions=['pdf','png'])
def testPlot():
    np.random.seed(11111)
    X = np.random.uniform(0, 1, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.GPRegression(X, Y)
    m.optimize()
    m.plot_data()
    m.plot_mean()
    m.plot_confidence()
    m.plot_density()
    m.plot_errorbars_trainset()

@image_comparison(baseline_images=['sparse_gp_{}'.format(sub) for sub in ["data", "mean", 'conf', 'density', 'error', 'inducing']], extensions=['pdf','png'])
def testPlotSparse():
    np.random.seed(11111)
    X = np.random.uniform(0, 1, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y)
    m.optimize()
    m.plot_data()
    m.plot_mean()
    m.plot_confidence()
    m.plot_density()
    m.plot_errorbars_trainset()
    m.plot_inducing()

@image_comparison(baseline_images=['gp_class_{}'.format(sub) for sub in ["", "raw", 'link', 'raw_link']], extensions=['pdf','png'])
def testPlotClassification():
    np.random.seed(11111)
    X = np.random.uniform(0, 1, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.GPClassification(X, Y>Y.mean())
    m.optimize()
    m.plot()
    m.plot(plot_raw=True)
    m.plot(plot_raw=False, apply_link=True)
    m.plot(plot_raw=True, apply_link=True)

@image_comparison(baseline_images=['sparse_gp_class_{}'.format(sub) for sub in ["", "raw", 'link', 'raw_link']], extensions=['pdf','png'])
def testPlotSparseClassification():
    np.random.seed(11111)
    X = np.random.uniform(0, 1, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPClassification(X, Y>Y.mean())
    m.optimize()
    m.plot()
    m.plot(plot_raw=True)
    m.plot(plot_raw=False, apply_link=True)
    m.plot(plot_raw=True, apply_link=True)
