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
import GPy, os
from nose import SkipTest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.noseclasses import ImageComparisonFailure

try:
    from matplotlib import cbook, pyplot as plt
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
except:
    raise SkipTest("Matplotlib not installed, not testing plots")

extensions = ['png']

def _image_directories():
    """
    Compute the baseline and result image directories for testing *func*.
    Create the result directory if it doesn't exist.
    """
    basedir = os.path.splitext(os.path.relpath(os.path.abspath(__file__)))[0]
    #module_name = __init__.__module__
    #mods = module_name.split('.')
    #basedir = os.path.join(*mods)
    result_dir = os.path.join(basedir, 'testresult')
    baseline_dir = os.path.join(basedir, 'baseline')
    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)
    return baseline_dir, result_dir


def _sequenceEqual(a, b):
    assert len(a) == len(b), "Sequences not same length"
    for i, [x, y], in enumerate(zip(a, b)):
        assert x == y, "element not matching {}".format(i) 

def _notFound(path):
    raise IOError('File {} not in baseline')

def _image_comparison(baseline_images, extensions=['pdf','svg','ong'], tol=1e-3):
    baseline_dir, result_dir = _image_directories()
    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            fig = plt.figure(num)
            fig.axes[0].set_axis_off()
            fig.set_frameon(False)
            fig.canvas.draw()
            fig.savefig(os.path.join(result_dir, "{}.{}".format(base, ext)))
    plt.close('all')
    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            #plt.close(num)
            actual = os.path.join(result_dir, "{}.{}".format(base, ext))
            expected = os.path.join(baseline_dir, "{}.{}".format(base, ext))
            def do_test():
                err = compare_images(expected, actual, tol)
                try:
                    if not os.path.exists(expected):
                        raise ImageComparisonFailure(
                            'image does not exist: %s' % expected)
                    if err:
                        raise ImageComparisonFailure(
                            'images not close: {err[actual]!s} vs. {err[expected]!s} (RMS {err[rms]:.3f})'.format(err=err))
                except ImageComparisonFailure:
                    pass
            yield do_test

def test_plot(self=None):
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
    m.plot_samples()
    for do_test in _image_comparison(baseline_images=['gp_{}'.format(sub) for sub in ["data", "mean", 'conf', 'density', 'error', 'samples']], extensions=extensions):
        yield (do_test, )

def test_plot_sparse(self=None):
    np.random.seed(11111)
    X = np.random.uniform(-1, 1, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y)
    m.optimize()
    m.plot_inducing()
    for do_test in _image_comparison(baseline_images=['sparse_gp_{}'.format(sub) for sub in ['inducing']], extensions=extensions):
        yield (do_test, )

def test_plot_classification(self=None):
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
    for do_test in _image_comparison(baseline_images=['gp_class_{}'.format(sub) for sub in ["", "raw", 'link', 'raw_link']], extensions=extensions):
        yield (do_test, )

 
def test_plot_sparse_classification(self=None):
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
    for do_test in _image_comparison(baseline_images=['sparse_gp_class_{}'.format(sub) for sub in ["", "raw", 'link', 'raw_link']], extensions=extensions):
        yield (do_test, )

def test_gplvm_plot(self=None):
    from ..examples.dimensionality_reduction import _simulate_matern
    from ..kern import RBF
    from ..models import GPLVM
    Q = 3
    _, _, Ylist = _simulate_matern(5, 1, 1, 100, num_inducing=5, plot_sim=False)
    Y = Ylist[0]
    k = RBF(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    # k = kern.RBF(Q, ARD=True, lengthscale=10.)
    m = GPLVM(Y, Q, init="PCA", kernel=k)
    m.likelihood.variance = .1
    m.optimize(messages=0)
    labels = np.random.multinomial(1, np.random.dirichlet([.3333333, .3333333, .3333333]), size=(m.Y.shape[0])).nonzero()[1]
    m.plot_prediction_fit(which_data_ycols=(0,1)) # ignore this test, as plotting is not consistent!!
    plt.close('all')
    m.plot_latent()
    m.plot_magnification(labels=labels)
    m.plot_steepest_gradient_map(resolution=5)
    for do_test in _image_comparison(baseline_images=['gplvm_{}'.format(sub) for sub in ["latent", "magnification", 'gradient']], extensions=extensions):
        yield (do_test, )
        
if __name__ == '__main__':
    import nose
    nose.main()
