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


#===============================================================================
# SKIPPING PLOTTING BECAUSE IT BEHAVES DIFFERENTLY ON DIFFERENT
# SYSTEMS, AND WILL MISBEHAVE
from nose import SkipTest
raise SkipTest("Skipping Matplotlib testing")
#===============================================================================

import matplotlib
from unittest.case import TestCase
matplotlib.use('agg')

import numpy as np
import GPy, os

from GPy.util.config import config
from GPy.plotting import change_plotting_library, plotting_library

class ConfigTest(TestCase):
    def tearDown(self):
        change_plotting_library('matplotlib')

    def test_change_plotting(self):
        self.assertRaises(ValueError, change_plotting_library, 'not+in9names')
        change_plotting_library('none')
        self.assertRaises(RuntimeError, plotting_library)

change_plotting_library('matplotlib')
if config.get('plotting', 'library') != 'matplotlib':
    raise SkipTest("Matplotlib not installed, not testing plots")

try:
    from matplotlib import cbook, pyplot as plt
    from matplotlib.testing.compare import compare_images
    from matplotlib.testing.noseclasses import ImageComparisonFailure
except ImportError:
    raise SkipTest("Matplotlib not installed, not testing plots")

extensions = ['png']

def _image_directories():
    """
    Compute the baseline and result image directories for testing *func*.
    Create the result directory if it doesn't exist.
    """
    basedir = os.path.dirname(os.path.relpath(os.path.abspath(__file__)))
    #module_name = __init__.__module__
    #mods = module_name.split('.')
    #basedir = os.path.join(*mods)
    result_dir = os.path.join(basedir, 'testresult','.')
    baseline_dir = os.path.join(basedir, 'baseline','.')
    if not os.path.exists(result_dir):
        cbook.mkdirs(result_dir)
    return baseline_dir, result_dir


def _sequenceEqual(a, b):
    assert len(a) == len(b), "Sequences not same length"
    for i, [x, y], in enumerate(zip(a, b)):
        assert x == y, "element not matching {}".format(i)

def _notFound(path):
    raise IOError('File {} not in baseline')

def _image_comparison(baseline_images, extensions=['pdf','svg','png'], tol=11):
    baseline_dir, result_dir = _image_directories()
    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            fig = plt.figure(num)
            fig.axes[0].set_axis_off()
            fig.set_frameon(False)
            fig.canvas.draw()
            fig.savefig(os.path.join(result_dir, "{}.{}".format(base, ext)), transparent=True, edgecolor='none', facecolor='none', bbox='tight')
    for num, base in zip(plt.get_fignums(), baseline_images):
        for ext in extensions:
            #plt.close(num)
            actual = os.path.join(result_dir, "{}.{}".format(base, ext))
            expected = os.path.join(baseline_dir, "{}.{}".format(base, ext))
            def do_test():
                err = compare_images(expected, actual, tol, in_decorator=True)
                if err:
                    raise ImageComparisonFailure("Error between {} and {} is {:.5f}, which is bigger then the tolerance of {:.5f}".format(actual, expected, err['rms'], tol))
            yield do_test
    plt.close('all')

def test_figure():
    np.random.seed(1239847)
    from GPy.plotting import plotting_library as pl
    #import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ax, _ = pl().new_canvas(num=1)
        def test_func(x):
            return x[:, 0].reshape(3,3)
        pl().imshow_interact(ax, test_func, extent=(-1,1,-1,1), resolution=3)

        ax, _ = pl().new_canvas()
        def test_func_2(x):
            y = x[:, 0].reshape(3,3)
            anno = np.argmax(x, axis=1).reshape(3,3)
            return y, anno

        pl().annotation_heatmap_interact(ax, test_func_2, extent=(-1,1,-1,1), resolution=3)
        pl().annotation_heatmap_interact(ax, test_func_2, extent=(-1,1,-1,1), resolution=3, imshow_kwargs=dict(interpolation='nearest'))

        ax, _ = pl().new_canvas(figsize=(4,3))
        x = np.linspace(0,1,100)
        y = [0,1,2]
        array = np.array([.4,.5])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('WhToColor', ('r', 'b'), N=array.size)

        pl().fill_gradient(ax, x, y, facecolors=['r', 'g'], array=array, cmap=cmap)

        ax, _ = pl().new_canvas(num=4, figsize=(4,3), projection='3d', xlabel='x', ylabel='y', zlabel='z', title='awsome title', xlim=(-1,1), ylim=(-1,1), zlim=(-3,3))
        z = 2-np.abs(np.linspace(-2,2,(100)))+1
        x, y = z*np.sin(np.linspace(-2*np.pi,2*np.pi,(100))), z*np.cos(np.linspace(-np.pi,np.pi,(100)))

        pl().plot(ax, x, y, z, linewidth=2)

        for do_test in _image_comparison(
                baseline_images=['coverage_{}'.format(sub) for sub in ["imshow_interact",'annotation_interact','gradient','3d_plot',]],
                extensions=extensions):
            yield (do_test, )


def test_kernel():
    np.random.seed(1239847)
    #import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = GPy.kern.RBF(5, ARD=True) * GPy.kern.Linear(3, active_dims=[0,2,4], ARD=True) + GPy.kern.Bias(2)
        k.randomize()
        k2 = GPy.kern.RBF(5, ARD=True) * GPy.kern.Linear(3, active_dims=[0,2,4], ARD=True) + GPy.kern.Bias(2) + GPy.kern.White(4)
        k2[:-1] = k[:]
        k2.plot_ARD(['rbf', 'linear', 'bias'], legend=True)
        k2.plot_covariance(visible_dims=[0, 3], plot_limits=(-1,3))
        k2.plot_covariance(visible_dims=[2], plot_limits=(-1, 3))
        k2.plot_covariance(visible_dims=[2, 4], plot_limits=((-1, 0), (5, 3)), projection='3d')
        k2.plot_covariance(visible_dims=[1, 4])
        for do_test in _image_comparison(
                baseline_images=['kern_{}'.format(sub) for sub in ["ARD", 'cov_2d', 'cov_1d', 'cov_3d', 'cov_no_lim']],
                extensions=extensions):
            yield (do_test, )

def test_plot():
    np.random.seed(111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.random.uniform(-2, 2, (40, 1))
        f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
        Y = f+np.random.normal(0, .1, f.shape)
        m = GPy.models.SparseGPRegression(X, Y, X_variance=np.ones_like(X)*[0.06])
        #m.optimize()
        m.plot_data()
        m.plot_mean()
        m.plot_confidence()
        m.plot_density()
        m.plot_errorbars_trainset()
        m.plot_samples()
        m.plot_data_error()
    for do_test in _image_comparison(baseline_images=['gp_{}'.format(sub) for sub in ["data", "mean", 'conf',
                                                                                      'density',
                                                                                      'out_error',
                                                                                      'samples', 'in_error']], extensions=extensions):
        yield (do_test, )

def test_twod():
    np.random.seed(11111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    X = np.random.uniform(-2, 2, (40, 2))
    f = .2 * np.sin(1.3*X[:,[0]]) + 1.3*np.cos(2*X[:,[1]])
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y, X_variance=np.ones_like(X)*[0.01, 0.2])
    #m.optimize()
    m.plot_data()
    m.plot_mean()
    m.plot_inducing()
    #m.plot_errorbars_trainset()
    m.plot_data_error()
    for do_test in _image_comparison(baseline_images=['gp_2d_{}'.format(sub) for sub in ["data", "mean",
                                                                                         'inducing',
                                                                                         #'out_error',
                                                                                         'in_error',
                                                                                         ]], extensions=extensions):
        yield (do_test, )

def test_threed():
    np.random.seed(11111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    X = np.random.uniform(-2, 2, (40, 2))
    f = .2 * np.sin(1.3*X[:,[0]]) + 1.3*np.cos(2*X[:,[1]])
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y)
    m.likelihood.variance = .1
    #m.optimize()
    m.plot_samples(projection='3d', samples=1)
    m.plot_samples(projection='3d', plot_raw=False, samples=1)
    plt.close('all')
    m.plot_data(projection='3d')
    m.plot_mean(projection='3d')
    m.plot_inducing(projection='3d')
    #m.plot_errorbars_trainset(projection='3d')
    for do_test in _image_comparison(baseline_images=['gp_3d_{}'.format(sub) for sub in ["data", "mean", 'inducing',
                                                                                         #'error',
                                                                                          #"samples", "samples_lik"
                                                                                          ]], extensions=extensions):
        yield (do_test, )

def test_sparse():
    np.random.seed(11111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    X = np.random.uniform(-2, 2, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPRegression(X, Y, X_variance=np.ones_like(X)*0.1)
    #m.optimize()
    #m.plot_inducing()
    m.plot_data()
    for do_test in _image_comparison(baseline_images=['sparse_gp_{}'.format(sub) for sub in ['data_error']], extensions=extensions):
        yield (do_test, )

def test_classification():
    np.random.seed(11111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    X = np.random.uniform(-2, 2, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.GPClassification(X, Y>Y.mean())
    #m.optimize()
    _, ax = plt.subplots()
    m.plot(plot_raw=False, apply_link=False, ax=ax)
    m.plot_errorbars_trainset(plot_raw=False, apply_link=False, ax=ax)
    _, ax = plt.subplots()
    m.plot(plot_raw=True, apply_link=False, ax=ax)
    m.plot_errorbars_trainset(plot_raw=True, apply_link=False, ax=ax)
    _, ax = plt.subplots()
    m.plot(plot_raw=True, apply_link=True, ax=ax)
    m.plot_errorbars_trainset(plot_raw=True, apply_link=True, ax=ax)
    for do_test in _image_comparison(baseline_images=['gp_class_{}'.format(sub) for sub in ["likelihood", "raw", 'raw_link']], extensions=extensions):
        yield (do_test, )


def test_sparse_classification():
    np.random.seed(11111)
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    X = np.random.uniform(-2, 2, (40, 1))
    f = .2 * np.sin(1.3*X) + 1.3*np.cos(2*X)
    Y = f+np.random.normal(0, .1, f.shape)
    m = GPy.models.SparseGPClassification(X, Y>Y.mean())
    #m.optimize()
    m.plot(plot_raw=False, apply_link=False, samples_likelihood=3)
    np.random.seed(111)
    m.plot(plot_raw=True, apply_link=False, samples=3)
    np.random.seed(111)
    m.plot(plot_raw=True, apply_link=True, samples=3)
    for do_test in _image_comparison(baseline_images=['sparse_gp_class_{}'.format(sub) for sub in ["likelihood", "raw", 'raw_link']], extensions=extensions):
        yield (do_test, )

def test_gplvm():
    from ..examples.dimensionality_reduction import _simulate_matern
    from ..kern import RBF
    from ..models import GPLVM
    np.random.seed(12345)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    Q = 3
    # Define dataset
    N = 10
    k1 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,10,10,0.1,0.1]), ARD=True)
    k2 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,0.1,10,0.1,10]), ARD=True)
    k3 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[0.1,0.1,10,10,10]), ARD=True)
    X = np.random.normal(0, 1, (N, 5))
    A = np.random.multivariate_normal(np.zeros(N), k1.K(X), Q).T
    B = np.random.multivariate_normal(np.zeros(N), k2.K(X), Q).T
    C = np.random.multivariate_normal(np.zeros(N), k3.K(X), Q).T

    Y = np.vstack((A,B,C))
    labels = np.hstack((np.zeros(A.shape[0]), np.ones(B.shape[0]), np.ones(C.shape[0])*2))

    k = RBF(Q, ARD=True, lengthscale=2)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    m = GPLVM(Y, Q, init="PCA", kernel=k)
    m.kern.lengthscale[:] = [1./.3, 1./.1, 1./.7]
    m.likelihood.variance = .001
    #m.optimize(messages=0)
    np.random.seed(111)
    m.plot_latent(labels=labels)
    np.random.seed(111)
    m.plot_scatter(projection='3d', labels=labels)
    np.random.seed(111)
    m.plot_magnification(labels=labels)
    m.plot_steepest_gradient_map(resolution=10, data_labels=labels)
    for do_test in _image_comparison(baseline_images=['gplvm_{}'.format(sub) for sub in ["latent", "latent_3d", "magnification", 'gradient']],
                                     extensions=extensions,
                                     tol=12):
        yield (do_test, )

def test_bayesian_gplvm():
    from ..examples.dimensionality_reduction import _simulate_matern
    from ..kern import RBF
    from ..models import BayesianGPLVM
    np.random.seed(12345)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams[u'figure.figsize'] = (4,3)
    matplotlib.rcParams[u'text.usetex'] = False
    Q = 3
    # Define dataset
    N = 10
    k1 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,10,10,0.1,0.1]), ARD=True)
    k2 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,0.1,10,0.1,10]), ARD=True)
    k3 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[0.1,0.1,10,10,10]), ARD=True)
    X = np.random.normal(0, 1, (N, 5))
    A = np.random.multivariate_normal(np.zeros(N), k1.K(X), Q).T
    B = np.random.multivariate_normal(np.zeros(N), k2.K(X), Q).T
    C = np.random.multivariate_normal(np.zeros(N), k3.K(X), Q).T

    Y = np.vstack((A,B,C))
    labels = np.hstack((np.zeros(A.shape[0]), np.ones(B.shape[0]), np.ones(C.shape[0])*2))

    k = RBF(Q, ARD=True, lengthscale=2)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    m = BayesianGPLVM(Y, Q, init="PCA", kernel=k)
    m.kern.lengthscale[:] = [1./.3, 1./.1, 1./.7]
    m.likelihood.variance = .001
    #m.optimize(messages=0)
    np.random.seed(111)
    m.plot_inducing(projection='2d')
    np.random.seed(111)
    m.plot_inducing(projection='3d')
    np.random.seed(111)
    m.plot_latent(projection='2d', labels=labels)
    np.random.seed(111)
    m.plot_scatter(projection='3d', labels=labels)
    np.random.seed(111)
    m.plot_magnification(labels=labels)
    np.random.seed(111)
    m.plot_steepest_gradient_map(resolution=10, data_labels=labels)
    for do_test in _image_comparison(baseline_images=['bayesian_gplvm_{}'.format(sub) for sub in ["inducing", "inducing_3d", "latent", "latent_3d", "magnification", 'gradient']], extensions=extensions):
        yield (do_test, )

if __name__ == '__main__':
    import nose
    nose.main()
