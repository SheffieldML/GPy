# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as _np
default_seed = 123344

# default_seed = _np.random.seed(123344)

def bgplvm_test_model(optimize=False, verbose=1, plot=False, output_dim=200, nan=False):
    """
    model for testing purposes. Samples from a GP with rbf kernel and learns
    the samples with a new kernel. Normally not for optimization, just model cheking
    """
    import GPy

    num_inputs = 13
    num_inducing = 5
    if plot:
        output_dim = 1
        input_dim = 3
    else:
        input_dim = 2
        output_dim = output_dim

    # generate GPLVM-like data
    X = _np.random.rand(num_inputs, input_dim)
    lengthscales = _np.random.rand(input_dim)
    k = GPy.kern.RBF(input_dim, .5, lengthscales, ARD=True)
    K = k.K(X)
    Y = _np.random.multivariate_normal(_np.zeros(num_inputs), K, (output_dim,)).T

    # k = GPy.kern.RBF_inv(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim)
    # k = GPy.kern.linear(input_dim)# + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim, 0.00001)
    # k = GPy.kern.RBF(input_dim, ARD = False)  + GPy.kern.white(input_dim, 0.00001)
    # k = GPy.kern.RBF(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.RBF(input_dim, .3, _np.ones(input_dim) * .2, ARD=True)
    # k = GPy.kern.RBF(input_dim, .5, 2., ARD=0) + GPy.kern.RBF(input_dim, .3, .2, ARD=0)
    # k = GPy.kern.RBF(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.linear(input_dim, _np.ones(input_dim) * .2, ARD=True)

    p = .3

    m = GPy.models.BayesianGPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)

    if nan:
        m.inference_method = GPy.inference.latent_function_inference.var_dtc.VarDTCMissingData()
        m.Y[_np.random.binomial(1, p, size=(Y.shape)).astype(bool)] = _np.nan
        m.parameters_changed()

    #===========================================================================
    # randomly obstruct data with percentage p
    #===========================================================================
    # m2 = GPy.models.BayesianGPLVMWithMissingData(Y_obstruct, input_dim, kernel=k, num_inducing=num_inducing)
    # m.lengthscales = lengthscales

    if plot:
        import matplotlib.pyplot as pb
        m.plot()
        pb.title('PCA initialisation')
        # m2.plot()
        # pb.title('PCA initialisation')

    if optimize:
        m.optimize('scg', messages=verbose)
        # m2.optimize('scg', messages=verbose)
        if plot:
            m.plot()
            pb.title('After optimisation')
            # m2.plot()
            # pb.title('After optimisation')

    return m

def gplvm_oil_100(optimize=True, verbose=1, plot=True):
    import GPy
    import pods
    data = pods.datasets.oil_100()
    Y = data['X']
    # create simple GP model
    kernel = GPy.kern.RBF(6, ARD=True) + GPy.kern.Bias(6)
    m = GPy.models.GPLVM(Y, 6, kernel=kernel)
    m.data_labels = data['Y'].argmax(axis=1)
    if optimize: m.optimize('scg', messages=verbose)
    if plot:
        m.plot_latent(labels=m.data_labels)
    return m

def sparse_gplvm_oil(optimize=True, verbose=0, plot=True, N=100, Q=6, num_inducing=15, max_iters=50):
    import GPy
    import pods

    _np.random.seed(0)
    data = pods.datasets.oil()
    Y = data['X'][:N]
    Y = Y - Y.mean(0)
    Y /= Y.std(0)
    # Create the model
    kernel = GPy.kern.RBF(Q, ARD=True) + GPy.kern.Bias(Q)
    m = GPy.models.SparseGPLVM(Y, Q, kernel=kernel, num_inducing=num_inducing)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    if optimize: m.optimize('scg', messages=verbose, max_iters=max_iters)
    if plot:
        m.plot_latent(labels=m.data_labels)
        m.kern.plot_ARD()
    return m

def swiss_roll(optimize=True, verbose=1, plot=True, N=1000, num_inducing=25, Q=4, sigma=.2):
    import GPy
    from pods.datasets import swiss_roll_generated
    from GPy.models import BayesianGPLVM

    data = swiss_roll_generated(num_samples=N, sigma=sigma)
    Y = data['Y']
    Y -= Y.mean()
    Y /= Y.std()

    t = data['t']
    c = data['colors']

    try:
        from sklearn.manifold.isomap import Isomap
        iso = Isomap().fit(Y)
        X = iso.embedding_
        if Q > 2:
            X = _np.hstack((X, _np.random.randn(N, Q - 2)))
    except ImportError:
        X = _np.random.randn(N, Q)

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
        fig = plt.figure("Swiss Roll Data")
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(*Y.T, c=c)
        ax.set_title("Swiss Roll")

        ax = fig.add_subplot(122)
        ax.scatter(*X.T[:2], c=c)
        ax.set_title("BGPLVM init")

    var = .5
    S = (var * _np.ones_like(X) + _np.clip(_np.random.randn(N, Q) * var ** 2,
                                         - (1 - var),
                                         (1 - var))) + .001
    Z = _np.random.permutation(X)[:num_inducing]

    kernel = GPy.kern.RBF(Q, ARD=True) + GPy.kern.Bias(Q, _np.exp(-2)) + GPy.kern.White(Q, _np.exp(-2))

    m = BayesianGPLVM(Y, Q, X=X, X_variance=S, num_inducing=num_inducing, Z=Z, kernel=kernel)
    m.data_colors = c
    m.data_t = t

    if optimize:
        m.optimize('bfgs', messages=verbose, max_iters=2e3)

    if plot:
        fig = plt.figure('fitted')
        ax = fig.add_subplot(111)
        s = m.input_sensitivity().argsort()[::-1][:2]
        ax.scatter(*m.X.mean.T[s], c=c)

    return m

def bgplvm_oil(optimize=True, verbose=1, plot=True, N=200, Q=7, num_inducing=40, max_iters=1000, **k):
    import GPy
    from matplotlib import pyplot as plt
    import numpy as np
    _np.random.seed(0)
    try:
        import pods
        data = pods.datasets.oil()
    except ImportError:
        data = GPy.util.datasets.oil()


    kernel = GPy.kern.RBF(Q, 1., 1. / _np.random.uniform(0, 1, (Q,)), ARD=True)  # + GPy.kern.Bias(Q, _np.exp(-2))
    Y = data['X'][:N]
    m = GPy.models.BayesianGPLVM(Y, Q, kernel=kernel, num_inducing=num_inducing, **k)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    if optimize:
        m.optimize('bfgs', messages=verbose, max_iters=max_iters, gtol=.05)

    if plot:
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        m.plot_latent(ax=latent_axes, labels=m.data_labels)
        data_show = GPy.plotting.matplot_dep.visualize.vector_show((m.Y[0, :]))
        lvm_visualizer = GPy.plotting.matplot_dep.visualize.lvm_dimselect(m.X.mean.values[0:1, :],  # @UnusedVariable
            m, data_show, latent_axes=latent_axes, sense_axes=sense_axes, labels=m.data_labels)
        input('Press enter to finish')
        plt.close(fig)
    return m

def ssgplvm_oil(optimize=True, verbose=1, plot=True, N=200, Q=7, num_inducing=40, max_iters=1000, **k):
    import GPy
    from matplotlib import pyplot as plt
    import pods

    _np.random.seed(0)
    data = pods.datasets.oil()

    kernel = GPy.kern.RBF(Q, 1., 1. / _np.random.uniform(0, 1, (Q,)), ARD=True)  # + GPy.kern.Bias(Q, _np.exp(-2))
    Y = data['X'][:N]
    m = GPy.models.SSGPLVM(Y, Q, kernel=kernel, num_inducing=num_inducing, **k)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    if optimize:
        m.optimize('bfgs', messages=verbose, max_iters=max_iters, gtol=.05)

    if plot:
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        m.plot_latent(ax=latent_axes, labels=m.data_labels)
        data_show = GPy.plotting.matplot_dep.visualize.vector_show((m.Y[0, :]))
        lvm_visualizer = GPy.plotting.matplot_dep.visualize.lvm_dimselect(m.X.mean.values[0:1, :],  # @UnusedVariable
            m, data_show, latent_axes=latent_axes, sense_axes=sense_axes, labels=m.data_labels)
        input('Press enter to finish')
        plt.close(fig)
    return m

def _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim=False):
    """Simulate some data drawn from a matern covariance and a periodic exponential for use in MRD demos."""
    Q_signal = 4
    import GPy
    import numpy as np
    np.random.seed(3000)

    k = GPy.kern.Matern32(Q_signal, 1., lengthscale=(np.random.uniform(1, 6, Q_signal)), ARD=1)
    for i in range(Q_signal):
        k += GPy.kern.PeriodicExponential(1, variance=1., active_dims=[i], period=3., lower=-2, upper=6)
    t = np.c_[[np.linspace(-1, 5, N) for _ in range(Q_signal)]].T
    K = k.K(t)
    s2, s1, s3, sS = np.random.multivariate_normal(np.zeros(K.shape[0]), K, size=(4))[:, :, None]

    Y1, Y2, Y3, S1, S2, S3 = _generate_high_dimensional_output(D1, D2, D3, s1, s2, s3, sS)

    slist = [sS, s1, s2, s3]
    slist_names = ["sS", "s1", "s2", "s3"]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        import itertools
        fig = plt.figure("MRD Simulation Data", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = slist_names
        for S, lab in zip(slist, labls):
            ax.plot(S, label=lab)
        ax.legend()
        for i, Y in enumerate(Ylist):
            ax = fig.add_subplot(2, len(Ylist), len(Ylist) + 1 + i)
            ax.imshow(Y, aspect='auto', cmap=cm.gray)  # @UndefinedVariable
            ax.set_title("Y{}".format(i + 1))
        plt.draw()
        plt.tight_layout()

    return slist, [S1, S2, S3], Ylist

def _simulate_sincos(D1, D2, D3, N, num_inducing, plot_sim=False):
    """Simulate some data drawn from sine and cosine for use in demos of MRD"""
    _np.random.seed(1234)

    x = _np.linspace(0, 4 * _np.pi, N)[:, None]
    s1 = _np.vectorize(lambda x: _np.sin(x))
    s2 = _np.vectorize(lambda x: _np.cos(x))
    s3 = _np.vectorize(lambda x:-_np.exp(-_np.cos(2 * x)))
    sS = _np.vectorize(lambda x: _np.cos(x))

    s1 = s1(x)
    s2 = s2(x)
    s3 = s3(x)
    sS = sS(x)

    s1 -= s1.mean(); s1 /= s1.std(0)
    s2 -= s2.mean(); s2 /= s2.std(0)
    s3 -= s3.mean(); s3 /= s3.std(0)
    sS -= sS.mean(); sS /= sS.std(0)

    Y1, Y2, Y3, S1, S2, S3 = _generate_high_dimensional_output(D1, D2, D3, s1, s2, s3, sS)

    slist = [sS, s1, s2, s3]
    slist_names = ["sS", "s1", "s2", "s3"]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        import itertools
        fig = plt.figure("MRD Simulation Data", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = slist_names
        for S, lab in zip(slist, labls):
            ax.plot(S, label=lab)
        ax.legend()
        for i, Y in enumerate(Ylist):
            ax = fig.add_subplot(2, len(Ylist), len(Ylist) + 1 + i)
            ax.imshow(Y, aspect='auto', cmap=cm.gray)  # @UndefinedVariable
            ax.set_title("Y{}".format(i + 1))
        plt.draw()
        plt.tight_layout()

    return slist, [S1, S2, S3], Ylist

def _generate_high_dimensional_output(D1, D2, D3, s1, s2, s3, sS):
    S1 = _np.hstack([s1, sS])
    S2 = _np.hstack([sS])
    S3 = _np.hstack([s1, s3, sS])
    Y1 = S1.dot(_np.random.randn(S1.shape[1], D1))
    Y2 = S2.dot(_np.random.randn(S2.shape[1], D2))
    Y3 = S3.dot(_np.random.randn(S3.shape[1], D3))
    Y1 += .3 * _np.random.randn(*Y1.shape)
    Y2 += .2 * _np.random.randn(*Y2.shape)
    Y3 += .25 * _np.random.randn(*Y3.shape)
    Y1 -= Y1.mean(0)
    Y2 -= Y2.mean(0)
    Y3 -= Y3.mean(0)
    Y1 /= Y1.std(0)
    Y2 /= Y2.std(0)
    Y3 /= Y3.std(0)
    return Y1, Y2, Y3, S1, S2, S3

def bgplvm_simulation(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4,
                      ):
    from GPy import kern
    from GPy.models import BayesianGPLVM

    D1, D2, D3, N, num_inducing, Q = 13, 5, 8, 45, 3, 9
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    # k = kern.RBF(Q, ARD=True, lengthscale=10.)
    m = BayesianGPLVM(Y, Q, init="PCA", num_inducing=num_inducing, kernel=k)
    m.X.variance[:] = _np.random.uniform(0, .01, m.X.shape)
    m.likelihood.variance = .1

    if optimize:
        print("Optimizing model:")
        m.optimize('bfgs', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m

def gplvm_simulation(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4,
                      ):
    from GPy import kern
    from GPy.models import GPLVM

    D1, D2, D3, N, num_inducing, Q = 13, 5, 8, 45, 3, 9
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    # k = kern.RBF(Q, ARD=True, lengthscale=10.)
    m = GPLVM(Y, Q, init="PCA", kernel=k)
    m.likelihood.variance = .1

    if optimize:
        print("Optimizing model:")
        m.optimize('bfgs', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m
def ssgplvm_simulation(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4, useGPU=False
                      ):
    from GPy import kern
    from GPy.models import SSGPLVM

    D1, D2, D3, N, num_inducing, Q = 13, 5, 8, 45, 3, 9
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
    # k = kern.RBF(Q, ARD=True, lengthscale=10.)
    m = SSGPLVM(Y, Q, init="rand", num_inducing=num_inducing, kernel=k, group_spike=True)
    m.X.variance[:] = _np.random.uniform(0, .01, m.X.shape)
    m.likelihood.variance = .01

    if optimize:
        print("Optimizing model:")
        m.optimize('bfgs', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.X.plot("SSGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m

def bgplvm_simulation_missing_data(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4, percent_missing=.1, d=13,
                      ):
    from GPy import kern
    from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch

    D1, D2, D3, N, num_inducing, Q = d, 5, 8, 400, 3, 4
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)

    inan = _np.random.binomial(1, percent_missing, size=Y.shape).astype(bool)  # 80% missing data
    Ymissing = Y.copy()
    Ymissing[inan] = _np.nan

    m = BayesianGPLVMMiniBatch(Ymissing, Q, init="random", num_inducing=num_inducing,
                      kernel=k, missing_data=True)

    m.Yreal = Y

    if optimize:
        print("Optimizing model:")
        m.optimize('bfgs', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m

def bgplvm_simulation_missing_data_stochastics(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4, percent_missing=.1, d=13, batchsize=2,
                      ):
    from GPy import kern
    from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch

    D1, D2, D3, N, num_inducing, Q = d, 5, 8, 400, 3, 4
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)
    Y = Ylist[0]
    k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)

    inan = _np.random.binomial(1, percent_missing, size=Y.shape).astype(bool)  # 80% missing data
    Ymissing = Y.copy()
    Ymissing[inan] = _np.nan

    m = BayesianGPLVMMiniBatch(Ymissing, Q, init="random", num_inducing=num_inducing,
                      kernel=k, missing_data=True, stochastic=True, batchsize=batchsize)

    m.Yreal = Y

    if optimize:
        print("Optimizing model:")
        m.optimize('bfgs', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.X.plot("BGPLVM Latent Space 1D")
        m.kern.plot_ARD()
    return m


def mrd_simulation(optimize=True, verbose=True, plot=True, plot_sim=True, **kw):
    from GPy import kern
    from GPy.models import MRD

    D1, D2, D3, N, num_inducing, Q = 60, 20, 36, 60, 6, 5
    _, _, Ylist = _simulate_sincos(D1, D2, D3, N, num_inducing, plot_sim)

    k = kern.Linear(Q, ARD=True) + kern.White(Q, variance=1e-4)
    m = MRD(Ylist, input_dim=Q, num_inducing=num_inducing, kernel=k, initx="PCA_concat", initz='permute', **kw)

    m['.*noise'] = [Y.var() / 40. for Y in Ylist]

    if optimize:
        print("Optimizing Model:")
        m.optimize(messages=verbose, max_iters=8e3)
    if plot:
        m.X.plot("MRD Latent Space 1D")
        m.plot_scales()
    return m

def mrd_simulation_missing_data(optimize=True, verbose=True, plot=True, plot_sim=True, **kw):
    from GPy import kern
    from GPy.models import MRD

    D1, D2, D3, N, num_inducing, Q = 60, 20, 36, 60, 6, 5
    _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, plot_sim)

    k = kern.Linear(Q, ARD=True) + kern.White(Q, variance=1e-4)
    inanlist = []

    for Y in Ylist:
        inan = _np.random.binomial(1, .6, size=Y.shape).astype(bool)
        inanlist.append(inan)
        Y[inan] = _np.nan

    m = MRD(Ylist, input_dim=Q, num_inducing=num_inducing,
            kernel=k, inference_method=None,
            initx="random", initz='permute', **kw)

    if optimize:
        print("Optimizing Model:")
        m.optimize('bfgs', messages=verbose, max_iters=8e3, gtol=.1)
    if plot:
        m.X.plot("MRD Latent Space 1D")
        m.plot_scales()
    return m

def brendan_faces(optimize=True, verbose=True, plot=True):
    import GPy
    import pods

    data = pods.datasets.brendan_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.BayesianGPLVM(Yn, Q, num_inducing=20)

    # optimize

    if optimize: m.optimize('bfgs', messages=verbose, max_iters=1000)

    if plot:
        ax = m.plot_latent(which_indices=(0, 1))
        y = m.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=(20, 28), transpose=True, order='F', invert=False, scale=False)
        lvm = GPy.plotting.matplot_dep.visualize.lvm(m.X.mean[0, :].copy(), m, data_show, ax)
        input('Press enter to finish')

    return m

def olivetti_faces(optimize=True, verbose=True, plot=True):
    import GPy
    import pods

    data = pods.datasets.olivetti_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.BayesianGPLVM(Yn, Q, num_inducing=20)

    if optimize: m.optimize('bfgs', messages=verbose, max_iters=1000)
    if plot:
        ax = m.plot_latent(which_indices=(0, 1))
        y = m.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.image_show(y[None, :], dimensions=(112, 92), transpose=False, invert=False, scale=False)
        lvm = GPy.plotting.matplot_dep.visualize.lvm(m.X.mean[0, :].copy(), m, data_show, ax)
        input('Press enter to finish')

    return m

def stick_play(range=None, frame_rate=15, optimize=False, verbose=True, plot=True):
    import GPy
    import pods

    data = pods.datasets.osu_run1()
    # optimize
    if range == None:
        Y = data['Y'].copy()
    else:
        Y = data['Y'][range[0]:range[1], :].copy()
    if plot:
        y = Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.plotting.matplot_dep.visualize.data_play(Y, data_show, frame_rate)
    return Y

def stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy
    import pods

    data = pods.datasets.osu_run1()
    # optimize
    m = GPy.models.GPLVM(data['Y'], 2, kernel=kernel)
    if optimize: m.optimize('bfgs', messages=verbose, max_f_eval=10000)
    if plot:
        plt.clf
        ax = m.plot_latent()
        y = m.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.stick_show(y[None, :], connect=data['connect'])
        lvm_visualizer = GPy.plotting.matplot_dep.visualize.lvm(m.X[:1, :].copy(), m, data_show, latent_axes=ax)
        input('Press enter to finish')
        lvm_visualizer.close()
        data_show.close()
    return m

def bcgplvm_linear_stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy
    import pods

    data = pods.datasets.osu_run1()
    # optimize
    mapping = GPy.mappings.Linear(data['Y'].shape[1], 2)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot and GPy.plotting.matplot_dep.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.plotting.matplot_dep.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        input('Press enter to finish')

    return m

def bcgplvm_stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy
    import pods

    data = pods.datasets.osu_run1()
    # optimize
    back_kernel = GPy.kern.RBF(data['Y'].shape[1], lengthscale=5.)
    mapping = GPy.mappings.Kernel(X=data['Y'], output_dim=2, kernel=back_kernel)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot and GPy.plotting.matplot_dep.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.plotting.matplot_dep.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        # input('Press enter to finish')

    return m

def robot_wireless(optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy
    import pods

    data = pods.datasets.robot_wireless()
    # optimize
    m = GPy.models.BayesianGPLVM(data['Y'], 4, num_inducing=25)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot:
        m.plot_latent()

    return m

def stick_bgplvm(model=None, optimize=True, verbose=True, plot=True):
    """Interactive visualisation of the Stick Man data from Ohio State University with the Bayesian GPLVM."""
    from GPy.models import BayesianGPLVM
    from matplotlib import pyplot as plt
    import numpy as np
    import GPy
    import pods

    data = pods.datasets.osu_run1()
    Q = 6
    kernel = GPy.kern.RBF(Q, lengthscale=np.repeat(.5, Q), ARD=True)
    m = BayesianGPLVM(data['Y'], Q, init="PCA", num_inducing=20, kernel=kernel)

    m.data = data
    m.likelihood.variance = 0.001

    # optimize
    try:
        if optimize: m.optimize('bfgs', messages=verbose, max_iters=5e3, bfgs_factor=10)
    except KeyboardInterrupt:
        print("Keyboard interrupt, continuing to plot and return")

    if plot:
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        plt.sca(latent_axes)
        m.plot_latent(ax=latent_axes)
        y = m.Y[:1, :].copy()
        data_show = GPy.plotting.matplot_dep.visualize.stick_show(y, connect=data['connect'])
        dim_select = GPy.plotting.matplot_dep.visualize.lvm_dimselect(m.X.mean[:1, :].copy(), m, data_show, latent_axes=latent_axes, sense_axes=sense_axes)
        fig.canvas.draw()
        # Canvas.show doesn't work on OSX.
        #fig.canvas.show()
        input('Press enter to finish')

    return m


def cmu_mocap(subject='35', motion=['01'], in_place=True, optimize=True, verbose=True, plot=True):
    import matplotlib.pyplot as plt
    import GPy
    import pods

    data = pods.datasets.cmu_mocap(subject, motion)
    if in_place:
        # Make figure move in place.
        data['Y'][:, 0:3] = 0.0
    Y = data['Y']
    Y_mean = Y.mean(0)
    Y_std = Y.std(0)
    m = GPy.models.GPLVM((Y - Y_mean) / Y_std, 2)

    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot:
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        m.plot_latent(ax=latent_axes)
        y = m.Y[0, :]
        data_show = GPy.plotting.matplot_dep.visualize.skeleton_show(y[None, :], data['skel'])
        lvm_visualizer = GPy.plotting.matplot_dep.visualize.lvm(m.X[0].copy(), m, data_show, latent_axes=ax)
        input('Press enter to finish')
        lvm_visualizer.close()
        data_show.close()

    return m

def ssgplvm_simulation_linear():
    import numpy as np
    import GPy
    N, D, Q = 1000, 20, 5
    pi = 0.2

    def sample_X(Q, pi):
        x = np.empty(Q)
        dies = np.random.rand(Q)
        for q in range(Q):
            if dies[q] < pi:
                x[q] = np.random.randn()
            else:
                x[q] = 0.
        return x

    Y = np.empty((N, D))
    X = np.empty((N, Q))
    # Generate data from random sampled weight matrices
    for n in range(N):
        X[n] = sample_X(Q, pi)
        w = np.random.randn(D, Q)
        Y[n] = np.dot(w, X[n])

