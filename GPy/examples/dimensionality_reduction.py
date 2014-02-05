# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as _np
default_seed = _np.random.seed(123344)

def bgplvm_test_model(seed=default_seed, optimize=False, verbose=1, plot=False):
    """
    model for testing purposes. Samples from a GP with rbf kernel and learns
    the samples with a new kernel. Normally not for optimization, just model cheking
    """
    from GPy.likelihoods.gaussian import Gaussian
    import GPy

    num_inputs = 13
    num_inducing = 5
    if plot:
        output_dim = 1
        input_dim = 2
    else:
        input_dim = 2
        output_dim = 25

    # generate GPLVM-like data
    X = _np.random.rand(num_inputs, input_dim)
    lengthscales = _np.random.rand(input_dim)
    k = (GPy.kern.rbf(input_dim, .5, lengthscales, ARD=True)
         + GPy.kern.white(input_dim, 0.01))
    K = k.K(X)
    Y = _np.random.multivariate_normal(_np.zeros(num_inputs), K, output_dim).T
    lik = Gaussian(Y, normalize=True)

    k = GPy.kern.rbf_inv(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim)
    # k = GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim, 0.00001)
    # k = GPy.kern.rbf(input_dim, ARD = False)  + GPy.kern.white(input_dim, 0.00001)
    # k = GPy.kern.rbf(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.rbf(input_dim, .3, _np.ones(input_dim) * .2, ARD=True)
    # k = GPy.kern.rbf(input_dim, .5, 2., ARD=0) + GPy.kern.rbf(input_dim, .3, .2, ARD=0)
    # k = GPy.kern.rbf(input_dim, .5, _np.ones(input_dim) * 2., ARD=True) + GPy.kern.linear(input_dim, _np.ones(input_dim) * .2, ARD=True)

    m = GPy.models.BayesianGPLVM(lik, input_dim, kernel=k, num_inducing=num_inducing)
    #===========================================================================
    # randomly obstruct data with percentage p
    p = .8
    Y_obstruct = Y.copy()
    Y_obstruct[_np.random.uniform(size=(Y.shape)) < p] = _np.nan
    #===========================================================================
    m2 = GPy.models.BayesianGPLVMWithMissingData(Y_obstruct, input_dim, kernel=k, num_inducing=num_inducing)
    m.lengthscales = lengthscales

    if plot:
        import matplotlib.pyplot as pb
        m.plot()
        pb.title('PCA initialisation')
        m2.plot()
        pb.title('PCA initialisation')

    if optimize:
        m.optimize('scg', messages=verbose)
        m2.optimize('scg', messages=verbose)
        if plot:
            m.plot()
            pb.title('After optimisation')
            m2.plot()
            pb.title('After optimisation')

    return m, m2

def gplvm_oil_100(optimize=True, verbose=1, plot=True):
    import GPy
    data = GPy.util.datasets.oil_100()
    Y = data['X']
    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD=True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(Y, 6, kernel=kernel)
    m.data_labels = data['Y'].argmax(axis=1)
    if optimize: m.optimize('scg', messages=verbose)
    if plot: m.plot_latent(labels=m.data_labels)
    return m

def sparse_gplvm_oil(optimize=True, verbose=0, plot=True, N=100, Q=6, num_inducing=15, max_iters=50):
    import GPy
    _np.random.seed(0)
    data = GPy.util.datasets.oil()
    Y = data['X'][:N]
    Y = Y - Y.mean(0)
    Y /= Y.std(0)
    # Create the model
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q)
    m = GPy.models.SparseGPLVM(Y, Q, kernel=kernel, num_inducing=num_inducing)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    if optimize: m.optimize('scg', messages=verbose, max_iters=max_iters)
    if plot:
        m.plot_latent(labels=m.data_labels)
        m.kern.plot_ARD()
    return m

def swiss_roll(optimize=True, verbose=1, plot=True, N=1000, num_inducing=15, Q=4, sigma=.2):
    import GPy
    from GPy.util.datasets import swiss_roll_generated
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

    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, _np.exp(-2)) + GPy.kern.white(Q, _np.exp(-2))

    m = BayesianGPLVM(Y, Q, X=X, X_variance=S, num_inducing=num_inducing, Z=Z, kernel=kernel)
    m.data_colors = c
    m.data_t = t
    m['noise_variance'] = Y.var() / 100.

    if optimize:
        m.optimize('scg', messages=verbose, max_iters=2e3)

    if plot:
        fig = plt.figure('fitted')
        ax = fig.add_subplot(111)
        s = m.input_sensitivity().argsort()[::-1][:2]
        ax.scatter(*m.X.T[s], c=c)

    return m

def bgplvm_oil(optimize=True, verbose=1, plot=True, N=200, Q=7, num_inducing=40, max_iters=1000, **k):
    import GPy
    from GPy.likelihoods import Gaussian
    from matplotlib import pyplot as plt

    _np.random.seed(0)
    data = GPy.util.datasets.oil()

    kernel = GPy.kern.rbf_inv(Q, 1., [.1] * Q, ARD=True) + GPy.kern.bias(Q, _np.exp(-2))
    Y = data['X'][:N]
    Yn = Gaussian(Y, normalize=True)
    m = GPy.models.BayesianGPLVM(Yn, Q, kernel=kernel, num_inducing=num_inducing, **k)
    m.data_labels = data['Y'][:N].argmax(axis=1)
    m['noise'] = Yn.Y.var() / 100.

    if optimize:
        m.optimize('scg', messages=verbose, max_iters=max_iters, gtol=.05)

    if plot:
        y = m.likelihood.Y[0, :]
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        m.plot_latent(ax=latent_axes)
        data_show = GPy.util.visualize.vector_show(y)
        lvm_visualizer = GPy.util.visualize.lvm_dimselect(m.X[0, :], # @UnusedVariable
            m, data_show, latent_axes=latent_axes, sense_axes=sense_axes)
        raw_input('Press enter to finish')
        plt.close(fig)
    return m

def _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim=False):
    x = _np.linspace(0, 4 * _np.pi, N)[:, None]
    s1 = _np.vectorize(lambda x: _np.sin(x))
    s2 = _np.vectorize(lambda x: _np.cos(x))
    s3 = _np.vectorize(lambda x:-_np.exp(-_np.cos(2 * x)))
    sS = _np.vectorize(lambda x: _np.sin(2 * x))

    s1 = s1(x)
    s2 = s2(x)
    s3 = s3(x)
    sS = sS(x)

    S1 = _np.hstack([s1, sS])
    S2 = _np.hstack([s2, s3, sS])
    S3 = _np.hstack([s3, sS])

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

    slist = [sS, s1, s2, s3]
    slist_names = ["sS", "s1", "s2", "s3"]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        import pylab
        import matplotlib.cm as cm
        import itertools
        fig = pylab.figure("MRD Simulation Data", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = slist_names
        for S, lab in itertools.izip(slist, labls):
            ax.plot(S, label=lab)
        ax.legend()
        for i, Y in enumerate(Ylist):
            ax = fig.add_subplot(2, len(Ylist), len(Ylist) + 1 + i)
            ax.imshow(Y, aspect='auto', cmap=cm.gray) # @UndefinedVariable
            ax.set_title("Y{}".format(i + 1))
        pylab.draw()
        pylab.tight_layout()

    return slist, [S1, S2, S3], Ylist

# def bgplvm_simulation_matlab_compare():
#     from GPy.util.datasets import simulation_BGPLVM
#     from GPy import kern
#     from GPy.models import BayesianGPLVM
#
#     sim_data = simulation_BGPLVM()
#     Y = sim_data['Y']
#     mu = sim_data['mu']
#     num_inducing, [_, Q] = 3, mu.shape
#
#     k = kern.linear(Q, ARD=True) + kern.bias(Q, _np.exp(-2)) + kern.white(Q, _np.exp(-2))
#     m = BayesianGPLVM(Y, Q, init="PCA", num_inducing=num_inducing, kernel=k,
#                        _debug=False)
#     m.auto_scale_factor = True
#     m['noise'] = Y.var() / 100.
#     m['linear_variance'] = .01
#     return m

def bgplvm_simulation(optimize=True, verbose=1,
                      plot=True, plot_sim=False,
                      max_iters=2e4,
                      ):
    from GPy import kern
    from GPy.models import BayesianGPLVM

    D1, D2, D3, N, num_inducing, Q = 49, 30, 10, 12, 3, 10
    _, _, Ylist = _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim)
    Y = Ylist[0]
    k = kern.linear(Q, ARD=True)
    m = BayesianGPLVM(Y, Q, init="PCA", num_inducing=num_inducing, kernel=k)
    m.X_variance = m.X_variance * .7
    m['noise'] = Y.var() / 100.

    if optimize:
        print "Optimizing model:"
        m.optimize('scg', messages=verbose, max_iters=max_iters,
                   gtol=.05)
    if plot:
        m.plot_X_1d("BGPLVM Latent Space 1D")
        m.kern.plot_ARD('BGPLVM Simulation ARD Parameters')
    return m

def mrd_simulation(optimize=True, verbose=True, plot=True, plot_sim=True, **kw):
    from GPy import kern
    from GPy.models import MRD
    from GPy.likelihoods import Gaussian

    D1, D2, D3, N, num_inducing, Q = 60, 20, 36, 60, 6, 5
    _, _, Ylist = _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim)
    likelihood_list = [Gaussian(x, normalize=True) for x in Ylist]

    k = kern.linear(Q, ARD=True)# + kern.bias(Q, _np.exp(-2)) + kern.white(Q, _np.exp(-2))
    m = MRD(likelihood_list, input_dim=Q, num_inducing=num_inducing, kernels=k, initx="", initz='permute', **kw)
    m.ensure_default_constraints()

    for i, bgplvm in enumerate(m.bgplvms):
        m['{}_noise'.format(i)] = 1 #bgplvm.likelihood.Y.var() / 500.
        bgplvm.X_variance = bgplvm.X_variance #* .1
    if optimize:
        print "Optimizing Model:"
        m.optimize(messages=verbose, max_iters=8e3, gtol=.1)
    if plot:
        m.plot_X_1d("MRD Latent Space 1D")
        m.plot_scales("MRD Scales")
    return m

def brendan_faces(optimize=True, verbose=True, plot=True):
    import GPy

    data = GPy.util.datasets.brendan_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.GPLVM(Yn, Q)

    # optimize
    m.constrain('rbf|noise|white', GPy.core.transformations.logexp_clipped())

    if optimize: m.optimize('scg', messages=verbose, max_iters=1000)

    if plot:
        ax = m.plot_latent(which_indices=(0, 1))
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(20, 28), transpose=True, order='F', invert=False, scale=False)
        GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def olivetti_faces(optimize=True, verbose=True, plot=True):
    import GPy

    data = GPy.util.datasets.olivetti_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.GPLVM(Yn, Q)
    if optimize: m.optimize('scg', messages=verbose, max_iters=1000)
    if plot:
        ax = m.plot_latent(which_indices=(0, 1))
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(112, 92), transpose=False, invert=False, scale=False)
        GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def stick_play(range=None, frame_rate=15, optimize=False, verbose=True, plot=True):
    import GPy
    data = GPy.util.datasets.osu_run1()
    # optimize
    if range == None:
        Y = data['Y'].copy()
    else:
        Y = data['Y'][range[0]:range[1], :].copy()
    if plot:
        y = Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.util.visualize.data_play(Y, data_show, frame_rate)
    return Y

def stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy

    data = GPy.util.datasets.osu_run1()
    # optimize
    m = GPy.models.GPLVM(data['Y'], 2, kernel=kernel)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot and GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def bcgplvm_linear_stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy

    data = GPy.util.datasets.osu_run1()
    # optimize
    mapping = GPy.mappings.Linear(data['Y'].shape[1], 2)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot and GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def bcgplvm_stick(kernel=None, optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy

    data = GPy.util.datasets.osu_run1()
    # optimize
    back_kernel=GPy.kern.rbf(data['Y'].shape[1], lengthscale=5.)
    mapping = GPy.mappings.Kernel(X=data['Y'], output_dim=2, kernel=back_kernel)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot and GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def robot_wireless(optimize=True, verbose=True, plot=True):
    from matplotlib import pyplot as plt
    import GPy

    data = GPy.util.datasets.robot_wireless()
    # optimize
    m = GPy.models.GPLVM(data['Y'], 2)
    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    m._set_params(m._get_params())
    if plot:
        m.plot_latent()

    return m

def stick_bgplvm(model=None, optimize=True, verbose=True, plot=True):
    from GPy.models import BayesianGPLVM
    from matplotlib import pyplot as plt
    import GPy

    data = GPy.util.datasets.osu_run1()
    Q = 6
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, _np.exp(-2)) + GPy.kern.white(Q, _np.exp(-2))
    m = BayesianGPLVM(data['Y'], Q, init="PCA", num_inducing=20, kernel=kernel)
    # optimize
    m.ensure_default_constraints()
    if optimize: m.optimize('scg', messages=verbose, max_iters=200, xtol=1e-300, ftol=1e-300)
    m._set_params(m._get_params())
    if plot:
        plt.clf, (latent_axes, sense_axes) = plt.subplots(1, 2)
        plt.sca(latent_axes)
        m.plot_latent()
        y = m.likelihood.Y[0, :].copy()
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        GPy.util.visualize.lvm_dimselect(m.X[0, :].copy(), m, data_show, latent_axes=latent_axes, sense_axes=sense_axes)
        raw_input('Press enter to finish')

    return m


def cmu_mocap(subject='35', motion=['01'], in_place=True, optimize=True, verbose=True, plot=True):
    import GPy

    data = GPy.util.datasets.cmu_mocap(subject, motion)
    if in_place:
        # Make figure move in place.
        data['Y'][:, 0:3] = 0.0
    m = GPy.models.GPLVM(data['Y'], 2, normalize_Y=True)

    if optimize: m.optimize(messages=verbose, max_f_eval=10000)
    if plot:
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.skeleton_show(y[None, :], data['skel'])
        lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')
        lvm_visualizer.close()

    return m
