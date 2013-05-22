# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from matplotlib import pyplot as plt

import GPy
from GPy.models.Bayesian_GPLVM import Bayesian_GPLVM
from GPy.util.datasets import swiss_roll_generated

default_seed = np.random.seed(123344)

def BGPLVM(seed=default_seed):
    N = 10
    M = 3
    Q = 2
    D = 4
    # generate GPLVM-like data
    X = np.random.rand(N, Q)
    k = GPy.kern.rbf(Q)  + GPy.kern.white(Q, 0.00001)
    K = k.K(X)
    Y = np.random.multivariate_normal(np.zeros(N), K, D).T

    k = GPy.kern.rbf(Q, ARD=True) + GPy.kern.linear(Q, ARD=True) + GPy.kern.rbf(Q, ARD=True)  + GPy.kern.white(Q)
    # k = GPy.kern.rbf(Q) + GPy.kern.rbf(Q) + GPy.kern.white(Q)
    # k = GPy.kern.rbf(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
    # k = GPy.kern.rbf(Q, ARD = False)  + GPy.kern.white(Q, 0.00001)

    m = GPy.models.Bayesian_GPLVM(Y, Q, kernel=k, M=M)
    m.constrain_positive('(rbf|bias|noise|white|S)')
    # m.constrain_fixed('S', 1)

    # pb.figure()
    # m.plot()
    # pb.title('PCA initialisation')
    # pb.figure()
    # m.optimize(messages = 1)
    # m.plot()
    # pb.title('After optimisation')
    m.ensure_default_constraints()
    m.randomize()
    m.checkgrad(verbose=1)

    return m

def GPLVM_oil_100(optimize=True):
    data = GPy.util.datasets.oil_100()
    Y = data['X']

    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD=True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(Y, 6, kernel=kernel)
    m.data_labels = data['Y'].argmax(axis=1)

    # optimize
    m.ensure_default_constraints()
    if optimize:
        m.optimize('scg', messages=1)

    # plot
    print(m)
    m.plot_latent(labels=m.data_labels)
    return m

def swiss_roll(optimize=True, N=1000, M=15, Q=4, sigma=.2, plot=False):
    from GPy.util.datasets import swiss_roll
    from GPy.core.transformations import logexp_clipped

    data = swiss_roll_generated(N=N, sigma=sigma)
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
            X = np.hstack((X, np.random.randn(N, Q - 2)))
    except ImportError:
        X = np.random.randn(N, Q)

    if plot:
        from mpl_toolkits import mplot3d
        import pylab
        fig = pylab.figure("Swiss Roll Data")
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(*Y.T, c=c)
        ax.set_title("Swiss Roll")

        ax = fig.add_subplot(122)
        ax.scatter(*X.T[:2], c=c)
        ax.set_title("Initialization")


    var = .5
    S = (var * np.ones_like(X) + np.clip(np.random.randn(N, Q) * var ** 2,
                                         - (1 - var),
                                         (1 - var))) + .001
    Z = np.random.permutation(X)[:M]

    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, np.exp(-2)) + GPy.kern.white(Q, np.exp(-2))

    m = Bayesian_GPLVM(Y, Q, X=X, X_variance=S, M=M, Z=Z, kernel=kernel)
    m.data_colors = c
    m.data_t = t

    m.constrain('variance|length', logexp_clipped())
    m['lengthscale'] = 1. # X.var(0).max() / X.var(0)
    m['noise'] = Y.var() / 100.
    m.ensure_default_constraints()

    if optimize:
        m.optimize('scg', messages=1)
    return m

def BGPLVM_oil(optimize=True, N=100, Q=5, M=25, max_f_eval=4e3, plot=False, **k):
    np.random.seed(0)
    data = GPy.util.datasets.oil()
    from GPy.core.transformations import logexp_clipped

    # create simple GP model
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, np.exp(-2)) + GPy.kern.white(Q, np.exp(-2))
    Y = data['X'][:N]
    Yn = Y - Y.mean(0)
    Yn /= Yn.std(0)

    m = GPy.models.Bayesian_GPLVM(Yn, Q, kernel=kernel, M=M, **k)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    m.constrain('variance|leng', logexp_clipped())
    m['lengt'] = m.X.var(0).max() / m.X.var(0)
    m['noise'] = Yn.var() / 100.

    m.ensure_default_constraints()

    # optimize
    if optimize:
        m.optimize('scg', messages=1, max_f_eval=max_f_eval)

    if plot:
        y = m.likelihood.Y[0, :]
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        plt.sca(latent_axes)
        m.plot_latent()
        data_show = GPy.util.visualize.vector_show(y)
        lvm_visualizer = GPy.util.visualize.lvm_dimselect(m.X[0, :].copy(), m, data_show, latent_axes=latent_axes) # , sense_axes=sense_axes)
        raw_input('Press enter to finish')
        plt.close('all')
    return m

def oil_100():
    data = GPy.util.datasets.oil_100()
    m = GPy.models.GPLVM(data['X'], 2)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1, max_iters=2)

    # plot
    print(m)
    # m.plot_latent(labels=data['Y'].argmax(axis=1))
    return m



def _simulate_sincos(D1, D2, D3, N, M, Q, plot_sim=False):
    x = np.linspace(0, 4 * np.pi, N)[:, None]
    s1 = np.vectorize(lambda x: np.sin(x))
    s2 = np.vectorize(lambda x: np.cos(x))
    s3 = np.vectorize(lambda x:-np.exp(-np.cos(2 * x)))
    sS = np.vectorize(lambda x: np.sin(2 * x))

    s1 = s1(x)
    s2 = s2(x)
    s3 = s3(x)
    sS = sS(x)

    S1 = np.hstack([s1, sS])
    S2 = np.hstack([s2, sS])
    S3 = np.hstack([s3, sS])

    Y1 = S1.dot(np.random.randn(S1.shape[1], D1))
    Y2 = S2.dot(np.random.randn(S2.shape[1], D2))
    Y3 = S3.dot(np.random.randn(S3.shape[1], D3))

    Y1 += .1 * np.random.randn(*Y1.shape)
    Y2 += .1 * np.random.randn(*Y2.shape)
    Y3 += .1 * np.random.randn(*Y3.shape)

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
            ax.imshow(Y)
            ax.set_title("Y{}".format(i + 1))
        pylab.draw()
        pylab.tight_layout()

    return slist, [S1, S2, S3], Ylist

def bgplvm_simulation_matlab_compare():
    from GPy.util.datasets import simulation_BGPLVM
    sim_data = simulation_BGPLVM()
    Y = sim_data['Y']
    S = sim_data['S']
    mu = sim_data['mu']
    M, [_, Q] = 3, mu.shape

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)
    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    m = Bayesian_GPLVM(Y, Q, init="PCA", M=M, kernel=k,
#                        X=mu,
#                        X_variance=S,
                       _debug=False)
    m.ensure_default_constraints()
    m.auto_scale_factor = True
    m['noise'] = Y.var() / 100.
    m['linear_variance'] = .01
    return m

def bgplvm_simulation(optimize='scg',
                      plot=True,
                      max_f_eval=2e4):
    from GPy.core.transformations import logexp_clipped
    D1, D2, D3, N, M, Q = 15, 8, 8, 100, 3, 5
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, M, Q, plot)

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)


    Y = Ylist[0]

    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2)) # + kern.bias(Q)
    m = Bayesian_GPLVM(Y, Q, init="PCA", M=M, kernel=k, _debug=True)
    m.constrain('variance|noise', logexp_clipped())
#     m.ensure_default_constraints()
    m['noise'] = Y.var() / 100.
    m['linear_variance'] = .01

    if optimize:
        print "Optimizing model:"
        m.optimize('scg', max_iters=max_f_eval, max_f_eval=max_f_eval, messages=True)
    if plot:
        import pylab
        m.plot_X_1d()
        pylab.figure(); pylab.axis(); m.kern.plot_ARD()
    return m

def mrd_simulation(optimize=True, plot_sim=False):
    D1, D2, D3, N, M, Q = 150, 250, 30, 300, 3, 7
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, M, Q, plot_sim)

    from GPy.models import mrd
    from GPy import kern
    from GPy.core.transformations import logexp_clipped

    reload(mrd); reload(kern)

    k = kern.linear(Q, [0.01] * Q, True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    m = mrd.MRD(*Ylist, Q=Q, M=M, kernel=k, initx="concat", initz='permute')

    for i, Y in enumerate(Ylist):
        m['{}_noise'.format(i + 1)] = Y.var() / 100.

    m.constrain('variance|noise', logexp_clipped())
    m.ensure_default_constraints()

    # DEBUG
    np.seterr("raise")

    if optimize:
        print "Optimizing Model:"
        m.optimize('scg', messages=1, max_iters=3e3)

    return m

def brendan_faces():
    from GPy import kern
    data = GPy.util.datasets.brendan_faces()
    Q = 2
    Y = data['Y'][0:-1:10, :]
    # Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.GPLVM(Yn, Q)
    # m = GPy.models.Bayesian_GPLVM(Yn, Q, M=100)

    # optimize
    m.constrain('rbf|noise|white', GPy.core.transformations.logexp_clipped())

    m.ensure_default_constraints()
    m.optimize('scg', messages=1, max_f_eval=10000)

    ax = m.plot_latent(which_indices=(0,1))
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(20, 28), transpose=True, invert=False, scale=False)
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')
    plt.close('all')

    return m

def stick():
    data = GPy.util.datasets.stick()
    m = GPy.models.GPLVM(data['Y'], 2)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1, max_f_eval=10000)
    m._set_params(m._get_params())

    ax = m.plot_latent()
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')
    plt.close('all')

    return m

def cmu_mocap(subject='35', motion=['01'], in_place=True):

    data = GPy.util.datasets.cmu_mocap(subject, motion)
    Y = data['Y']
    if in_place:
        # Make figure move in place.
        data['Y'][:, 0:3] = 0.0
    m = GPy.models.GPLVM(data['Y'], 2, normalize_Y=True)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1, max_f_eval=10000)

    ax = m.plot_latent()
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.skeleton_show(y[None, :], data['skel'])
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')
    plt.close('all')

    return m

# def BGPLVM_oil():
#     data = GPy.util.datasets.oil()
#     Y, X = data['Y'], data['X']
#     X -= X.mean(axis=0)
#     X /= X.std(axis=0)
#
#     Q = 10
#     M = 30
#
#     kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q)
#     m = GPy.models.Bayesian_GPLVM(X, Q, kernel=kernel, M=M)
#     # m.scale_factor = 100.0
#     m.constrain_positive('(white|noise|bias|X_variance|rbf_variance|rbf_length)')
#     from sklearn import cluster
#     km = cluster.KMeans(M, verbose=10)
#     Z = km.fit(m.X).cluster_centers_
#     # Z = GPy.util.misc.kmm_init(m.X, M)
#     m.set('iip', Z)
#     m.set('bias', 1e-4)
#     # optimize
#     # m.ensure_default_constraints()
#
#     import pdb; pdb.set_trace()
#     m.optimize('tnc', messages=1)
#     print m
#     m.plot_latent(labels=data['Y'].argmax(axis=1))
#     return m

