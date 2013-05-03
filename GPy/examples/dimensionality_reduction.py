# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from matplotlib import pyplot as plt, pyplot

import GPy
from GPy.models.Bayesian_GPLVM import Bayesian_GPLVM
from GPy.util.datasets import simulation_BGPLVM

default_seed = np.random.seed(123344)

def BGPLVM(seed=default_seed):
    N = 10
    M = 3
    Q = 2
    D = 4
    # generate GPLVM-like data
    X = np.random.rand(N, Q)
    k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
    K = k.K(X)
    Y = np.random.multivariate_normal(np.zeros(N), K, D).T

    k = GPy.kern.linear(Q, ARD=True) + GPy.kern.white(Q)
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

    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD=True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(data['X'], 6, kernel=kernel)
    m.data_labels = data['Y'].argmax(axis=1)

    # optimize
    m.ensure_default_constraints()
    if optimize:
        m.optimize('scg', messages=1)

    # plot
    print(m)
    m.plot_latent(labels=m.data_labels)
    return m

def BGPLVM_oil(optimize=True, N=100, Q=10, M=15, max_f_eval=300):
    data = GPy.util.datasets.oil()

    # create simple GP model
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.001)
    m = GPy.models.Bayesian_GPLVM(data['X'][:N], Q, kernel=kernel, M=M)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    # optimize
    if optimize:
        m.constrain_fixed('noise', 0.05)
        m.ensure_default_constraints()
        m.optimize('scg', messages=1, max_f_eval=max(80, max_f_eval))
        m.unconstrain('noise')
        m.constrain_positive('noise')
        m.optimize('scg', messages=1, max_f_eval=max(0, max_f_eval - 80))
    else:
        m.ensure_default_constraints()

    y = m.likelihood.Y[0, :]
    fig, (latent_axes, hist_axes) = plt.subplots(1, 2)
    plt.sca(latent_axes)
    m.plot_latent()
    data_show = GPy.util.visualize.vector_show(y)
    lvm_visualizer = GPy.util.visualize.lvm_dimselect(m.X[0, :], m, data_show, latent_axes=latent_axes, hist_axes=hist_axes)
    raw_input('Press enter to finish')
    plt.close('all')
    # # plot
    # print(m)
    # m.plot_latent(labels=m.data_labels)
    # pb.figure()
    # pb.bar(np.arange(m.kern.D), 1. / m.input_sensitivity())
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

#     s1 -= s1.mean()
#     s2 -= s2.mean()
#     s3 -= s3.mean()
#     sS -= sS.mean()
#     s1 /= .5 * (np.abs(s1).max() - np.abs(s1).min())
#     s2 /= .5 * (np.abs(s2).max() - np.abs(s2).min())
#     s3 /= .5 * (np.abs(s3).max() - np.abs(s3).min())
#     sS /= .5 * (np.abs(sS).max() - np.abs(sS).min())

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

    slist = [s1, s2, s3, sS]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        import pylab
        import itertools
        fig = pylab.figure("MRD Simulation", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = sorted(filter(lambda x: x.startswith("s"), locals()))
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
    sim_data = simulation_BGPLVM()
    Y = sim_data['Y']
    S = sim_data['S']
    mu = sim_data['mu']
    M, [_, Q] = 20, mu.shape

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)
    # k = kern.rbf(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    m = Bayesian_GPLVM(Y, Q, init="PCA", M=M, kernel=k,
#                        X=mu,
#                        X_variance=S,
                       _debug=True)
    m.ensure_default_constraints()
    m.auto_scale_factor = True
    m['noise'] = Y.var() / 100.
    m['linear_variance'] = .01

#     lscstr = 'X_variance'
#     m[lscstr] = .01
#     m.unconstrain(lscstr); m.constrain_fixed(lscstr, .1)

#     cstr = 'white'
#     m.unconstrain(cstr); m.constrain_bounded(cstr, .01, 1.)

#     cstr = 'noise'
#     m.unconstrain(cstr); m.constrain_bounded(cstr, .01, 1.)
    return m

def bgplvm_simulation(burnin='scg', plot_sim=False,
                      max_burnin=100, true_X=False,
                      do_opt=True,
                      max_f_eval=1000):
    D1, D2, D3, N, M, Q = 10, 8, 8, 250, 10, 6
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, M, Q, plot_sim)

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)


    Y = Ylist[0]

    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))  # + kern.bias(Q)
#     k = kern.white(Q, .00001) + kern.bias(Q)
    m = Bayesian_GPLVM(Y, Q, init="PCA", M=M, kernel=k, _debug=True)
    # m.set('noise',)
    m.ensure_default_constraints()
    m['noise'] = Y.var() / 100.
    m['linear_variance'] = .001
#     m.auto_scale_factor = True
#     m.scale_factor = 1.


    if burnin:
        print "initializing beta"
        cstr = "noise"
        m.unconstrain(cstr); m.constrain_fixed(cstr, Y.var() / 70.)
        m.optimize(burnin, messages=1, max_f_eval=max_burnin)

        print "releasing beta"
        cstr = "noise"
        m.unconstrain(cstr);  m.constrain_positive(cstr)

    if true_X:
        true_X = np.hstack((slist[0], slist[3], 0. * np.ones((N, Q - 2))))
        m.set('X_\d', true_X)
        m.constrain_fixed("X_\d")

        cstr = 'X_variance'
#         m.unconstrain(cstr), m.constrain_fixed(cstr, .0001)
        m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-7, .1)

#     cstr = 'X_variance'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-3, 1.)

    # m['X_var'] = np.ones(N * Q) * .5 + np.random.randn(N * Q) * .01

#     cstr = "iip"
#     m.unconstrain(cstr); m.constrain_fixed(cstr)

#     cstr = 'variance'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-10, 1.)
#     cstr = 'X_\d'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, -10., 10.)
#
#     cstr = 'noise'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-5, 1.)
#
#     cstr = 'white'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-6, 1.)
#
#     cstr = 'linear_variance'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-10, 10.)

#     cstr = 'variance'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-10, 10.)

#     np.seterr(all='call')
#     def ipdbonerr(errtype, flags):
#         import ipdb; ipdb.set_trace()
#     np.seterrcall(ipdbonerr)

    if do_opt and burnin:
        try:
            m.optimize(burnin, messages=1, max_f_eval=max_f_eval)
        except:
            pass
        finally:
            return m
    return m

def mrd_simulation(plot_sim=False):
    # num = 2
#     ard1 = np.array([1., 1, 0, 0], dtype=float)
#     ard2 = np.array([0., 1, 1, 0], dtype=float)
#     ard1[ard1 == 0] = 1E-10
#     ard2[ard2 == 0] = 1E-10

#     ard1i = 1. / ard1
#     ard2i = 1. / ard2

#     k = GPy.kern.rbf(Q, ARD=True, lengthscale=ard1i) + GPy.kern.bias(Q, 0) + GPy.kern.white(Q, 0.0001)
#     Y1 = np.random.multivariate_normal(np.zeros(N), k.K(X), D1).T
#     Y1 -= Y1.mean(0)
#
#     k = GPy.kern.rbf(Q, ARD=True, lengthscale=ard2i) + GPy.kern.bias(Q, 0) + GPy.kern.white(Q, 0.0001)
#     Y2 = np.random.multivariate_normal(np.zeros(N), k.K(X), D2).T
#     Y2 -= Y2.mean(0)
#     make_params = lambda ard: np.hstack([[1], ard, [1, .3]])
    D1, D2, D3, N, M, Q = 2000, 34, 8, 500, 3, 6
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, M, Q, plot_sim)

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)

#    k = kern.rbf(2, ARD=True) + kern.bias(2) + kern.white(2)
#     Y1 = np.random.multivariate_normal(np.zeros(N), k.K(S1), D1).T
#     Y2 = np.random.multivariate_normal(np.zeros(N), k.K(S2), D2).T
#     Y3 = np.random.multivariate_normal(np.zeros(N), k.K(S3), D3).T

    Ylist = Ylist[0:2]

    # k = kern.rbf(Q, ARD=True) + kern.bias(Q) + kern.white(Q)

    k = kern.linear(Q, ARD=True) + kern.bias(Q, .01) + kern.white(Q, .001)
    m = mrd.MRD(*Ylist, Q=Q, M=M, kernel=k, initx="concat", initz='permute', _debug=False)

    for i, Y in enumerate(Ylist):
        m.set('{}_noise'.format(i + 1), Y.var() / 100.)

    m.ensure_default_constraints()
    m.auto_scale_factor = True

#     cstr = 'variance'
#     m.unconstrain(cstr), m.constrain_bounded(cstr, 1e-12, 1.)
#
#     cstr = 'linear_variance'
#     m.unconstrain(cstr), m.constrain_positive(cstr)

#     print "initializing beta"
#     cstr = "noise"
#     m.unconstrain(cstr); m.constrain_fixed(cstr)
#     m.optimize('scg', messages=1, max_f_eval=100)

#     print "releasing beta"
#     cstr = "noise"
#     m.unconstrain(cstr);  m.constrain_positive(cstr)

    np.seterr(all='call')
    def ipdbonerr(errtype, flags):
        import ipdb; ipdb.set_trace()
    np.seterrcall(ipdbonerr)

    return m  # , mtest

def mrd_silhouette():

    pass

def brendan_faces():
    data = GPy.util.datasets.brendan_faces()
    Y = data['Y'][0:-1:10, :]
    m = GPy.models.GPLVM(data['Y'], 2)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1, max_f_eval=10000)

    ax = m.plot_latent()
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(20, 28), transpose=True, invert=False, scale=False)
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :], m, data_show, ax)
    raw_input('Press enter to finish')
    plt.close('all')

    return m

def stick():
    data = GPy.util.datasets.stick()
    m = GPy.models.GPLVM(data['Y'], 2)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1, max_f_eval=10000)

    ax = m.plot_latent()
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :], m, data_show, ax)
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
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :], m, data_show, ax)
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

