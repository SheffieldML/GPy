# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from matplotlib import pyplot as plt

import GPy
from GPy.models.mrd import MRD

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

def GPLVM_oil_100(optimize=True, M=15):
    data = GPy.util.datasets.oil_100()

    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD=True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(data['X'], 6, kernel=kernel, M=M)
    m.data_labels = data['Y'].argmax(axis=1)

    # optimize
    m.ensure_default_constraints()
    if optimize:
        m.optimize('scg', messages=1)

    # plot
    print(m)
    m.plot_latent(labels=m.data_labels)
    return m

def BGPLVM_oil(optimize=True, N=100, Q=10, M=15):
    data = GPy.util.datasets.oil()

    # create simple GP model
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.001)
    m = GPy.models.Bayesian_GPLVM(data['X'][:N], Q, kernel=kernel, M=M)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    # optimize
    if optimize:
        m.constrain_fixed('noise', 0.05)
        m.ensure_default_constraints()
        m.optimize('scg', messages=1)
        m.unconstrain('noise')
        m.constrain_positive('noise')
        m.optimize('scg', messages=1)
    else:
        m.ensure_default_constraints()

    # plot
    print(m)
    m.plot_latent(labels=m.data_labels)
    pb.figure()
    pb.bar(np.arange(m.kern.D), 1. / m.input_sensitivity())
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

def mrd_simulation():
    num = 2
    ard1 = np.array([1., 1, 0, 0], dtype=float)
    ard2 = np.array([0., 1, 1, 0], dtype=float)
    ard1[ard1 == 0] = 1E+10
    ard2[ard2 == 0] = 1E+10

    make_params = lambda ard: np.hstack([[1], ard, [1, .3]])

    D1, D2, N, M, Q = 50, 100, 150, 15, 4
    X = np.random.randn(N, Q)

    k = GPy.kern.rbf(Q, ARD=True, lengthscale=ard1) + GPy.kern.bias(Q, 1) + GPy.kern.white(Q, 0.0001)
    Y1 = np.random.multivariate_normal(np.zeros(N), k.K(X), D1).T
    Y1 -= Y1.mean(0)

    k = GPy.kern.rbf(Q, ARD=True, lengthscale=ard2) + GPy.kern.bias(Q, 1) + GPy.kern.white(Q, 0.0001)
    Y2 = np.random.multivariate_normal(np.zeros(N), k.K(X), D2).T
    Y2 -= Y2.mean(0)

    k = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q, 1.0)

    m = MRD(Y1, Y2, Q=Q, M=M, kernel=k, _debug=False)
    m.ensure_default_constraints()

    m.optimize(messages=1, max_f_eval=5000)

    import ipdb;ipdb.set_trace()
    return m

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
    lvm_visualizer = GPy.util.visualize.lvm(m, data_show, ax)
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
    lvm_visualizer = GPy.util.visualize.lvm(m, data_show, ax)
    raw_input('Press enter to finish')
    plt.close('all')

    return m
