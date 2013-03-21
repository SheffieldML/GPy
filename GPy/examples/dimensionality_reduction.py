# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
import GPy

default_seed = np.random.seed(123344)

def BGPLVM(seed = default_seed):
    N = 10
    M = 3
    Q = 2
    D = 4
    #generate GPLVM-like data
    X = np.random.rand(N, Q)
    k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
    K = k.K(X)
    Y = np.random.multivariate_normal(np.zeros(N),K,D).T

    k = GPy.kern.linear(Q, ARD = True) + GPy.kern.white(Q)
    # k = GPy.kern.rbf(Q) + GPy.kern.rbf(Q) + GPy.kern.white(Q)
    # k = GPy.kern.rbf(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
    # k = GPy.kern.rbf(Q, ARD = False)  + GPy.kern.white(Q, 0.00001)

    m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
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
    m.checkgrad(verbose = 1)

    return m

def GPLVM_oil_100():
    data = GPy.util.datasets.oil_100()

    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD = True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(data['X'], 6, kernel = kernel)

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=1)

    # plot
    print(m)
    m.plot_latent(labels=data['Y'].argmax(axis=1))
    return m


def BGPLVM_oil():
    data = GPy.util.datasets.oil()
    Y, X = data['Y'], data['X']
    X -= X.mean(axis=0)
    # X /= X.std(axis=0)

    Q = 10
    M = 30

    kernel = GPy.kern.rbf(Q, ARD = True) + GPy.kern.bias(Q) + GPy.kern.white(Q)
    m = GPy.models.Bayesian_GPLVM(X, Q, kernel=kernel, M=M)
    m.scale_factor = 10000.0
    m.constrain_positive('(white|noise|bias|X_variance|rbf_variance|rbf_length)')
    from sklearn import cluster
    km = cluster.KMeans(M, verbose=10)
    Z = km.fit(m.X).cluster_centers_
    # Z = GPy.util.misc.kmm_init(m.X, M)
    m.set('iip', Z)
    # optimize
    # m.ensure_default_constraints()

    import pdb; pdb.set_trace()
    m.optimize('tnc', messages=1)
    print m
    m.plot_latent(labels=data['Y'].argmax(axis=1))
    return m
