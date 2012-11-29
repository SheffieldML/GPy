# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes regression examples
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')


def toy_rbf_1d():
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize
    m.optimize()

    # plot
    m.plot()
    print(m)
    return m

def rogers_girolami_olympics():
    """Run a standard Gaussian process regression on the Rogers and Girolami olympics data."""
    data = GPy.util.datasets.rogers_girolami_olympics()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize
    m.optimize()

    # plot
    m.plot(plot_limits = (1850, 2050))
    print(m)
    return m

def toy_rbf_1d_50():
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d_50()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize
    m.optimize()

    # plot
    m.plot()
    print(m)
    return m

def silhouette():
    """Predict the pose of a figure given a silhouette. This is a task from Agarwal and Triggs 2004 ICML paper."""
    data = GPy.util.datasets.silhouette()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize
    m.optimize()

    print(m)
    return m
