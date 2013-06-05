# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes + Expectation Propagation - Poisson Likelihood
"""
import pylab as pb
import numpy as np
import GPy

default_seed=10000

def  toy_poisson_1d(seed=default_seed):
    """
    Simple 1D classification example
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    """

    X = np.arange(0,100,5)[:,None]
    F = np.round(np.sin(X/18.) + .1*X) + np.arange(5,25)[:,None]
    E = np.random.randint(-5,5,20)[:,None]
    Y = F + E

    kernel = GPy.kern.rbf(1)
    distribution = GPy.likelihoods.likelihood_functions.Poisson()
    likelihood = GPy.likelihoods.EP(Y,distribution)

    m = GPy.models.GP(X,likelihood,kernel)
    m.ensure_default_constraints()

    # Approximate likelihood
    m.update_likelihood_approximation()

    # Optimize and plot
    m.optimize()
    #m.EPEM FIXME
    print m

    # Plot
    pb.subplot(211)
    m.plot_f() #GP plot
    pb.subplot(212)
    m.plot() #Output plot

    return m
