# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes classification
"""
import pylab as pb
import numpy as np
import GPy

default_seed=10000

def crescent_data(model_type='Full', inducing=10, seed=default_seed): #FIXME
    """Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """
    data = GPy.util.datasets.crescent_data(seed=seed)
    likelihood = GPy.inference.likelihoods.probit(data['Y'])

    if model_type=='Full':
        m = GPy.models.GP_EP(data['X'],likelihood)
    else:
        # create sparse GP EP model
        m = GPy.models.sparse_GP_EP(data['X'],likelihood=likelihood,inducing=inducing,ep_proxy=model_type)

    m.update_likelihood_approximation()
    print(m)

    # optimize
    m.em()
    print(m)

    # plot
    m.plot()
    return m

def oil():
    """
    Run a Gaussian process classification on the oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.
    """
    data = GPy.util.datasets.oil()
    # Kernel object
    kernel = GPy.kern.rbf(12)

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(data['Y'][:, 0:1],distribution)

    # Create GP model
    m = GPy.models.GP(data['X'],kernel,likelihood=likelihood)

    # Contrain all parameters to be positive
    m.constrain_positive('')
    m.tie_param('lengthscale')
    m.update_likelihood_approximation()

    # Optimize
    m.optimize()

    print(m)
    return m

def toy_linear_1d_classification(seed=default_seed):
    """
    Simple 1D classification example
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    :type inducing: int
    """

    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)

    # Kernel object
    kernel = GPy.kern.rbf(1)

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(data['Y'][:, 0:1],distribution)

    # Model definition
    m = GPy.models.GP(data['X'],kernel,likelihood=likelihood)

    # Optimize
    """
    EPEM runs a loop that consists of two steps:
    1) EP likelihood approximation:
        m.update_likelihood_approximation()
    2) Parameters optimization:
        m.optimize()
    """
    m.EPEM()

    # Plot
    pb.subplot(211)
    m.plot_GP()
    pb.subplot(212)
    m.plot_output()
    print(m)

    return m
