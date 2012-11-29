# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Simple Gaussian Processes classification
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

default_seed=10000
######################################
## 2 dimensional example
def crescent_data(model_type='Full', inducing=10, seed=default_seed):
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
        m = GPy.models.simple_GP_EP(data['X'],likelihood)
    else:
        # create sparse GP EP model
        m = GPy.models.sparse_GP_EP(data['X'],likelihood=likelihood,inducing=inducing,ep_proxy=model_type)

    m.approximate_likelihood()
    print(m)

    # optimize
    m.em()
    print(m)

    # plot
    m.plot()
    return m

def oil():
    """Run a Gaussian process classification on the oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood."""
    data = GPy.util.datasets.oil()
    likelihood = GPy.inference.likelihoods.probit(data['Y'][:, 0:1])

    # create simple GP model
    m = GPy.models.simple_GP_EP(data['X'],likelihood)

    # contrain all parameters to be positive
    m.constrain_positive('')
    m.tie_param('lengthscale')
    m.approximate_likelihood()

    # optimize
    m.optimize()

    # plot
    #m.plot()
    print(m)
    return m

def toy_linear_1d_classification(model_type='Full', inducing=4, seed=default_seed):
    """Simple 1D classification example.
    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """
    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
    likelihood = GPy.inference.likelihoods.probit(data['Y'][:, 0:1])
    assert model_type in ('Full','DTC','FITC')

    # create simple GP model
    if model_type=='Full':
        m = GPy.models.simple_GP_EP(data['X'],likelihood)
    else:
        # create sparse GP EP model
        m = GPy.models.sparse_GP_EP(data['X'],likelihood=likelihood,inducing=inducing,ep_proxy=model_type)
            

    m.constrain_positive('var')
    m.constrain_positive('len')
    m.tie_param('lengthscale')
    m.approximate_likelihood()

    # Optimize and plot
    m.em(plot_all=False) # EM algorithm
    m.plot()

    print(m)
    return m
