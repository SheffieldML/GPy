# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes classification
"""
import pylab as pb
import numpy as np
import GPy

default_seed=10000
def crescent_data(seed=default_seed): #FIXME
    """Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """

    data = GPy.util.datasets.crescent_data(seed=seed)

    # Kernel object
    kernel = GPy.kern.rbf(data['X'].shape[1])

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(data['Y'],distribution)


    m = GPy.models.GP(data['X'],likelihood,kernel)
    m.ensure_default_constraints()

    m.update_likelihood_approximation()
    print(m)

    # optimize
    m.optimize()
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
    m = GPy.models.GP(data['X'],likelihood=likelihood,kernel=kernel)

    # Contrain all parameters to be positive
    m.constrain_positive('')
    m.tie_params('lengthscale')
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
    """

    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]

    # Kernel object
    kernel = GPy.kern.rbf(1)

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(Y,distribution)

    # Model definition
    m = GPy.models.GP(data['X'],likelihood=likelihood,kernel=kernel)
    m.ensure_default_constraints()

    # Optimize
    m.update_likelihood_approximation()
    # Parameters optimization:
    m.optimize()
    #m.pseudo_EM() #FIXME

    # Plot
    pb.subplot(211)
    m.plot_f()
    pb.subplot(212)
    m.plot()
    print(m)

    return m

def sparse_toy_linear_1d_classification(seed=default_seed):
    """
    Sparse 1D classification example
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    """

    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]

    # Kernel object
    kernel = GPy.kern.rbf(1) + GPy.kern.white(1)

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(Y,distribution)

    Z = np.random.uniform(data['X'].min(),data['X'].max(),(10,1))

    # Model definition
    m = GPy.models.sparse_GP(data['X'],likelihood=likelihood,kernel=kernel,Z=Z,normalize_X=False)
    m.set('len',2.)

    m.ensure_default_constraints()
    # Optimize
    m.update_likelihood_approximation()
    # Parameters optimization:
    m.optimize()
    #m.EPEM() #FIXME

    # Plot
    pb.subplot(211)
    m.plot_f()
    pb.subplot(212)
    m.plot()
    print(m)

    return m

def sparse_crescent_data(inducing=10, seed=default_seed):
    """Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """

    data = GPy.util.datasets.crescent_data(seed=seed)

    # Kernel object
    kernel = GPy.kern.rbf(data['X'].shape[1]) + GPy.kern.white(data['X'].shape[1])

    # Likelihood object
    distribution = GPy.likelihoods.likelihood_functions.probit()
    likelihood = GPy.likelihoods.EP(data['Y'],distribution)

    sample = np.random.randint(0,data['X'].shape[0],inducing)
    Z = data['X'][sample,:]

    # create sparse GP EP model
    m = GPy.models.sparse_GP(data['X'],likelihood=likelihood,kernel=kernel,Z=Z)
    m.ensure_default_constraints()
    m.set('len',10.)

    m.update_likelihood_approximation()

    # optimize
    m.optimize()
    print(m)

    # plot
    m.plot()
    return m
