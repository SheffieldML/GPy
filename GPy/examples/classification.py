# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes classification
"""
import pylab as pb
import numpy as np
import GPy

default_seed = 10000
def crescent_data(seed=default_seed): # FIXME
    """Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """

    data = GPy.util.datasets.crescent_data(seed=seed)
    Y = data['Y']
    Y[Y.flatten()==-1] = 0

    m = GPy.models.GPClassification(data['X'], Y)
    m.ensure_default_constraints()
    #m.update_likelihood_approximation()
    #m.optimize()
    m.pseudo_EM()
    print(m)
    m.plot()
    return m

def oil(num_inducing=50):
    """
    Run a Gaussian process classification on the oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.
    """
    data = GPy.util.datasets.oil()
    X = data['X'][:600,:]
    X_test = data['X'][600:,:]
    Y = data['Y'][:600, 0:1]
    Y[Y.flatten()==-1] = 0
    Y_test = data['Y'][600:, 0:1]

    # Create GP model
    m = GPy.models.SparseGPClassification(X, Y,num_inducing=num_inducing)

    # Contrain all parameters to be positive
    m.constrain_positive('')
    m.tie_params('.*len')
    m['.*len'] = 10.
    m.update_likelihood_approximation()

    # Optimize
    m.optimize()
    print(m)

    #Test
    probs = m.predict(X_test)[0]
    GPy.util.classification.conf_matrix(probs,Y_test)
    return m

def toy_linear_1d_classification(seed=default_seed):
    """
    Simple 1D classification example
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    """

    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.GPClassification(data['X'], Y)
    m.ensure_default_constraints()

    # Optimize
    #m.update_likelihood_approximation()
    # Parameters optimization:
    #m.optimize()
    m.pseudo_EM()

    # Plot
    fig, axes = pb.subplots(2,1)
    m.plot_f(ax=axes[0])
    m.plot(ax=axes[1])
    print(m)

    return m

def sparse_toy_linear_1d_classification(num_inducing=10,seed=default_seed):
    """
    Sparse 1D classification example
    :param seed : seed value for data generation (default is 4).
    :type seed: int
    """

    data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.SparseGPClassification(data['X'], Y,num_inducing=num_inducing)
    m['.*len']= 4.

    m.ensure_default_constraints()
    # Optimize
    #m.update_likelihood_approximation()
    # Parameters optimization:
    #m.optimize()
    m.pseudo_EM()

    # Plot
    fig, axes = pb.subplots(2,1)
    m.plot_f(ax=axes[0])
    m.plot(ax=axes[1])
    print(m)

    return m

def sparse_crescent_data(num_inducing=10, seed=default_seed):
    """
    Run a Gaussian process classification with DTC approxiamtion on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    """

    data = GPy.util.datasets.crescent_data(seed=seed)
    Y = data['Y']
    Y[Y.flatten()==-1]=0

    m = GPy.models.SparseGPClassification(data['X'], Y,num_inducing=num_inducing)
    m.ensure_default_constraints()
    m['.*len'] = 10.
    #m.update_likelihood_approximation()
    #m.optimize()
    m.pseudo_EM()
    print(m)
    m.plot()
    return m

def FITC_crescent_data(num_inducing=10, seed=default_seed):
    """
    Run a Gaussian process classification with FITC approximation on the crescent data. The demonstration uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param seed : seed value for data generation.
    :type seed: int
    :param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
    :type num_inducing: int
    """

    data = GPy.util.datasets.crescent_data(seed=seed)
    Y = data['Y']
    Y[Y.flatten()==-1]=0

    m = GPy.models.FITCClassification(data['X'], Y,num_inducing=num_inducing)
    m.constrain_bounded('.*len',1.,1e3)
    m.ensure_default_constraints()
    m['.*len'] = 3.
    #m.update_likelihood_approximation()
    #m.optimize()
    m.pseudo_EM()
    print(m)
    m.plot()
    return m
