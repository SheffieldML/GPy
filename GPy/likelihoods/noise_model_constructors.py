# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import noise_models

def binomial(gp_link=None):
    """
    Construct a binomial likelihood

    :param gp_link: a GPy gp_link function
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Probit()
    #else:
    #    assert isinstance(gp_link,noise_models.gp_transformations.GPTransformation), 'gp_link function is not valid.'

    if isinstance(gp_link,noise_models.gp_transformations.Probit):
        analytical_mean = True
        analytical_variance = False

    elif isinstance(gp_link,noise_models.gp_transformations.Heaviside):
        analytical_mean = True
        analytical_variance = True

    else:
        analytical_mean = False
        analytical_variance = False

    return noise_models.binomial_noise.Binomial(gp_link,analytical_mean,analytical_variance)

def exponential(gp_link=None):
    """
    Construct a binomial likelihood

    :param gp_link: a GPy gp_link function
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Identity()

    analytical_mean = False
    analytical_variance = False
    return noise_models.exponential_noise.Exponential(gp_link,analytical_mean,analytical_variance)

def gaussian_ep(gp_link=None,variance=1.):
    """
    Construct a gaussian likelihood

    :param gp_link: a GPy gp_link function
    :param variance: scalar
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Identity()
    #else:
    #    assert isinstance(gp_link,noise_models.gp_transformations.GPTransformation), 'gp_link function is not valid.'

    analytical_mean = False
    analytical_variance = False
    return noise_models.gaussian_noise.Gaussian(gp_link,analytical_mean,analytical_variance,variance)

def poisson(gp_link=None):
    """
    Construct a Poisson likelihood

    :param gp_link: a GPy gp_link function
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Log_ex_1()
    #else:
    #    assert isinstance(gp_link,noise_models.gp_transformations.GPTransformation), 'gp_link function is not valid.'
    analytical_mean = False
    analytical_variance = False
    return noise_models.poisson_noise.Poisson(gp_link,analytical_mean,analytical_variance)

def gamma(gp_link=None,beta=1.):
    """
    Construct a Gamma likelihood

    :param gp_link: a GPy gp_link function
    :param beta: scalar
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Log_ex_1()
    analytical_mean = False
    analytical_variance = False
    return noise_models.gamma_noise.Gamma(gp_link,analytical_mean,analytical_variance,beta)


