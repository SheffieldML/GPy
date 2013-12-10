# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import noise_models

def bernoulli(gp_link=None):
    """
    Construct a bernoulli likelihood

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

    return noise_models.bernoulli_noise.Bernoulli(gp_link,analytical_mean,analytical_variance)

def exponential(gp_link=None):

    """
    Construct a exponential likelihood

    :param gp_link: a GPy gp_link function
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Log_ex_1()

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

def gaussian(gp_link=None, variance=2, D=None, N=None):
    """
    Construct a Gaussian likelihood

    :param gp_link: a GPy gp_link function
    :param variance: variance
    :type variance: scalar
    :returns: Gaussian noise model:
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Identity()
    analytical_mean = True
    analytical_variance = True # ?
    return noise_models.gaussian_noise.Gaussian(gp_link, analytical_mean,
            analytical_variance, variance=variance, D=D, N=N)

def student_t(gp_link=None, deg_free=5, sigma2=2):
    """
    Construct a Student t likelihood

    :param gp_link: a GPy gp_link function
    :param deg_free: degrees of freedom of student-t
    :type deg_free: scalar
    :param sigma2: variance
    :type sigma2: scalar
    :returns: Student-T noise model
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Identity()
    analytical_mean = True
    analytical_variance = True
    return noise_models.student_t_noise.StudentT(gp_link, analytical_mean,
            analytical_variance,deg_free, sigma2)
