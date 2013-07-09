# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import noise_models
#from likelihood_functions import LikelihoodFunction
#import gp_transformations

def binomial(gp_link=None):
    """
    Construct a binomial likelihood

    :param gp_link: a GPy gp_link function
    """
    #self.discrete = True
    #self.support_limits = (0,1)

    if gp_link is None:
        gp_link = noise_models.gp_transformations.Probit()
    else:
        assert isinstance(gp_link,noise_models.gp_transformations.GPTransformation), 'gp_link function is not valid.'

    if isinstance(gp_link,noise_models.gp_transformations.Probit):
        analytical_moments = True
    else:
        analytical_moments = False
    return noise_models.binomial_noise.Binomial(gp_link,analytical_moments)


def poisson(gp_link=None):
    """
    Construct a Poisson likelihood

    :param gp_link: a GPy gp_link function
    """
    if gp_link is None:
        gp_link = noise_models.gp_transformations.Log_ex_1()
    else:
        assert isinstance(gp_link,noise_models.gp_transformations.GPTransformation), 'gp_link function is not valid.'
    #assert isinstance(gp_link,gp_transformations.GPTransformation), 'gp_link function is not valid.'
    analytical_moments = False
    return noise_models.poisson_noise.Poisson(gp_link,analytical_moments)
