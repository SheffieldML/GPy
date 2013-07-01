# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from likelihood_functions import LikelihoodFunction
import noise_models
import link_functions

def binomial(link=None):
    """
    Construct a binomial likelihood

    :param link: a GPy link function
    """
    #self.discrete = True
    #self.support_limits = (0,1)

    if link is None:
        link = link_functions.Probit()
    else:
        assert isinstance(link,link_functions.LinkFunction), 'link function is not valid.'

    if isinstance(link,link_functions.Probit):
        analytical_moments = True
    else:
        analytical_moments = False
    return noise_models.binomial_likelihood.Binomial(link,analytical_moments)


def poisson(link=None):
    """
    Construct a Poisson likelihood

    :param link: a GPy link function
    """
    if link is None:
        link = link_functions.Log_ex_1()
    else:
        assert isinstance(link,link_functions.LinkFunction), 'link function is not valid.'
    #assert isinstance(link,link_functions.LinkFunction), 'link function is not valid.'
    analytical_moments = False
    return noise_models.poisson_likelihood.Poisson(link,analytical_moments)
