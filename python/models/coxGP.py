# Copyright (c) 2013, Alan Saul

from GPy.models import GP
from .. import likelihoods
from GPy import kern


class cox_GP_regression(GP):
    """
    Cox Gaussian Process model for regression
    """

    def __init__(self,X,Y,kernel=None,normalize_X=False,normalize_Y=False, Xslices=None):
        if kernel is None:
            kernel = kern.rbf(X.shape[1])

        likelihood = likelihoods.cox_piecewise(Y, normalize=normalize_Y)

        GP.__init__(self, X, likelihood, kernel, normalize_X=normalize_X, Xslices=Xslices)
