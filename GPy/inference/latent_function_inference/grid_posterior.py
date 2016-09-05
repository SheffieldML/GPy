# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import numpy as np

class GridPosterior(object):
    """
    Specially intended for the Grid Regression case
    An object to represent a Gaussian posterior over latent function values, p(f|D).

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes.

    """
    def __init__(self, alpha_kron=None, QTs=None, Qs=None, V_kron=None):
        """
        alpha_kron : 
        QTs : transpose of eigen vectors resulting from decomposition of single dimension covariance matrices
        Qs : eigen vectors resulting from decomposition of single dimension covariance matrices
        V_kron : kronecker product of eigenvalues reulting decomposition of single dimension covariance matrices
        """

        if ((alpha_kron is not None) and (QTs is not None) 
            and (Qs is not None) and (V_kron is not None)):
            pass # we have sufficient to compute the posterior
        else:
            raise ValueError("insufficient information for predictions")

        self._alpha_kron = alpha_kron
        self._qTs = QTs
        self._qs = Qs
        self._v_kron = V_kron

    @property
    def alpha(self):
        """
        """
        return self._alpha_kron

    @property
    def QTs(self):
        """
        array of transposed eigenvectors resulting for single dimension covariance
        """
        return self._qTs

    @property
    def Qs(self):
        """
        array of eigenvectors resulting for single dimension covariance
        """
        return self._qs

    @property
    def V_kron(self):
        """
        kronecker product of eigenvalues s
        """
        return self._v_kron
    
