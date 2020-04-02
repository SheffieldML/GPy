# Copyright (c) 2014, James Hensman, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import Model
from paramz import ObsAr
from .. import likelihoods

class GPKroneckerGaussianRegression(Model):
    """
    Kronecker GP regression

    Take two kernels computed on separate spaces K1(X1), K2(X2), and a data
    matrix Y which is f size (N1, N2).

    The effective covaraince is np.kron(K2, K1)
    The effective data is vec(Y) = Y.flatten(order='F')

    The noise must be iid Gaussian.

    See [stegle_et_al_2011]_.

    .. rubric:: References

    .. [stegle_et_al_2011] Stegle, O.; Lippert, C.; Mooij, J.M.; Lawrence, N.D.; Borgwardt, K.:Efficient inference in matrix-variate Gaussian models with \iid observation noise. In: Advances in Neural Information Processing Systems, 2011, Pages 630-638

    """
    def __init__(self, X1, X2, Y, kern1, kern2, noise_var=1., name='KGPR'):
        Model.__init__(self, name=name)

        # accept the construction arguments
        self.X1 = ObsAr(X1)
        self.X2 = ObsAr(X2)
        self.Y = Y
        self.kern1, self.kern2 = kern1, kern2
        self.link_parameter(self.kern1)
        self.link_parameter(self.kern2)

        self.likelihood = likelihoods.Gaussian()
        self.likelihood.variance = noise_var
        self.link_parameter(self.likelihood)

        self.num_data1, self.input_dim1 = self.X1.shape
        self.num_data2, self.input_dim2 = self.X2.shape

        assert kern1.input_dim == self.input_dim1
        assert kern2.input_dim == self.input_dim2
        assert Y.shape == (self.num_data1, self.num_data2)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def parameters_changed(self):
        (N1, D1), (N2, D2) = self.X1.shape, self.X2.shape
        K1, K2 = self.kern1.K(self.X1), self.kern2.K(self.X2)

        # eigendecompositon
        S1, U1 = np.linalg.eigh(K1)
        S2, U2 = np.linalg.eigh(K2)
        W = np.kron(S2, S1) + self.likelihood.variance

        Y_ = U1.T.dot(self.Y).dot(U2)

        # store these quantities: needed for prediction
        Wi = 1./W
        Ytilde = Y_.flatten(order='F')*Wi

        self._log_marginal_likelihood = -0.5*self.num_data1*self.num_data2*np.log(2*np.pi)\
                                        -0.5*np.sum(np.log(W))\
                                        -0.5*np.dot(Y_.flatten(order='F'), Ytilde)

        # gradients for data fit part
        Yt_reshaped = Ytilde.reshape(N1, N2, order='F')
        tmp = U1.dot(Yt_reshaped)
        dL_dK1 = .5*(tmp*S2).dot(tmp.T)
        tmp = U2.dot(Yt_reshaped.T)
        dL_dK2 = .5*(tmp*S1).dot(tmp.T)

        # gradients for logdet
        Wi_reshaped = Wi.reshape(N1, N2, order='F')
        tmp = np.dot(Wi_reshaped, S2)
        dL_dK1 += -0.5*(U1*tmp).dot(U1.T)
        tmp = np.dot(Wi_reshaped.T, S1)
        dL_dK2 += -0.5*(U2*tmp).dot(U2.T)

        self.kern1.update_gradients_full(dL_dK1, self.X1)
        self.kern2.update_gradients_full(dL_dK2, self.X2)

        # gradients for noise variance
        dL_dsigma2 = -0.5*Wi.sum() + 0.5*np.sum(np.square(Ytilde))
        self.likelihood.variance.gradient = dL_dsigma2

        # store these quantities for prediction:
        self.Wi, self.Ytilde, self.U1, self.U2 = Wi, Ytilde, U1, U2

    def predict(self, X1new, X2new):
        """
        Return the predictive mean and variance at a series of new points X1new, X2new
        Only returns the diagonal of the predictive variance, for now.

        :param X1new: The points at which to make a prediction
        :type X1new: np.ndarray, Nnew x self.input_dim1
        :param X2new: The points at which to make a prediction
        :type X2new: np.ndarray, Nnew x self.input_dim2

        """
        k1xf = self.kern1.K(X1new, self.X1)
        k2xf = self.kern2.K(X2new, self.X2)
        A = k1xf.dot(self.U1)
        B = k2xf.dot(self.U2)
        mu = A.dot(self.Ytilde.reshape(self.num_data1, self.num_data2, order='F')).dot(B.T).flatten(order='F')
        k1xx = self.kern1.Kdiag(X1new)
        k2xx = self.kern2.Kdiag(X2new)
        BA = np.kron(B, A)
        var = np.kron(k2xx, k1xx) - np.sum(BA**2*self.Wi, 1) + self.likelihood.variance

        return mu[:, None], var[:, None]
