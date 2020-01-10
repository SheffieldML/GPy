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

    See Stegle et al.
    @inproceedings{stegle2011efficient,
      title={Efficient inference in matrix-variate gaussian models with $\\backslash$ iid observation noise},
      author={Stegle, Oliver and Lippert, Christoph and Mooij, Joris M and Lawrence, Neil D and Borgwardt, Karsten M},
      booktitle={Advances in Neural Information Processing Systems},
      pages={630--638},
      year={2011}
    }

    """
    def __init__(self, Xs, Y, kerns, noise_var=1., name='KGPR'):
        Model.__init__(self, name=name)

        # accept the construction arguments
        for i, (X, kern) in enumerate(zip(Xs, kerns)):
            assert X.shape[1] == Y.shape[i], "Invalid shape in dimension %d of Y"%i
            assert kern.input_dim == X.shape[1], "Invalid shape dimension %d of kernel"%i

            setattr(self, "num_data%d"%i, X.shape[0])
            setattr(self, "input_dim%d"%i, X.shape[1])

            setattr(self, "X%d"%i, ObsAr(X))
            setattr(self, "kern%d"%i, kern)

            self.link_parameter(kern)

        self.Y = Y

        self.likelihood = likelihoods.Gaussian()
        self.likelihood.variance = noise_var
        self.link_parameter(self.likelihood)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def parameters_changed(self):

        Ss, Us = [], []

        for i in xrange(len(self.Y.shape)):
            X = getattr(self, "X%d"%i)
            kern = getattr(self, "kern%d"%i)
            K = kern(X)
            S, U = np.linalg.eigh(K)

            Ss.append(S)
            Us.append(U)

        W = np.kron(Ss[1], Ss[0])

        for s in Ss[2:]:
            W = np.kron(s, W)

        W+=self.likelihood.variance

        Y_ =np.tensordot(Us[0], self.Y, axes = [[-1], [-1]])

        # TODO reversed order?
        for u in Us[1:]:
            Y_ = np.tensordot(u, Y_, axes  = [[-1], [-1]])

        # store these quantities: needed for prediction
        Wi = 1./W
        Ytilde = Y_.flatten(order='F')*Wi

        num_data_prod = np.prod([getattr(self, "num_data%d"%i) for i in xrange(len(self.Y.shape))])

        self._log_marginal_likelihood = -0.5*num_data_prod*np.log(2*np.pi)\
                                        -0.5*np.sum(np.log(W))\
                                        -0.5*np.dot(Y_.flatten(order='F'), Ytilde)

        # gradients for data fit part
        Yt_reshaped = np.reshape(Ytilde, self.Y.shape, order='F')
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
