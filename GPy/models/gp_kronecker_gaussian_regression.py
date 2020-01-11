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


        dims = len(self.Y.shape)
        Ss, Us = [], []

        for i in xrange(dims):
            X = getattr(self, "X%d"%i)
            kern = getattr(self, "kern%d"%i)
            K = kern(X)
            S, U = np.linalg.eigh(K)

            Ss.append(S)
            Us.append(U)

        W = reduce(np.kron, reversed(Ss))
        W+=self.likelihood.variance

        Y_list = [self.Y].extend(Us)
        Y_ = reduce(lambda x,y: np.tensordot(y,x, axes = [[1], [-1]]), Y_list)


        # store these quantities: needed for prediction
        Wi = 1./W
        Ytilde = Y_.flatten(order='F')*Wi

        num_data_prod = np.prod([getattr(self, "num_data%d"%i) for i in xrange(len(self.Y.shape))])

        self._log_marginal_likelihood = -0.5*num_data_prod*np.log(2*np.pi)\
                                        -0.5*np.sum(np.log(W))\
                                        -0.5*np.dot(Y_.flatten(order='F'), Ytilde)

        # gradients for data fit part
        Yt_reshaped = np.reshape(Ytilde, self.Y.shape, order='F')
        Wi_reshaped = np.reshape(Wi, self.Y.shape, order='F')

        for i in xrange(dims):
            U = Us[i]
            tmp =np.tensordot(U, Yt_reshaped.T, axes = [[1], [i]])
            S = reduce(np.multiply.outer, [s for j,s in enumerate(Ss) if i!=j])

            tmps = tmp*S
            dL_dK = .5 * (np.tensordot(tmps, tmp.T, axes = dims-1))

            axes = [[k for k in xrange(dims - 1, 0, -1)], [j for j in xrange(dims - 1)]]
            tmp = np.tensordot(Wi_reshaped, S.T, axes=axes)

            dL_dK+=-0.5*np.dot(U*tmp, U.T)

            getattr(self, "kern%d"%i).update_gradients_full(dL_dK, getattr(self, "X%d"%i))


        # gradients for noise variance
        dL_dsigma2 = -0.5*Wi.sum() + 0.5*np.sum(np.square(Ytilde))
        self.likelihood.variance.gradient = dL_dsigma2

        # store these quantities for prediction:
        self.Wi, self.Ytilde = Wi, Ytilde

        for i,u in enumerate(Us):
            setattr(self, "U%d"%i, u)

    def predict(self, Xnews):
        """
        Return the predictive mean and variance at a series of new points X1new, X2new
        Only returns the diagonal of the predictive variance, for now.

        :param X1new: The points at which to make a prediction
        :type X1new: np.ndarray, Nnew x self.input_dim1
        :param X2new: The points at which to make a prediction
        :type X2new: np.ndarray, Nnew x self.input_dim2

        """

        embeds = []
        kxxs = []
        dims = len(self.Y.shape)
        for i in xrange(dims):
            kxf = getattr(self, "kern%d"%i)(Xnews[i], getattr(self, "X%d"%i))

            embeds.append(kxf.dot(getattr(self, "U%d"%i)))
            kxxs.append(kxf.Kdiag(Xnews[i]))




        Y_list = [self.Y].extend(embeds)
        mu = reduce(lambda x, y: np.tensordot(y, x, axes=[[1], [-1]]), Y_list).flatten(order='F')
        #mu = A.dot(self.Ytilde.reshape(self.num_data1, self.num_data2, order='F')).dot(B.T).flatten(order='F')

        kron_embeds = reduce(np.kron, reversed(embeds))
        var = reduce(np.kron, reversed(kxxs))- np.sum(kron_embeds**2*self.Wi, 1) + self.likelihood.variance

        return mu[:, None], var[:, None]
