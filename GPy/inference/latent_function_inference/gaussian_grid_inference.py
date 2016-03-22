# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

# This implementation of converting GPs to state space models is based on the article:

#@article{Gilboa:2015,
#  title={Scaling multidimensional inference for structured Gaussian processes},
#  author={Gilboa, Elad and Saat{\c{c}}i, Yunus and Cunningham, John P},
#  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
#  volume={37},
#  number={2},
#  pages={424--436},
#  year={2015},
#  publisher={IEEE}
#}

from grid_posterior import GridPosterior
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class GaussianGridInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian and inputs are on a grid.

    The function self.inference returns a GridPosterior object, which summarizes
    the posterior.

    """
    def __init__(self):
        pass

    def kron_mvprod(self, A, b):
        x = b
        N = 1
        D = len(A)
        G = np.zeros((D,1))
        for d in xrange(0, D):
            G[d] = len(A[d])
        N = np.prod(G)
        for d in xrange(D-1, -1, -1):
            X = np.reshape(x, (G[d], round(N/G[d])), order='F')
            Z = np.dot(A[d], X)
            Z = Z.T
            x = np.reshape(Z, (-1, 1), order='F')
        return x

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        """
        Returns a GridPosterior class containing essential quantities of the posterior
        """
        N = X.shape[0] #number of training points
        D = X.shape[1] #number of dimensions

        Kds = np.zeros(D, dtype=object) #vector for holding covariance per dimension
        Qs = np.zeros(D, dtype=object) #vector for holding eigenvectors of covariance per dimension
        QTs = np.zeros(D, dtype=object) #vector for holding transposed eigenvectors of covariance per dimension
        V_kron = 1 # kronecker product of eigenvalues

        # retrieve the one-dimensional variation of the designated kernel
        oneDkernel = kern.getOneDimensionalKernel(D)

        for d in xrange(D):
            xg = list(set(X[:,d])) #extract unique values for a dimension
            xg = np.reshape(xg, (len(xg), 1))
            oneDkernel.lengthscale = kern.lengthscale[d]
            Kds[d] = oneDkernel.K(xg)
            [V, Q] = np.linalg.eig(Kds[d])
            V_kron = np.kron(V_kron, V)
            Qs[d] = Q
            QTs[d] = Q.T

        noise = likelihood.variance + 1e-8

        alpha_kron = self.kron_mvprod(QTs, Y)
        V_kron = V_kron.reshape(-1, 1)
        alpha_kron = alpha_kron / (V_kron + noise)
        alpha_kron = self.kron_mvprod(Qs, alpha_kron)

        log_likelihood = -0.5 * (np.dot(Y.T, alpha_kron) + np.sum((np.log(V_kron + noise))) + N*log_2_pi)

        # compute derivatives wrt parameters Thete
        derivs = np.zeros(D+2, dtype='object')
        for t in xrange(len(derivs)):
            dKd_dTheta = np.zeros(D, dtype='object')
            gamma = np.zeros(D, dtype='object')
            gam = 1
            for d in xrange(D):
                xg = list(set(X[:,d]))
                xg = np.reshape(xg, (len(xg), 1))
                oneDkernel.lengthscale = kern.lengthscale[d]
                if t < D:
                    dKd_dTheta[d] = oneDkernel.dKd_dLen(xg, (t==d), lengthscale=kern.lengthscale[t]) #derivative wrt lengthscale
                elif (t == D):
                    dKd_dTheta[d] = oneDkernel.dKd_dVar(xg) #derivative wrt variance
                else:
                    dKd_dTheta[d] = np.identity(len(xg)) #derivative wrt noise
                gamma[d] = np.diag(np.dot(np.dot(QTs[d], dKd_dTheta[d].T), Qs[d]))
                gam = np.kron(gam, gamma[d])
            
            gam = gam.reshape(-1,1)
            kappa = self.kron_mvprod(dKd_dTheta, alpha_kron)
            derivs[t] = 0.5*np.dot(alpha_kron.T,kappa) - 0.5*np.sum(gam / (V_kron + noise))

        # separate derivatives
        dL_dLen = derivs[:D]
        dL_dVar = derivs[D]
        dL_dThetaL = derivs[D+1]

        return GridPosterior(alpha_kron=alpha_kron, QTs=QTs, Qs=Qs, V_kron=V_kron), log_likelihood, {'dL_dLen':dL_dLen, 'dL_dVar':dL_dVar, 'dL_dthetaL':dL_dThetaL}
