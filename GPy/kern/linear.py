# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
from ..util.linalg import tdot

class linear(kernpart):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^D \sigma^2_i x_iy_i

    :param D: the number of input dimensions
    :type D: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there is only one variance parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel has only one variance parameter \sigma^2, otherwise there is one variance parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object
    """

    def __init__(self, D, variances=None, ARD=False):
        self.D = D
        self.ARD = ARD
        if ARD == False:
            self.Nparam = 1
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
            self._Xcache, self._X2cache = np.empty(shape=(2,))
        else:
            self.Nparam = self.D
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.D, "bad number of lengthscales"
            else:
                variances = np.ones(self.D)
        self._set_params(variances.flatten())

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._params = np.empty(shape=(3, 1))

    def _get_params(self):
        return self.variances

    def _set_params(self, x):
        assert x.size == (self.Nparam)
        self.variances = x
        self.variances2 = np.square(self.variances)

    def _get_param_names(self):
        if self.Nparam == 1:
            return ['variance']
        else:
            return ['variance_%i' % i for i in range(self.variances.size)]

    def K(self, X, X2, target):
        if self.ARD:
            XX = X * np.sqrt(self.variances)
            if X2 is None:
                target += tdot(XX)
            else:
                XX2 = X2 * np.sqrt(self.variances)
                target += np.dot(XX, XX2.T)
        else:
            self._K_computations(X, X2)
            target += self.variances * self._dot_product

    def Kdiag(self, X, target):
        np.add(target, np.sum(self.variances * np.square(X), -1), target)

    def dK_dtheta(self, dL_dK, X, X2, target):
        if self.ARD:
            if X2 is None:
                [np.add(target[i:i + 1], np.sum(dL_dK * tdot(X[:, i:i + 1])), target[i:i + 1]) for i in range(self.D)]
            else:
                product = X[:, None, :] * X2[None, :, :]
                target += (dL_dK[:, :, None] * product).sum(0).sum(0)
        else:
            self._K_computations(X, X2)
            target += np.sum(self._dot_product * dL_dK)

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        tmp = dL_dKdiag[:, None] * X ** 2
        if self.ARD:
            target += tmp.sum(0)
        else:
            target += tmp.sum()

    def dK_dX(self, dL_dK, X, X2, target):
        target += (((X2[:, None, :] * self.variances)) * dL_dK[:, :, None]).sum(0)

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += np.sum(self.variances * self.mu2_S, 1)

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        tmp = dL_dpsi0[:, None] * self.mu2_S
        if self.ARD:
            target += tmp.sum(0)
        else:
            target += tmp.sum()

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S, target_mu, target_S):
        target_mu += dL_dpsi0[:, None] * (2.0 * mu * self.variances)
        target_S += dL_dpsi0[:, None] * self.variances

    def psi1(self, Z, mu, S, target):
        """the variance, it does nothing"""
        self._psi1 = self.K(mu, Z, target)

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S, target):
        """the variance, it does nothing"""
        self.dK_dtheta(dL_dpsi1, mu, Z, target)

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S, target_mu, target_S):
        """Do nothing for S, it does not affect psi1"""
        self._psi_computations(Z, mu, S)
        target_mu += (dL_dpsi1.T[:, :, None] * (Z * self.variances)).sum(1)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S, target):
        self.dK_dX(dL_dpsi1.T, Z, mu, target)

    def psi2(self, Z, mu, S, target):
        """
        returns N,M,M matrix
        """
        self._psi_computations(Z, mu, S)
#         psi2_old = self.ZZ * np.square(self.variances) * self.mu2_S[:, None, None, :]
#         target += psi2.sum(-1)
        # slow way of doing it, but right
#         psi2_real = rm np.zeros((mu.shape[0], Z.shape[0], Z.shape[0]))
#         for n in range(mu.shape[0]):
#             for m_prime in range(Z.shape[0]):
#                 for m in range(Z.shape[0]):
#                     tmp = self._Z[m:m + 1] * self.variances
#                     tmp = np.dot(tmp, (tdot(self._mu[n:n + 1].T) + np.diag(S[n])))
#                     psi2_real[n, m, m_prime] = np.dot(tmp, (
#                             self._Z[m_prime:m_prime + 1] * self.variances).T)
#         mu2_S = (self._mu[:, None, :] * self._mu[:, :, None])
#         mu2_S[:, np.arange(self.D), np.arange(self.D)] += self._S
#         psi2 = (self.ZA[None, :, None, :] * mu2_S[:, None]).sum(-1)
#         psi2 = (psi2[:, :, None] * self.ZA[None, None]).sum(-1)
#         psi2_tensor = np.tensordot(self.ZZ[None, :, :, :] * np.square(self.variances), self.mu2_S[:, None, None, :], ((3), (3))).squeeze().T
        target += self._psi2

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        tmp = dL_dpsi2[:, :, :, None] * (self.ZAinner[:, :, None, :] * (2 * Z)[None, None, :, :])
        if self.ARD:
            target += tmp.sum(0).sum(0).sum(0)
        else:
            target += tmp.sum()

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        """Think N,M,M,Q """
        self._psi_computations(Z, mu, S)
        AZZA = self.ZA.T[:, None, :, None] * self.ZA[None, :, None, :]
        AZZA = AZZA + AZZA.swapaxes(1, 2)
        target_S += (dL_dpsi2[:, :, :, None] * self.ZA[None, :, None, :] * self.ZA[None, None, :, :]).sum(1).sum(1)
        dpsi2_dmu = (dL_dpsi2[:, :, :, None] * np.tensordot(mu, AZZA, (-1, 0))).sum(1).sum(1)
        target_mu += dpsi2_dmu

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
#         mu2_S = np.sum(self.mu2_S, 0)  # Q,
#         import ipdb;ipdb.set_trace()
#         psi2_dZ_real = np.zeros((mu.shape[0], Z.shape[0], Z.shape[1]))
#         for n in range(mu.shape[0]):
#             for m in range(Z.shape[0]):
#                 tmp = self.variances * (tdot(self._mu[n:n + 1].T) + np.diag(S[n]))
#                 psi2_dZ_real[n, m, :] = np.dot(tmp, (
#                             self._Z[m:m + 1] * self.variances).T).T
#                 tmp = self._Z[m:m + 1] * self.variances
#                 tmp = np.dot(tmp, (tdot(self._mu[n:n + 1].T) + np.diag(S[n])))
#                 psi2_dZ_real[n, m, :] = tmp * self.variances
#                 for m_prime in range(Z.shape[0]):
#                     if m == m_prime:
#                         psi2_dZ_real[n, m, :] *= 2
#         prod = (dL_dpsi2[:, :, :, None] * np.eye(Z.shape[0])[None, :, :, None] * (self.ZAinner * self.variances).swapaxes(0, 1)[:, :, None, :])
#         psi2_dZ = prod.swapaxes(1, 2) + prod
        psi2_dZ = dL_dpsi2[:, :, :, None] * self.variances * self.ZAinner[:, :, None, :]
        target += psi2_dZ.sum(0).sum(0)
#         import ipdb;ipdb.set_trace()
#         psi2_dZ_old = (dL_dpsi2[:, :, :, None] * (self.mu2_S[:, None, None, :] * (Z * np.square(self.variances)[None, :])[None, None, :, :])).sum(0).sum(1)
#         target += (dL_dpsi2[:, :, :, None] * psi2_dZ_real[:, :, None, :]).sum(0).sum(0) * 2  # (self.variances * np.dot(self.inner, self.ZA.T)).sum(1)

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self, X, X2):
        if not (np.array_equal(X, self._Xcache) and np.array_equal(X2, self._X2cache)):
            self._Xcache = X.copy()
            if X2 is None:
                self._dot_product = tdot(X)
                self._X2cache = None
            else:
                self._X2cache = X2.copy()
                self._dot_product = np.dot(X, X2.T)

    def _psi_computations(self, Z, mu, S):
        # here are the "statistics" for psi1 and psi2
        Zv_changed = not (np.array_equal(Z, self._Z) and np.array_equal(self.variances, self._variances))
        muS_changed = not (np.array_equal(mu, self._mu) and np.array_equal(S, self._S))
        if Zv_changed:
            # Z has changed, compute Z specific stuff
            # self.ZZ = Z[:,None,:]*Z[None,:,:] # M,M,Q
#             self.ZZ = np.empty((Z.shape[0], Z.shape[0], Z.shape[1]), order='F')
#             [tdot(Z[:, i:i + 1], self.ZZ[:, :, i].T) for i in xrange(Z.shape[1])]
            self.ZA = Z * self.variances
            self._Z = Z.copy()
            self._variances = self.variances.copy()
        if muS_changed:
            self.mu2_S = np.square(mu) + S
            self.inner = (mu[:, None, :] * mu[:, :, None])
            diag_indices = np.diag_indices(mu.shape[1], 2)
            self.inner[:, diag_indices[0], diag_indices[1]] += S
            self._mu, self._S = mu.copy(), S.copy()
        if Zv_changed or muS_changed:
            self.ZAinner = np.dot(self.ZA, self.inner).swapaxes(0, 1)  # NOTE: self.ZAinner \in [M x N x Q]!
            self._psi2 = np.dot(self.ZAinner, self.ZA.T)
