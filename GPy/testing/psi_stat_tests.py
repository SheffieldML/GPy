'''
Created on 22 Apr 2013

@author: maxz
'''
import unittest
import numpy

from GPy.models.Bayesian_GPLVM import Bayesian_GPLVM
import GPy
import itertools
from GPy.core import model

class PsiStatModel(model):
    def __init__(self, which, X, X_variance, Z, M, kernel, mu_or_S, dL_=numpy.ones((1, 1))):
        self.which = which
        self.dL_ = dL_
        self.X = X
        self.X_variance = X_variance
        self.Z = Z
        self.N, self.Q = X.shape
        self.M, Q = Z.shape
        self.mu_or_S = mu_or_S
        assert self.Q == Q, "shape missmatch: Z:{!s} X:{!s}".format(Z.shape, X.shape)
        self.kern = kernel
        super(PsiStatModel, self).__init__()
    def _get_param_names(self):
        Xnames = ["{}_{}_{}".format(what, i, j) for what, i, j in itertools.product(['X', 'X_variance'], range(self.N), range(self.Q))]
        Znames = ["Z_{}_{}".format(i, j) for i, j in itertools.product(range(self.M), range(self.Q))]
        return Xnames + Znames + self.kern._get_param_names()
    def _get_params(self):
        return numpy.hstack([self.X.flatten(), self.X_variance.flatten(), self.Z.flatten(), self.kern._get_params()])
    def _set_params(self, x, save_old=True, save_count=0):
        start, end = 0, self.X.size
        self.X = x[start:end].reshape(self.N, self.Q)
        start, end = end, end + self.X_variance.size
        self.X_variance = x[start: end].reshape(self.N, self.Q)
        start, end = end, end + self.Z.size
        self.Z = x[start: end].reshape(self.M, self.Q)
        self.kern._set_params(x[end:])
    def log_likelihood(self):
#         if '2' in self.which:
#             norm = self.N ** 2
#         else:  # '0', '1' in self.which:
#             norm = self.N
        return self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance).sum()
    def _log_likelihood_gradients(self):
        psi_ = self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance)
        psimu, psiS = self.kern.__getattribute__("d" + self.which + "_dmuS")(numpy.ones_like(psi_), self.Z, self.X, self.X_variance)
        try:
            psiZ = self.kern.__getattribute__("d" + self.which + "_dZ")(numpy.ones_like(psi_), self.Z, self.X, self.X_variance)
        except AttributeError:
            psiZ = numpy.zeros(self.M * self.Q)
        thetagrad = self.kern.__getattribute__("d" + self.which + "_dtheta")(numpy.ones_like(psi_), self.Z, self.X, self.X_variance).flatten()
        return numpy.hstack((psimu.flatten(), psiS.flatten(), psiZ.flatten(), thetagrad))

class Test(unittest.TestCase):
    Q = 5
    N = 50
    M = 10
    D = 10
    X = numpy.random.randn(N, Q)
    X_var = .5 * numpy.ones_like(X) + .4 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
    Z = numpy.random.permutation(X)[:M]
    Y = X.dot(numpy.random.randn(Q, D))

    def testPsi0(self):
        kernel = GPy.kern.linear(Q)
        m = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL=numpy.ones((1)))
        assert m.checkgrad(), "linear x psi0"

    def testPsi1(self):
        kernel = GPy.kern.linear(Q)
        m = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL=numpy.ones((1, 1)))
        assert(m.checkgrad())

    def testPsi2(self):
        kernel = GPy.kern.linear(Q)
        m = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL=numpy.ones((1, 1, 1)))
        assert(m.checkgrad())


if __name__ == "__main__":
    Q = 5
    N = 50
    M = 10
    D = 10
    X = numpy.random.randn(N, Q)
    X_var = .5 * numpy.ones_like(X) + .4 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
    Z = numpy.random.permutation(X)[:M]
    Y = X.dot(numpy.random.randn(Q, D))
    kernel = GPy.kern.linear(Q)  # GPy.kern.bias(Q)  # GPy.kern.linear(Q) + GPy.kern.rbf(Q)
    m0 = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL_=numpy.ones((1)))
    m1 = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL_=numpy.ones((1)))
    m2 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
                     M=M, kernel=kernel, mu_or_S=0, dL_=numpy.ones((1, 1, 1)))

