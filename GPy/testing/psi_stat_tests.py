'''
Created on 22 Apr 2013

@author: maxz
'''
import unittest
import numpy

import GPy
import itertools
from GPy.core import model

class PsiStatModel(model):
    def __init__(self, which, X, X_variance, Z, M, kernel):
        self.which = which
        self.X = X
        self.X_variance = X_variance
        self.Z = Z
        self.N, self.Q = X.shape
        self.M, Q = Z.shape
        assert self.Q == Q, "shape missmatch: Z:{!s} X:{!s}".format(Z.shape, X.shape)
        self.kern = kernel
        super(PsiStatModel, self).__init__()
        self.psi_ = self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance)
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
        return self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance).sum()
    def _log_likelihood_gradients(self):
        psimu, psiS = self.kern.__getattribute__("d" + self.which + "_dmuS")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance)
        try:
            psiZ = self.kern.__getattribute__("d" + self.which + "_dZ")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance)
        except AttributeError:
            psiZ = numpy.zeros(self.M * self.Q)
        thetagrad = self.kern.__getattribute__("d" + self.which + "_dtheta")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance).flatten()
        return numpy.hstack((psimu.flatten(), psiS.flatten(), psiZ.flatten(), thetagrad))

class DPsiStatTest(unittest.TestCase):
    Q = 5
    N = 50
    M = 10
    D = 20
    X = numpy.random.randn(N, Q)
    X_var = .5 * numpy.ones_like(X) + .4 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
    Z = numpy.random.permutation(X)[:M]
    Y = X.dot(numpy.random.randn(Q, D))
#     kernels = [GPy.kern.linear(Q, ARD=True, variances=numpy.random.rand(Q)), GPy.kern.rbf(Q, ARD=True), GPy.kern.bias(Q)]

    kernels = [GPy.kern.linear(Q), GPy.kern.rbf(Q), GPy.kern.bias(Q),
               GPy.kern.linear(Q) + GPy.kern.bias(Q),
               GPy.kern.rbf(Q) + GPy.kern.bias(Q)]

    def testPsi0(self):
        for k in self.kernels:
            m = PsiStatModel('psi0', X=self.X, X_variance=self.X_var, Z=self.Z,
                         M=self.M, kernel=k)
            try:
                assert m.checkgrad(), "{} x psi0".format("+".join(map(lambda x: x.name, k.parts)))
            except:
                import ipdb;ipdb.set_trace()

#     def testPsi1(self):
#         for k in self.kernels:
#             m = PsiStatModel('psi1', X=self.X, X_variance=self.X_var, Z=self.Z,
#                      M=self.M, kernel=k)
#             assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k.parts)))

    def testPsi2_lin(self):
        k = self.kernels[0]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     M=self.M, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_lin_bia(self):
        k = self.kernels[3]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     M=self.M, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_rbf(self):
        k = self.kernels[1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     M=self.M, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_rbf_bia(self):
        k = self.kernels[-1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     M=self.M, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_bia(self):
        k = self.kernels[2]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     M=self.M, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))


if __name__ == "__main__":
    import sys
    interactive = 'i' in sys.argv
    if interactive:
#         N, M, Q, D = 30, 5, 4, 30
#         X = numpy.random.rand(N, Q)
#         k = GPy.kern.linear(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
#         K = k.K(X)
#         Y = numpy.random.multivariate_normal(numpy.zeros(N), K, D).T
#         Y -= Y.mean(axis=0)
#         k = GPy.kern.linear(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
#         m = GPy.models.Bayesian_GPLVM(Y, Q, kernel=k, M=M)
#         m.ensure_default_constraints()
#         m.randomize()
# #         self.assertTrue(m.checkgrad())
        numpy.random.seed(0)
        Q = 5
        N = 50
        M = 10
        D = 15
        X = numpy.random.randn(N, Q)
        X_var = .5 * numpy.ones_like(X) + .1 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
        Z = numpy.random.permutation(X)[:M]
        Y = X.dot(numpy.random.randn(Q, D))
#         kernel = GPy.kern.bias(Q)
#
#         kernels = [GPy.kern.linear(Q), GPy.kern.rbf(Q), GPy.kern.bias(Q),
#                GPy.kern.linear(Q) + GPy.kern.bias(Q),
#                GPy.kern.rbf(Q) + GPy.kern.bias(Q)]

#         for k in kernels:
#             m = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                      M=M, kernel=k)
#             assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k.parts)))
#
#         m0 = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
#                          M=M, kernel=GPy.kern.linear(Q))
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          M=M, kernel=kernel)
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          M=M, kernel=kernel)
#         m2 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          M=M, kernel=GPy.kern.rbf(Q))
        m3 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
                         M=M, kernel=GPy.kern.linear(Q, ARD=True, variances=numpy.random.rand(Q)))
        m3.ensure_default_constraints()
        # + GPy.kern.bias(Q))
#         m4 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          M=M, kernel=GPy.kern.rbf(Q) + GPy.kern.bias(Q))
    else:
        unittest.main()
