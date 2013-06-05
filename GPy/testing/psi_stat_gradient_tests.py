'''
Created on 22 Apr 2013

@author: maxz
'''
import unittest
import numpy

import GPy
import itertools
from GPy.core import Model

class PsiStatModel(Model):
    def __init__(self, which, X, X_variance, Z, num_inducing, kernel):
        self.which = which
        self.X = X
        self.X_variance = X_variance
        self.Z = Z
        self.N, self.input_dim = X.shape
        self.num_inducing, input_dim = Z.shape
        assert self.input_dim == input_dim, "shape missmatch: Z:{!s} X:{!s}".format(Z.shape, X.shape)
        self.kern = kernel
        super(PsiStatModel, self).__init__()
        self.psi_ = self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance)
    def _get_param_names(self):
        Xnames = ["{}_{}_{}".format(what, i, j) for what, i, j in itertools.product(['X', 'X_variance'], range(self.N), range(self.input_dim))]
        Znames = ["Z_{}_{}".format(i, j) for i, j in itertools.product(range(self.num_inducing), range(self.input_dim))]
        return Xnames + Znames + self.kern._get_param_names()
    def _get_params(self):
        return numpy.hstack([self.X.flatten(), self.X_variance.flatten(), self.Z.flatten(), self.kern._get_params()])
    def _set_params(self, x, save_old=True, save_count=0):
        start, end = 0, self.X.size
        self.X = x[start:end].reshape(self.N, self.input_dim)
        start, end = end, end + self.X_variance.size
        self.X_variance = x[start: end].reshape(self.N, self.input_dim)
        start, end = end, end + self.Z.size
        self.Z = x[start: end].reshape(self.num_inducing, self.input_dim)
        self.kern._set_params(x[end:])
    def log_likelihood(self):
        return self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance).sum()
    def _log_likelihood_gradients(self):
        psimu, psiS = self.kern.__getattribute__("d" + self.which + "_dmuS")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance)
        try:
            psiZ = self.kern.__getattribute__("d" + self.which + "_dZ")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance)
        except AttributeError:
            psiZ = numpy.zeros(self.num_inducing * self.input_dim)
        thetagrad = self.kern.__getattribute__("d" + self.which + "_dtheta")(numpy.ones_like(self.psi_), self.Z, self.X, self.X_variance).flatten()
        return numpy.hstack((psimu.flatten(), psiS.flatten(), psiZ.flatten(), thetagrad))

class DPsiStatTest(unittest.TestCase):
    input_dim = 5
    N = 50
    num_inducing = 10
    input_dim = 20
    X = numpy.random.randn(N, input_dim)
    X_var = .5 * numpy.ones_like(X) + .4 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
    Z = numpy.random.permutation(X)[:num_inducing]
    Y = X.dot(numpy.random.randn(input_dim, input_dim))
#     kernels = [GPy.kern.linear(input_dim, ARD=True, variances=numpy.random.rand(input_dim)), GPy.kern.rbf(input_dim, ARD=True), GPy.kern.bias(input_dim)]

    kernels = [GPy.kern.linear(input_dim), GPy.kern.rbf(input_dim), GPy.kern.bias(input_dim),
               GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim),
               GPy.kern.rbf(input_dim) + GPy.kern.bias(input_dim)]

    def testPsi0(self):
        for k in self.kernels:
            m = PsiStatModel('psi0', X=self.X, X_variance=self.X_var, Z=self.Z,
                             num_inducing=self.num_inducing, kernel=k)
            assert m.checkgrad(), "{} x psi0".format("+".join(map(lambda x: x.name, k.parts)))

#     def testPsi1(self):
#         for k in self.kernels:
#             m = PsiStatModel('psi1', X=self.X, X_variance=self.X_var, Z=self.Z,
#                      num_inducing=self.num_inducing, kernel=k)
#             assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k.parts)))

    def testPsi2_lin(self):
        k = self.kernels[0]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_lin_bia(self):
        k = self.kernels[3]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_rbf(self):
        k = self.kernels[1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_rbf_bia(self):
        k = self.kernels[-1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))
    def testPsi2_bia(self):
        k = self.kernels[2]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k.parts)))


if __name__ == "__main__":
    import sys
    interactive = 'i' in sys.argv
    if interactive:
#         N, num_inducing, input_dim, input_dim = 30, 5, 4, 30
#         X = numpy.random.rand(N, input_dim)
#         k = GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim, 0.00001)
#         K = k.K(X)
#         Y = numpy.random.multivariate_normal(numpy.zeros(N), K, input_dim).T
#         Y -= Y.mean(axis=0)
#         k = GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim) + GPy.kern.white(input_dim, 0.00001)
#         m = GPy.models.Bayesian_GPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)
#         m.ensure_default_constraints()
#         m.randomize()
# #         self.assertTrue(m.checkgrad())
        numpy.random.seed(0)
        input_dim = 5
        N = 50
        num_inducing = 10
        D = 15
        X = numpy.random.randn(N, input_dim)
        X_var = .5 * numpy.ones_like(X) + .1 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
        Z = numpy.random.permutation(X)[:num_inducing]
        Y = X.dot(numpy.random.randn(input_dim, D))
#         kernel = GPy.kern.bias(input_dim)
#
#         kernels = [GPy.kern.linear(input_dim), GPy.kern.rbf(input_dim), GPy.kern.bias(input_dim),
#                GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim),
#                GPy.kern.rbf(input_dim) + GPy.kern.bias(input_dim)]

#         for k in kernels:
#             m = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                      num_inducing=num_inducing, kernel=k)
#             assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k.parts)))
#
#         m0 = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=GPy.kern.linear(input_dim))
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=kernel)
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=kernel)
#         m2 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=GPy.kern.rbf(input_dim))
        m3 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
                         num_inducing=num_inducing, kernel=GPy.kern.linear(input_dim, ARD=True, variances=numpy.random.rand(input_dim)))
        m3.ensure_default_constraints()
        # + GPy.kern.bias(input_dim))
#         m4 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=GPy.kern.rbf(input_dim) + GPy.kern.bias(input_dim))
    else:
        unittest.main()
