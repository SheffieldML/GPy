'''
Created on 22 Apr 2013

@author: maxz
'''
import unittest
import numpy

import GPy
import itertools
from GPy.core import Model
from GPy.core.parameterization.param import Param
from GPy.core.parameterization.transformations import Logexp
from GPy.core.parameterization.variational import NormalPosterior

class PsiStatModel(Model):
    def __init__(self, which, X, X_variance, Z, num_inducing, kernel):
        super(PsiStatModel, self).__init__(name='psi stat test')
        self.which = which
        self.X = Param("X", X)
        self.X_variance = Param('X_variance', X_variance, Logexp())
        self.q = NormalPosterior(self.X, self.X_variance)
        self.Z = Param("Z", Z)
        self.N, self.input_dim = X.shape
        self.num_inducing, input_dim = Z.shape
        assert self.input_dim == input_dim, "shape missmatch: Z:{!s} X:{!s}".format(Z.shape, X.shape)
        self.kern = kernel
        self.psi_ = self.kern.__getattribute__(self.which)(self.Z, self.q)
        self.add_parameters(self.q, self.Z, self.kern)

    def log_likelihood(self):
        return self.kern.__getattribute__(self.which)(self.Z, self.X, self.X_variance).sum()

    def parameters_changed(self):
        psimu, psiS = self.kern.__getattribute__("d" + self.which + "_dmuS")(numpy.ones_like(self.psi_), self.Z, self.q)
        self.X.gradient = psimu
        self.X_variance.gradient = psiS
        #psimu, psiS = numpy.ones(self.N * self.input_dim), numpy.ones(self.N * self.input_dim)
        try: psiZ = self.kern.__getattribute__("d" + self.which + "_dZ")(numpy.ones_like(self.psi_), self.Z, self.q)
        except AttributeError: psiZ = numpy.zeros_like(self.Z)
        self.Z.gradient = psiZ
        #psiZ = numpy.ones(self.num_inducing * self.input_dim)
        N,M = self.X.shape[0], self.Z.shape[0]
        dL_dpsi0, dL_dpsi1, dL_dpsi2 = numpy.zeros([N]), numpy.zeros([N,M]), numpy.zeros([N,M,M])
        if self.which == 'psi0': dL_dpsi0 += 1
        if self.which == 'psi1': dL_dpsi1 += 1
        if self.which == 'psi2': dL_dpsi2 += 1
        self.kern.update_gradients_variational(numpy.zeros([1,1]),
                                               dL_dpsi0,
                                               dL_dpsi1,
                                               dL_dpsi2, self.X, self.X_variance, self.Z)

class DPsiStatTest(unittest.TestCase):
    input_dim = 5
    N = 50
    num_inducing = 10
    input_dim = 20
    X = numpy.random.randn(N, input_dim)
    X_var = .5 * numpy.ones_like(X) + .4 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
    Z = numpy.random.permutation(X)[:num_inducing]
    Y = X.dot(numpy.random.randn(input_dim, input_dim))
#     kernels = [GPy.kern.Linear(input_dim, ARD=True, variances=numpy.random.rand(input_dim)), GPy.kern.RBF(input_dim, ARD=True), GPy.kern.Bias(input_dim)]

    kernels = [
               GPy.kern.Linear(input_dim),
               GPy.kern.RBF(input_dim),
               #GPy.kern.Bias(input_dim),
               #GPy.kern.Linear(input_dim) + GPy.kern.Bias(input_dim),
               #GPy.kern.RBF(input_dim) + GPy.kern.Bias(input_dim)
               ]

    def testPsi0(self):
        for k in self.kernels:
            m = PsiStatModel('psi0', X=self.X, X_variance=self.X_var, Z=self.Z,\
                             num_inducing=self.num_inducing, kernel=k)
            m.randomize()
            assert m.checkgrad(), "{} x psi0".format("+".join(map(lambda x: x.name, k._parameters_)))

    def testPsi1(self):
        for k in self.kernels:
            m = PsiStatModel('psi1', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
            m.randomize()
            assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k._parameters_)))

    def testPsi2_lin(self):
        k = self.kernels[0]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                 num_inducing=self.num_inducing, kernel=k)
        m.randomize()
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k._parameters_)))
    def testPsi2_lin_bia(self):
        k = self.kernels[3]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        m.randomize()
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k._parameters_)))
    def testPsi2_rbf(self):
        k = self.kernels[1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        m.randomize()
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k._parameters_)))
    def testPsi2_rbf_bia(self):
        k = self.kernels[-1]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        m.randomize()
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k._parameters_)))
    def testPsi2_bia(self):
        k = self.kernels[2]
        m = PsiStatModel('psi2', X=self.X, X_variance=self.X_var, Z=self.Z,
                     num_inducing=self.num_inducing, kernel=k)
        m.randomize()
        assert m.checkgrad(), "{} x psi2".format("+".join(map(lambda x: x.name, k._parameters_)))


if __name__ == "__main__":
    import sys
    interactive = 'i' in sys.argv
    if interactive:
#         N, num_inducing, input_dim, input_dim = 30, 5, 4, 30
#         X = numpy.random.rand(N, input_dim)
#         k = GPy.kern.Linear(input_dim) + GPy.kern.Bias(input_dim) + GPy.kern.White(input_dim, 0.00001)
#         K = k.K(X)
#         Y = numpy.random.multivariate_normal(numpy.zeros(N), K, input_dim).T
#         Y -= Y.mean(axis=0)
#         k = GPy.kern.Linear(input_dim) + GPy.kern.Bias(input_dim) + GPy.kern.White(input_dim, 0.00001)
#         m = GPy.models.Bayesian_GPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)
#         m.randomize()
# #         self.assertTrue(m.checkgrad())
        numpy.random.seed(0)
        input_dim = 3
        N = 3
        num_inducing = 2
        D = 15
        X = numpy.random.randn(N, input_dim)
        X_var = .5 * numpy.ones_like(X) + .1 * numpy.clip(numpy.random.randn(*X.shape), 0, 1)
        Z = numpy.random.permutation(X)[:num_inducing]
        Y = X.dot(numpy.random.randn(input_dim, D))
#         kernel = GPy.kern.Bias(input_dim)
#
#         kernels = [GPy.kern.Linear(input_dim), GPy.kern.RBF(input_dim), GPy.kern.Bias(input_dim),
#                GPy.kern.Linear(input_dim) + GPy.kern.Bias(input_dim),
#                GPy.kern.RBF(input_dim) + GPy.kern.Bias(input_dim)]

#         for k in kernels:
#             m = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                      num_inducing=num_inducing, kernel=k)
#             assert m.checkgrad(), "{} x psi1".format("+".join(map(lambda x: x.name, k.parts)))
#
        m0 = PsiStatModel('psi0', X=X, X_variance=X_var, Z=Z,
                         num_inducing=num_inducing, kernel=GPy.kern.RBF(input_dim)+GPy.kern.Bias(input_dim))
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=kernel)
#         m1 = PsiStatModel('psi1', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=kernel)
#         m2 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=GPy.kern.RBF(input_dim))
#         m3 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing, kernel=GPy.kern.Linear(input_dim, ARD=True, variances=numpy.random.rand(input_dim)))
        # + GPy.kern.Bias(input_dim))
#         m = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
#                          num_inducing=num_inducing,
#                          kernel=(
#             GPy.kern.RBF(input_dim, ARD=1)
#             +GPy.kern.Linear(input_dim, ARD=1)
#             +GPy.kern.Bias(input_dim))
#                          )
#         m.ensure_default_constraints()
        m2 = PsiStatModel('psi2', X=X, X_variance=X_var, Z=Z,
                         num_inducing=num_inducing, kernel=(
            GPy.kern.RBF(input_dim, numpy.random.rand(), numpy.random.rand(input_dim), ARD=1)
            #+GPy.kern.Linear(input_dim, numpy.random.rand(input_dim), ARD=1)
            #+GPy.kern.RBF(input_dim, numpy.random.rand(), numpy.random.rand(input_dim), ARD=1)
            #+GPy.kern.RBF(input_dim, numpy.random.rand(), numpy.random.rand(), ARD=0)
            +GPy.kern.Bias(input_dim)
            +GPy.kern.White(input_dim)
            )
            )
        #m2.ensure_default_constraints()
    else:
        unittest.main()
