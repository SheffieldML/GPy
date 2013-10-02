'''
Created on 4 Sep 2013

@author: maxz
'''
import unittest
from GPy.kern.constructors import rbf, linear, white
from GPy.models.gp_regression import GPRegression
import numpy
from GPy.models.bayesian_gplvm import BayesianGPLVM
from GPy.core.parameter import Parameter, Parameters


class Test(unittest.TestCase):
    N, D, Q = 100, 1, 2
    def setUp(self):
        self.rbf_variance = numpy.random.rand()
        self.rbf_lengthscale = numpy.random.rand(self.Q)
        self.linear_variance = numpy.random.rand(self.Q)
        self.kern = (rbf(self.Q, self.rbf_variance, self.rbf_lengthscale, ARD=True)
                     + linear(self.Q, self.linear_variance, ARD=True))
        self.X = numpy.random.rand(self.N, self.Q) + 10
        self.X_variance = numpy.random.rand(self.N, self.Q) * .2

        K = self.kern.K(self.X)

        self.Y = numpy.random.multivariate_normal(numpy.zeros(self.N), K + numpy.eye(self.N) * .2, self.D).T

        self.bgplvm = BayesianGPLVM(self.Y, self.Q, self.X, self.X_variance, kernel=self.kern)
        self.bgplvm.ensure_default_constraints()

        self.parameter = Parameters([
                                     Parameters([
                                                 Parameter('X', self.X),
                                                 Parameter('X_variance', self.X_variance),
                                                 ],
                                                prefix='X'),
                                     Parameter('iip', self.bgplvm.Z),
                                     Parameters([
                                                 Parameter('rbf_variance', self.rbf_variance),
                                                 Parameter('rbf_lengthscale', self.rbf_lengthscale)
                                                 ],
                                                'rbf'
                                                ),
                                     Parameter('linear_variance', self.linear_variance),
                                     Parameter('noise_variance', self.linear_variance),
                                     ])

    def tearDown(self):
        pass


    def testGrepParamNames(self):
        assert(self.bgplvm.grep_param_names('X_\d') == self.parameter.grep_param_names('X_\d'))
        assert(self.bgplvm.grep_param_names('X_\d+_1') == self.parameter.grep_param_names('X_\d+_1'))
        assert(self.bgplvm.grep_param_names('X_\d_1') == self.parameter.grep_param_names('X_\d_1'))
        assert(self.bgplvm.grep_param_names('X_.+_1') == self.parameter.grep_param_names('X_.+_1'))
        assert(self.bgplvm.grep_param_names('X_1_1') == self.parameter.grep_param_names('X_1_1'))
        assert(self.bgplvm.grep_param_names('X') == self.parameter.grep_param_names('X'))
        assert(self.bgplvm.grep_param_names('rbf') == self.parameter.grep_param_names('rbf'))
        assert(self.bgplvm.grep_param_names('rbf_l.*_1') == self.parameter.grep_param_names('rbf_l.*_1'))
        assert(self.bgplvm.grep_param_names('l') == self.parameter.grep_param_names('l'))
        assert(self.bgplvm.grep_param_names('dont_match') == self.parameter.grep_param_names('dont_match'))
        assert(self.bgplvm.grep_param_names('.*') == self.parameter.grep_param_names('.*'))

    def testConstraints(self):
        assert(self.bgplvm.constraints)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
