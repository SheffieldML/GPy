'''
Created on 4 Sep 2013

@author: maxz
'''
import unittest
from GPy.kern.constructors import rbf, linear, white
from GPy.models.gp_regression import GPRegression
import numpy
from GPy.models.bayesian_gplvm import BayesianGPLVM
from GPy.core.parameter import Param, Parameterized
from GPy.likelihoods.gaussian import Gaussian


class Test(unittest.TestCase):
    N, D, Q = 10, 6, 4
    def setUp(self):
        self.rbf_variance = numpy.random.rand()
        self.rbf_lengthscale = numpy.random.rand(self.Q)
        self.linear_variance = numpy.random.rand(self.Q)
        self.noise_variance = numpy.random.rand(1)
        self.kern = (rbf(self.Q, self.rbf_variance, self.rbf_lengthscale, ARD=True)
                     + linear(self.Q, self.linear_variance, ARD=True)
                     + white(self.Q, self.noise_variance))
        self.X = numpy.random.rand(self.N, self.Q) + 10
        self.X_variance = numpy.random.rand(self.N, self.Q) * .2

        K = self.kern.K(self.X)

        self.Y = numpy.random.multivariate_normal(numpy.zeros(self.N), K + numpy.eye(self.N) * .2, self.D).T
        
        self.bgplvm = BayesianGPLVM(Gaussian(self.Y, variance=self.noise_variance), self.Q, self.X, self.X_variance, kernel=self.kern)
        self.bgplvm.ensure_default_constraints(warning=False)
        self.bgplvm.tie_params("noise_variance|white_variance")
        self.bgplvm.constrain_fixed("rbf_var", warning=False)
        self.parameter = Parameterized([
                                    Parameterized([
                                                Param('X', self.X),
                                                Param('X_variance', self.X_variance),
                                                ]),
                                    Param('iip', self.bgplvm.Z),
                                    Parameterized([
                                                Param('rbf_variance', self.rbf_variance),
                                                Param('rbf_lengthscale', self.rbf_lengthscale)
                                                ]),
                                    Param('linear_variance', self.linear_variance),
                                    Param('white_variance', self.noise_variance),
                                    Param('noise_variance', self.noise_variance),
                                     ])
        
        self.parameter['.*variance'].constrain_positive(False)
        self.parameter['.*length'].constrain_positive(False)
        self.parameter.white.tie_to(self.parameter.noise)
        self.parameter.rbf_var.constrain_fixed(False)

    def tearDown(self):
        pass

#     def testGrepParamNamesTest(self):
#         assert(self.bgplvm.grep_param_names('X_\d') == self.parameter.grep_param_names('X_\d'))
#         assert(self.bgplvm.grep_param_names('X_\d+_1') == self.parameter.grep_param_names('X_\d+_1'))
#         assert(self.bgplvm.grep_param_names('X_\d_1') == self.parameter.grep_param_names('X_\d_1'))
#         assert(self.bgplvm.grep_param_names('X_.+_1') == self.parameter.grep_param_names('X_.+_1'))
#         assert(self.bgplvm.grep_param_names('X_1_1') == self.parameter.grep_param_names('X_1_1'))
#         assert(self.bgplvm.grep_param_names('X') == self.parameter.grep_param_names('X'))
#         assert(self.bgplvm.grep_param_names('rbf') == self.parameter.grep_param_names('rbf'))
#         assert(self.bgplvm.grep_param_names('rbf_l.*_1') == self.parameter.grep_param_names('rbf_l.*_1'))
#         assert(self.bgplvm.grep_param_names('l') == self.parameter.grep_param_names('l'))
#         assert(self.bgplvm.grep_param_names('dont_match') == self.parameter.grep_param_names('dont_match'))
#         assert(self.bgplvm.grep_param_names('.*') == self.parameter.grep_param_names('.*'))

    def testGetParams(self):
        assert(numpy.allclose(self.bgplvm._get_params(), self.parameter._get_params()))
        assert(numpy.allclose(self.bgplvm._get_params_transformed(), self.parameter._get_params_transformed()))

    def testSetParams(self):
        self.bgplvm.randomize()
        self.parameter._set_params(self.bgplvm._get_params())
        assert(numpy.allclose(self.bgplvm._get_params(), self.parameter._get_params()))
        assert(numpy.allclose(self.bgplvm._get_params_transformed(), self.parameter._get_params_transformed()))
        self.bgplvm.randomize()
        self.parameter._set_params_transformed(self.bgplvm._get_params_transformed())
        assert(numpy.allclose(self.bgplvm._get_params(), self.parameter._get_params()))
        assert(numpy.allclose(self.bgplvm._get_params_transformed(), self.parameter._get_params_transformed()))

    def testSlicing(self):
        assert(numpy.allclose(self.parameter.X[:,1], self.X[:,1]))
        assert(numpy.allclose(self.parameter.X[:,1], self.X[:,1]))
        assert(numpy.allclose(self.parameter.X_variance[1,1], self.X_variance[1,1]))
        assert(numpy.allclose(self.parameter.X_variance[:], self.X_variance[:]))
        assert(numpy.allclose(self.parameter.X[:,:][:,0:2][:,1], self.X[:,1]))
        assert(numpy.allclose(self.parameter.X[:,1], self.X[:,1]))
        assert(numpy.allclose(self.parameter.X_variance[1,1], self.X_variance[1,1]))
        assert(numpy.allclose(self.parameter.X_variance[:], self.X_variance[:]))

    def testSlicingSet(self):
        self.parameter['.*variance'] = 1.
        assert(numpy.alltrue(self.parameter['.*variance'] == 1.))
        self.parameter.X[0,:3] = 2
        assert(numpy.alltrue(self.parameter.X[0,:3] == 2))
        X = self.parameter.X.copy()
        self.parameter.X[[0,4,9],[0,1,3]] -= 1
        assert(numpy.alltrue((X[[0,4,9],[0,1,3]] - 1) == self.parameter.X[[0,4,9],[0,1,3]]))
        self.parameter[''] = 10
        assert(numpy.alltrue(self.parameter[''] == 10))
            
    def testConstraints(self):
        self.parameter[''].unconstrain()
        self.parameter.X.constrain_positive()
        self.parameter.X[:,numpy.s_[0::2]].unconstrain_positive()
        assert(numpy.alltrue(self.parameter._constraints.indices()[0] == numpy.r_[1:self.N*self.Q:2]))

    def testNdarrayFunc(self):
        assert(numpy.alltrue(self.parameter.X * self.parameter.X == self.X * self.X))
        assert(numpy.alltrue(self.parameter.X[0,:] * self.parameter.X[1,:] == self.X[0,:] * self.X[1,:]))
        
    


if __name__ == "__main__":
    import sys;sys.argv = ['', 
                           'Test.testSlicing',
                           'Test.testGetParams',
                           'Test.testNdarrayFunc',
                           'Test.testSetParams',
                           'Test.testConstraints',
                           'Test.testSlicingSet',
                           ]
    unittest.main()
