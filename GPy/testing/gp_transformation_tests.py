from nose.tools import with_setup
from GPy.models import GradientChecker
from GPy.likelihoods.noise_models import gp_transformations
import inspect
import unittest
import numpy as np

class TestTransformations(object):
    """
    Generic transformations checker
    """
    def setUp(self):
        N = 30
        self.fs = [np.random.rand(N, 1), float(np.random.rand(1))]


    def tearDown(self):
        self.fs = None

    def test_transformations(self):
        self.setUp()
        transformations = [gp_transformations.Identity(),
                           gp_transformations.Log(),
                           gp_transformations.Probit(),
                           gp_transformations.Log_ex_1(),
                           gp_transformations.Reciprocal(),
                           ]

        for transformation in transformations:
            for f in self.fs:
                yield self.t_dtransf_df, transformation, f
                yield self.t_d2transf_df2, transformation, f
                yield self.t_d3transf_df3, transformation, f

    @with_setup(setUp, tearDown)
    def t_dtransf_df(self, transformation, f):
        print "\n{}".format(inspect.stack()[0][3])
        grad = GradientChecker(transformation.transf, transformation.dtransf_df, f, 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d2transf_df2(self, transformation, f):
        print "\n{}".format(inspect.stack()[0][3])
        grad = GradientChecker(transformation.dtransf_df, transformation.d2transf_df2, f, 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d3transf_df3(self, transformation, f):
        print "\n{}".format(inspect.stack()[0][3])
        grad = GradientChecker(transformation.d2transf_df2, transformation.d3transf_df3, f, 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

#if __name__ == "__main__":
    #print "Running unit tests"
    #unittest.main()
