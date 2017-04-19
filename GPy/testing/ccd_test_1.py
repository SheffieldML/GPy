import numpy as np
import unittest
from nose.tools import nottest
import matplotlib.pyplot as plt
import GPy
import itertools
np.set_printoptions(suppress=True, precision=10)
from paramz.transformations import Logexp



class SimpleModel(GPy.Model):
    def __init__(self, name, dims, priors=False):
        super(SimpleModel, self).__init__(name)
        self.params = []
        self.peak_loc = range(1, dims + 1, 1)
        for i, pos in enumerate(self.peak_loc):
            if priors:
                p = GPy.Param('param%d' % i, 1.0, Logexp())
            else:
                p = GPy.Param('param%d' % i, 1.0)
            self.params.append(p)
            self.link_parameter(p)

    def log_likelihood(self):
        like = 0
        assert len(self.params) == len(self.peak_loc), "Dimensions do not match!"
        for i, pos in enumerate(self.peak_loc):
            like = like - (self.params[i] - pos) ** 2
        return like

    def parameters_changed(self):
        for i, pos in enumerate(self.peak_loc):
            self.params[i].gradient = -2 * ((self.params[i]) - pos)


class BiQuadModel(SimpleModel):
    def log_likelihood(self):
        like = 0
        for i, pos in enumerate(self.peak_loc):
            like = like - (self.params[i] - pos) ** 4
        return like

    def parameters_changed(self):
        pass


class CCDTests(unittest.TestCase):
    """
    general ccd test for a particular model-scenario.

    """
    def setUp(self):
        self.m = SimpleModel("Quad",3)
        self.ndims = self.m.num_params
        self.true_mode = self.m.peak_loc
        self.stepsize = 0.2

    def tearDown(self):
        # model could be memory intensive- so remove after test!
        self.m = None

    def test_ccd_placement(self):
        # m2 = SimpleModel('simple', 2)
        self.m.optimize()

        # assert np.all(np.isclose(self.m.numerical_parameter_hessian(), np.array([[2,0],[0,2]]))), "Numerical approximation to Hessian doesn't match."
        assert np.all(np.isclose(self.m.numerical_parameter_hessian(), np.eye(self.ndims)*2)), "Numerical approximation to Hessian doesn't match. Error in numerical_parameter_hessian()."

        #check the optimizing step-found the maximum correctly.
        for i in range(self.m.num_params):
            assert np.isclose(self.m.params[i], i+1, atol=0.01), "Failed to find likelihood maximum of test model's parameter while testing CCD"
        # get the CCD positions and log likelihoods at those locations.
        ccdpos, ccdres, scalings, z = self.m.CCD()
        # (optimise again as CCD moves the parameter values)
        self.m.optimize()
        # calculate euclidean distance of ccdpoints except centralpoint from true mode.
        dists = np.sum((ccdpos - self.true_mode) ** 2, 1)[1:]
        dists - np.mean(dists)
        # basically checks that all CCD points are equidistant from the central point.
        assert np.all(np.isclose(dists, np.mean(dists),
                                 atol=0.01)), "CCD placement error - Points should be equidistant!"
        assert np.all(np.isclose(np.sum(ccdres[1:] / ccdres[0]), 4.7619,
                                 atol=0.1)), "CCD placement error - off-centre locations should have log likelihood" \
                                             " ratios to central point summing to 4.76 times the centre, for nd" \
                                             " symmetrical Quadratic test case"

    @nottest
    def find_likes(self, stepsize=0.3,rangemin=-2,rangemax=7):
        params = self.m.parameter_names_flat()
        param_ranges = []
        for param in params:
            param_ranges.append(np.arange(rangemin, rangemax, stepsize))
        combs = itertools.product(*param_ranges)
        llsum = 0
        for el in combs:
            llsum += np.exp(-self.m._objective(el))
        return llsum

    def test_ccd_integration(self):
        ls = self.find_likes(self.stepsize)
        numsum = ls*(self.stepsize**self.ndims)
        self.m.optimize()
        hes = self.m.numerical_parameter_hessian()
        hessum = np.exp(self.m.log_likelihood()) * 1 / np.sqrt(np.linalg.det(1 / (2 * np.pi) * hes))
        print hessum, hes
        print numsum
        assert np.isclose(hessum, numsum,
                          atol=0.2), "Laplace approximation using numerical_parameter_hessian()=%0.4f not" \
                                     " equal to numerical grid sum=%0.4f" % (
        hessum, numsum)


if __name__ == "__main__":
    unittest.main()
    # ndims = 6
    # for i in range(1, ndims):
    #     m = SimpleModel(i)
