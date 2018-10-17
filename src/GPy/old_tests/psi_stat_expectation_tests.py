'''
Created on 26 Apr 2013

@author: maxz
'''
import unittest
import GPy
import numpy as np
from GPy import testing
import sys
import numpy
from GPy.kern import RBF
from GPy.kern import Linear
from copy import deepcopy
from GPy.core.parameterization.variational import NormalPosterior

__test__ = lambda: 'deep' in sys.argv
# np.random.seed(0)

def ard(p):
    try:
        if p.ARD:
            return "ARD"
    except:
        pass
    return ""

@testing.deepTest(__test__())
class Test(unittest.TestCase):
    input_dim = 9
    num_inducing = 13
    N = 1000
    Nsamples = 1e6

    def setUp(self):
        self.kerns = (
                      #GPy.kern.RBF([0,1,2], ARD=True)+GPy.kern.Bias(self.input_dim)+GPy.kern.White(self.input_dim),
                      #GPy.kern.RBF(self.input_dim)+GPy.kern.Bias(self.input_dim)+GPy.kern.White(self.input_dim),
                      #GPy.kern.Linear(self.input_dim) + GPy.kern.Bias(self.input_dim) + GPy.kern.White(self.input_dim),
                      #GPy.kern.Linear(self.input_dim, ARD=True) + GPy.kern.Bias(self.input_dim) + GPy.kern.White(self.input_dim),
                      GPy.kern.Linear([1,3,6,7], ARD=True) + GPy.kern.RBF([0,5,8], ARD=True) + GPy.kern.White(self.input_dim),
                      )
        self.q_x_mean = np.random.randn(self.input_dim)[None]
        self.q_x_variance = np.exp(.5*np.random.randn(self.input_dim))[None]
        self.q_x_samples = np.random.randn(self.Nsamples, self.input_dim) * np.sqrt(self.q_x_variance) + self.q_x_mean
        self.q_x = NormalPosterior(self.q_x_mean, self.q_x_variance)
        self.Z = np.random.randn(self.num_inducing, self.input_dim)
        self.q_x_mean.shape = (1, self.input_dim)
        self.q_x_variance.shape = (1, self.input_dim)

    def test_psi0(self):
        for kern in self.kerns:
            psi0 = kern.psi0(self.Z, self.q_x_mean, self.q_x_variance)
            Kdiag = kern.Kdiag(self.q_x_samples)
            self.assertAlmostEqual(psi0, np.mean(Kdiag), 1)
            # print kern.parts[0].name, np.allclose(psi0, np.mean(Kdiag))

    def test_psi1(self):
        for kern in self.kerns:
            Nsamples = np.floor(self.Nsamples/self.N)
            psi1 = kern.psi1(self.Z, self.q_x_mean, self.q_x_variance)
            K_ = np.zeros((Nsamples, self.num_inducing))
            diffs = []
            for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
                K = kern.K(q_x_sample_stripe[:Nsamples], self.Z)
                K_ += K
                diffs.append((np.abs(psi1 - (K_ / (i + 1)))**2).mean())
            K_ /= self.Nsamples / Nsamples
            msg = "psi1: " + "+".join([p.name + ard(p) for p in kern.parts])
            try:
                import pylab
                pylab.figure(msg)
                pylab.plot(diffs)
#                 print msg, ((psi1.squeeze() - K_)**2).mean() < .01
                self.assertTrue(((psi1.squeeze() - K_)**2).mean() < .01,
                                msg=msg + ": not matching")
#                 sys.stdout.write(".")
            except:
#                 import ipdb;ipdb.set_trace()
#                 kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
#                 sys.stdout.write("E")  # msg + ": not matching"
                pass

    def test_psi2(self):
        for kern in self.kerns:
            kern.randomize()
            Nsamples = int(np.floor(self.Nsamples/self.N))
            psi2 = kern.psi2(self.Z, self.q_x)
            K_ = np.zeros((self.num_inducing, self.num_inducing))
            diffs = []
            for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
                K = kern.K(q_x_sample_stripe, self.Z)
                K = (K[:, :, None] * K[:, None, :])
                K_ += K.sum(0) / self.Nsamples
                diffs.append(((psi2 - (K_*self.Nsamples/((i+1)*Nsamples)))**2).mean())
            #K_ /= self.Nsamples / Nsamples
            msg = "psi2: {}".format("+".join([p.name + ard(p) for p in kern.parts]))
            try:
                import pylab
                pylab.figure(msg)
                pylab.plot(diffs, marker='x', mew=.2)
#                 print msg, np.allclose(psi2.squeeze(), K_, rtol=1e-1, atol=.1)
                self.assertTrue(np.allclose(psi2.squeeze(), K_,
                                            atol=.1, rtol=1),
                                msg=msg + ": not matching")
#                 sys.stdout.write(".")
            except:
#                 kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
#                 sys.stdout.write("E")
                print msg + ": not matching"
                import ipdb;ipdb.set_trace()
                pass

if __name__ == "__main__":
    sys.argv = ['',
         #'Test.test_psi0',
         #'Test.test_psi1',
         'Test.test_psi2',
         ]
    unittest.main()
