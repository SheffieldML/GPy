'''
Created on 26 Apr 2013

@author: maxz
'''
import unittest
import GPy
import numpy as np
import sys
from .. import testing

__test__ = True
np.random.seed(0)

def ard(p):
    try:
        if p.ARD:
            return "ARD"
    except:
        pass
    return ""

@testing.deepTest
class Test(unittest.TestCase):
    D = 9
    M = 4
    N = 3
    Nsamples = 6e6

    def setUp(self):
        self.kerns = (
                      GPy.kern.rbf(self.D), GPy.kern.rbf(self.D, ARD=True),
                      GPy.kern.linear(self.D, ARD=False), GPy.kern.linear(self.D, ARD=True),
                      GPy.kern.linear(self.D) + GPy.kern.bias(self.D),
                      GPy.kern.rbf(self.D) + GPy.kern.bias(self.D),
                      GPy.kern.linear(self.D) + GPy.kern.bias(self.D) + GPy.kern.white(self.D),
                      GPy.kern.rbf(self.D) + GPy.kern.bias(self.D) + GPy.kern.white(self.D),
                      GPy.kern.bias(self.D), GPy.kern.white(self.D),
                      )
        self.q_x_mean = np.random.randn(self.D)
        self.q_x_variance = np.exp(np.random.randn(self.D))
        self.q_x_samples = np.random.randn(self.Nsamples, self.D) * np.sqrt(self.q_x_variance) + self.q_x_mean
        self.Z = np.random.randn(self.M, self.D)
        self.q_x_mean.shape = (1, self.D)
        self.q_x_variance.shape = (1, self.D)

    def test_psi0(self):
        for kern in self.kerns:
            psi0 = kern.psi0(self.Z, self.q_x_mean, self.q_x_variance)
            Kdiag = kern.Kdiag(self.q_x_samples)
            self.assertAlmostEqual(psi0, np.mean(Kdiag), 1)
            # print kern.parts[0].name, np.allclose(psi0, np.mean(Kdiag))

    def test_psi1(self):
        for kern in self.kerns:
            Nsamples = 100
            psi1 = kern.psi1(self.Z, self.q_x_mean, self.q_x_variance)
            K_ = np.zeros((Nsamples, self.M))
            diffs = []
            for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
                K = kern.K(q_x_sample_stripe, self.Z)
                K_ += K
                diffs.append(((psi1 - (K_ / (i + 1)))).mean())
            K_ /= self.Nsamples / Nsamples
            msg = "psi1: " + "+".join([p.name + ard(p) for p in kern.parts])
            try:
#                 pylab.figure(msg)
#                 pylab.plot(diffs)
                self.assertTrue(np.allclose(psi1.squeeze(), K_,
                                            rtol=1e-1, atol=.1),
                                msg=msg + ": not matching")
#                 sys.stdout.write(".")
            except:
#                 import ipdb;ipdb.set_trace()
#                 kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
#                 sys.stdout.write("E")  # msg + ": not matching"
                pass

    def test_psi2(self):
        for kern in self.kerns:
            Nsamples = 100
            psi2 = kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
            K_ = np.zeros((self.M, self.M))
            diffs = []
            for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
                K = kern.K(q_x_sample_stripe, self.Z)
                K = (K[:, :, None] * K[:, None, :]).mean(0)
                K_ += K
                diffs.append(((psi2 - (K_ / (i + 1)))).mean())
            K_ /= self.Nsamples / Nsamples
            msg = "psi2: {}".format("+".join([p.name + ard(p) for p in kern.parts]))
            try:
#                 pylab.figure(msg)
#                 pylab.plot(diffs)
                self.assertTrue(np.allclose(psi2.squeeze(), K_,
                                            rtol=1e-1, atol=.1),
                                msg=msg + ": not matching")
#                 sys.stdout.write(".")
            except:
#                 import ipdb;ipdb.set_trace()
#                 kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
#                 sys.stdout.write("E")
                print msg + ": not matching"
                pass

if __name__ == "__main__":
    import sys;sys.argv = ['',
                            'Test.test_psi0',
                            'Test.test_psi1',
                            'Test.test_psi2',
                           ]
    unittest.main()
