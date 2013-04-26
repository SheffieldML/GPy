'''
Created on 26 Apr 2013

@author: maxz
'''
import unittest
import GPy
import numpy as np
import pylab

class Test(unittest.TestCase):
    D = 9
    M = 5
    Nsamples = 3e6

    def setUp(self):
        self.kerns = (
                      GPy.kern.rbf(self.D), GPy.kern.rbf(self.D, ARD=True),
                      GPy.kern.linear(self.D), GPy.kern.linear(self.D, ARD=True),
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

#     def test_psi0(self):
#         for kern in self.kerns:
#             psi0 = kern.psi0(self.Z, self.q_x_mean, self.q_x_variance)
#             Kdiag = kern.Kdiag(self.q_x_samples)
#             self.assertAlmostEqual(psi0, np.mean(Kdiag), 1)
#             # print kern.parts[0].name, np.allclose(psi0, np.mean(Kdiag))
#
#     def test_psi1(self):
#         for kern in self.kerns:
#             Nsamples = 100
#             psi1 = kern.psi1(self.Z, self.q_x_mean, self.q_x_variance)
#             K_ = np.zeros((self.N, self.M))
#             diffs = []
#             for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
#                 K = kern.K(q_x_sample_stripe, self.Z)
#                 K_ += K
#                 diffs.append(((psi1 - (K_ / (i + 1))) ** 2).mean())
#             K_ /= self.Nsamples / Nsamples
# #             pylab.figure("+".join([p.name for p in kern.parts]) + "psi1")
# #             pylab.plot(diffs)
#             self.assertTrue(np.allclose(psi1.flatten() , K.mean(0), rtol=1e-1))
#
#     def test_psi2(self):
#         for kern in self.kerns:
#             Nsamples = 100
#             psi2 = kern.psi2(self.Z, self.q_x_mean, self.q_x_variance)
#             K_ = np.zeros((self.M, self.M))
#             diffs = []
#             for i, q_x_sample_stripe in enumerate(np.array_split(self.q_x_samples, self.Nsamples / Nsamples)):
#                 K = kern.K(q_x_sample_stripe, self.Z)
#                 K = (K[:, :, None] * K[:, None, :]).mean(0)
#                 K_ += K
#                 diffs.append(((psi2 - (K_ / (i + 1))) ** 2).mean())
#             K_ /= self.Nsamples / Nsamples
#             try:
# #                 pylab.figure("+".join([p.name for p in kern.parts]) + "psi2")
# #                 pylab.plot(diffs)
#                 self.assertTrue(np.allclose(psi2.squeeze(), K_,
#                                             rtol=1e-1, atol=.1),
#                                 msg="{}: not matching".format("+".join([p.name for p in kern.parts])))
#             except:
#                 print "{}: not matching".format(kern.parts[0].name)

if __name__ == "__main__":
    import sys;sys.argv = ['', 'Test.test_psi2']
    unittest.main()
