# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The test cases for various inference algorithms
"""

import unittest, itertools
import numpy as np
import GPy
#np.seterr(invalid='raise')

class InferenceXTestCase(unittest.TestCase):

    def genData(self):
        np.random.seed(1)
        D1,D2,N = 12,12,50

        x = np.linspace(0, 4 * np.pi, N)[:, None]
        s1 = np.vectorize(lambda x: np.sin(x))
        s2 = np.vectorize(lambda x: np.cos(x)**2)
        s3 = np.vectorize(lambda x:-np.exp(-np.cos(2 * x)))
        sS = np.vectorize(lambda x: np.cos(x))

        s1 = s1(x)
        s2 = s2(x)
        s3 = s3(x)
        sS = sS(x)

        s1 -= s1.mean(); s1 /= s1.std(0)
        s2 -= s2.mean(); s2 /= s2.std(0)
        s3 -= s3.mean(); s3 /= s3.std(0)
        sS -= sS.mean(); sS /= sS.std(0)

        S1 = np.hstack([s1, sS])
        S2 = np.hstack([s3, sS])

        P1 = np.random.randn(S1.shape[1], D1)
        P2 = np.random.randn(S2.shape[1], D2)

        Y1 = S1.dot(P1)
        Y2 = S2.dot(P2)

        Y1 += .01 * np.random.randn(*Y1.shape)
        Y2 += .01 * np.random.randn(*Y2.shape)

        Y1 -= Y1.mean(0)
        Y2 -= Y2.mean(0)
        Y1 /= Y1.std(0)
        Y2 /= Y2.std(0)

        slist = [s1, s2, s3, sS]
        slist_names = ["s1", "s2", "s3", "sS"]
        Ylist = [Y1, Y2]

        return Ylist

    def test_inferenceX_BGPLVM(self):
        Ys = self.genData()
        m = GPy.models.BayesianGPLVM(Ys[0],5,kernel=GPy.kern.Linear(5,ARD=True))

        x,mi = m.infer_newX(m.Y, optimize=False)
        self.assertTrue(mi.checkgrad())

        m.optimize(max_iters=10000)
        x, mi = m.infer_newX(m.Y)

        print(m.X.mean - mi.X.mean)
        self.assertTrue(np.allclose(m.X.mean, mi.X.mean, rtol=1e-4, atol=1e-4))
        self.assertTrue(np.allclose(m.X.variance, mi.X.variance, rtol=1e-4, atol=1e-4))

    def test_inferenceX_GPLVM(self):
        Ys = self.genData()
        m = GPy.models.GPLVM(Ys[0],3,kernel=GPy.kern.RBF(3,ARD=True))

        x,mi = m.infer_newX(m.Y, optimize=False)
        self.assertTrue(mi.checkgrad())

#         m.optimize(max_iters=10000)
#         x,mi = m.infer_newX(m.Y)
#         self.assertTrue(np.allclose(m.X, x))


if __name__ == "__main__":
    unittest.main()
