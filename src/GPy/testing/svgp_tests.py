import numpy as np
import scipy as sp
import GPy

class SVGP_nonconvex(np.testing.TestCase):
    """
    Inference in the SVGP with a student-T likelihood
    """
    def setUp(self):
        X = np.linspace(0,10,100).reshape(-1,1)
        Z = np.linspace(0,10,10).reshape(-1,1)
        Y = np.sin(X) + np.random.randn(*X.shape)*0.1
        Y[50] += 3

        lik = GPy.likelihoods.StudentT(deg_free=2)
        k = GPy.kern.RBF(1, lengthscale=5.) + GPy.kern.White(1, 1e-6)
        self.m = GPy.core.SVGP(X, Y, Z=Z, likelihood=lik, kernel=k)
    def test_grad(self):
        assert self.m.checkgrad(step=1e-4)

class SVGP_classification(np.testing.TestCase):
    """
    Inference in the SVGP with a Bernoulli likelihood
    """
    def setUp(self):
        X = np.linspace(0,10,100).reshape(-1,1)
        Z = np.linspace(0,10,10).reshape(-1,1)
        Y = np.where((np.sin(X) + np.random.randn(*X.shape)*0.1)>0, 1,0)

        lik = GPy.likelihoods.Bernoulli()
        k = GPy.kern.RBF(1, lengthscale=5.) + GPy.kern.White(1, 1e-6)
        self.m = GPy.core.SVGP(X, Y, Z=Z, likelihood=lik, kernel=k)
    def test_grad(self):
        assert self.m.checkgrad(step=1e-4)

class SVGP_Poisson_with_meanfunction(np.testing.TestCase):
    """
    Inference in the SVGP with a Bernoulli likelihood
    """
    def setUp(self):
        X = np.linspace(0,10,100).reshape(-1,1)
        Z = np.linspace(0,10,10).reshape(-1,1)
        latent_f = np.exp(0.1*X * 0.05*X**2)
        Y = np.array([np.random.poisson(f) for f in latent_f.flatten()]).reshape(-1,1)

        mf = GPy.mappings.Linear(1,1)

        lik = GPy.likelihoods.Poisson()
        k = GPy.kern.RBF(1, lengthscale=5.) + GPy.kern.White(1, 1e-6)
        self.m = GPy.core.SVGP(X, Y, Z=Z, likelihood=lik, kernel=k, mean_function=mf)
    def test_grad(self):
        assert self.m.checkgrad(step=1e-4)


