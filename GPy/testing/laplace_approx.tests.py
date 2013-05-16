import unittest
import numpy as np

import GPy
from GPy.models import GP
from GPy.util.linalg import pdinv, tdot
from scipy import linalg

class LikelihoodGradParam(GP):
    def __init__(self, X, likelihood_function, kernel, param_name=None, function=None, **kwargs):
        super(LikelihoodGradParam, self).__init__(X, likelihood_function, kernel)
        self.param_name = param_name
        self.func = function
        #self.func_params = kwargs
        #self.parameter = self.likelihood.__getattribute__(self.param_name)

    def _get_param_names(self):
        f_hats = ["f_{}".format(i) for i in range(len(self.likelihood.f_hat))]
        return f_hats

    def _get_params(self):
        return np.hstack([np.squeeze(self.likelihood.f_hat)])
        #return np.hstack([self.likelihood.__getattribute__(self.param_name)])

    def hack_dL_dK(self):
        self.K = self.kern.K(self.X)
        self.K += self.likelihood.covariance_matrix

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        # the gradient of the likelihood wrt the covariance matrix
        if self.likelihood.YYT is None:
            alpha, _ = linalg.lapack.flapack.dpotrs(self.L, self.likelihood.Y, lower=1)
            self.dL_dK = 0.5 * (tdot(alpha) - self.D * self.Ki)
        else:
            tmp, _ = linalg.lapack.flapack.dpotrs(self.L, np.asfortranarray(self.likelihood.YYT), lower=1)
            tmp, _ = linalg.lapack.flapack.dpotrs(self.L, np.asfortranarray(tmp.T), lower=1)
            self.dL_dK = 0.5 * (tmp - self.D * self.Ki)

    def _set_params(self, x):
        self.likelihood.f_hat = x.reshape(self.N, 1)
        self.likelihood._compute_likelihood_variables()
        self.hack_dL_dK()

    def log_likelihood(self):
        return self.func(self.likelihood)[0, 0]

    def _log_likelihood_gradients(self):
        #gradient = self.likelihood.__getattribute__(self.param_name)
        self.likelihood._compute_likelihood_variables()
        self.likelihood._gradients(partial=np.diag(self.dL_dK))
        gradient = getattr(self.likelihood, self.param_name)
        #Need to sum over fhats? For dytil_dfhat...
        #gradient = np.flatten(gradient, axis=0)
        #return gradient[:, 0]
        return gradient[0, :]


class LaplaceTests(unittest.TestCase):
    def setUp(self):
        real_var = 0.1
        #Start a function, any function
        #self.X = np.linspace(0.0, 10.0, 30)[:, None]
        self.X = np.random.randn(9,1)
        #self.X = np.ones((10,1))
        Y = np.sin(self.X) + np.random.randn(*self.X.shape)*real_var
        self.Y = Y/Y.max()
        self.kernel = GPy.kern.rbf(self.X.shape[1])

        deg_free = 10000
        real_sd = np.sqrt(real_var)
        initial_sd_guess = 1

        t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=initial_sd_guess)
        self.stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, rasm=True)
        self.stu_t_likelihood.fit_full(self.kernel.K(self.X))
        self.m = LikelihoodGradParam(self.X, self.stu_t_likelihood, self.kernel, None, None)
        self.m.constrain_fixed('rbf_v', 1.0898)
        self.m.constrain_fixed('rbf_l', 1.8651)

    def tearDown(self):
        self.m = None

    def test_dy_dfhat(self):
        def ytil(likelihood):
            Sigma_tilde = likelihood.Sigma_tilde
            K = likelihood.K
            Ki, _, _, _ = pdinv(K)
            f_hat = likelihood.f_hat
            Sigma, _, _, _ = pdinv(Sigma_tilde)
            return np.dot(np.dot(Sigma_tilde, (Ki + Sigma)), f_hat)

        self.m.func = ytil
        self.m.param_name = 'dytil_dfhat'
        self.m.randomize()
        #try:
        self.m.checkgrad(verbose=1)
        assert self.m.checkgrad()
        #except:
            #import ipdb;ipdb.set_trace()


    #def test_dL_dytil(self):
        #def L(likelihood):
            ##-0.5 * self.D * self.K_logdet + self._model_fit_term() + self.likelihood.Z
            #Sigma_tilde = likelihood.Sigma_tilde
            #Ki = likelihood.K
            #f_hat = likelihood.f_hat
            #Sigma, _, _, _ = pdinv(Sigma_tilde)
            #return np.dot(np.dot(Sigma_tilde, (Ki + Sigma)), f_hat)

        #self.m.func = L
        #self.m.param_name = 'dL_dytil'
        #m.randomize()
        ##try:
        #m.checkgrad(verbose=1)
        #assert m.checkgrad()
        #except:
            #import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    unittest.main()

