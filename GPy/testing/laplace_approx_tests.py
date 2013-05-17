import unittest
import numpy as np
np.random.seed(82)

import GPy
from GPy.models import GP
from GPy.util.linalg import pdinv, tdot
from scipy import linalg

class LikelihoodParamGrad(GP):
    def __init__(self, X=None, likelihood_function=None, kernel=None, param_name=None, function=None, dparam_name=None, **kwargs):
        self.param_name = param_name
        self.dparam_name = dparam_name
        self.func = function
        super(LikelihoodParamGrad, self).__init__(X, likelihood_function, kernel)
        #self.func_params = kwargs
        #self.parameter = self.likelihood.__getattribute__(self.param_name)

    def _get_param_names(self):
        params = getattr(self.likelihood, self.dparam_name)
        params_names = ["{}_{}".format(self.dparam_name, i) for i in range(len(params))]
        return params_names

    def _get_params(self):
        params = getattr(self.likelihood, self.dparam_name)
        return np.hstack([params])

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
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    def _log_likelihood_gradients(self):
        raise NotImplementedError


class Likelihood_F_Grad(LikelihoodParamGrad):
    def __init__(self, **kwargs):
        super(Likelihood_F_Grad, self).__init__(**kwargs)

    def _set_params(self, x):
        params = getattr(self.likelihood, self.dparam_name)
        setattr(self.likelihood, self.dparam_name, x.reshape(*params.shape))
        self.likelihood._compute_likelihood_variables()
        self.hack_dL_dK()

    def log_likelihood(self):
        ll = self.func(self)
        if self.param_name == "dL_dfhat_":
            import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
        if len(ll.shape) == 0 or len(ll.shape) == 1:
            return ll.sum()
        elif len(ll.shape) == 2:
            #print "Only checking first likelihood"
            return ll[0, 0]
        else:
            raise ValueError('Not implemented for larger matricies yet')
        return ll

    def _log_likelihood_gradients(self):
        self.likelihood._compute_likelihood_variables()
        self.likelihood._gradients(partial=np.diag(self.dL_dK))
        gradient = getattr(self.likelihood, self.param_name)
        if len(gradient.shape) == 1:
            return gradient
        elif len(gradient.shape) == 2:
            #print "Only checking first gradients"
            return gradient[0,: ]
        else:
            raise ValueError('Not implemented for larger matricies yet')


class LaplaceTests(unittest.TestCase):
    def setUp(self):
        real_var = 0.1
        #Start a function, any function
        self.X = np.linspace(0.0, 10.0, 4)[:, None]
        #self.X = np.random.randn(,1)
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

    def tearDown(self):
        self.m = None

    def test_dy_dfhat(self):
        def ytil(self):
            Sigma_tilde = self.likelihood.Sigma_tilde
            K = self.likelihood.K
            Ki, _, _, _ = pdinv(K)
            f_hat = self.likelihood.f_hat
            Sigma, _, _, _ = pdinv(Sigma_tilde)
            return np.dot(np.dot(Sigma_tilde, (Ki + Sigma)), f_hat)

        self.m = Likelihood_F_Grad(X=self.X, likelihood_function=self.stu_t_likelihood,
                                   kernel=self.kernel, param_name='dytil_dfhat',
                                   function=ytil, dparam_name='f_hat')
        #self.m.constrain_fixed('rbf_v', 1.0898)
        #self.m.constrain_fixed('rbf_l', 1.8651)
        self.m.randomize()
        self.m.checkgrad(verbose=1)
        assert self.m.checkgrad()

    def test_dL_dfhat(self):
        def L(self):
            return np.array(-0.5 * self.D * self.K_logdet + self._model_fit_term() + self.likelihood.Z)

        self.m = Likelihood_F_Grad(X=self.X, likelihood_function=self.stu_t_likelihood,
                                    kernel=self.kernel, param_name='dL_dfhat',
                                    function=L, dparam_name='f_hat')
        self.m.constrain_fixed('rbf_v', 1.0898)
        self.m.constrain_fixed('rbf_l', 1.8651)
        self.m.randomize()
        self.m.checkgrad(verbose=1)
        assert self.m.checkgrad()

if __name__ == "__main__":
    unittest.main()

