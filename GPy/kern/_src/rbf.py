# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import weave
from kern import Kern
from ...util.linalg import tdot
from ...util.misc import fast_array_equal, param_to_array
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from stationary import Stationary

class RBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='RBF'):
        super(RBF, self).__init__(input_dim, variance, lengthscale, ARD, name)
        self.weave_options = {}

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def parameters_changed(self):
        # reset cached results
        self._Z, self._mu, self._S = np.empty(shape=(3, 1)) # cached versions of Z,mu,S


    def psi0(self, Z, variational_posterior):
        return self.Kdiag(variational_posterior.mean)

    def psi1(self, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        self._psi_computations(Z, mu, S)
        return self._psi1

    def psi2(self, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        self._psi_computations(Z, mu, S)
        return self._psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        self._psi_computations(Z, mu, S)
        l2 = self.lengthscale **2

        #contributions from psi0:
        self.variance.gradient += np.sum(dL_dpsi0)

        #from psi1
        self.variance.gradient += np.sum(dL_dpsi1 * self._psi1 / self.variance)
        d_length = self._psi1[:,:,None] * ((self._psi1_dist_sq - 1.)/(self.lengthscale*self._psi1_denom) +1./self.lengthscale)
        dpsi1_dlength = d_length * dL_dpsi1[:, :, None]
        if not self.ARD:
            self.lengthscale.gradient += dpsi1_dlength.sum()
        else:
            self.lengthscale.gradient += dpsi1_dlength.sum(0).sum(0)

        #from psi2
        d_var = 2.*self._psi2 / self.variance
        d_length = 2.*self._psi2[:, :, :, None] * (self._psi2_Zdist_sq * self._psi2_denom + self._psi2_mudist_sq + S[:, None, None, :] / l2) / (self.lengthscale * self._psi2_denom)

        self.variance.gradient += np.sum(dL_dpsi2 * d_var)
        dpsi2_dlength = d_length * dL_dpsi2[:, :, :, None]
        if not self.ARD:
            self.lengthscale.gradient += dpsi2_dlength.sum()
        else:
            self.lengthscale.gradient += dpsi2_dlength.sum(0).sum(0).sum(0)

    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        self._psi_computations(Z, mu, S)
        l2 = self.lengthscale **2

        #psi1
        denominator = (l2 * (self._psi1_denom))
        dpsi1_dZ = -self._psi1[:, :, None] * ((self._psi1_dist / denominator))
        grad = np.sum(dL_dpsi1[:, :, None] * dpsi1_dZ, 0)

        #psi2
        term1 = self._psi2_Zdist / l2 # num_inducing, num_inducing, input_dim
        term2 = self._psi2_mudist / self._psi2_denom / l2 # N, num_inducing, num_inducing, input_dim
        dZ = self._psi2[:, :, :, None] * (term1[None] + term2)
        grad += 2*(dL_dpsi2[:, :, :, None] * dZ).sum(0).sum(0)

        return grad

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        self._psi_computations(Z, mu, S)
        l2 = self.lengthscale **2
        #psi1
        tmp = self._psi1[:, :, None] / l2 / self._psi1_denom
        grad_mu = np.sum(dL_dpsi1[:, :, None] * tmp * self._psi1_dist, 1)
        grad_S = np.sum(dL_dpsi1[:, :, None] * 0.5 * tmp * (self._psi1_dist_sq - 1), 1)
        #psi2
        tmp = self._psi2[:, :, :, None] / l2 / self._psi2_denom
        grad_mu += -2.*(dL_dpsi2[:, :, :, None] * tmp * self._psi2_mudist).sum(1).sum(1)
        grad_S += (dL_dpsi2[:, :, :, None] * tmp * (2.*self._psi2_mudist_sq - 1)).sum(1).sum(1)

        return grad_mu, grad_S

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _dL_dlengthscales_via_K(self, dL_dK, X, X2):
        """
        A helper function for update_gradients_* methods

        Computes the derivative of the objective L wrt the lengthscales via

        dL_dl = sum_{i,j}(dL_dK_{ij} dK_dl)

        assumes self._K_computations has just been called.

        This is only valid if self.ARD=True
        """
        target = np.zeros(self.input_dim)
        dvardLdK = self._K_dvar * dL_dK
        var_len3 = self.variance / np.power(self.lengthscale, 3)
        if X2 is None:
            # save computation for the symmetrical case
            dvardLdK = dvardLdK + dvardLdK.T
            code = """
            int q,i,j;
            double tmp;
            for(q=0; q<input_dim; q++){
              tmp = 0;
              for(i=0; i<num_data; i++){
                for(j=0; j<i; j++){
                  tmp += (X(i,q)-X(j,q))*(X(i,q)-X(j,q))*dvardLdK(i,j);
                }
              }
              target(q) += var_len3(q)*tmp;
            }
            """
            num_data, num_inducing, input_dim = X.shape[0], X.shape[0], self.input_dim
            X, dvardLdK, var_len3 = param_to_array(X, dvardLdK, var_len3)
            weave.inline(code, arg_names=['num_data', 'num_inducing', 'input_dim', 'X', 'target', 'dvardLdK', 'var_len3'], type_converters=weave.converters.blitz, **self.weave_options)
        else:
            code = """
            int q,i,j;
            double tmp;
            for(q=0; q<input_dim; q++){
              tmp = 0;
              for(i=0; i<num_data; i++){
                for(j=0; j<num_inducing; j++){
                  tmp += (X(i,q)-X2(j,q))*(X(i,q)-X2(j,q))*dvardLdK(i,j);
                }
              }
              target(q) += var_len3(q)*tmp;
            }
            """
            num_data, num_inducing, input_dim = X.shape[0], X2.shape[0], self.input_dim
            X, X2, dvardLdK, var_len3 = param_to_array(X, X2, dvardLdK, var_len3)
            weave.inline(code, arg_names=['num_data', 'num_inducing', 'input_dim', 'X', 'X2', 'target', 'dvardLdK', 'var_len3'], type_converters=weave.converters.blitz, **self.weave_options)
        return target



    def _psi_computations(self, Z, mu, S):
        # here are the "statistics" for psi1 and psi2
        Z_changed = not fast_array_equal(Z, self._Z)
        if Z_changed:
            # Z has changed, compute Z specific stuff
            self._psi2_Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
            self._psi2_Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
            self._psi2_Zdist_sq = np.square(self._psi2_Zdist / self.lengthscale) # M,M,Q

        if Z_changed or not fast_array_equal(mu, self._mu) or not fast_array_equal(S, self._S):
            # something's changed. recompute EVERYTHING
            l2 = self.lengthscale **2

            # psi1
            self._psi1_denom = S[:, None, :] / l2 + 1.
            self._psi1_dist = Z[None, :, :] - mu[:, None, :]
            self._psi1_dist_sq = np.square(self._psi1_dist) / l2 / self._psi1_denom
            self._psi1_exponent = -0.5 * np.sum(self._psi1_dist_sq + np.log(self._psi1_denom), -1)
            self._psi1 = self.variance * np.exp(self._psi1_exponent)

            # psi2
            self._psi2_denom = 2.*S[:, None, None, :] / l2 + 1. # N,M,M,Q
            self._psi2_mudist, self._psi2_mudist_sq, self._psi2_exponent, _ = self.weave_psi2(mu, self._psi2_Zhat)
            # self._psi2_mudist = mu[:,None,None,:]-self._psi2_Zhat #N,M,M,Q
            # self._psi2_mudist_sq = np.square(self._psi2_mudist)/(l2*self._psi2_denom)
            # self._psi2_exponent = np.sum(-self._psi2_Zdist_sq -self._psi2_mudist_sq -0.5*np.log(self._psi2_denom),-1) #N,M,M,Q
            self._psi2 = np.square(self.variance) * np.exp(self._psi2_exponent) # N,M,M,Q

            # store matrices for caching
            self._Z, self._mu, self._S = Z, mu, S

    def weave_psi2(self, mu, Zhat):
        N, input_dim = mu.shape
        num_inducing = Zhat.shape[0]

        mudist = np.empty((N, num_inducing, num_inducing, input_dim))
        mudist_sq = np.empty((N, num_inducing, num_inducing, input_dim))
        psi2_exponent = np.zeros((N, num_inducing, num_inducing))
        psi2 = np.empty((N, num_inducing, num_inducing))

        psi2_Zdist_sq = self._psi2_Zdist_sq
        _psi2_denom = self._psi2_denom.squeeze().reshape(N, self.input_dim)
        half_log_psi2_denom = 0.5 * np.log(self._psi2_denom).squeeze().reshape(N, self.input_dim)
        variance_sq = np.float64(np.square(self.variance))
        if self.ARD:
            lengthscale2 = self.lengthscale **2
        else:
            lengthscale2 = np.ones(input_dim) * self.lengthscale2**2
        code = """
        double tmp;

        #pragma omp parallel for private(tmp)
        for (int n=0; n<N; n++){
            for (int m=0; m<num_inducing; m++){
               for (int mm=0; mm<(m+1); mm++){
                   for (int q=0; q<input_dim; q++){
                       //compute mudist
                       tmp = mu(n,q) - Zhat(m,mm,q);
                       mudist(n,m,mm,q) = tmp;
                       mudist(n,mm,m,q) = tmp;

                       //now mudist_sq
                       tmp = tmp*tmp/lengthscale2(q)/_psi2_denom(n,q);
                       mudist_sq(n,m,mm,q) = tmp;
                       mudist_sq(n,mm,m,q) = tmp;

                       //now psi2_exponent
                       tmp = -psi2_Zdist_sq(m,mm,q) - tmp - half_log_psi2_denom(n,q);
                       psi2_exponent(n,mm,m) += tmp;
                       if (m !=mm){
                           psi2_exponent(n,m,mm) += tmp;
                       }
                   //psi2 would be computed like this, but np is faster
                   //tmp = variance_sq*exp(psi2_exponent(n,m,mm));
                   //psi2(n,m,mm) = tmp;
                   //psi2(n,mm,m) = tmp;
                   }
                }
            }
        }

        """

        support_code = """
        #include <omp.h>
        #include <math.h>
        """
        mu = param_to_array(mu)
        weave.inline(code, support_code=support_code, libraries=['gomp'],
                     arg_names=['N', 'num_inducing', 'input_dim', 'mu', 'Zhat', 'mudist_sq', 'mudist', 'lengthscale2', '_psi2_denom', 'psi2_Zdist_sq', 'psi2_exponent', 'half_log_psi2_denom', 'psi2', 'variance_sq'],
                     type_converters=weave.converters.blitz, **self.weave_options)

        return mudist, mudist_sq, psi2_exponent, psi2

    def input_sensitivity(self):
        if self.ARD: return 1./self.lengthscale
        else: return (1./self.lengthscale).repeat(self.input_dim)
