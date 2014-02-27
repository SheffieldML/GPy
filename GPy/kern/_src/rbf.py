# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import weave
from ...util.misc import param_to_array
from stationary import Stationary
from GPy.util.caching import Cache_this

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

    def psi0(self, Z, variational_posterior):
        return self.Kdiag(variational_posterior.mean)

    def psi1(self, Z, variational_posterior):
        _, _, _, psi1 = self._psi1computations(Z, variational_posterior)
        return psi1

    def psi2(self, Z, variational_posterior):
        _, _, _, _, _, psi2 = self._psi2computations(Z, variational_posterior)
        return psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        l2 = self.lengthscale **2

        #contributions from psi0:
        self.variance.gradient = np.sum(dL_dpsi0)
        self.lengthscale.gradient = 0.

        #from psi1
        denom, _, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
        d_length = psi1[:,:,None] * ((dist_sq - 1.)/(self.lengthscale*denom) +1./self.lengthscale)
        dpsi1_dlength = d_length * dL_dpsi1[:, :, None]
        if not self.ARD:
            self.lengthscale.gradient += dpsi1_dlength.sum()
        else:
            self.lengthscale.gradient += dpsi1_dlength.sum(0).sum(0)
        self.variance.gradient += np.sum(dL_dpsi1 * psi1) / self.variance

        #from psi2
        S = variational_posterior.variance
        denom, _, Zdist_sq, _, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
        d_length = 2.*psi2[:, :, :, None] * (Zdist_sq * denom + mudist_sq + S[:, None, None, :] / l2) / (self.lengthscale * denom)
        #TODO: combine denom and l2 as denom_l2??
        #TODO: tidy the above!
        #TODO: tensordot below?

        dpsi2_dlength = d_length * dL_dpsi2[:, :, :, None]
        if not self.ARD:
            self.lengthscale.gradient += dpsi2_dlength.sum()
        else:
            self.lengthscale.gradient += dpsi2_dlength.sum(0).sum(0).sum(0)

        self.variance.gradient += 2.*np.sum(dL_dpsi2 * psi2)/self.variance

    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        l2 = self.lengthscale **2

        #psi1
        denom, dist, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
        denominator = l2 * denom
        dpsi1_dZ = -psi1[:, :, None] * (dist / denominator)
        grad = np.sum(dL_dpsi1[:, :, None] * dpsi1_dZ, 0)

        #psi2
        denom, Zdist, Zdist_sq, mudist, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
        term1 = Zdist / l2 # M, M, Q
        term2 = mudist / denom / l2 # N, M, M, Q
        dZ = psi2[:, :, :, None] * (term1[None, :, :, :] + term2) #N,M,M,Q
        grad += 2*(dL_dpsi2[:, :, :, None] * dZ).sum(0).sum(0)

        return grad

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        l2 = self.lengthscale **2
        #psi1
        denom, dist, dist_sq, psi1 = self._psi1computations(Z, variational_posterior)
        tmp = psi1[:, :, None] / l2 / denom
        grad_mu = np.sum(dL_dpsi1[:, :, None] * tmp * dist, 1)
        grad_S = np.sum(dL_dpsi1[:, :, None] * 0.5 * tmp * (dist_sq - 1), 1)
        #psi2
        denom, Zdist, Zdist_sq, mudist, mudist_sq, psi2 = self._psi2computations(Z, variational_posterior)
        tmp = psi2[:, :, :, None] / l2 / denom
        grad_mu += -2.*(dL_dpsi2[:, :, :, None] * tmp * mudist).sum(1).sum(1)
        grad_S += (dL_dpsi2[:, :, :, None] * tmp * (2.*mudist_sq - 1)).sum(1).sum(1)

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


    @Cache_this(limit=1)
    def _psi1computations(self, Z, vp):
        mu, S = vp.mean, vp.variance
        l2 = self.lengthscale **2
        denom = S[:, None, :] / l2 + 1. # N,1,Q
        dist = Z[None, :, :] - mu[:, None, :] # N,M,Q
        dist_sq = np.square(dist) / l2 / denom # N,M,Q
        exponent = -0.5 * np.sum(dist_sq + np.log(denom), -1)#N,M
        psi1 = self.variance * np.exp(exponent) # N,M
        return denom, dist, dist_sq, psi1


    #@cache_this(ignore_args=(1,))
    def _Z_distances(self, Z):
        Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
        Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
        return Zhat, Zdist

    @Cache_this(limit=1)
    def _psi2computations(self, Z, vp):
        mu, S = vp.mean, vp.variance

        N, Q = mu.shape
        M = Z.shape[0]

        #compute required distances
        Zhat, Zdist = self._Z_distances(Z)
        Zdist_sq = np.square(Zdist / self.lengthscale) # M,M,Q

        #allocate memory for the things we want to compute
        mudist = np.empty((N, M, M, Q))
        mudist_sq = np.empty((N, M, M, Q))
        exponent = np.zeros((N,M,M))
        psi2 = np.empty((N, M, M))

        l2 = self.lengthscale **2
        denom = (2.*S[:,None,None,:] / l2) + 1. # N,Q
        half_log_denom = 0.5 * np.log(denom[:,0,0,:])
        denom_l2 = denom[:,0,0,:]*l2

        variance_sq = float(np.square(self.variance))
        code = """
        double tmp, exponent_tmp;

        //#pragma omp parallel for private(tmp, exponent_tmp)
        for (int n=0; n<N; n++)
        {
            for (int m=0; m<M; m++)
            {
                for (int mm=0; mm<(m+1); mm++)
                {
                    exponent_tmp = 0.0;
                    for (int q=0; q<Q; q++)
                    {
                        //compute mudist
                        tmp = mu(n,q) - Zhat(m,mm,q);
                        mudist(n,m,mm,q) = tmp;
                        mudist(n,mm,m,q) = tmp;

                        //now mudist_sq
                        tmp = tmp*tmp/denom_l2(n,q);
                        mudist_sq(n,m,mm,q) = tmp;
                        mudist_sq(n,mm,m,q) = tmp;

                        //now exponent
                        tmp = -Zdist_sq(m,mm,q) - tmp - half_log_denom(n,q);
                        exponent_tmp += tmp;
                    }
                    //compute psi2 by exponontiating
                    psi2(n,m,mm) = variance_sq * exp(exponent_tmp);
                    psi2(n,mm,m) = psi2(n,m,mm);
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
                     arg_names=['N', 'M', 'Q', 'mu', 'Zhat', 'mudist_sq', 'mudist', 'denom_l2', 'Zdist_sq', 'half_log_denom', 'psi2', 'variance_sq'],
                     type_converters=weave.converters.blitz, **self.weave_options)

        return denom, Zdist, Zdist_sq, mudist, mudist_sq, psi2

    def input_sensitivity(self):
        if self.ARD: return 1./self.lengthscale
        else: return (1./self.lengthscale).repeat(self.input_dim)
