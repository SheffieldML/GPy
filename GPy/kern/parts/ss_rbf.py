# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from kernpart import Kernpart
from ...util.linalg import tdot
from ...util.misc import fast_array_equal, param_to_array
from ...core.parameterization import Param

class SS_RBF(Kernpart):
    """
    The RBF kernel for Spike-and-Slab GPLVM
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \ \ \ \ \  \\text{ where  } r^2 = \sum_{i=1}^d \\frac{ (x_i-x^\prime_i)^2}{\ell_i^2}

    where \ell_i is the lengthscale, \sigma^2 the variance and d the dimensionality of the input.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the vector of lengthscale of the kernel
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :rtype: kernel object
    """

    def __init__(self, input_dim, variance=1., lengthscale=None, name='rbf'):
        super(RBF, self).__init__(input_dim, name)
        self.input_dim = input_dim

        if lengthscale is not None:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size == self.input_dim, "bad number of lengthscales"
        else:
            lengthscale = np.ones(self.input_dim)

        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengthscale', lengthscale)
        self.lengthscale.add_observer(self, self.update_lengthscale)
        self.add_parameters(self.variance, self.lengthscale)
        self.parameters_changed() # initializes cache

    def on_input_change(self, X):
        #self._K_computations(X, None)
        pass

    def update_lengthscale(self, l):
        self.lengthscale2 = np.square(self.lengthscale)

    def parameters_changed(self):
        # reset cached results
        self._X, self._X2 = np.empty(shape=(2, 1))
        self._Z, self._mu, self._S = np.empty(shape=(3, 1)) # cached versions of Z,mu,S

    def K(self, X, X2, target):
        self._K_computations(X, X2)
        target += self.variance * self._K_dvar

    def Kdiag(self, X, target):
        np.add(target, self.variance, target)

    def psi0(self, Z, mu, S, target):
        target += self.variance

    def psi1(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += self._psi1

    def psi2(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += self._psi2

    def update_gradients_full(self, dL_dK, X):
        self._K_computations(X, None)
        self.variance.gradient = np.sum(self._K_dvar * dL_dK)
        if self.ARD:
            self.lengthscale.gradient = self._dL_dlengthscales_via_K(dL_dK, X, None)
        else:
            self.lengthscale.gradient = (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dK)

    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        #contributions from Kdiag
        self.variance.gradient = np.sum(dL_dKdiag)

        #from Knm
        self._K_computations(X, Z)
        self.variance.gradient += np.sum(dL_dKnm * self._K_dvar)
        if self.ARD:
            self.lengthscales.gradient = self._dL_dlengthscales_via_K(dL_dKnm, X, Z)

        else:
            self.lengthscale.gradient = (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dKmm)

        #from Kmm
        self._K_computations(Z, None)
        self.variance.gradient += np.sum(dL_dKmm * self._K_dvar)
        if self.ARD:
            self.lengthscales.gradient += self._dL_dlengthscales_via_K(dL_dKmm, Z, None)
        else:
            self.lengthscale.gradient += (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dKmm)

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        self._psi_computations(Z, mu, S)

        #contributions from psi0:
        self.variance.gradient = np.sum(dL_dpsi0)

        #from psi1
        self.variance.gradient += np.sum(dL_dpsi1 * self._psi1 / self.variance)
        d_length = self._psi1[:,:,None] * ((self._psi1_dist_sq - 1.)/(self.lengthscale*self._psi1_denom) +1./self.lengthscale)
        dpsi1_dlength = d_length * dL_dpsi1[:, :, None]
        if not self.ARD:
            self.lengthscale.gradeint = dpsi1_dlength.sum()
        else:
            self.lengthscale.gradient = dpsi1_dlength.sum(0).sum(0)

        #from psi2
        d_var = 2.*self._psi2 / self.variance
        d_length = 2.*self._psi2[:, :, :, None] * (self._psi2_Zdist_sq * self._psi2_denom + self._psi2_mudist_sq + S[:, None, None, :] / self.lengthscale2) / (self.lengthscale * self._psi2_denom)

        self.variance.gradient += np.sum(dL_dpsi2 * d_var)
        dpsi2_dlength = d_length * dL_dpsi2[:, :, :, None]
        if not self.ARD:
            self.lengthscale.gradient += dpsi2_dlength.sum()
        else:
            self.lengthscale.gradient += dpsi2_dlength.sum(0).sum(0).sum(0)

        #from Kmm
        self._K_computations(Z, None)
        self.variance.gradient += np.sum(dL_dKmm * self._K_dvar)
        if self.ARD:
            self.lengthscales.gradient += self._dL_dlengthscales_via_K(dL_dKmm, Z, None)
        else:
            self.lengthscale.gradient += (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dK)

    def gradients_X(self, dL_dK, X, X2, target):
        #if self._X is None or X.base is not self._X.base or X2 is not None:
        self._K_computations(X, X2)
        if X2 is None:
            _K_dist = 2*(X[:, None, :] - X[None, :, :])
        else:
            _K_dist = X[:, None, :] - X2[None, :, :] # don't cache this in _K_computations because it is high memory. If this function is being called, chances are we're not in the high memory arena.
        dK_dX = (-self.variance / self.lengthscale2) * np.transpose(self._K_dvar[:, :, np.newaxis] * _K_dist, (1, 0, 2))
        target += np.sum(dK_dX * dL_dK.T[:, :, None], 0)

    def dKdiag_dX(self, dL_dKdiag, X, target):
        pass

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S, target_mu, target_S):
        pass

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        denominator = (self.lengthscale2 * (self._psi1_denom))
        dpsi1_dZ = -self._psi1[:, :, None] * ((self._psi1_dist / denominator))
        target += np.sum(dL_dpsi1[:, :, None] * dpsi1_dZ, 0)

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S, target_mu, target_S):
        self._psi_computations(Z, mu, S)
        tmp = self._psi1[:, :, None] / self.lengthscale2 / self._psi1_denom
        target_mu += np.sum(dL_dpsi1[:, :, None] * tmp * self._psi1_dist, 1)
        target_S += np.sum(dL_dpsi1[:, :, None] * 0.5 * tmp * (self._psi1_dist_sq - 1), 1)

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        term1 = self._psi2_Zdist / self.lengthscale2 # num_inducing, num_inducing, input_dim
        term2 = self._psi2_mudist / self._psi2_denom / self.lengthscale2 # N, num_inducing, num_inducing, input_dim
        dZ = self._psi2[:, :, :, None] * (term1[None] + term2)
        target += (dL_dpsi2[:, :, :, None] * dZ).sum(0).sum(0)

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        """Think N,num_inducing,num_inducing,input_dim """
        self._psi_computations(Z, mu, S)
        tmp = self._psi2[:, :, :, None] / self.lengthscale2 / self._psi2_denom
        target_mu += -2.*(dL_dpsi2[:, :, :, None] * tmp * self._psi2_mudist).sum(1).sum(1)
        target_S += (dL_dpsi2[:, :, :, None] * tmp * (2.*self._psi2_mudist_sq - 1)).sum(1).sum(1)

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self, X, X2):
        #params = self._get_params()
        if not (fast_array_equal(X, self._X) and fast_array_equal(X2, self._X2)):# and fast_array_equal(self._params_save , params)):
            #self._X = X.copy()
            #self._params_save = params.copy()
            if X2 is None:
                self._X2 = None
                X = X / self.lengthscale
                Xsquare = np.sum(np.square(X), 1)
                self._K_dist2 = -2.*tdot(X) + (Xsquare[:, None] + Xsquare[None, :])
            else:
                self._X2 = X2.copy()
                X = X / self.lengthscale
                X2 = X2 / self.lengthscale
                self._K_dist2 = -2.*np.dot(X, X2.T) + (np.sum(np.square(X), 1)[:, None] + np.sum(np.square(X2), 1)[None, :])
            self._K_dvar = np.exp(-0.5 * self._K_dist2)

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
            X, dvardLdK = param_to_array(X, dvardLdK)
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
            X, X2, dvardLdK = param_to_array(X, X2, dvardLdK)
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

            # psi1
            self._psi1_denom = S[:, None, :] / self.lengthscale2 + 1.
            self._psi1_dist = Z[None, :, :] - mu[:, None, :]
            self._psi1_dist_sq = np.square(self._psi1_dist) / self.lengthscale2 / self._psi1_denom
            self._psi1_exponent = -0.5 * np.sum(self._psi1_dist_sq + np.log(self._psi1_denom), -1)
            self._psi1 = self.variance * np.exp(self._psi1_exponent)

            # psi2
            self._psi2_denom = 2.*S[:, None, None, :] / self.lengthscale2 + 1. # N,M,M,Q
            self._psi2_mudist, self._psi2_mudist_sq, self._psi2_exponent, _ = self.weave_psi2(mu, self._psi2_Zhat)
            # self._psi2_mudist = mu[:,None,None,:]-self._psi2_Zhat #N,M,M,Q
            # self._psi2_mudist_sq = np.square(self._psi2_mudist)/(self.lengthscale2*self._psi2_denom)
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
        variance_sq = float(np.square(self.variance))
        if self.ARD:
            lengthscale2 = self.lengthscale2
        else:
            lengthscale2 = np.ones(input_dim) * self.lengthscale2
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
        weave.inline(code, support_code=support_code, libraries=['gomp'],
                     arg_names=['N', 'num_inducing', 'input_dim', 'mu', 'Zhat', 'mudist_sq', 'mudist', 'lengthscale2', '_psi2_denom', 'psi2_Zdist_sq', 'psi2_exponent', 'half_log_psi2_denom', 'psi2', 'variance_sq'],
                     type_converters=weave.converters.blitz, **self.weave_options)

        return mudist, mudist_sq, psi2_exponent, psi2
