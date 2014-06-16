# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import weave
from kern import Kern
from ...util.linalg import tdot
from ...util.misc import param_to_array
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ...util.caching import Cache_this
from ...core.parameterization import variational
from ...util.config import *
from .psi_comp import PSICOMP_Linear

class Linear(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^input_dim \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, variances=None, ARD=False, active_dims=None, name='linear'):
        super(Linear, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
        else:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances = np.ones(self.input_dim)

        self.variances = Param('variances', variances, Logexp())
        self.add_parameter(self.variances)
        self.psicomp = PSICOMP_Linear()
    
    @Cache_this(limit=2)
    def K(self, X, X2=None):
        if self.ARD:
            if X2 is None:
                return tdot(X*np.sqrt(self.variances))
            else:
                rv = np.sqrt(self.variances)
                return np.dot(X*rv, (X2*rv).T)
        else:
            return self._dot_product(X, X2) * self.variances

    @Cache_this(limit=1, ignore_args=(0,))
    def _dot_product(self, X, X2=None):
        if X2 is None:
            return tdot(X)
        else:
            return np.dot(X, X2.T)

    def Kdiag(self, X):
        return np.sum(self.variances * np.square(X), -1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if self.ARD:
            if X2 is None:
                self.variances.gradient = np.array([np.sum(dL_dK * tdot(X[:, i:i + 1])) for i in range(self.input_dim)])
            else:
                product = X[:, None, :] * X2[None, :, :]
                self.variances.gradient = (dL_dK[:, :, None] * product).sum(0).sum(0)
        else:
            self.variances.gradient = np.sum(self._dot_product(X, X2) * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        tmp = dL_dKdiag[:, None] * X ** 2
        if self.ARD:
            self.variances.gradient = tmp.sum(0)
        else:
            self.variances.gradient = np.atleast_1d(tmp.sum())


    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            return np.einsum('mq,nm->nq',X*self.variances,dL_dK)+np.einsum('nq,nm->mq',X*self.variances,dL_dK)
        else:
            return (((X2[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)

    def gradients_X_diag(self, dL_dKdiag, X):
        return 2.*self.variances*dL_dKdiag[:,None]*X

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return self.psicomp.psicomputations(self.variances, Z, variational_posterior)[0]
        else:
            return np.sum(self.variances * self._mu2S(variational_posterior), 1)

    def psi1(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return self.psicomp.psicomputations(self.variances, Z, variational_posterior)[1]
        else:
            return self.K(variational_posterior.mean, Z) #the variance, it does nothing

    @Cache_this(limit=1)
    def psi2(self, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            return self.psicomp.psicomputations(self.variances, Z, variational_posterior)[2]
        else:
            ZA = Z * self.variances
            ZAinner = self._ZAinner(variational_posterior, Z)
            return np.dot(ZAinner, ZA.T)

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            dL_dvar,_,_,_,_ = self.psicomp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variances, Z, variational_posterior)
            if self.ARD:
                self.variances.gradient = dL_dvar
            else:
                self.variances.gradient = dL_dvar.sum()
        else:
            #psi1
            self.update_gradients_full(dL_dpsi1, variational_posterior.mean, Z)
            # psi0:
            tmp = dL_dpsi0[:, None] * self._mu2S(variational_posterior)
            if self.ARD: self.variances.gradient += tmp.sum(0)
            else: self.variances.gradient += tmp.sum()
            #psi2
            if self.ARD:
                tmp = dL_dpsi2[:, :, :, None] * (self._ZAinner(variational_posterior, Z)[:, :, None, :] * Z[None, None, :, :])
                self.variances.gradient += 2.*tmp.sum(0).sum(0).sum(0)
            else:
                self.variances.gradient += 2.*np.sum(dL_dpsi2 * self.psi2(Z, variational_posterior))/self.variances

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            _,dL_dZ,_,_,_ = self.psicomp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variances, Z, variational_posterior)
            return dL_dZ
        else:
            #psi1
            grad = self.gradients_X(dL_dpsi1.T, Z, variational_posterior.mean)
            #psi2
            self._weave_dpsi2_dZ(dL_dpsi2, Z, variational_posterior, grad)
            return grad

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if isinstance(variational_posterior, variational.SpikeAndSlabPosterior):
            _,_,dL_dmu, dL_dS, dL_dgamma = self.psicomp.psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, self.variances, Z, variational_posterior)
            return dL_dmu, dL_dS, dL_dgamma
        else:
            grad_mu, grad_S = np.zeros(variational_posterior.mean.shape), np.zeros(variational_posterior.mean.shape)
            # psi0
            grad_mu += dL_dpsi0[:, None] * (2.0 * variational_posterior.mean * self.variances)
            grad_S += dL_dpsi0[:, None] * self.variances
            # psi1
            grad_mu += (dL_dpsi1[:, :, None] * (Z * self.variances)).sum(1)
            # psi2
            self._weave_dpsi2_dmuS(dL_dpsi2, Z, variational_posterior, grad_mu, grad_S)

            return grad_mu, grad_S

    #--------------------------------------------------#
    #            Helpers for psi statistics            #
    #--------------------------------------------------#


    def _weave_dpsi2_dmuS(self, dL_dpsi2, Z, vp, target_mu, target_S):
        # Think N,num_inducing,num_inducing,input_dim
        ZA = Z * self.variances
        AZZA = ZA.T[:, None, :, None] * ZA[None, :, None, :]
        AZZA = AZZA + AZZA.swapaxes(1, 2)
        AZZA_2 = AZZA/2.
        if config.getboolean('parallel', 'openmp'):
            pragma_string = '#pragma omp parallel for private(m,mm,q,qq,factor,tmp)'
            header_string = '#include <omp.h>'
            weave_options = {'headers'           : ['<omp.h>'],
                                'extra_compile_args': ['-fopenmp -O3'],
                                'extra_link_args'   : ['-lgomp'],
                                'libraries': ['gomp']}
        else:
            pragma_string = ''
            header_string = ''
            weave_options = {'extra_compile_args': ['-O3']}

        #Using weave, we can exploit the symmetry of this problem:
        code = """
        int n, m, mm,q,qq;
        double factor,tmp;
        %s
        for(n=0;n<N;n++){
          for(m=0;m<num_inducing;m++){
            for(mm=0;mm<=m;mm++){
              //add in a factor of 2 for the off-diagonal terms (and then count them only once)
              if(m==mm)
                factor = dL_dpsi2(n,m,mm);
              else
                factor = 2.0*dL_dpsi2(n,m,mm);

              for(q=0;q<input_dim;q++){

                //take the dot product of mu[n,:] and AZZA[:,m,mm,q] TODO: blas!
                tmp = 0.0;
                for(qq=0;qq<input_dim;qq++){
                  tmp += mu(n,qq)*AZZA(qq,m,mm,q);
                }

                target_mu(n,q) += factor*tmp;
                target_S(n,q) += factor*AZZA_2(q,m,mm,q);
              }
            }
          }
        }
        """ % pragma_string
        support_code = """
        %s
        #include <math.h>
        """ % header_string
        mu = vp.mean
        N,num_inducing,input_dim,mu = mu.shape[0],Z.shape[0],mu.shape[1],param_to_array(mu)
        weave.inline(code, support_code=support_code,
                     arg_names=['N','num_inducing','input_dim','mu','AZZA','AZZA_2','target_mu','target_S','dL_dpsi2'],
                     type_converters=weave.converters.blitz,**weave_options)


    def _weave_dpsi2_dZ(self, dL_dpsi2, Z, vp, target):
        AZA = self.variances*self._ZAinner(vp, Z)

        if config.getboolean('parallel', 'openmp'):
            pragma_string = '#pragma omp parallel for private(n,mm,q)'
            header_string = '#include <omp.h>'
            weave_options =  {'headers'           : ['<omp.h>'],
                              'extra_compile_args': ['-fopenmp -O3'],
                              'extra_link_args'   : ['-lgomp'],
                              'libraries': ['gomp']}
        else:
            pragma_string = ''
            header_string = ''
            weave_options = {'extra_compile_args': ['-O3']}

        code="""
        int n,m,mm,q;
        %s
        for(m=0;m<num_inducing;m++){
          for(q=0;q<input_dim;q++){
            for(mm=0;mm<num_inducing;mm++){
              for(n=0;n<N;n++){
                target(m,q) += 2*dL_dpsi2(n,m,mm)*AZA(n,mm,q);
              }
            }
          }
        }
        """ % pragma_string
        support_code = """
        %s
        #include <math.h>
        """ % header_string

        N,num_inducing,input_dim = vp.mean.shape[0],Z.shape[0],vp.mean.shape[1]
        mu = param_to_array(vp.mean)
        weave.inline(code, support_code=support_code,
                     arg_names=['N','num_inducing','input_dim','AZA','target','dL_dpsi2'],
                     type_converters=weave.converters.blitz,**weave_options)


    @Cache_this(limit=1, ignore_args=(0,))
    def _mu2S(self, vp):
        return np.square(vp.mean) + vp.variance

    @Cache_this(limit=1)
    def _ZAinner(self, vp, Z):
        ZA = Z*self.variances
        inner = (vp.mean[:, None, :] * vp.mean[:, :, None])
        diag_indices = np.diag_indices(vp.mean.shape[1], 2)
        inner[:, diag_indices[0], diag_indices[1]] += vp.variance

        return np.dot(ZA, inner).swapaxes(0, 1)  # NOTE: self.ZAinner \in [num_inducing x num_data x input_dim]!

    def input_sensitivity(self):
        return np.ones(self.input_dim) * self.variances

class LinearFull(Kern):
    def __init__(self, input_dim, rank, W=None, kappa=None, active_dims=None, name='linear_full'):
        super(LinearFull, self).__init__(input_dim, active_dims, name)
        if W is None:
            W = np.ones((input_dim, rank))
        if kappa is None:
            kappa = np.ones(input_dim)
        assert W.shape == (input_dim, rank)
        assert kappa.shape == (input_dim,)

        self.W = Param('W', W)
        self.kappa = Param('kappa', kappa, Logexp())
        self.add_parameters(self.W, self.kappa)

    def K(self, X, X2=None):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,lk->il', X, P, X if X2 is None else X2)

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.kappa.gradient = np.einsum('ij,ik,kj->j', X, dL_dK, X if X2 is None else X2)
        self.W.gradient = np.einsum('ij,kl,ik,lm->jm', X, X if X2 is None else X2, dL_dK, self.W)
        self.W.gradient += np.einsum('ij,kl,ik,jm->lm', X, X if X2 is None else X2, dL_dK, self.W)

    def Kdiag(self, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,ik->i', X, P, X)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.kappa.gradient = np.einsum('ij,i->j', np.square(X), dL_dKdiag)
        self.W.gradient = 2.*np.einsum('ij,ik,jl,i->kl', X, X, self.W, dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2=None):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        if X2 is None:
            return 2.*np.einsum('ij,jk,kl->il', dL_dK, X, P)
        else:
            return np.einsum('ij,jk,kl->il', dL_dK, X2, P)

    def gradients_X_diag(self, dL_dKdiag, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return 2.*np.einsum('jk,i,ij->ik', P, dL_dKdiag, X)


