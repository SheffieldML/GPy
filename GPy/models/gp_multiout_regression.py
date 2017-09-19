# Copyright (c) 2017  Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from .. import util
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from ..core.parameterization.param import Param
from paramz.transformations import Logexp
from ..util.linalg import tdot

class GPMultioutRegression(SparseGP):
    """
    Gaussian Process model for scalable multioutput regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X, Y, Xr_dim, kernel=None, kernel_row=None, likelihood=None, Z=None, Z_row=None, X_row=None, Xvariance_row=None, num_inducing=(10,10), qU_var_r_W_dim=None, qU_var_c_W_dim=None, init='GP', name='GPMR'):

        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if kernel_row is None:
            kernel_row = kern.RBF(Xr_dim,name='kern_row')

        if init=='GP':
            from . import SparseGPRegression, BayesianGPLVM
            from ..util.linalg import jitchol
            Mc, Mr = num_inducing
            print('Intializing with GP...')
            print('Fit Sparse GP...')
            m_sgp = SparseGPRegression(X,Y,kernel=kernel.copy(),num_inducing=Mc)
            m_sgp.likelihood.variance[:] = Y.var()*0.01
            m_sgp.optimize(max_iters=1000)
            print('Fit BGPLVM...')
            m_lvm = BayesianGPLVM(m_sgp.posterior.mean.copy().T,Xr_dim,kernel=kernel_row.copy(), num_inducing=Mr)
            m_lvm.likelihood.variance[:] = m_lvm.Y.var()*0.01
            m_lvm.optimize(max_iters=10000)

            kernel[:] = m_sgp.kern.param_array.copy()
            kernel.variance[:] = np.sqrt(kernel.variance)
            Z = m_sgp.Z.values.copy()
            kernel_row[:] = m_lvm.kern.param_array.copy()
            kernel_row.variance[:] = np.sqrt(kernel_row.variance)
            Z_row = m_lvm.Z.values.copy()
            X_row = m_lvm.X.mean.values.copy()
            Xvariance_row = m_lvm.X.variance.values

            qU_mean = m_lvm.posterior.mean.T.copy()
            qU_var_col_W = jitchol(m_sgp.posterior.covariance)
            qU_var_col_diag = np.full(Mc,1e-5)
            qU_var_row_W = jitchol(m_lvm.posterior.covariance)
            qU_var_row_diag = np.full(Mr,1e-5)
            print('Done.')
        else:
            qU_mean = np.zeros(num_inducing)
            qU_var_col_W = np.random.randn(num_inducing[0],num_inducing[0] if qU_var_c_W_dim is None else qU_var_c_W_dim)*0.01
            qU_var_col_diag = np.full(num_inducing[0],1e-5)
            qU_var_row_W = np.random.randn(num_inducing[1],num_inducing[1] if qU_var_r_W_dim is None else qU_var_r_W_dim)*0.01
            qU_var_row_diag = np.full(num_inducing[1],1e-5)

        if X_row is None:
            u,s,v = np.linalg.svd(Y)
            X_row = Y.T.dot(u[:,:Xr_dim])#*np.sqrt(s)[:Xr_dim])
            X_row = X_row/X_row.std(0)
        if Xvariance_row is None:
            Xvariance_row = np.ones((Y.shape[1],Xr_dim))*0.0001
        if Z is None:
            Z = X[np.random.permutation(X.shape[0])[:num_inducing[0]]].copy()
        if Z_row is None:
            Z_row = X_row[np.random.permutation(X_row.shape[0])[:num_inducing[1]]].copy()

        self.kern_row = kernel_row
        self.X_row = NormalPosterior(X_row, Xvariance_row,name='Xr')
        self.Z_row = Param('Zr', Z_row)
        self.variational_prior_row = NormalPrior()

        self.qU_mean = Param('qU_mean', qU_mean)
        self.qU_var_c_W = Param('qU_var_col_W', qU_var_col_W)
        self.qU_var_c_diag = Param('qU_var_col_diag', qU_var_col_diag, Logexp())
        self.qU_var_r_W = Param('qU_var_row_W',qU_var_row_W)
        self.qU_var_r_diag = Param('qU_var_row_diag', qU_var_row_diag, Logexp())

        #Likelihood
        likelihood = likelihoods.Gaussian(variance=np.var(Y)*0.01)
        from ..inference.latent_function_inference import VarDTC_SVI_Multiout
        inference_method = VarDTC_SVI_Multiout()

        super(GPMultioutRegression,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method)

        self.link_parameters(self.kern_row, self.X_row, self.Z_row,self.qU_mean, self.qU_var_c_W, self.qU_var_c_diag, self.qU_var_r_W, self.qU_var_r_diag)

        self._log_marginal_likelihood = np.nan

    def parameters_changed(self):
        qU_var_c = tdot(self.qU_var_c_W) + np.diag(self.qU_var_c_diag)
        qU_var_r = tdot(self.qU_var_r_W) + np.diag(self.qU_var_r_diag)
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern_row, self.kern, self.X_row, self.X, self.Z_row, self.Z, self.likelihood, self.Y, self.qU_mean ,qU_var_r, qU_var_c)

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.qU_mean.gradient[:] = self.grad_dict['dL_dqU_mean']
        self.qU_var_c_diag.gradient[:] = np.diag(self.grad_dict['dL_dqU_var_c'])
        self.qU_var_c_W.gradient[:] = (self.grad_dict['dL_dqU_var_c']+self.grad_dict['dL_dqU_var_c'].T).dot(self.qU_var_c_W)
        self.qU_var_r_diag.gradient[:] = np.diag(self.grad_dict['dL_dqU_var_r'])
        self.qU_var_r_W.gradient[:] = (self.grad_dict['dL_dqU_var_r']+self.grad_dict['dL_dqU_var_r'].T).dot(self.qU_var_r_W)

        self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag_c'], self.X)
        kerngrad = self.kern.gradient.copy()
        self.kern.update_gradients_full(self.grad_dict['dL_dKfu_c'], self.X, self.Z)
        kerngrad += self.kern.gradient
        self.kern.update_gradients_full(self.grad_dict['dL_dKuu_c'], self.Z, None)
        self.kern.gradient += kerngrad
        #gradients wrt Z
        self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKuu_c'], self.Z)
        self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKfu_c'].T, self.Z, self.X)


        #gradients wrt kernel
        self.kern_row.update_gradients_full(self.grad_dict['dL_dKuu_r'], self.Z_row, None)
        kerngrad = self.kern_row.gradient.copy()
        self.kern_row.update_gradients_expectations(variational_posterior=self.X_row,
                                                Z=self.Z_row,
                                                dL_dpsi0=self.grad_dict['dL_dpsi0_r'],
                                                dL_dpsi1=self.grad_dict['dL_dpsi1_r'],
                                                dL_dpsi2=self.grad_dict['dL_dpsi2_r'])
        self.kern_row.gradient += kerngrad

        #gradients wrt Z
        self.Z_row.gradient = self.kern_row.gradients_X(self.grad_dict['dL_dKuu_r'], self.Z_row)
        self.Z_row.gradient += self.kern_row.gradients_Z_expectations(
                           self.grad_dict['dL_dpsi0_r'],
                           self.grad_dict['dL_dpsi1_r'],
                           self.grad_dict['dL_dpsi2_r'],
                           Z=self.Z_row,
                           variational_posterior=self.X_row)

        self._log_marginal_likelihood -= self.variational_prior_row.KL_divergence(self.X_row)

        self.X_row.mean.gradient, self.X_row.variance.gradient = self.kern_row.gradients_qX_expectations(
                                            variational_posterior=self.X_row,
                                            Z=self.Z_row,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0_r'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1_r'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2_r'])

        self.variational_prior_row.update_gradients_KL(self.X_row)

    def optimize_auto(self,max_iters=10000,verbose=True):
        self.Z.fix(warning=False)
        self.kern.fix(warning=False)
        self.kern_row.fix(warning=False)
        self.Zr.fix(warning=False)
        self.Xr.fix(warning=False)
        self.optimize(max_iters=int(0.1*max_iters),messages=verbose)
        self.unfix()
        self.optimize(max_iters=max_iters,messages=verbose)
