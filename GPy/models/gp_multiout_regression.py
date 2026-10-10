# Copyright (c) 2017  Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import warnings

import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from ..core.parameterization.param import Param
from paramz.transformations import Logexp
from ..util.linalg import tdot


def _match_qU_mean(mean, Mc, Mr):
    """Ensure qU_mean is shaped (Mc, Mr), padding or truncating as needed."""
    mean = np.asarray(mean, dtype=float)
    out = np.zeros((Mc, Mr))
    if mean.ndim != 2:
        return out
    mc = min(Mc, mean.shape[0])
    mr = min(Mr, mean.shape[1])
    out[:mc, :mr] = mean[:mc, :mr]
    return out


def _qU_cov_factors(cov, M, W_dim=None):
    """
    Low-rank-plus-diag factors for a M×M covariance.

    ``cov`` may come from a warm-start posterior whose size differs from ``M``
    (e.g. BGPLVM posterior over D outputs while Mr inducing points were requested).
    """
    from ..util.linalg import jitchol

    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        n = cov.shape[0]
        if n == M:
            base = cov
        elif n > M:
            base = cov[:M, :M]
        else:
            base = np.eye(M) * 1e-2
            base[:n, :n] = cov
        W = jitchol(base)
        if W_dim is not None and W_dim < M:
            W = W[:, :W_dim]
    else:
        width = M if W_dim is None else W_dim
        W = np.random.randn(M, width) * 0.01
    diag = np.full(M, 1e-5)
    return W, diag


class GPMultioutRegression(SparseGP):
    """
    Gaussian Process model for multi-output regression without missing data

    This is an implementation of Latent Variable Multiple Output Gaussian Processes (LVMOGP) in [Dai_et_al_2017]_.

    .. rubric:: References

    .. [Dai_et_al_2017] Dai, Z.; Alvarez, M.A.; Lawrence, N.D: Efficient Modeling of Latent Information in Supervised Learning using Gaussian Processes. In NIPS, 2017.

    :param X: input observations.
    :type X: numpy.ndarray
    :param Y: output observations, each column corresponding to an output dimension.
    :type Y: numpy.ndarray
    :param int Xr_dim: the dimensionality of a latent space, in which output dimensions are embedded in
    :param kernel: a GPy kernel for GP of individual output dimensions ** defaults to RBF **
    :type kernel: GPy.kern.Kern or None
    :param kernel_row: a GPy kernel for the GP of the latent space ** defaults to RBF **
    :type kernel_row: GPy.kern.Kern or None
    :param Z: inducing inputs
    :type Z: numpy.ndarray or None
    :param Z_row: inducing inputs for the latent space
    :type Z_row: numpy.ndarray or None
    :param X_row: the initial value of the mean of the variational posterior distribution of points in the latent space
    :type X_row: numpy.ndarray or None
    :param Xvariance_row: the initial value of the variance of the variational posterior distribution of points in the latent space
    :type Xvariance_row: numpy.ndarray or None
    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space (must be ``<=`` number of outputs; capped with a warning if larger — #733).
    :type num_inducing: (int, int)
    :param int qU_var_r_W_dim: the dimensionality of the covariance of q(U) for the latent space. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param int qU_var_c_W_dim: the dimensionality of the covariance of q(U) for the GP regression. If it is smaller than the number of inducing points, it represents a low-rank parameterization of the covariance matrix.
    :param str init: the choice of initialization: 'GP' or 'rand'. With 'rand', the model is initialized randomly. With 'GP', the model is initialized through a protocol as follows: (1) fits a sparse GP (2) fits a BGPLVM based on the outcome of sparse GP (3) initialize the model based on the outcome of the BGPLVM.
    :param str name: the name of the model

    """
    def __init__(self, X, Y, Xr_dim, kernel=None, kernel_row=None, Z=None, Z_row=None, X_row=None, Xvariance_row=None, num_inducing=(10,10), qU_var_r_W_dim=None, qU_var_c_W_dim=None, init='GP', name='GPMR'):

        D = Y.shape[1]
        Mc, Mr = int(num_inducing[0]), int(num_inducing[1])
        if Mr > D:
            warnings.warn(
                "num_inducing[1]=%d exceeds the number of outputs D=%d; "
                "capping Mr to D so q(U) row covariance matches Z_row (#733)."
                % (Mr, D),
                UserWarning,
                stacklevel=2,
            )
            Mr = D

        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if kernel_row is None:
            kernel_row = kern.RBF(Xr_dim,name='kern_row')

        qU_mean = None
        qU_var_col_W = qU_var_col_diag = None
        qU_var_row_W = qU_var_row_diag = None

        if init=='GP':
            from . import SparseGPRegression, BayesianGPLVM
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

            Mc = Z.shape[0]
            Mr = Z_row.shape[0]
            qU_mean = _match_qU_mean(m_lvm.posterior.mean.T.copy(), Mc, Mr)
            qU_var_col_W, qU_var_col_diag = _qU_cov_factors(
                m_sgp.posterior.covariance, Mc, qU_var_c_W_dim
            )
            qU_var_row_W, qU_var_row_diag = _qU_cov_factors(
                m_lvm.posterior.covariance, Mr, qU_var_r_W_dim
            )
            print('Done.')
        else:
            if X_row is None:
                u,s,v = np.linalg.svd(Y)
                X_row = Y.T.dot(u[:,:Xr_dim])#*np.sqrt(s)[:Xr_dim])
                X_row = X_row/X_row.std(0)
            if Xvariance_row is None:
                Xvariance_row = np.ones((Y.shape[1],Xr_dim))*0.0001
            if Z is None:
                Z = X[np.random.permutation(X.shape[0])[:Mc]].copy()
            if Z_row is None:
                Z_row = X_row[np.random.permutation(X_row.shape[0])[:Mr]].copy()
            Mc = Z.shape[0]
            Mr = Z_row.shape[0]
            qU_mean = np.zeros((Mc, Mr))
            qU_var_col_W, qU_var_col_diag = _qU_cov_factors(
                None, Mc, qU_var_c_W_dim
            )
            qU_var_row_W, qU_var_row_diag = _qU_cov_factors(
                None, Mr, qU_var_r_W_dim
            )

        if X_row is None:
            u,s,v = np.linalg.svd(Y)
            X_row = Y.T.dot(u[:,:Xr_dim])#*np.sqrt(s)[:Xr_dim])
            X_row = X_row/X_row.std(0)
        if Xvariance_row is None:
            Xvariance_row = np.ones((Y.shape[1],Xr_dim))*0.0001
        if Z is None:
            Z = X[np.random.permutation(X.shape[0])[:Mc]].copy()
        if Z_row is None:
            Z_row = X_row[np.random.permutation(X_row.shape[0])[:Mr]].copy()

        # Final sizes always follow the inducing sets actually used (#733).
        Mc = Z.shape[0]
        Mr = Z_row.shape[0]
        qU_mean = _match_qU_mean(qU_mean, Mc, Mr)
        if qU_var_col_W.shape[0] != Mc:
            qU_var_col_W, qU_var_col_diag = _qU_cov_factors(
                None, Mc, qU_var_c_W_dim
            )
        if qU_var_row_W.shape[0] != Mr:
            qU_var_row_W, qU_var_row_diag = _qU_cov_factors(
                None, Mr, qU_var_r_W_dim
            )
        if qU_var_col_diag.shape[0] != Mc:
            qU_var_col_diag = np.full(Mc, 1e-5)
        if qU_var_row_diag.shape[0] != Mr:
            qU_var_row_diag = np.full(Mr, 1e-5)

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
        """
        Optimize the model parameters through a pre-defined protocol.

        :param int max_iters: the maximum number of iterations.
        :param boolean verbose: print the progress of optimization or not.
        """
        self.Z.fix(warning=False)
        self.kern.fix(warning=False)
        self.kern_row.fix(warning=False)
        self.Zr.fix(warning=False)
        self.Xr.fix(warning=False)
        self.optimize(max_iters=int(0.1*max_iters),messages=verbose)
        self.unfix()
        self.optimize(max_iters=max_iters,messages=verbose)
