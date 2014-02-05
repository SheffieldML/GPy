# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, tdot, symmetrify, backsub_both_sides, chol_inv, dtrtrs, dpotrs, dpotri
from scipy import linalg
from ..likelihoods import Gaussian, EP,EP_Mixed_Noise
from gp_base import GPBase

class SparseGP(GPBase):
    """
    Variational sparse GP model

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing: Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int
    :param normalize_(X|Y): whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool

    """

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)

        self.Z = Z
        self.num_inducing = Z.shape[0]
        self.backsub = 0
        
        if X_variance is None:
            self.has_uncertain_inputs = False
            self.X_variance = None
        else:
            assert X_variance.shape == X.shape
            self.has_uncertain_inputs = True
            self.X_variance = X_variance

        if normalize_X:
            self.Z = (self.Z.copy() - self._Xoffset) / self._Xscale

        # normalize X uncertainty also
        if self.has_uncertain_inputs:
            self.X_variance /= np.square(self._Xscale)

        self._const_jitter = None

    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z, self.X, self.X_variance)
            self.psi1 = self.kern.psi1(self.Z, self.X, self.X_variance)
            self.psi2 = self.kern.psi2(self.Z, self.X, self.X_variance)
        else:
            self.psi0 = self.kern.Kdiag(self.X)
            self.psi1 = self.kern.K(self.X, self.Z)
            self.psi2 = None

    def _computations(self):
        if self._const_jitter is None or not(self._const_jitter.shape[0] == self.num_inducing):
            self._const_jitter = np.eye(self.num_inducing) * 1e-7

        # factor Kmm
        self._Lm = jitchol(self.Kmm + self._const_jitter)    
        if not self.backsub:
            self._LmInv = linalg.lapack.dtrtri(self._Lm, lower=1)[0] # TODO: not needed in old version
        
        # The rather complex computations of self._A
        if self.has_uncertain_inputs:
            if self.likelihood.is_heteroscedastic:
                psi2_beta = (self.psi2 * (self.likelihood.precision.flatten().reshape(self.num_data, 1, 1))).sum(0)
            else:
                psi2_beta = self.psi2.sum(0) * self.likelihood.precision
            if self.backsub:
                evals, evecs = linalg.eigh(psi2_beta)
                clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
                if not np.array_equal(evals, clipped_evals):
                    pass # print evals
                tmp = evecs * np.sqrt(clipped_evals)
                tmp = tmp.T
                tmp, _ = dtrtrs(self._Lm, np.asfortranarray(tmp.T), lower=1)
                self._A = tdot(tmp) 
            else:
                self._A = np.dot(np.dot(self._LmInv,
                                        psi2_beta),
                                 self._LmInv.T)
        else:
            if self.likelihood.is_heteroscedastic:
                tmp = self.psi1 * (np.sqrt(self.likelihood.precision.flatten().reshape(self.num_data, 1)))
            else:
                tmp = self.psi1 * (np.sqrt(self.likelihood.precision))
            tmp, _ = dtrtrs(self._Lm, np.asfortranarray(tmp.T), lower=1)
            self._A = tdot(tmp)        
        
        # factor B
        self.B = np.eye(self.num_inducing) + self._A
        self.LB = jitchol(self.B)

        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        self.psi1Vf = np.dot(self.psi1.T, self.likelihood.VVT_factor)

        if 1:#self.backsub:
            # back substutue C into psi1Vf
            tmp, info1 = dtrtrs(self._Lm, np.asfortranarray(self.psi1Vf), lower=1, trans=0)
            self._LBi_Lmi_psi1Vf, _ = dtrtrs(self.LB, np.asfortranarray(tmp), lower=1, trans=0)
            # tmp, info2 = dpotrs(self.LB, tmp, lower=1)
            tmp, info2 = dtrtrs(self.LB, self._LBi_Lmi_psi1Vf, lower=1, trans=1)
            self.Cpsi1Vf, info3 = dtrtrs(self._Lm, tmp, lower=1, trans=1)
        else:
            # slower, but more stable (?) version:
            tmp = np.dot(self._LmInv, self.psi1Vf)
            self._LBInv = linalg.lapack.dtrtri(self.LB, lower=True)[0]
            self._LBi_Lmi_psi1Vf = np.dot(self._LBInv, tmp)
            tmp = np.dot(self._LBInv.T, self._LBi_Lmi_psi1Vf)
            self.Cpsi1Vf = np.dot(self._LmInv.T, tmp)
        
        #import ipdb;ipdb.set_trace()
        
        # Compute dL_dKmm
        tmp = tdot(self._LBi_Lmi_psi1Vf)
        self.data_fit = np.trace(tmp)
        self.DBi_plus_BiPBi = backsub_both_sides(self.LB, self.output_dim * np.eye(self.num_inducing) + tmp)
        tmp = -0.5 * self.DBi_plus_BiPBi
        tmp += -0.5 * self.B * self.output_dim
        tmp += self.output_dim * np.eye(self.num_inducing)
        self.dL_dKmm = backsub_both_sides(self._Lm, tmp)

        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertain inputs case
        self.dL_dpsi0 = -0.5 * self.output_dim * (self.likelihood.precision * np.ones([self.num_data, 1])).flatten()
        self.dL_dpsi1 = np.dot(self.likelihood.VVT_factor, self.Cpsi1Vf.T)
        dL_dpsi2_beta = 0.5 * backsub_both_sides(self._Lm, self.output_dim * np.eye(self.num_inducing) - self.DBi_plus_BiPBi)

        if self.likelihood.is_heteroscedastic:

            if self.has_uncertain_inputs:
                self.dL_dpsi2 = self.likelihood.precision.flatten()[:, None, None] * dL_dpsi2_beta[None, :, :]
            else:
                self.dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (self.psi1 * self.likelihood.precision.reshape(self.num_data, 1)).T).T
                self.dL_dpsi2 = None
        else:
            dL_dpsi2 = self.likelihood.precision * dL_dpsi2_beta
            if self.has_uncertain_inputs:
                # repeat for each of the N psi_2 matrices
                self.dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], self.num_data, axis=0)
            else:
                # subsume back into psi1 (==Kmn)
                self.dL_dpsi1 += 2.*np.dot(self.psi1, dL_dpsi2)
                self.dL_dpsi2 = None


        # the partial derivative vector for the likelihood
        if self.likelihood.num_params == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:

            if self.has_uncertain_inputs:
                raise NotImplementedError, "heteroscedatic derivates with uncertain inputs not implemented"

            else:

                LBi = chol_inv(self.LB)
                Lmi_psi1, nil = dtrtrs(self._Lm, np.asfortranarray(self.psi1.T), lower=1, trans=0)
                _LBi_Lmi_psi1, _ = dtrtrs(self.LB, np.asfortranarray(Lmi_psi1), lower=1, trans=0)


                self.partial_for_likelihood = -0.5 * self.likelihood.precision + 0.5 * self.likelihood.V**2
                self.partial_for_likelihood += 0.5 * self.output_dim * (self.psi0 - np.sum(Lmi_psi1**2,0))[:,None] * self.likelihood.precision**2

                self.partial_for_likelihood += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*self.likelihood.precision**2

                self.partial_for_likelihood += -np.dot(self._LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * self.likelihood.Y * self.likelihood.precision**2
                self.partial_for_likelihood += 0.5*np.dot(self._LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * self.likelihood.precision**2

        else:
            # likelihood is not heteroscedatic
            self.partial_for_likelihood = -0.5 * self.num_data * self.output_dim * self.likelihood.precision + 0.5 * self.likelihood.trYYT * self.likelihood.precision ** 2
            self.partial_for_likelihood += 0.5 * self.output_dim * (self.psi0.sum() * self.likelihood.precision ** 2 - np.trace(self._A) * self.likelihood.precision)
            self.partial_for_likelihood += self.likelihood.precision * (0.5 * np.sum(self._A * self.DBi_plus_BiPBi) - self.data_fit)

    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        if self.likelihood.is_heteroscedastic:
            A = -0.5 * self.num_data * self.output_dim * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.likelihood.precision)) - 0.5 * np.sum(self.likelihood.V * self.likelihood.Y)
            B = -0.5 * self.output_dim * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self._A))
        else:
            A = -0.5 * self.num_data * self.output_dim * (np.log(2.*np.pi) - np.log(self.likelihood.precision)) - 0.5 * self.likelihood.precision * self.likelihood.trYYT
            B = -0.5 * self.output_dim * (np.sum(self.likelihood.precision * self.psi0) - np.trace(self._A))
        C = -self.output_dim * (np.sum(np.log(np.diag(self.LB)))) # + 0.5 * self.num_inducing * np.log(sf2))
        D = 0.5 * self.data_fit
        self._A_part, self._B_part, self._C_part, self._D_part = A, B, C, D
        return A + B + C + D + self.likelihood.Z

    def _set_params(self, p):
        self.Z = p[:self.num_inducing * self.input_dim].reshape(self.num_inducing, self.input_dim)
        self.kern._set_params(p[self.Z.size:self.Z.size + self.kern.num_params])
        self.likelihood._set_params(p[self.Z.size + self.kern.num_params:])
        self._compute_kernel_matrices()
        self._computations()
        self.Cpsi1V = None

    def _get_params(self):
        return np.hstack([self.Z.flatten(), self.kern._get_params_transformed(), self.likelihood._get_params()])

    def _get_param_names(self):
        return sum([['iip_%i_%i' % (i, j) for j in range(self.Z.shape[1])] for i in range(self.Z.shape[0])], [])\
            + self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    #def _get_print_names(self):
    #    return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def update_likelihood_approximation(self, **kwargs):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        if not isinstance(self.likelihood, Gaussian): # Updates not needed for Gaussian likelihood
            self.likelihood.restart()
            if self.has_uncertain_inputs:
                Lmi = chol_inv(self._Lm)
                Kmmi = tdot(Lmi.T)
                diag_tr_psi2Kmmi = np.array([np.trace(psi2_Kmmi) for psi2_Kmmi in np.dot(self.psi2, Kmmi)])

                self.likelihood.fit_FITC(self.Kmm, self.psi1.T, diag_tr_psi2Kmmi, **kwargs) # This uses the fit_FITC code, but does not perfomr a FITC-EP.#TODO solve potential confusion
                # raise NotImplementedError, "EP approximation not implemented for uncertain inputs"
            else:
                self.likelihood.fit_DTC(self.Kmm, self.psi1.T, **kwargs)
                # self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
                self._set_params(self._get_params()) # update the GP

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1, self.X, self.Z)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = self.kern.dK_dX(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dZ += self.kern.dK_dX(self.dL_dpsi1.T, self.Z, self.X)
        return dL_dZ

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        """
        Internal helper function for making predictions, does not account for
        normalization or likelihood function
        """

        Bi, _ = dpotri(self.LB, lower=0) # WTH? this lower switch should be 1, but that doesn't work!
        symmetrify(Bi)
        Kmmi_LmiBLmi = backsub_both_sides(self._Lm, np.eye(self.num_inducing) - Bi)

        if self.Cpsi1V is None:
            psi1V = np.dot(self.psi1.T, self.likelihood.V)
            tmp, _ = dtrtrs(self._Lm, np.asfortranarray(psi1V), lower=1, trans=0)
            tmp, _ = dpotrs(self.LB, tmp, lower=1)
            self.Cpsi1V, _ = dtrtrs(self._Lm, tmp, lower=1, trans=1)

        if X_variance_new is None:
            Kx = self.kern.K(self.Z, Xnew, which_parts=which_parts)
            mu = np.dot(Kx.T, self.Cpsi1V)
            if full_cov:
                Kxx = self.kern.K(Xnew, which_parts=which_parts)
                var = Kxx - mdot(Kx.T, Kmmi_LmiBLmi, Kx) # NOTE this won't work for plotting
            else:
                Kxx = self.kern.Kdiag(Xnew, which_parts=which_parts)
                var = Kxx - np.sum(Kx * np.dot(Kmmi_LmiBLmi, Kx), 0)
        else:
            # assert which_parts=='all', "swithching out parts of variational kernels is not implemented"
            Kx = self.kern.psi1(self.Z, Xnew, X_variance_new) # , which_parts=which_parts) TODO: which_parts
            mu = np.dot(Kx, self.Cpsi1V)
            if full_cov:
                raise NotImplementedError, "TODO"
            else:
                Kxx = self.kern.psi0(self.Z, Xnew, X_variance_new)
                psi2 = self.kern.psi2(self.Z, Xnew, X_variance_new)
                var = Kxx - np.sum(np.sum(psi2 * Kmmi_LmiBLmi[None, :, :], 1), 1)

        return mu, var[:, None]

    def predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False, **likelihood_args):
        """
        Predict the function(s) at the new point(s) Xnew.

        **Arguments**

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param X_variance_new: The uncertainty in the prediction points
        :type X_variance_new: np.ndarray, Nnew x self.input_dim
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.input_dim
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim


           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        # normalize X values
        Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        if X_variance_new is not None:
            X_variance_new = X_variance_new / self._Xscale ** 2

        # here's the actual prediction by the GP model
        mu, var = self._raw_predict(Xnew, X_variance_new, full_cov=full_cov, which_parts=which_parts)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov, **likelihood_args)

        return mean, var, _025pm, _975pm


    def plot_f(self, samples=0, plot_limits=None, which_data_rows='all',
            which_data_ycols='all', which_parts='all', resolution=None,
            full_cov=False, fignum=None, ax=None):

        """
        Plot the GP's view of the world, where the data is normalized and the
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - Not implemented in higher dimensions

        :param samples: the number of a posteriori samples to plot
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_data_rows: which if the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param full_cov:
        :type full_cov: bool
                :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle

        :param output: which output to plot (for multiple output models only)
        :type output: integer (first output is 0)
        """
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data_rows is 'all':
            which_data_rows = slice(None)

        GPBase.plot_f(self, samples=samples, plot_limits=plot_limits, which_data_rows=which_data_rows, which_data_ycols=which_data_ycols, which_parts=which_parts, resolution=resolution, fignum=fignum, ax=ax)

        if self.X.shape[1] == 1:
            if self.has_uncertain_inputs:
                Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now
                ax.errorbar(Xu[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu, np.zeros_like(Zu) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self, plot_limits=None, which_data_rows='all',
            which_data_ycols='all', which_parts='all', fixed_inputs=[],
            plot_raw=False,
            levels=20, samples=0, fignum=None, ax=None, resolution=None):
        """
        Plot the posterior of the sparse GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, use fixed_inputs to plot the GP  with some of the inputs fixed.

        Can plot only part of the data and part of the posterior functions
        using which_data_rowsm which_data_ycols and which_parts

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data_rows: which of the training data to plot (default all)
        :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
        :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        :type which_data_rows: 'all' or a list of integers
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :type output: integer (first output is 0)
        :param linecol: color of line to plot.
        :type linecol:
        :param fillcol: color of fill
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        """
        #deal work out which ax to plot on
        #Need these because we use which_data_rows in this function not just base
        if which_data_rows == 'all':
            which_data_rows = slice(None)
        if which_data_ycols == 'all':
            which_data_ycols = np.arange(self.output_dim)
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        #work out what the inputs are for plotting (1D or 2D)
        fixed_dims = np.array([i for i,v in fixed_inputs])
        free_dims = np.setdiff1d(np.arange(self.input_dim),fixed_dims)

        #call the base plotting
        GPBase.plot(self, samples=samples, plot_limits=plot_limits,
                which_data_rows=which_data_rows,
                which_data_ycols=which_data_ycols, fixed_inputs=fixed_inputs,
                which_parts=which_parts, resolution=resolution, levels=20,
                fignum=fignum, ax=ax)

        if len(free_dims) == 1:
            #plot errorbars for the uncertain inputs
            if self.has_uncertain_inputs:
                Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now
                ax.errorbar(Xu[which_data_rows, 0], self.likelihood.data[which_data_rows, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data_rows, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)

            #plot the inducing inputs
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu, np.zeros_like(Zu) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif len(free_dims) == 2:
            Zu = self.Z * self._Xscale + self._Xoffset
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return GPBase.getstate(self) + [self.Z,
                self.num_inducing,
                self.has_uncertain_inputs,
                self.X_variance]

    def setstate(self, state):
        self.X_variance = state.pop()
        self.has_uncertain_inputs = state.pop()
        self.num_inducing = state.pop()
        self.Z = state.pop()
        GPBase.setstate(self, state)


