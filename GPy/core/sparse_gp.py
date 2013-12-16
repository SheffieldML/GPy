# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, tdot, symmetrify, backsub_both_sides, chol_inv, dtrtrs, dpotrs, dpotri
from gp_base import GPBase
from GPy.core import Param

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

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False, name='sparse gp'):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=normalize_X, name=name)

        self.Z = Z
        self.num_inducing = Z.shape[0]

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

        self.Z = Param('inducing inputs', self.Z)
        self.add_parameter(self.Z, gradient=self.dL_dZ, index=0)
        self.add_parameter(self.kern, gradient=self.dL_dtheta)
        self.add_parameter(self.likelihood, gradient=lambda:self.likelihood._gradients(partial=self.partial_for_likelihood))
        #self.Z.add_observer(self, lambda Z: self._compute_kernel_matrices() or self._computations())

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
    def parameters_changed(self):
        self._compute_kernel_matrices()
        self._computations()
        self.Cpsi1V = None
        self.dL_dK = self.dL_dKmm
        super(SparseGP, self).parameters_changed()


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


    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False, fignum=None, ax=None):
        """
        Plot the GP's view of the world, where the data is normalized and the
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - Not implemented in higher dimensions

        :param samples: the number of a posteriori samples to plot
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
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
        if which_data is 'all':
            which_data = slice(None)

        GPBase.plot_f(self, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, full_cov=full_cov, fignum=fignum, ax=ax)

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

    def plot(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, fignum=None, ax=None):
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data is 'all':
            which_data = slice(None)

        GPBase.plot(self, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, levels=20, fignum=fignum, ax=ax)

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

    def predict_single_output(self, Xnew, output=0, which_parts='all', full_cov=False):
        """
        For a specific output, predict the function at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param output: output to predict
        :type output: integer in {0,..., num_outputs-1}
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.input_dim
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim

        .. Note:: For multiple output models only
        """

        assert hasattr(self,'multioutput')
        index = np.ones_like(Xnew)*output
        Xnew = np.hstack((Xnew,index))

        # normalize X values
        Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, which_parts=which_parts)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov, noise_model = output)
        return mean, var, _025pm, _975pm

    def _raw_predict_single_output(self, _Xnew, output=0, X_variance_new=None, which_parts='all', full_cov=False,stop=False):
        """
        Internal helper function for making predictions for a specific output,
        does not account for normalization or likelihood
        ---------

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param output: output to predict
        :type output: integer in {0,..., num_outputs-1}
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal

        .. Note:: For multiple output models only
        """
        Bi, _ = dpotri(self.LB, lower=0)  # WTH? this lower switch should be 1, but that doesn't work!
        symmetrify(Bi)
        Kmmi_LmiBLmi = backsub_both_sides(self._Lm, np.eye(self.num_inducing) - Bi)

        if self.Cpsi1V is None:
            psi1V = np.dot(self.psi1.T,self.likelihood.V)
            tmp, _ = dtrtrs(self._Lm, np.asfortranarray(psi1V), lower=1, trans=0)
            tmp, _ = dpotrs(self.LB, tmp, lower=1)
            self.Cpsi1V, _ = dtrtrs(self._Lm, tmp, lower=1, trans=1)

        assert hasattr(self,'multioutput')
        index = np.ones_like(_Xnew)*output
        _Xnew = np.hstack((_Xnew,index))

        if X_variance_new is None:
            Kx = self.kern.K(self.Z, _Xnew, which_parts=which_parts)
            mu = np.dot(Kx.T, self.Cpsi1V)
            if full_cov:
                Kxx = self.kern.K(_Xnew, which_parts=which_parts)
                var = Kxx - mdot(Kx.T, Kmmi_LmiBLmi, Kx) # NOTE this won't work for plotting
            else:
                Kxx = self.kern.Kdiag(_Xnew, which_parts=which_parts)
                var = Kxx - np.sum(Kx * np.dot(Kmmi_LmiBLmi, Kx), 0)
        else:
            Kx = self.kern.psi1(self.Z, _Xnew, X_variance_new)
            mu = np.dot(Kx, self.Cpsi1V)
            if full_cov:
                raise NotImplementedError, "TODO"
            else:
                Kxx = self.kern.psi0(self.Z, _Xnew, X_variance_new)
                psi2 = self.kern.psi2(self.Z, _Xnew, X_variance_new)
                var = Kxx - np.sum(np.sum(psi2 * Kmmi_LmiBLmi[None, :, :], 1), 1)

        return mu, var[:, None]


    def plot_single_output_f(self, output=None, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False, fignum=None, ax=None):

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data is 'all':
            which_data = slice(None)

        GPBase.plot_single_output_f(self, output=output, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, full_cov=full_cov, fignum=fignum, ax=ax)

        if self.X.shape[1] == 2:
            if self.has_uncertain_inputs:
                Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now
                ax.errorbar(Xu[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)
            Zu = self.Z * self._Xscale + self._Xoffset
            Zu = Zu[Zu[:,1]==output,0:1]
            ax.plot(Zu[:,0], np.zeros_like(Zu[:,0]) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            Zu = self.Z * self._Xscale + self._Xoffset
            Zu = Zu[Zu[:,1]==output,0:2]
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')


        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot_single_output(self, output=None, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, fignum=None, ax=None):
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)
        if fignum is None and ax is None:
                fignum = fig.num
        if which_data is 'all':
            which_data = slice(None)

        GPBase.plot_single_output(self, samples=samples, plot_limits=plot_limits, which_data='all', which_parts='all', resolution=resolution, levels=20, fignum=fignum, ax=ax, output=output)

        if self.X.shape[1] == 2:
            if self.has_uncertain_inputs:
                Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now
                ax.errorbar(Xu[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)
            Zu = self.Z * self._Xscale + self._Xoffset
            Zu = Zu[Zu[:,1]==output,0:1]
            ax.plot(Zu, np.zeros_like(Zu) + ax.get_ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 3:
            Zu = self.Z * self._Xscale + self._Xoffset
            Zu = Zu[Zu[:,1]==output,0:1]
            ax.plot(Zu[:, 0], Zu[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"
