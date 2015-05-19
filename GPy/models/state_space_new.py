# Copyright (c) 2013, Arno Solin.
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
# This implementation of converting GPs to state space models is based on the article:
#
#  @article{Sarkka+Solin+Hartikainen:2013,
#     author = {Simo S\"arkk\"a and Arno Solin and Jouni Hartikainen},
#       year = {2013},
#      title = {Spatiotemporal learning via infinite-dimensional {B}ayesian filtering and smoothing},
#    journal = {IEEE Signal Processing Magazine},
#     volume = {30},
#     number = {4},
#      pages = {51--61}
#  }
#

import numpy as np
from scipy import linalg
from ..core import Model
from .. import kern
from GPy.plotting.matplot_dep.models_plots import gpplot
from GPy.plotting.matplot_dep.base_plots import x_frame1D
from GPy.plotting.matplot_dep import Tango
import pylab as pb
from GPy.core.parameterization.param import Param

import GPy
from .. import likelihoods
import GPy.models.state_space_main as ssm
#import state_space_main as ssm
reload(ssm)
print ssm.__file__

class StateSpace(Model):
    def __init__(self, X, Y, kernel=None, sigma2=1.0, name='StateSpace'):
        super(StateSpace, self).__init__(name=name)
        self.num_data, input_dim = X.shape
        assert input_dim==1, "State space methods for time only"
        num_data_Y, self.output_dim = Y.shape
        assert num_data_Y == self.num_data, "X and Y data don't match"
        assert self.output_dim == 1, "State space methods for single outputs only"

        # Make sure the observations are ordered in time
        sort_index = np.argsort(X[:,0])
        self.X = X[sort_index]
        self.Y = Y[sort_index]

        # Noise variance
        self.likelihood = likelihoods.Gaussian()
        
        # Default kernel
        if kernel is None:
            self.kern = kern.Matern32(1)
        else:
            self.kern = kernel
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior = None

        # Assert that the kernel is supported
        if not hasattr(self.kern, 'sde'):
            raise NotImplementedError('SDE must be implemented for the kernel being used')
        #assert self.kern.sde() not False, "This kernel is not supported for state space estimation"

    def parameters_changed(self):
        """
        Parameters have now changed
        """
        
        # Get the model matrices from the kernel
        (F,L,Qc,H,P_inf,dFt,dQct,dP_inft) = self.kern.sde()

        # necessary parameters
        measurement_dim = self.output_dim
        grad_params_no = dFt.shape[2]+1 # we also add measurement noise as a parameter

        # add measurement noise as a parameter and get the gradient matrices
        dF    = np.zeros([dFt.shape[0],dFt.shape[1],grad_params_no])
        dQc   = np.zeros([dQct.shape[0],dQct.shape[1],grad_params_no])
        dP_inf = np.zeros([dP_inft.shape[0],dP_inft.shape[1],grad_params_no])

        # Assign the values for the kernel function
        dF[:,:,:-1] = dFt
        dQc[:,:,:-1] = dQct
        dP_inf[:,:,:-1] = dP_inft

        # The sigma2 derivative
        dR = np.zeros([measurement_dim,measurement_dim,grad_params_no])
        dR[:,:,-1] = np.eye(measurement_dim)


        # Use the Kalman filter to evaluate the likelihood
        
        grad_calc_params = {}
        grad_calc_params['dP_inf'] = dP_inf
        grad_calc_params['dF'] = dF
        grad_calc_params['dQc'] = dQc
        grad_calc_params['dR'] = dR
        
        (filter_means, filter_covs, log_likelihood, 
         grad_log_likelihood,SmootherMatrObject) = ssm.ContDescrStateSpace.cont_discr_kalman_filter(F,L,Qc,H,self.Gaussian_noise.variance,P_inf,self.X,self.Y,m_init=None,
                                      P_init=None, calc_log_likelihood=True, 
                                      calc_grad_log_likelihood=True, 
                                      grad_params_no=grad_params_no, 
                                      grad_calc_params=grad_calc_params)
                      
        self._log_marginal_likelihood = log_likelihood
        #gradients  = self.compute_gradients()
        self.likelihood.update_gradients(grad_log_likelihood[-1,0])
        
        self.kern.sde_update_gradient_full(grad_log_likelihood[:-1,0])
        
    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _predict_raw(self, Xnew, Ynew=None, filteronly=False):
        """
        Inner function. It is called only from inside this class
        """
        
        # Set defaults
        if Ynew is None:
            Ynew = self.Y

        # Make a single matrix containing training and testing points
        X = np.vstack((self.X, Xnew))
        Y = np.vstack((Ynew, np.nan*np.zeros(Xnew.shape)))

        # Sort the matrix (save the order)
        _, return_index, return_inverse = np.unique(X,True,True)
        X = X[return_index]
        Y = Y[return_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,P_inf,dF,dQc,dP_inf) = self.kern.sde()
        state_dim = F.shape[0]        
        
        # Run the Kalman filter
        (M, P, tmp_log_likelihood, 
         tmp_grad_log_likelihood,SmootherMatrObject) = ssm.ContDescrStateSpace.cont_discr_kalman_filter(F,L,Qc,H,self.sigma2,P_inf,self.X,self.Y,m_init=None,
                                      P_init=None, calc_log_likelihood=False, 
                                      calc_grad_log_likelihood=False) 
                                      
        # Run the Rauch-Tung-Striebel smoother
        if not filteronly:
            (M, P) = ssm.ContDescrStateSpace.cont_discr_rts_smoother(state_dim, M, P, 
                                AQcomp=SmootherMatrObject, X=X, F=F,L=L,Qc=Qc)
        
        # remove initial values        
        M = M[:,1:]
        P = P[:,:,1:]        
        
        # Put the data back in the original order
        M = M[:,return_inverse]
        P = P[:,:,return_inverse]

        # Only return the values for Xnew
        M = M[:,self.num_data:]
        P = P[:,:,self.num_data:]

        # Calculate the mean and variance
        m = H.dot(M).T
        V = np.tensordot(H[0],P,(0,0))
        V = np.tensordot(V,H[0],(0,0))
        V = V[:,None]

        # Return the posterior of the state
        return (m, V)

    def predict(self, Xnew, filteronly=False):

        # Run the Kalman filter to get the state
        (m, V) = self._predict_raw(Xnew,filteronly=filteronly)

        # Add the noise variance to the state variance
        V += self.sigma2

        # Lower and upper bounds
        lower = m - 2*np.sqrt(V)
        upper = m + 2*np.sqrt(V)

        # Return mean and variance
        return (m, V, lower, upper)

    def plot(self, plot_limits=None, levels=20, samples=0, fignum=None,
            ax=None, resolution=None, plot_raw=False, plot_filter=False,
            linecol=Tango.colorsHex['darkBlue'],fillcol=Tango.colorsHex['lightBlue']):

        # Deal with optional parameters
        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        # Define the frame on which to plot
        resolution = resolution or 200
        Xgrid, xmin, xmax = x_frame1D(self.X, plot_limits=plot_limits)

        # Make a prediction on the frame and plot it
        if plot_raw:
            m, v = self.predict_raw(Xgrid,filteronly=plot_filter)
            lower = m - 2*np.sqrt(v)
            upper = m + 2*np.sqrt(v)
            Y = self.Y
        else:
            m, v, lower, upper = self.predict(Xgrid,filteronly=plot_filter)
            Y = self.Y

        # Plot the values
        gpplot(Xgrid, m, lower, upper, axes=ax, edgecol=linecol, fillcol=fillcol)
        ax.plot(self.X, self.Y, 'kx', mew=1.5)

        # Optionally plot some samples
        if samples:
            if plot_raw:
                Ysim = self.posterior_samples_f(Xgrid, samples)
            else:
                Ysim = self.posterior_samples(Xgrid, samples)
            for yi in Ysim.T:
                ax.plot(Xgrid, yi, Tango.colorsHex['darkBlue'], linewidth=0.25)

        # Set the limits of the plot to some sensible values
        ymin, ymax = min(np.append(Y.flatten(), lower.flatten())), max(np.append(Y.flatten(), upper.flatten()))
        ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def prior_samples_f(self,X,size=10):

        # Sort the matrix (save the order)
        (_, return_index, return_inverse) = np.unique(X,True,True)
        X = X[return_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.kern.sde()

        # Allocate space for results
        Y = np.empty((size,X.shape[0]))

        # Simulate random draws
        #for j in range(0,size):
        #    Y[j,:] = H.dot(self.simulate(F,L,Qc,Pinf,X.T))
        Y = self.simulate(F,L,Qc,Pinf,X.T,size)

        # Only observations
        Y = np.tensordot(H[0],Y,(0,0))

        # Reorder simulated values
        Y = Y[:,return_inverse]

        # Return trajectory
        return Y.T

    def posterior_samples_f(self,X,size=10):

        # Sort the matrix (save the order)
        (_, return_index, return_inverse) = np.unique(X,True,True)
        X = X[return_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.kern.sde()

        # Run smoother on original data
        (m,V) = self.predict_raw(X)

        # Simulate random draws from the GP prior
        y = self.prior_samples_f(np.vstack((self.X, X)),size)

        # Allocate space for sample trajectories
        Y = np.empty((size,X.shape[0]))

        # Run the RTS smoother on each of these values
        for j in range(0,size):
            yobs =  y[0:self.num_data,j:j+1] + np.sqrt(self.sigma2)*np.random.randn(self.num_data,1)
            (m2,V2) = self.predict_raw(X,Ynew=yobs)
            Y[j,:] = m.T + y[self.num_data:,j].T - m2.T

        # Reorder simulated values
        Y = Y[:,return_inverse]

        # Return posterior sample trajectories
        return Y.T

    def posterior_samples(self, X, size=10):

        # Make samples of f
        Y = self.posterior_samples_f(X,size)

        # Add noise
        Y += np.sqrt(self.sigma2)*np.random.randn(Y.shape[0],Y.shape[1])

        # Return trajectory
        return Y
        
        
    def simulate(self,F,L,Qc,Pinf,X,size=1):
        # Simulate a trajectory using the state space model

        # Allocate space for results
        f = np.zeros((F.shape[0],size,X.shape[1]))

        # Initial state
        f[:,:,1] = np.linalg.cholesky(Pinf).dot(np.random.randn(F.shape[0],size))

        # Time step lengths
        dt = np.empty(X.shape)
        dt[:,0] = X[:,1]-X[:,0]
        dt[:,1:] = np.diff(X)

        # Solve the LTI SDE for these time steps
        As, Qs, index = ssm.ContDescrStateSpace.lti_sde_to_descrete(F,L,Qc,dt)

        # Sweep through remaining time points
        for k in range(1,X.shape[1]):

            # Form discrete-time model
            A = As[:,:,index[1-k]]
            Q = Qs[:,:,index[1-k]]

            # Draw the state
            f[:,:,k] = A.dot(f[:,:,k-1]) + np.dot(np.linalg.cholesky(Q),np.random.randn(A.shape[0],size))

        # Return values
        return f
