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
from GPy.util.plot import gpplot, Tango, x_frame1D
import pylab as pb

class StateSpace(Model):
    def __init__(self, X, Y, kernel=None):
        super(StateSpace, self).__init__()
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
        self.sigma2 = 1.

        # Default kernel
        if kernel is None:
            self.kern = kern.Matern32(1)
        else:
            self.kern = kernel

        # Make sure all parameters are positive
        self.ensure_default_constraints()

        # Assert that the kernel is supported
        #assert self.kern.sde() not False, "This kernel is not supported for state space estimation"

    def _set_params(self, x):
        self.kern._set_params(x[:self.kern.num_params_transformed()])
        self.sigma2 = x[-1]

    def _get_params(self):
        return np.append(self.kern._get_params_transformed(), self.sigma2)

    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + ['noise_variance']

    def log_likelihood(self):

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf) = self.kern.sde()

        # Use the Kalman filter to evaluate the likelihood
        return self.kf_likelihood(F,L,Qc,H,self.sigma2,Pinf,self.X.T,self.Y.T)

    def _log_likelihood_gradients(self):

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.kern.sde()

        # Calculate the likelihood gradients TODO
        #return self.kf_likelihood_g(F,L,Qc,self.sigma2,H,Pinf,dF,dQc,dPinf,self.X,self.Y) 
        return False

    def predict_raw(self, Xnew):

        # Make a single matrix containing training and testing points
        X = np.vstack((self.X, Xnew))
        Y = np.vstack((self.Y, np.nan*np.zeros(Xnew.shape)))

        # Sort the matrix (save the order)
        _, return_index, return_inverse = np.unique(X,True,True)
        X = X[return_index]
        Y = Y[return_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf) = self.kern.sde()

        # Run the Kalman filter
        (M, P) = self.kalman_filter(F,L,Qc,H,self.sigma2,Pinf,X.T,Y.T)

        # Run the Rauch-Tung-Striebel smoother
        (M, P) = self.rts_smoother(F,L,Qc,X.T,M,P)

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

    def predict(self, Xnew):

        # Run the Kalman filter to get the state
        (m, V) = self.predict_raw(Xnew)

        # Add the noise variance to the state variance
        V += self.sigma2

        # Lower and upper bounds
        lower = m - 2*np.sqrt(V)
        upper = m + 2*np.sqrt(V)

        # Return mean and variance
        return (m, V, lower, upper)

    def plot(self, plot_limits=None, levels=20, samples=0, fignum=None,
            ax=None, resolution=None, plot_raw=False,
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
            m, v = self.predict_raw(Xgrid)
            lower = m - 2*np.sqrt(v)
            upper = m + 2*np.sqrt(v)
            Y = self.Y
        else:
            m, v, lower, upper = self.predict(Xgrid)
            Y = self.Y

        # Plot the values
        gpplot(Xgrid, m, lower, upper, axes=ax, edgecol=linecol, fillcol=fillcol)
        ax.plot(self.X, self.Y, 'kx', mew=1.5)

        # Optionally plot some samples
        if samples:
            Ysim = self.posterior_samples(Xgrid, samples)
            for yi in Ysim.T:
                ax.plot(Xgrid, yi, Tango.colorsHex['darkBlue'], linewidth=0.25)

        # Set the limits of the plot to some sensible values
        ymin, ymax = min(np.append(Y.flatten(), lower.flatten())), max(np.append(Y.flatten(), upper.flatten()))
        ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def posterior_samples_f(self,X,size=10):

        # Reorder X values
        sort_index = np.argsort(X[:,0])
        X = X[sort_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf) = self.kern.sde()

        # Allocate space for results
        Y = np.empty((size,X.shape[0]))

        # Simulate random draws
        for j in range(0,size):
            Y[j,:] = H.dot(self.simulate(F,L,Qc,Pinf,X.T))

        # Reorder simulated values
        Y[:,sort_index] = Y[:,:]

        # Return trajectory
        return Y.T

    def posterior_samples(self, X, size=10):

        # Make samples of f
        Y = self.posterior_samples_f(X,size)

        # Add noise
        Y += np.sqrt(self.sigma2)*np.random.randn(Y.shape[0],Y.shape[1])

        # Return trajectory
        return Y

    def kalman_filter(self,F,L,Qc,H,R,Pinf,X,Y):
        # KALMAN_FILTER - Run the Kalman filter for a given model and data

        # Allocate space for results
        MF = np.empty((F.shape[0],Y.shape[1]))
        PF = np.empty((F.shape[0],F.shape[0],Y.shape[1]))

        # Initialize
        MF[:,-1] = np.zeros(F.shape[0])
        PF[:,:,-1] = Pinf.copy()

        # Time step lengths
        dt = np.empty(X.shape)
        dt[:,0] = X[:,1]-X[:,0]
        dt[:,1:] = np.diff(X)

        # Solve the LTI SDE for these time steps
        As, Qs, index = self.lti_disc(F,L,Qc,dt)

        # Kalman filter
        for k in range(0,Y.shape[1]):

            # Form discrete-time model
            #(A, Q) = self.lti_disc(F,L,Qc,dt[:,k])
            A = As[:,:,index[k]];
            Q = Qs[:,:,index[k]];

            # Prediction step
            MF[:,k] = A.dot(MF[:,k-1])
            PF[:,:,k] = A.dot(PF[:,:,k-1]).dot(A.T) + Q

            # Update step (only if there is data)
            if not np.isnan(Y[:,k]):
                 LL = linalg.cho_factor(H.dot(PF[:,:,k]).dot(H.T) + R)
                 K = linalg.cho_solve(LL, H.dot(PF[:,:,k].T)).T
                 MF[:,k] += K.dot(Y[:,k]-H.dot(MF[:,k]))
                 PF[:,:,k] -= K.dot(H).dot(PF[:,:,k])

        # Return values
        return (MF, PF)

    def rts_smoother(self,F,L,Qc,X,MS,PS):
        # RTS_SMOOTHER - Run the RTS smoother for a given model and data

        # Time step lengths
        dt = np.empty(X.shape)
        dt[:,0] = X[:,1]-X[:,0]
        dt[:,1:] = np.diff(X)

        # Solve the LTI SDE for these time steps
        As, Qs, index = self.lti_disc(F,L,Qc,dt)

        # Sequentially smooth states starting from the end
        for k in range(2,X.shape[1]+1):

            # Form discrete-time model
            #(A, Q) = self.lti_disc(F,L,Qc,dt[:,1-k])
            A = As[:,:,index[1-k]];
            Q = Qs[:,:,index[1-k]];

            # Smoothing step
            LL = linalg.cho_factor(A.dot(PS[:,:,-k]).dot(A.T)+Q)
            G = linalg.cho_solve(LL,A.dot(PS[:,:,-k])).T
            MS[:,-k] += G.dot(MS[:,1-k]-A.dot(MS[:,-k]))
            PS[:,:,-k] += G.dot(PS[:,:,1-k]-A.dot(PS[:,:,-k]).dot(A.T)-Q).dot(G.T)

        # Return
        return (MS, PS)

    def kf_likelihood(self,F,L,Qc,H,R,Pinf,X,Y):
        # Evaluate marginal likelihood

        # Initialize
        lik = 0
        m = np.zeros((F.shape[0],1))
        P = Pinf.copy()

        # Time step lengths
        dt = np.empty(X.shape)
        dt[:,0] = X[:,1]-X[:,0]
        dt[:,1:] = np.diff(X)

        # Solve the LTI SDE for these time steps
        As, Qs, index = self.lti_disc(F,L,Qc,dt)

        # Kalman filter for likelihood evaluation
        for k in range(0,Y.shape[1]):

            # Form discrete-time model
            #(A,Q) = self.lti_disc(F,L,Qc,dt[:,k])
            A = As[:,:,index[k]];
            Q = Qs[:,:,index[k]];

            # Prediction step
            m = A.dot(m)
            P = A.dot(P).dot(A.T) + Q

            # Update step only if there is data
            if not np.isnan(Y[:,k]):
                 v = Y[:,k]-H.dot(m)
                 LL, isupper = linalg.cho_factor(H.dot(P).dot(H.T) + R)
                 lik -= np.sum(np.log(np.diag(LL)))
                 lik -= 0.5*v.shape[0]*np.log(2*np.pi)
                 lik -= 0.5*linalg.cho_solve((LL, isupper),v).dot(v)
                 K = linalg.cho_solve((LL, isupper), H.dot(P.T)).T
                 m += K.dot(v)
                 P -= K.dot(H).dot(P)

        # Return likelihood
        return lik[0,0]

    def simulate(self,F,L,Qc,Pinf,X):
        # Simulate a trajectory using the state space model

        # Allocate space for results
        f = np.zeros((F.shape[0],X.shape[1]))

        # Initial state
        f[:,0:1] = np.linalg.cholesky(Pinf).dot(np.random.randn(F.shape[0],1))

        # Sweep through remaining time points
        for k in range(1,X.shape[1]):

            # Form discrete-time model
            (A,Q) = self.lti_disc(F,L,Qc,X[:,k]-X[:,k-1])

            # Draw the state
            f[:,k] = A.dot(f[:,k-1]).T + np.dot(np.linalg.cholesky(Q),np.random.randn(A.shape[0],1)).T

        # Return values
        return f

    def lti_disc(self,F,L,Qc,dt):
        # Discrete-time solution to the LTI SDE

        # Dimensionality
        n = F.shape[0]
        index = 0

        # Check for numbers of time steps
        if dt.flatten().shape[0]==1:

            # The covariance matrix by matrix fraction decomposition
            Phi = np.zeros((2*n,2*n))
            Phi[:n,:n] = F
            Phi[:n,n:] = L.dot(Qc).dot(L.T)
            Phi[n:,n:] = -F.T
            AB = linalg.expm(Phi*dt).dot(np.vstack((np.zeros((n,n)),np.eye(n))))
            Q = linalg.solve(AB[n:,:].T,AB[:n,:].T)

            # The dynamical model
            A  = linalg.expm(F*dt)

            # Return
            return A, Q

        # Optimize for cases where time steps occur repeatedly
        else:

            # Time discretizations (round to 14 decimals to avoid problems)
            dt, _, index = np.unique(np.round(dt,14),True,True)

            # Allocate space for A and Q
            A = np.empty((n,n,dt.shape[0]))
            Q = np.empty((n,n,dt.shape[0]))

            # Call this function for each dt
            for j in range(0,dt.shape[0]):
                A[:,:,j], Q[:,:,j] = self.lti_disc(F,L,Qc,dt[j])

            # Return
            return A, Q, index

