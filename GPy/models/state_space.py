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
        self.sigma2 = Param('Gaussian_noise', sigma2)
        self.link_parameter(self.sigma2)

        # Default kernel
        if kernel is None:
            self.kern = kern.Matern32(1)
        else:
            self.kern = kernel
        self.link_parameter(self.kern)

        self.sigma2.constrain_positive()

        # Assert that the kernel is supported
        if not hasattr(self.kern, 'sde'):
            raise NotImplementedError('SDE must be implemented for the kernel being used')
        #assert self.kern.sde() not False, "This kernel is not supported for state space estimation"

    def parameters_changed(self):
        """
        Parameters have now changed
        """
        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.kern.sde()

        # Use the Kalman filter to evaluate the likelihood
        self._log_marginal_likelihood = self.kf_likelihood(F,L,Qc,H,self.sigma2,Pinf,self.X.T,self.Y.T)
        gradients  = self.compute_gradients()
        self.sigma2.gradient_full[:] = gradients[-1]
        self.kern.gradient_full[:] = gradients[:-1]

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def compute_gradients(self):
        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dFt,dQct,dPinft) = self.kern.sde()

        # Allocate space for the full partial derivative matrices
        dF    = np.zeros([dFt.shape[0],dFt.shape[1],dFt.shape[2]+1])
        dQc   = np.zeros([dQct.shape[0],dQct.shape[1],dQct.shape[2]+1])
        dPinf = np.zeros([dPinft.shape[0],dPinft.shape[1],dPinft.shape[2]+1])

        # Assign the values for the kernel function
        dF[:,:,:-1] = dFt
        dQc[:,:,:-1] = dQct
        dPinf[:,:,:-1] = dPinft

        # The sigma2 derivative
        dR = np.zeros([1,1,dF.shape[2]])
        dR[:,:,-1] = 1

        # Calculate the likelihood gradients
        gradients = self.kf_likelihood_g(F,L,Qc,H,self.sigma2,Pinf,dF,dQc,dPinf,dR,self.X.T,self.Y.T)
        return gradients

    def predict_raw(self, Xnew, Ynew=None, filteronly=False):

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
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.kern.sde()

        # Run the Kalman filter
        (M, P) = self.kalman_filter(F,L,Qc,H,self.sigma2,Pinf,X.T,Y.T)

        # Run the Rauch-Tung-Striebel smoother
        if not filteronly:
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

    def predict(self, Xnew, filteronly=False):

        # Run the Kalman filter to get the state
        (m, V) = self.predict_raw(Xnew,filteronly=filteronly)

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
                 if Y.shape[0]==1:
                     K = PF[:,:,k].dot(H.T)/(H.dot(PF[:,:,k]).dot(H.T) + R)
                 else:
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
                 if Y.shape[0]==1:
                     S = H.dot(P).dot(H.T) + R
                     K = P.dot(H.T)/S
                     lik -= 0.5*np.log(S)
                     lik -= 0.5*v.shape[0]*np.log(2*np.pi)
                     lik -= 0.5*v*v/S
                 else:
                     LL, isupper = linalg.cho_factor(H.dot(P).dot(H.T) + R)
                     lik -= np.sum(np.log(np.diag(LL)))
                     lik -= 0.5*v.shape[0]*np.log(2*np.pi)
                     lik -= 0.5*linalg.cho_solve((LL, isupper),v).dot(v)
                     K = linalg.cho_solve((LL, isupper), H.dot(P.T)).T
                 m += K.dot(v)
                 P -= K.dot(H).dot(P)

        # Return likelihood
        return lik[0,0]

    def kf_likelihood_g(self,F,L,Qc,H,R,Pinf,dF,dQc,dPinf,dR,X,Y):
        # Evaluate marginal likelihood gradient

        # State dimension, number of data points and number of parameters
        n      = F.shape[0]
        steps  = Y.shape[1]
        nparam = dF.shape[2]

        # Time steps
        t = X.squeeze()
  
        # Allocate space
        e  = 0
        eg = np.zeros(nparam)
  
        # Set up
        m  = np.zeros([n,1])
        P  = Pinf.copy()
        dm = np.zeros([n,nparam])
        dP = dPinf.copy()
        mm = m.copy()
        PP = P.copy()

        # Initial dt
        dt = -np.Inf

        # Allocate space for expm results
        AA  = np.zeros([2*n, 2*n, nparam])
        FF  = np.zeros([2*n, 2*n])

        # Loop over all observations
        for k in range(0,steps):

            # The previous time step
            dt_old = dt;

            # The time discretization step length
            if k>0:
                dt = t[k]-t[k-1]
            else:
                dt = 0

            # Loop through all parameters (Kalman filter prediction step)
            for j in range(0,nparam):

                # Should we recalculate the matrix exponential?
                if abs(dt-dt_old) > 1e-9:

                    # The first matrix for the matrix factor decomposition
                    FF[:n,:n] = F
                    FF[n:,:n] = dF[:,:,j]
                    FF[n:,n:] = F

                    # Solve the matrix exponential
                    AA[:,:,j] = linalg.expm3(FF*dt)

                # Solve the differential equation
                foo         = AA[:,:,j].dot(np.vstack([m, dm[:,j:j+1]]))
                mm          = foo[:n,:]
                dm[:,j:j+1] = foo[n:,:]

                # The discrete-time dynamical model
                if j==0:
                    A  = AA[:n,:n,j]
                    Q  = Pinf - A.dot(Pinf).dot(A.T)
                    PP = A.dot(P).dot(A.T) + Q

                # The derivatives of A and Q
                dA = AA[n:,:n,j]
                dQ = dPinf[:,:,j] - dA.dot(Pinf).dot(A.T) \
                   - A.dot(dPinf[:,:,j]).dot(A.T) - A.dot(Pinf).dot(dA.T)

                # The derivatives of P
                dP[:,:,j] = dA.dot(P).dot(A.T) + A.dot(dP[:,:,j]).dot(A.T) \
                   + A.dot(P).dot(dA.T) + dQ

            # Set predicted m and P
            m = mm
            P = PP

            # Start the Kalman filter update step and precalculate variables
            S = H.dot(P).dot(H.T) + R

            # We should calculate the Cholesky factor if S is a matrix
            # [LS,notposdef] = chol(S,'lower');

            # The Kalman filter update (S is scalar)
            HtiS = H.T/S
            iS   = 1/S
            K    = P.dot(HtiS)
            v    = Y[:,k]-H.dot(m)
            vtiS = v.T/S

            # Loop through all parameters (Kalman filter update step derivative)
            for j in range(0,nparam):

                # Innovation covariance derivative
                dS = H.dot(dP[:,:,j]).dot(H.T) + dR[:,:,j];

                # Evaluate the energy derivative for j
                eg[j] = eg[j]                           \
                    - .5*np.sum(iS*dS)                  \
                    + .5*H.dot(dm[:,j:j+1]).dot(vtiS.T) \
                    + .5*vtiS.dot(dS).dot(vtiS.T)       \
                    + .5*vtiS.dot(H.dot(dm[:,j:j+1]))

                # Kalman filter update step derivatives
                dK          = dP[:,:,j].dot(HtiS) - P.dot(HtiS).dot(dS)/S
                dm[:,j:j+1] = dm[:,j:j+1] + dK.dot(v) - K.dot(H).dot(dm[:,j:j+1])
                dKSKt       = dK.dot(S).dot(K.T)
                dP[:,:,j]   = dP[:,:,j] - dKSKt - K.dot(dS).dot(K.T) - dKSKt.T

            # Evaluate the energy
            # e = e - .5*S.shape[0]*np.log(2*np.pi) - np.sum(np.log(np.diag(LS))) - .5*vtiS.dot(v);
            e = e - .5*S.shape[0]*np.log(2*np.pi) - np.sum(np.log(np.sqrt(S))) - .5*vtiS.dot(v)

            # Finish Kalman filter update step
            m = m + K.dot(v)
            P = P - K.dot(S).dot(K.T)

            # Make sure the covariances stay symmetric
            P  = (P+P.T)/2
            dP = (dP + dP.transpose([1,0,2]))/2

            # raise NameError('Debug me')

        # Return the gradient
        return eg

    def kf_likelihood_g_notstable(self,F,L,Qc,H,R,Pinf,dF,dQc,dPinf,dR,X,Y):
        # Evaluate marginal likelihood gradient

        # State dimension, number of data points and number of parameters
        steps  = Y.shape[1]
        nparam = dF.shape[2]
        n      = F.shape[0]

        # Time steps
        t = X.squeeze()

        # Allocate space
        e  = 0
        eg = np.zeros(nparam)

        # Set up
        Z  = np.zeros(F.shape)
        QC = L.dot(Qc).dot(L.T)
        m  = np.zeros([n,1])
        P  = Pinf.copy()
        dm = np.zeros([n,nparam])
        dP = dPinf.copy()
        mm = m.copy()
        PP = P.copy()

        # % Initial dt
        dt = -np.Inf

        # Allocate space for expm results
        AA  = np.zeros([2*F.shape[0], 2*F.shape[0], nparam])
        AAA = np.zeros([4*F.shape[0], 4*F.shape[0], nparam])
        FF  = np.zeros([2*F.shape[0], 2*F.shape[0]])
        FFF = np.zeros([4*F.shape[0], 4*F.shape[0]])

        # Loop over all observations
        for k in range(0,steps):

            # The previous time step
            dt_old = dt;

            # The time discretization step length
            if k>0:
                dt = t[k]-t[k-1]
            else:
                dt = t[1]-t[0]

            # Loop through all parameters (Kalman filter prediction step)
            for j in range(0,nparam):

                # Should we recalculate the matrix exponential?
                if abs(dt-dt_old) > 1e-9:

                    # The first matrix for the matrix factor decomposition
                    FF[:n,:n] = F
                    FF[n:,:n] = dF[:,:,j]
                    FF[n:,n:] = F

                    # Solve the matrix exponential
                    AA[:,:,j] = linalg.expm3(FF*dt)

                # Solve using matrix fraction decomposition
                foo = AA[:,:,j].dot(np.vstack([m, dm[:,j:j+1]]))

                # Pick the parts
                mm          = foo[:n,:]
                dm[:,j:j+1] = foo[n:,:]

                # Should we recalculate the matrix exponential?
                if abs(dt-dt_old) > 1e-9:

                    # Define W and G
                    W = L.dot(dQc[:,:,j]).dot(L.T)
                    G = dF[:,:,j];

                    # The second matrix for the matrix factor decomposition
                    FFF[:n,:n]         =  F
                    FFF[2*n:-n,:n]     =  G
                    FFF[:n,    n:2*n]  =  QC
                    FFF[n:2*n, n:2*n]  = -F.T
                    FFF[2*n:-n,n:2*n]  =  W
                    FFF[-n:,   n:2*n]  = -G.T
                    FFF[2*n:-n,2*n:-n] =  F
                    FFF[2*n:-n,-n:]    =  QC
                    FFF[-n:,-n:]       = -F.T

                    # Solve the matrix exponential
                    AAA[:,:,j] = linalg.expm3(FFF*dt)

                # Solve using matrix fraction decomposition
                foo = AAA[:,:,j].dot(np.vstack([P, np.eye(n), dP[:,:,j], np.zeros([n,n])]))

                # Pick the parts
                C  = foo[:n,    :]
                D  = foo[n:2*n, :]
                dC = foo[2*n:-n,:]
                dD = foo[-n:,   :]

                # The prediction step covariance (PP = C/D)
                if j==0:
                    PP = linalg.solve(D.T,C.T).T
                    PP = (PP + PP.T)/2

                # Sove dP for j (C/D == P_{k|k-1})
                dP[:,:,j] = linalg.solve(D.T,(dC - PP.dot(dD)).T).T

            # Set predicted m and P
            m = mm
            P = PP

            # Start the Kalman filter update step and precalculate variables
            S = H.dot(P).dot(H.T) + R

            # We should calculate the Cholesky factor if S is a matrix
            # [LS,notposdef] = chol(S,'lower');

            # The Kalman filter update (S is scalar)
            HtiS = H.T/S
            iS   = 1/S
            K    = P.dot(HtiS)
            v    = Y[:,k]-H.dot(m)
            vtiS = v.T/S

            # Loop through all parameters (Kalman filter update step derivative)
            for j in range(0,nparam):

                # Innovation covariance derivative
                dS = H.dot(dP[:,:,j]).dot(H.T) + dR[:,:,j];

                # Evaluate the energy derivative for j
                eg[j] = eg[j]                           \
                    - .5*np.sum(iS*dS)                  \
                    + .5*H.dot(dm[:,j:j+1]).dot(vtiS.T) \
                    + .5*vtiS.dot(dS).dot(vtiS.T)       \
                    + .5*vtiS.dot(H.dot(dm[:,j:j+1]))

                # Kalman filter update step derivatives
                dK          = dP[:,:,j].dot(HtiS) - P.dot(HtiS).dot(dS)/S
                dm[:,j:j+1] = dm[:,j:j+1] + dK.dot(v) - K.dot(H).dot(dm[:,j:j+1])
                dKSKt       = dK.dot(S).dot(K.T)
                dP[:,:,j]   = dP[:,:,j] - dKSKt - K.dot(dS).dot(K.T) - dKSKt.T

            # Evaluate the energy
            # e = e - .5*S.shape[0]*np.log(2*np.pi) - np.sum(np.log(np.diag(LS))) - .5*vtiS.dot(v);
            e = e - .5*S.shape[0]*np.log(2*np.pi) - np.sum(np.log(np.sqrt(S))) - .5*vtiS.dot(v)

            # Finish Kalman filter update step
            m = m + K.dot(v)
            P = P - K.dot(S).dot(K.T)

            # Make sure the covariances stay symmetric
            P  = (P+P.T)/2
            dP = (dP + dP.transpose([1,0,2]))/2

            # raise NameError('Debug me')

        # Report
        #print e
        #print eg

        # Return the gradient
        return eg

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
        As, Qs, index = self.lti_disc(F,L,Qc,dt)

        # Sweep through remaining time points
        for k in range(1,X.shape[1]):

            # Form discrete-time model
            A = As[:,:,index[1-k]]
            Q = Qs[:,:,index[1-k]]

            # Draw the state
            f[:,:,k] = A.dot(f[:,:,k-1]) + np.dot(np.linalg.cholesky(Q),np.random.randn(A.shape[0],size))

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

