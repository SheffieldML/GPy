# Copyright (c) 2013,   Mu Niu,Arno Solin, Simo Sarkka.
# Licensed under the BSD 3-clause license (see LICENSE.txt) ??
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
# Input parameter :   
# X: temporal coordinate of data point     Y: spatio-temporal data   SXP: spatial coordinate grid
# SI: indicate the spatial coordinate of the data point from the spatial grid. 
#
# The spatial coordinate of data point do not change over time
# Kernel structure: separatable kernel   
# 
# Spatial kernel  : rbf
# Temporal kernel : state space of of Matern32
import numpy as np
from scipy import linalg
from ..core import Model
from .. import kern
from GPy.plotting.matplot_dep.models_plots import gpplot
from GPy.plotting.matplot_dep.base_plots import x_frame1D
from GPy.plotting.matplot_dep import Tango
import pylab as pb
from GPy.core.parameterization.param import Param

class StateSpace_xt(Model):
    def __init__(self,SXP,SI,X,Y, tempokernel=None,spacekernel=None,sigma2=1.0,name='StateSpace_xt'):
        super(StateSpace_xt, self).__init__(name=name)
        self.num_data, input_dim = X.shape
        assert input_dim==1, "State space methods for time and space  2"
        num_data_Y, self.output_dim = Y.shape
        assert num_data_Y == self.num_data, "X and Y data don't match"
        #assert self.output_dim == 1, "State space methods for single outputs only"

        # Make sure the observations are ordered in time
        sort_index = np.argsort(X[:,0])
        self.X = X[sort_index]
        self.Y = Y[sort_index]

        self.SXP = SXP
        self.SI = SI

        # Noise variance
        self.sigma2 = Param('Gaussian_noise', sigma2)
        self.link_parameter(self.sigma2)

        # Default kernel
        if tempokernel is None:
            self.tempokern = kern.Matern32(1,lengthscale=1)
        else:
            self.tempokern = tempokernel

        if spacekernel is None:
            #self.kern = kern.Matern32(1,lengthscale=1)
            #self.spacekern = kern.rbf(1,lengthscale=0.1)
            self.spacekern = kern.exponential(1,lengthscale=1)
            #self.spacekern = kern.Matern52(1,lengthscale=1)
        else:
            self.spacekern = spacekernel

        self.link_parameter(self.tempokern)
        self.link_parameter(self.spacekern)

        self.sigma2.constrain_positive()

        # Assert that the kernel is supported
        if not hasattr(self.tempokern, 'sde'):
            raise NotImplementedError('SDE must be implemented for the kernel being used')
        #assert self.kern.sde() not False, "This kernel is not supported for state space estimation"

    def parameters_changed(self):
        """
        Parameters have now changed
        """
        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dF,dQc,dPinf) = self.tempokern.sde()
        
        #X=self.X
        SX=self.SXP[self.SI]

        n=SX.shape[0]
        F1 = np.kron(np.eye(n),F)
        L1 = np.kron(np.eye(n),L)
        K1=self.spacekern.K(SX)
        Qc1 = K1*Qc               #kron(K,Qc1);
        H1 = np.kron(np.eye(n),H)
        Pinf1 = np.kron(K1,Pinf)

        # Use the Kalman filter to evaluate the likelihood
        self._log_marginal_likelihood = self.kf_likelihood(F1,L1,Qc1,H1,self.sigma2,Pinf1)#,self.X.T,self.Y.T)
        gradients  = self.compute_gradients()
        self.sigma2.gradient_full[:] = gradients[-1]     # the very last for noise
        #self.tempokern.gradient_full[:] = gradients[:-1]
        self.tempokern.gradient_full[:] = gradients[:2]# frist 2 for temporal grad
        self.spacekern.gradient_full[:] = gradients[2:-1]  # second 2 for spatiol grad

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def compute_gradients(self):
        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,dFt,dQct,dPinft) = self.tempokern.sde()

        #X=self.X
        SX=self.SXP[self.SI]

        n=SX.shape[0]
        F1 = np.kron(np.eye(n),F)
        L1 = np.kron(np.eye(n),L)
        K1=self.spacekern.K(SX)
        Qc1 = K1*Qc               #kron(K,Qc1);
        H1 = np.kron(np.eye(n),H)
        Pinf1 = np.kron(K1,Pinf)

        # Allocate space for the derivatives 
        dF1    = np.zeros([F1.shape[0],F1.shape[1],dFt.shape[2]+1+2])
        dQc1  = np.zeros([Qc1.shape[0],Qc1.shape[1],dQct.shape[2]+1+2]) 
        dPinf1 = np.zeros([Pinf1.shape[0],Pinf1.shape[1],dPinft.shape[2]+1+2])
        dK1_dl = self.spacekern.dK_dtheta(SX)
        dK1_ds = K1/self.spacekern.variance

        # Assign the values for the kernel function
        dF1[:,:,0] = np.kron(np.eye(n),dFt[:,:,0])
        dF1[:,:,1] = np.kron(np.eye(n),dFt[:,:,1])
        dQc1[:,:,0] = K1*dQct[:,:,0]
        dQc1[:,:,1] = K1*dQct[:,:,1]
        dPinf1[:,:,0] = np.kron(K1,dPinft[:,:,0])
        dPinf1[:,:,1] = np.kron(K1,dPinft[:,:,1])     

        dPinf1[:,:,2] = np.kron(dK1_dl,Pinf)   
        dPinf1[:,:,3] = np.kron(dK1_ds,Pinf)  
        dQc1[:,:,2] = dK1_dl*Qc
        dQc1[:,:,3] = dK1_ds*Qc  
        
        # The sigma2 derivative
        dR = np.zeros([1,1,dF1.shape[2]])#dR = np.zeros([1,1,dF1.shape[2]])
        dR[:,:,-1] = 1

        # Calculate the likelihood gradients
        gradients = self.kf_likelihood_g(F1,L1,Qc1,H1,self.sigma2,Pinf1,dF1,dQc1,dPinf1,dR)
    	return gradients


    def predict_raw(self, Xnew, filteronly=False):

        # Make a single matrix containing training and testing points
        #X = np.vstack((self.X, Xnew))
        #Y = np.vstack((self.Y, np.nan*np.zeros(Xnew.shape)))
        X=self.X        
        Y=self.Y
        SXP=self.SXP
        SI=self.SI

        # Sort the matrix (save the order)
        _, return_index, return_inverse = np.unique(X,True,True)
        X = X[return_index]
        Y = Y[return_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf,use1,use2,use3) = self.tempokern.sde()

        n=SXP.shape[0]
        F1 = np.kron(np.eye(n),F)
        L1 = np.kron(np.eye(n),L)
        K1=self.spacekern.K(SXP)
        Qc1 = K1*Qc               #kron(K,Qc1);
        H2 = np.zeros([len(SI),SXP.shape[0]])
        count = 0
        for index in SI:
            H2[count,index] = 1
            count = count+1
#        H1 = np.kron(np.eye(n),H)
        H1 = np.kron(H2,H)
        Pinf1 = np.kron(K1,Pinf)
                  
        
        # Run the Kalman filter
        #(M, P) = self.kalman_filter(F,L,Qc,H,self.sigma2,Pinf,X.T,Y.T)
        #(M, P) = self.kalman_filter(F1,L1,Qc1,H1,self.sigma2,Pinf1,X.T,Y)

        # indexed the data poin within grid
        NY = np.zeros([Y.shape[1],Xnew.shape[0]+X.shape[0]]) * np.nan
        NX = np.zeros([Xnew.shape[0] + X.shape[0],1])
        # Assume that Xmax is ordered !!!
        oi = 0
        ni = 0
        xni = 0
        for xni in range(Xnew.shape[0]):
            if oi < X.shape[0]:
                if (xni == 0 and X[oi] < Xnew[xni]) or (xni > 0 and X[oi] >= Xnew[xni-1] and X[oi] < Xnew[xni]):
                    NY[:,ni] = Y.T[:,oi]
                    NX[ni]   = X[oi]
                    ni = ni + 1
                    oi = oi + 1
            NX[ni] = Xnew[xni]
            ni = ni + 1
            count = count+1

        (M, P) = self.kalman_filter(F1,L1,Qc1,H1,self.sigma2,Pinf1,NX.T,NY)
        #stop
        # Run the Rauch-Tung-Striebel smoother
        #if not filter:
        #(M, P) = self.rts_smoother(F,L,Qc,X.T,M,P)
        (M, P) = self.rts_smoother(F1,L1,Qc1,NX.T,M,P)

        # Put the data back in the original order
        #M = M[:,return_inverse]  # Do not use with Xnew
        #P = P[:,:,return_inverse]

        # Only return the values for Xnew
        #M = M[:,self.num_data:]
        #P = P[:,:,self.num_data:]
        
        # Calculate the mean and variance
        #m = H.dot(M).T
        #m = H1.dot(M)

        n=SXP.shape[0]
        H3 = np.kron(np.eye(n),H)
        m = H3.dot(M)

        #V1 = np.tensordot(H[0],P,(0,0))
        #V2 = np.tensordot(V1,H[0],(0,0))

        #1st and 2nd dim, pick every 2nd elements
        V=P[::F.shape[0],::F.shape[0],:]  
        
        #V1 = np.tensordot(H1.T,P,(0,0))
        #V2 = np.tensordot(V1,H1,(1,1))
        #stop
        #V3 = V2[:,None]
     
        # Return the posterior of the state
        return (m, V)

    def predict(self, Xnew, filteronly=False):

        # Run the Kalman filter to get the state
        (m, V) = self.predict_raw(Xnew,filteronly=filteronly)

        # Add the noise variance to the state variance
        V += self.sigma2*np.eye(m.shape[0])
        #stop
        # Lower and upper bounds
        lower = m - 2*np.sqrt(V)
        upper = m + 2*np.sqrt(V)
        
        # Return mean and variance
        return (m, V, lower, upper)

    def plot(self, plot_limits=None, levels=20, samples=0, fignum=None,
            ax=None, resolution=None, plot_raw=False, plot_filter=False,
            linecol=Tango.colorsHex['darkBlue'],fillcol=Tango.colorsHex['lightBlue']):

        # Deal with optional parameters
        #if ax is None:
            #fig = pb.figure(num=fignum)
            #ax = fig.add_subplot(111)

        # Define the frame on which to plot
        resolution = resolution or 200
        Xgrid, xmin, xmax = x_frame1D(self.X, plot_limits=plot_limits)
        # T grid???
        #stop

        # Make a prediction on the frame and plot it
        if plot_raw:
            #m, v = self.predict_raw(Xgrid,filteronly=plot_filter)
            
            m, v = self.predict_raw(Xgrid,filteronly=plot_filter)
            
            Y = self.Y
            
            #allocate space for realisation
            reli = np.empty((Y.shape[0],Y.shape[1]))

            
            def forceAspect(ax,aspect=1):
                im = ax.get_images()
                extent =  im[0].get_extent()
                ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
            # mean
            fig = pb.figure(100)
            ax = fig.add_subplot(111)
            pb.imshow(m,interpolation="nearest")
            #pb.contour(m)
            forceAspect(ax,aspect=1)
            #pb.tight_layout()
            # data Y
            pb.figure(200)
            pb.imshow(Y,interpolation="nearest")

            #realisation
            #for i in range(0,Y.shape[1]):
            #    reli[:,i] = np.random.multivariate_normal(m[:,i],v[:,:,i])            
            #pb.figure(3)
            #pb.imshow(reli,interpolation="nearest")

            #for i in range(0,Y.shape[1]):
            #    reli[:,i] = np.random.multivariate_normal(m[:,i],v[:,:,i])
            #pb.figure(4)
            #pb.imshow(reli,interpolation="nearest")            

            
            #lower = m - 2*np.sqrt(v)
            #upper = m + 2*np.sqrt(v)
            
            


        else:
            m, v, lower, upper = self.predict(Xgrid,filteronly=plot_filter)
            Y = self.Y

        # Plot the values
        #gpplot(Xgrid, m, lower, upper, axes=ax, edgecol=linecol, fillcol=fillcol)
        #gpplot(self.X, m, lower, upper, axes=ax, edgecol=linecol, fillcol=fillcol)
        #ax.plot(self.X, self.Y, 'kx', mew=1.5)

        # Optionally plot some samples
        if samples:
            Ysim = self.posterior_samples(Xgrid, samples)
            for yi in Ysim.T:
                ax.plot(Xgrid, yi, Tango.colorsHex['darkBlue'], linewidth=0.25)

        # Set the limits of the plot to some sensible values
        #ymin, ymax = min(np.append(Y.flatten(), lower.flatten())), max(np.append(Y.flatten(), upper.flatten()))
        #ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
        #ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)

    def posterior_samples_f(self,X,size=10):

        # Reorder X values
        sort_index = np.argsort(X[:,0])
        X = X[sort_index]

        # Get the model matrices from the kernel
        (F,L,Qc,H,Pinf) = self.tempokern.sde()

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
            if not np.isnan(Y[0,k]):
                 if Y.shape[0]==1:
                     K = PF[:,:,k].dot(H.T)/(H.dot(PF[:,:,k]).dot(H.T) + R)
                 else:
#                     LL = linalg.cho_factor(H.dot(PF[:,:,k]).dot(H.T) + R*np.eye(Y.shape[0]))
#                     K = linalg.cho_solve(LL, H.dot(PF[:,:,k].T)).T
                     S = H.dot(PF[:,:,k]).dot(H.T) + R*np.eye(Y.shape[0])                
                     LL = linalg.cho_factor(S)
                     K = linalg.cho_solve(LL, H.dot(PF[:,:,k].T)).T                
                 
                 MF[:,k] += K.dot(Y[:,k]-H.dot(MF[:,k]))
#                 PF[:,:,k] -= K.dot(H).dot(PF[:,:,k])
                 PF[:,:,k] -= K.dot(S).dot(K.T)
                 PF[:,:,k] = 0.5 * (PF[:,:,k] + PF[:,:,k].T)
            
#            LL = linalg.cho_factor(H.dot(PF[:,:,k]).dot(H.T) + R*np.eye(Y.shape[1]))
#            K = linalg.cho_solve(LL, H.dot(PF[:,:,k].T)).T                
#            MF[:,k] += K.dot(Y[:,k]-H.dot(MF[:,k]))
#            PF[:,:,k] -= K.dot(H).dot(PF[:,:,k])
                 
           
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

        try:

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

        except linalg.LinAlgError:
            stop

        # Return
        return (MS, PS)

    def kf_likelihood(self,F,L,Qc,H,R,Pinf):
        # Evaluate marginal likelihood

        X=self.X.T
        Y=self.Y.T
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
            if not np.isnan(Y[0,k]):
                 if Y.shape[0]==1:
                     v = Y[:,k]-H.dot(m)
                     S = H.dot(P).dot(H.T) + R
                     K = P.dot(H.T)/S
                     lik -= 0.5*np.log(S)
                     lik -= 0.5*v.shape[0]*np.log(2*np.pi)
                     lik -= 0.5*(v*v/S)[0,0]  # !!!
                 else:
                     v = Y[:,k][None].T-H.dot(m)
                     S = H.dot(P).dot(H.T) + R*np.eye(Y.shape[0])
                     #Should be LL, isupper = ...                
                     #LL = linalg.cho_factor(S)
                     #K = linalg.cho_solve(LL, H.dot(P)).T                
                     LL, isupper = linalg.cho_factor(H.dot(P).dot(H.T) + R*np.eye(Y.shape[0]))
                     K = linalg.cho_solve((LL, isupper), H.dot(P)).T
                     lik -= np.sum(np.log(np.diag(LL)))
                     lik -= 0.5*v.shape[0]*np.log(2*np.pi)
                     lik -= 0.5*linalg.cho_solve((LL, isupper),v).T.dot(v)[0,0]
                 m += K.dot(v)
#                 P -= K.dot(H).dot(P)
                 P -= K.dot(S).dot(K.T)
                 P = 0.5 * (P + P.T)

            #stop
#            v = Y[:,k][None].T-H.dot(m)
#            LL, isupper = linalg.cho_factor(H.dot(P).dot(H.T) + R*np.eye(Y.shape[1]))
#            K = linalg.cho_solve((LL, isupper), H.dot(P)).T
#            lik -= np.sum(np.log(np.diag(LL)))
#            lik -= 0.5*v.shape[0]*np.log(2*np.pi)
#            lik -= 0.5*linalg.cho_solve((LL, isupper),v).T.dot(v)[0,0]
#            m += K.dot(v)
#            P -= K.dot(H).dot(P)
            
       
        # Return likelihood
        return lik

    def kf_likelihood_g(self,F,L,Qc,H,R,Pinf,dF,dQc,dPinf,dR):
        # Evaluate marginal likelihood gradient

        # State dimension, number of data points and number of parameters
        Y=self.Y.T
        X=self.X.T
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
            
            #stop

            # Set predicted m and P
            m = mm
            P = PP

            # Start the Kalman filter update step and precalculate variables
            #S = H.dot(P).dot(H.T) + R
            S = H.dot(P).dot(H.T) + R*np.eye(Y.shape[0])
            LL, isupper = linalg.cho_factor(H.dot(P).dot(H.T) + R*np.eye(Y.shape[0]))
            v = Y[:,k][None].T-H.dot(m)
            K = linalg.cho_solve((LL, isupper), H.dot(P)).T
            
            Vst = linalg.cho_solve((LL, isupper),v)

            # We should calculate the Cholesky factor if S is a matrix
            # [LS,notposdef] = chol(S,'lower');

            # The Kalman filter update (S is scalar)
            #iS   = 1/S
            #HtiS = H.T.dot(iS)
            #K    = P.dot(HtiS)
            #v    = Y[:,k]-H.dot(m)
            #vtiS = v.T.dot(iS)

            # Loop through all parameters (Kalman filter update step derivative)
            for j in range(0,nparam):

                # Innovation covariance derivative
                dS = H.dot(dP[:,:,j]).dot(H.T) + dR[:,:,j]*np.eye(Y.shape[0])
                # s^(-1)*ds
                iSd= linalg.cho_solve((LL, isupper),dS)
                # Evaluate the energy derivative for j
                eg[j] = eg[j]                           \
                    - .5*np.sum(np.diag(iSd))                  \
                    + .5*(H.dot(dm[:,j:j+1])).T.dot(Vst) \
                    + .5*Vst.T.dot(dS).dot(Vst)       \
                    + .5*Vst.T.dot(H.dot(dm[:,j:j+1]))

                # Kalman filter update step derivatives
                #dK          = dP[:,:,j].dot(HtiS) - P.dot(HtiS).dot(dS)/S
                dK          = dP[:,:,j].dot(linalg.cho_solve((LL, isupper),H).T) - P.dot(linalg.cho_solve((LL, isupper),H).T).dot(dS).dot(linalg.cho_solve((LL, isupper),np.eye(Y.shape[0])))
                dm[:,j:j+1] = dm[:,j:j+1] + dK.dot(v) - K.dot(H).dot(dm[:,j:j+1])
                dKSKt       = dK.dot(S).dot(K.T)
                dP[:,:,j]   = dP[:,:,j] - dKSKt - K.dot(dS).dot(K.T) - dKSKt.T

            # Evaluate the energy
            #e = e - .5*S.shape[0]*np.log(2*np.pi) - np.sum(np.log(np.sqrt(S))) - .5*vtiS.dot(v)

            # Finish Kalman filter update step
            m = m + K.dot(v)
            P = P - K.dot(S).dot(K.T)

            # Make sure the covariances stay symmetric
            P  = (P+P.T)/2
            dP = (dP + dP.transpose([1,0,2]))/2


        #stop
    	return eg

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



