# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from ..util.plot import gpplot

class likelihood:
    def __init__(self,Y):
        """
        Likelihood class for doing Expectation propagation

        :param Y: observed output (Nx1 numpy.darray)
        ..Note:: Y values allowed depend on the likelihood used
        """
        self.Y = Y
        self.N = self.Y.shape[0]

    def plot1Da(self,X_new,Mean_new,Var_new,X_u,Mean_u,Var_u):
        """
        Plot the predictive distribution of the GP model for 1-dimensional inputs

        :param X_new: The points at which to make a prediction
        :param Mean_new: mean values at X_new
        :param Var_new: variance values at X_new
        :param X_u: input (inducing)  points used to train the model
        :param Mean_u: mean values at X_u
        :param Var_new: variance values at X_u
        """
        assert X_new.shape[1] == 1, 'Number of dimensions must be 1'
        gpplot(X_new,Mean_new,Var_new)
        pb.errorbar(X_u,Mean_u,2*np.sqrt(Var_u),fmt='r+')
        pb.plot(X_u,Mean_u,'ro')

    def plot2D(self,X,X_new,F_new,U=None):
        """
        Predictive distribution of the fitted GP model for 2-dimensional inputs

        :param X_new: The points at which to make a prediction
        :param Mean_new: mean values at X_new
        :param Var_new: variance values at X_new
        :param X_u: input points used to train the model
        :param Mean_u: mean values at X_u
        :param Var_new: variance values at X_u
        """
        N,D = X_new.shape
        assert D == 2, 'Number of dimensions must be 2'
        n = np.sqrt(N)
        x1min = X_new[:,0].min()
        x1max = X_new[:,0].max()
        x2min = X_new[:,1].min()
        x2max = X_new[:,1].max()
        pb.imshow(F_new.reshape(n,n),extent=(x1min,x1max,x2max,x2min),vmin=0,vmax=1)
        pb.colorbar()
        C1 = np.arange(self.N)[self.Y.flatten()==1]
        C2 = np.arange(self.N)[self.Y.flatten()==-1]
        [pb.plot(X[i,0],X[i,1],'ro') for i in C1]
        [pb.plot(X[i,0],X[i,1],'bo') for i in C2]
        pb.xlim(x1min,x1max)
        pb.ylim(x2min,x2max)
        if U is not None:
            [pb.plot(a,b,'wo') for a,b in U]

class probit(likelihood):
    """
    Probit likelihood
    Y is expected to take values in {-1,1}
    -----
    $$
    L(x) = \\Phi (Y_i*f_i)
    $$
    """
    def moments_match(self,i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        z = self.Y[i]*v_i/np.sqrt(tau_i**2 + tau_i)
        Z_hat = stats.norm.cdf(z)
        phi = stats.norm.pdf(z)
        mu_hat = v_i/tau_i + self.Y[i]*phi/(Z_hat*np.sqrt(tau_i**2 + tau_i))
        sigma2_hat = 1./tau_i - (phi/((tau_i**2+tau_i)*Z_hat))*(z+phi/Z_hat)
        return Z_hat, mu_hat, sigma2_hat

    def plot1Db(self,X,X_new,F_new,U=None):
        assert X.shape[1] == 1, 'Number of dimensions must be 1'
        gpplot(X_new,F_new,np.zeros(X_new.shape[0]))
        pb.plot(X,(self.Y+1)/2,'kx',mew=1.5)
        pb.ylim(-0.2,1.2)
        if U is not None:
            pb.plot(U,U*0+.5,'r|',mew=1.5,markersize=12)

    def predictive_mean(self,mu,variance):
        return stats.norm.cdf(mu/np.sqrt(1+variance))

    def _log_likelihood_gradients():
        raise NotImplementedError

