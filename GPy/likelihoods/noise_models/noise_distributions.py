# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
import pylab as pb
from GPy.util.plot import gpplot
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations


class NoiseDistribution(object):
    """
    Likelihood class for doing Expectation propagation

    :param Y: observed output (Nx1 numpy.darray)

    .. note:: Y values allowed depend on the LikelihoodFunction used
    """
    def __init__(self,gp_link,analytical_mean=False,analytical_variance=False):
        assert isinstance(gp_link,gp_transformations.GPTransformation), "gp_link is not a valid GPTransformation."
        self.gp_link = gp_link
        self.analytical_mean = analytical_mean
        self.analytical_variance = analytical_variance
        if self.analytical_mean:
            self.moments_match = self._moments_match_analytical
            self.predictive_mean = self._predictive_mean_analytical
        else:
            self.moments_match = self._moments_match_numerical
            self.predictive_mean = self._predictive_mean_numerical
        if self.analytical_variance:
            self.predictive_variance = self._predictive_variance_analytical
        else:
            self.predictive_variance = self._predictive_variance_numerical

    def _get_params(self):
        return np.zeros(0)

    def _get_param_names(self):
        return []

    def _set_params(self,p):
        pass

    def _gradients(self,partial):
        return np.zeros(0)

    def _preprocess_values(self,Y):
        """
        In case it is needed, this function assess the output values or makes any pertinent transformation on them.

        :param Y: observed output
        :type Y: Nx1 numpy.darray

        """
        return Y

    def _product(self,gp,obs,mu,sigma):
        """
        Product between the cavity distribution and a likelihood factor.

        :param gp: latent variable
        :param obs: observed output
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return stats.norm.pdf(gp,loc=mu,scale=sigma) * self._mass(gp,obs)

    def _nlog_product_scaled(self,gp,obs,mu,sigma):
        """
        Negative log-product between the cavity distribution and a likelihood factor.

        .. note:: The constant term in the Gaussian distribution is ignored.

        :param gp: latent variable
        :param obs: observed output
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return .5*((gp-mu)/sigma)**2 + self._nlog_mass(gp,obs)

    def _dnlog_product_dgp(self,gp,obs,mu,sigma):
        """
        Derivative wrt latent variable of the log-product between the cavity distribution and a likelihood factor.

        :param gp: latent variable
        :param obs: observed output
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return (gp - mu)/sigma**2 + self._dnlog_mass_dgp(gp,obs)

    def _d2nlog_product_dgp2(self,gp,obs,mu,sigma):
        """
        Second derivative wrt latent variable of the log-product between the cavity distribution and a likelihood factor.

        :param gp: latent variable
        :param obs: observed output
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return 1./sigma**2 + self._d2nlog_mass_dgp2(gp,obs)

    def _product_mode(self,obs,mu,sigma):
        """
        Newton's CG method to find the mode in _product (cavity x likelihood factor).

        :param obs: observed output
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return sp.optimize.fmin_ncg(self._nlog_product_scaled,x0=mu,fprime=self._dnlog_product_dgp,fhess=self._d2nlog_product_dgp2,args=(obs,mu,sigma),disp=False)

    def _moments_match_analytical(self,obs,tau,v):
        """
        If available, this function computes the moments analytically.
        """
        pass

    def _moments_match_numerical(self,obs,tau,v):
        """
        Lapace approximation to calculate the moments.

        :param obs: observed output
        :param tau: cavity distribution 1st natural parameter (precision)
        :param v: cavity distribution 2nd natural paramenter (mu*precision)

        """
        mu = v/tau
        mu_hat = self._product_mode(obs,mu,np.sqrt(1./tau))
        sigma2_hat = 1./(tau + self._d2nlog_mass_dgp2(mu_hat,obs))
        Z_hat = np.exp(-.5*tau*(mu_hat-mu)**2) * self._mass(mu_hat,obs)*np.sqrt(tau*sigma2_hat)
        return Z_hat,mu_hat,sigma2_hat

    def _nlog_conditional_mean_scaled(self,gp,mu,sigma):
        """
        Negative logarithm of the l.v.'s predictive distribution times the output's mean given the l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        .. note:: This function helps computing E(Y_star) = E(E(Y_star|f_star))

        """
        return .5*((gp - mu)/sigma)**2 - np.log(self._mean(gp))

    def _dnlog_conditional_mean_dgp(self,gp,mu,sigma):
        """
        Derivative of _nlog_conditional_mean_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return (gp - mu)/sigma**2 - self._dmean_dgp(gp)/self._mean(gp)

    def _d2nlog_conditional_mean_dgp2(self,gp,mu,sigma):
        """
        Second derivative of _nlog_conditional_mean_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return 1./sigma**2 - self._d2mean_dgp2(gp)/self._mean(gp) + (self._dmean_dgp(gp)/self._mean(gp))**2

    def _nlog_exp_conditional_variance_scaled(self,gp,mu,sigma):
        """
        Negative logarithm of the l.v.'s predictive distribution times the output's variance given the l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        .. note:: This function helps computing E(V(Y_star|f_star))

        """
        return .5*((gp - mu)/sigma)**2 - np.log(self._variance(gp))

    def _dnlog_exp_conditional_variance_dgp(self,gp,mu,sigma):
        """
        Derivative of _nlog_exp_conditional_variance_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return (gp - mu)/sigma**2 - self._dvariance_dgp(gp)/self._variance(gp)

    def _d2nlog_exp_conditional_variance_dgp2(self,gp,mu,sigma):
        """
        Second derivative of _nlog_exp_conditional_variance_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return 1./sigma**2 - self._d2variance_dgp2(gp)/self._variance(gp) + (self._dvariance_dgp(gp)/self._variance(gp))**2

    def _nlog_exp_conditional_mean_sq_scaled(self,gp,mu,sigma):
        """
        Negative logarithm of the l.v.'s predictive distribution times the output's mean squared given the l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        .. note:: This function helps computing E( E(Y_star|f_star)**2 )

        """
        return .5*((gp - mu)/sigma)**2 - 2*np.log(self._mean(gp))

    def _dnlog_exp_conditional_mean_sq_dgp(self,gp,mu,sigma):
        """
        Derivative of _nlog_exp_conditional_mean_sq_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return (gp - mu)/sigma**2 - 2*self._dmean_dgp(gp)/self._mean(gp)

    def _d2nlog_exp_conditional_mean_sq_dgp2(self,gp,mu,sigma):
        """
        Second derivative of _nlog_exp_conditional_mean_sq_scaled wrt. l.v.

        :param gp: latent variable
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        return 1./sigma**2 - 2*( self._d2mean_dgp2(gp)/self._mean(gp) - (self._dmean_dgp(gp)/self._mean(gp))**2 )

    def _predictive_mean_analytical(self,mu,sigma):
        """
        If available, this function computes the predictive mean analytically.
        """
        pass

    def _predictive_variance_analytical(self,mu,sigma):
        """
        If available, this function computes the predictive variance analytically.
        """
        pass

    def _predictive_mean_numerical(self,mu,sigma):
        """
        Laplace approximation to the predictive mean: E(Y_star) = E( E(Y_star|f_star) )

        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        maximum = sp.optimize.fmin_ncg(self._nlog_conditional_mean_scaled,x0=self._mean(mu),fprime=self._dnlog_conditional_mean_dgp,fhess=self._d2nlog_conditional_mean_dgp2,args=(mu,sigma),disp=False)
        mean = np.exp(-self._nlog_conditional_mean_scaled(maximum,mu,sigma))/(np.sqrt(self._d2nlog_conditional_mean_dgp2(maximum,mu,sigma))*sigma)
        """

        pb.figure()
        x = np.array([mu + step*sigma for step in np.linspace(-7,7,100)])
        f = np.array([np.exp(-self._nlog_conditional_mean_scaled(xi,mu,sigma))/np.sqrt(2*np.pi*sigma**2) for xi in x])
        pb.plot(x,f,'b-')
        sigma2 = 1./self._d2nlog_conditional_mean_dgp2(maximum,mu,sigma)
        f2 = np.exp(-.5*(x-maximum)**2/sigma2)/np.sqrt(2*np.pi*sigma2)
        k = np.exp(-self._nlog_conditional_mean_scaled(maximum,mu,sigma))*np.sqrt(sigma2)/np.sqrt(sigma**2)
        pb.plot(x,f2*mean,'r-')
        pb.vlines(maximum,0,f.max())
        """
        return mean

    def _predictive_mean_sq(self,mu,sigma):
        """
        Laplace approximation to the predictive mean squared: E(Y_star**2) = E( E(Y_star|f_star)**2 )

        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation

        """
        maximum = sp.optimize.fmin_ncg(self._nlog_exp_conditional_mean_sq_scaled,x0=self._mean(mu),fprime=self._dnlog_exp_conditional_mean_sq_dgp,fhess=self._d2nlog_exp_conditional_mean_sq_dgp2,args=(mu,sigma),disp=False)
        mean_squared = np.exp(-self._nlog_exp_conditional_mean_sq_scaled(maximum,mu,sigma))/(np.sqrt(self._d2nlog_exp_conditional_mean_sq_dgp2(maximum,mu,sigma))*sigma)
        return mean_squared

    def _predictive_variance_numerical(self,mu,sigma,predictive_mean=None):
        """
        Laplace approximation to the predictive variance: V(Y_star) = E( V(Y_star|f_star) ) + V( E(Y_star|f_star) )

        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation
        :predictive_mean: output's predictive mean, if None _predictive_mean function will be called.

        """
        # E( V(Y_star|f_star) )
        maximum = sp.optimize.fmin_ncg(self._nlog_exp_conditional_variance_scaled,x0=self._variance(mu),fprime=self._dnlog_exp_conditional_variance_dgp,fhess=self._d2nlog_exp_conditional_variance_dgp2,args=(mu,sigma),disp=False)
        exp_var = np.exp(-self._nlog_exp_conditional_variance_scaled(maximum,mu,sigma))/(np.sqrt(self._d2nlog_exp_conditional_variance_dgp2(maximum,mu,sigma))*sigma)

        """
        pb.figure()
        x = np.array([mu + step*sigma for step in np.linspace(-7,7,100)])
        f = np.array([np.exp(-self._nlog_exp_conditional_variance_scaled(xi,mu,sigma))/np.sqrt(2*np.pi*sigma**2) for xi in x])
        pb.plot(x,f,'b-')
        sigma2 = 1./self._d2nlog_exp_conditional_variance_dgp2(maximum,mu,sigma)
        f2 = np.exp(-.5*(x-maximum)**2/sigma2)/np.sqrt(2*np.pi*sigma2)
        k = np.exp(-self._nlog_exp_conditional_variance_scaled(maximum,mu,sigma))*np.sqrt(sigma2)/np.sqrt(sigma**2)
        pb.plot(x,f2*exp_var,'r--')
        pb.vlines(maximum,0,f.max())
        """

        #V( E(Y_star|f_star) ) =  E( E(Y_star|f_star)**2 ) - E( E(Y_star|f_star)**2 )
        exp_exp2 = self._predictive_mean_sq(mu,sigma)
        if predictive_mean is None:
            predictive_mean = self.predictive_mean(mu,sigma)
        var_exp = exp_exp2 - predictive_mean**2
        return exp_var + var_exp

    def _predictive_percentiles(self,p,mu,sigma):
        """
        Percentiles of the predictive distribution

        :parm p: lower tail probability
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation
        :predictive_mean: output's predictive mean, if None _predictive_mean function will be called.

        """
        qf = stats.norm.ppf(p,mu,sigma)
        return self.gp_link.transf(qf)

    def _nlog_joint_predictive_scaled(self,x,mu,sigma):
        """
        Negative logarithm of the joint predictive distribution (latent variable and output).

        :param x: tuple (latent variable,output)
        :param mu: latent variable's predictive mean
        :param sigma: latent variable's predictive standard deviation

        """
        return self._nlog_product_scaled(x[0],x[1],mu,sigma)

    def _gradient_nlog_joint_predictive(self,x,mu,sigma):
        """
        Gradient of _nlog_joint_predictive_scaled.

        :param x: tuple (latent variable,output)
        :param mu: latent variable's predictive mean
        :param sigma: latent variable's predictive standard deviation

        .. note: Only available when the output is continuous

        """
        assert not self.discrete, "Gradient not available for discrete outputs."
        return np.array((self._dnlog_product_dgp(gp=x[0],obs=x[1],mu=mu,sigma=sigma),self._dnlog_mass_dobs(obs=x[1],gp=x[0])))

    def _hessian_nlog_joint_predictive(self,x,mu,sigma):
        """
        Hessian of _nlog_joint_predictive_scaled.

        :param x: tuple (latent variable,output)
        :param mu: latent variable's predictive mean
        :param sigma: latent variable's predictive standard deviation

        .. note: Only available when the output is continuous

        """
        assert not self.discrete, "Hessian not available for discrete outputs."
        cross_derivative = self._d2nlog_mass_dcross(gp=x[0],obs=x[1])
        return np.array((self._d2nlog_product_dgp2(gp=x[0],obs=x[1],mu=mu,sigma=sigma),cross_derivative,cross_derivative,self._d2nlog_mass_dobs2(obs=x[1],gp=x[0]))).reshape(2,2)

    def _joint_predictive_mode(self,mu,sigma):
        """
        Negative logarithm of the joint predictive distribution (latent variable and output).

        :param x: tuple (latent variable,output)
        :param mu: latent variable's predictive mean
        :param sigma: latent variable's predictive standard deviation

        """
        return sp.optimize.fmin_ncg(self._nlog_joint_predictive_scaled,x0=(mu,self.gp_link.transf(mu)),fprime=self._gradient_nlog_joint_predictive,fhess=self._hessian_nlog_joint_predictive,args=(mu,sigma),disp=False)

    def predictive_values(self,mu,var):
        """
        Compute  mean, variance and conficence interval (percentiles 5 and 95) of the  prediction.

        :param mu: mean of the latent variable
        :param var: variance of the latent variable

        """
        if isinstance(mu,float) or isinstance(mu,int):
            mu = [mu]
            var = [var]
        pred_mean = []
        pred_var = []
        q1 = []
        q3 = []
        for m,s in zip(mu,np.sqrt(var)):
            pred_mean.append(self.predictive_mean(m,s))
            pred_var.append(self.predictive_variance(m,s,pred_mean[-1]))
            q1.append(self._predictive_percentiles(.025,m,s))
            q3.append(self._predictive_percentiles(.975,m,s))
        pred_mean = np.vstack(pred_mean)
        pred_var = np.vstack(pred_var)
        q1 = np.vstack(q1)
        q3 = np.vstack(q3)
        return pred_mean, pred_var, q1, q3


    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        pass

