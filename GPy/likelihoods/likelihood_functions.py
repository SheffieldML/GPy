# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, integrate
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from scipy.special import gammaln, gamma
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf

class likelihood_function:
    """ Likelihood class for doing Expectation propagation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self,location=0,scale=1):
        self.location = location
        self.scale = scale
        self.log_concave = True

    def _get_params(self):
        return np.zeros(0)

    def _get_param_names(self):
        return []

    def _set_params(self, p):
        pass

class probit(likelihood_function):
    """
    Probit likelihood
    Y is expected to take values in {-1,1}
    -----
    $$
    L(x) = \\Phi (Y_i*f_i)
    $$
    """

    def moments_match(self,data_i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        #if data_i == 0: data_i = -1 #NOTE Binary classification algorithm works better with classes {-1,1}, 1D-plotting works better with classes {0,1}.
        # TODO: some version of assert
        z = data_i*v_i/np.sqrt(tau_i**2 + tau_i)
        Z_hat = std_norm_cdf(z)
        phi = std_norm_pdf(z)
        mu_hat = v_i/tau_i + data_i*phi/(Z_hat*np.sqrt(tau_i**2 + tau_i))
        sigma2_hat = 1./tau_i - (phi/((tau_i**2+tau_i)*Z_hat))*(z+phi/Z_hat)
        return Z_hat, mu_hat, sigma2_hat

    def predictive_values(self,mu,var):
        """
        Compute  mean, variance and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mu = mu.flatten()
        var = var.flatten()
        mean = stats.norm.cdf(mu/np.sqrt(1+var))
        norm_025 = [stats.norm.ppf(.025,m,v) for m,v in zip(mu,var)]
        norm_975 = [stats.norm.ppf(.975,m,v) for m,v in zip(mu,var)]
        p_025 = stats.norm.cdf(norm_025/np.sqrt(1+var))
        p_975 = stats.norm.cdf(norm_975/np.sqrt(1+var))
        return mean, np.nan*var, p_025, p_975 # TODO: var

class Poisson(likelihood_function):
    """
    Poisson likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def moments_match(self,data_i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        mu = v_i/tau_i
        sigma = np.sqrt(1./tau_i)
        def poisson_norm(f):
            """
            Product of the likelihood and the cavity distribution
            """
            pdf_norm_f = stats.norm.pdf(f,loc=mu,scale=sigma)
            rate = np.exp( (f*self.scale)+self.location)
            poisson = stats.poisson.pmf(float(data_i),rate)
            return pdf_norm_f*poisson

        def log_pnm(f):
            """
            Log of poisson_norm
            """
            return -(-.5*(f-mu)**2/sigma**2 - np.exp( (f*self.scale)+self.location) + ( (f*self.scale)+self.location)*data_i)

        """
        Golden Search and Simpson's Rule
        --------------------------------
        Simpson's Rule is used to calculate the moments mumerically, it needs a grid of points as input.
        Golden Search is used to find the mode in the poisson_norm distribution and define around it the grid for Simpson's Rule
        """
        #TODO golden search & simpson's rule can be defined in the general likelihood class, rather than in each specific case.

        #Golden search
        golden_A = -1 if data_i == 0 else np.array([np.log(data_i),mu]).min() #Lower limit
        golden_B = np.array([np.log(data_i),mu]).max() #Upper limit
        golden_A = (golden_A - self.location)/self.scale
        golden_B = (golden_B - self.location)/self.scale
        opt = sp.optimize.golden(log_pnm,brack=(golden_A,golden_B)) #Better to work with log_pnm than with poisson_norm

        # Simpson's approximation
        width = 3./np.log(max(data_i,2))
        A = opt - width #Lower limit
        B = opt + width #Upper limit
        K =  10*int(np.log(max(data_i,150))) #Number of points in the grid, we DON'T want K to be the same number for every case
        h = (B-A)/K # length of the intervals
        grid_x = np.hstack([np.linspace(opt-width,opt,K/2+1)[1:-1], np.linspace(opt,opt+width,K/2+1)]) # grid of points (X axis)
        x = np.hstack([A,B,grid_x[range(1,K,2)],grid_x[range(2,K-1,2)]]) # grid_x rearranged, just to make Simpson's algorithm easier
        zeroth = np.hstack([poisson_norm(A),poisson_norm(B),[4*poisson_norm(f) for f in grid_x[range(1,K,2)]],[2*poisson_norm(f) for f in grid_x[range(2,K-1,2)]]]) # grid of points (Y axis) rearranged like x
        first = zeroth*x
        second = first*x
        Z_hat = sum(zeroth)*h/3 # Zero-th moment
        mu_hat = sum(first)*h/(3*Z_hat) # First moment
        m2 = sum(second)*h/(3*Z_hat) # Second moment
        sigma2_hat = m2 - mu_hat**2 # Second central moment
        return float(Z_hat), float(mu_hat), float(sigma2_hat)

    def predictive_values(self,mu,var):
        """
        Compute  mean, and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mean = np.exp(mu*self.scale + self.location)
        tmp = stats.poisson.ppf(np.array([.025,.975]),mean)
        p_025 = tmp[:,0]
        p_975 = tmp[:,1]
        return mean,np.nan*mean,p_025,p_975 # better variance here TODO


class student_t(likelihood_function):
    """Student t likelihood distribution
    For nomanclature see Bayesian Data Analysis 2003 p576

    $$\ln p(y_{i}|f_{i}) = \ln \Gamma(\frac{v+1}{2}) - \ln \Gamma(\frac{v}{2})\sqrt{v \pi}\sigma - \frac{v+1}{2}\ln (1 + \frac{1}{v}\left(\frac{y_{i} - f_{i}}{\sigma}\right)^2)$$

    Laplace:
    Needs functions to calculate
    ln p(yi|fi)
    dln p(yi|fi)_dfi
    d2ln p(yi|fi)_d2fifj
    """
    def __init__(self, deg_free, sigma=2):
        #super(student_t, self).__init__()
        self.v = deg_free
        self.sigma = sigma
        self.log_concave = False

        self._set_params(np.asarray(sigma))

    def _get_params(self):
        return np.asarray(self.sigma)

    def _get_param_names(self):
        return ["t_noise_std"]

    def _set_params(self, x):
        self.sigma = float(x)

    @property
    def variance(self, extra_data=None):
        return (self.v / float(self.v - 2)) * (self.sigma**2)

    def link_function(self, y, f, extra_data=None):
        """link_function $\ln p(y|f)$
        $$\ln p(y_{i}|f_{i}) = \ln \Gamma(\frac{v+1}{2}) - \ln \Gamma(\frac{v}{2})\sqrt{v \pi}\sigma - \frac{v+1}{2}\ln (1 + \frac{1}{v}\left(\frac{y_{i} - f_{i}}{\sigma}\right)^2$$

        For wolfram alpha import parts for derivative of sigma are -log(sqrt(v*pi)*s) -(1/2)*(v + 1)*log(1 + (1/v)*((y-f)/(s))^2))

        :y: data
        :f: latent variables f
        :extra_data: extra_data which is not used in student t distribution
        :returns: float(likelihood evaluated for this point)

        """
        assert y.shape == f.shape
        e = y - f
        objective = (+ gammaln((self.v + 1) * 0.5)
                     - gammaln(self.v * 0.5)
                     - 0.5*np.log((self.sigma**2) * self.v * np.pi)
                     #- (self.v + 1) * 0.5 * np.log(1 + (((e / self.sigma)**2) / self.v))
                     - (self.v + 1) * 0.5 * np.log(1 + (e**2)/(self.v*(self.sigma**2)))
                    )
        return np.sum(objective)

    def dlik_df(self, y, f, extra_data=None):
        """
        Gradient of the link function at y, given f w.r.t f

        $$\frac{dp(y_{i}|f_{i})}{df} = \frac{(v+1)(y_{i}-f_{i})}{(y_{i}-f_{i})^{2} + \sigma^{2}v}$$

        :y: data
        :f: latent variables f
        :extra_data: extra_data which is not used in student t distribution
        :returns: gradient of likelihood evaluated at points

        """
        assert y.shape == f.shape
        e = y - f
        grad = ((self.v + 1) * e) / (self.v * (self.sigma**2) + (e**2))
        return grad

    def d2lik_d2f(self, y, f, extra_data=None):
        """
        Hessian at this point (if we are only looking at the link function not the prior) the hessian will be 0 unless i == j
        i.e. second derivative link_function at y given f f_j  w.r.t f and f_j

        Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
        (the distribution for y_{i} depends only on f_{i} not on f_{j!=i}

        $$\frac{d^{2}p(y_{i}|f_{i})}{d^{3}f} = \frac{(v+1)((y_{i}-f_{i})^{2} - \sigma^{2}v)}{((y_{i}-f_{i})^{2} + \sigma^{2}v)^{2}}$$

        :y: data
        :f: latent variables f
        :extra_data: extra_data which is not used in student t distribution
        :returns: array which is diagonal of covariance matrix (second derivative of likelihood evaluated at points)
        """
        assert y.shape == f.shape
        e = y - f
        hess = ((self.v + 1)*(e**2 - self.v*(self.sigma**2))) / ((((self.sigma**2)*self.v) + e**2)**2)
        return hess

    def d3lik_d3f(self, y, f, extra_data=None):
        """
        Third order derivative link_function (log-likelihood ) at y given f f_j w.r.t f and f_j

        $$\frac{d^{3}p(y_{i}|f_{i})}{d^{3}f} = \frac{-2(v+1)((y_{i} - f_{i})^3 - 3(y_{i} - f_{i}) \sigma^{2} v))}{((y_{i} - f_{i}) + \sigma^{2} v)^3}$$
        """
        assert y.shape == f.shape
        e = y - f
        d3lik_d3f = ( -(2*(self.v + 1)*(-e)*(e**2 - 3*self.v*(self.sigma**2))) /
                       ((e**2 + (self.sigma**2)*self.v)**3)
                    )
        return d3lik_d3f

    def lik_dstd(self, y, f, extra_data=None):
        """
        Gradient of the likelihood (lik) w.r.t sigma parameter (standard deviation)

        Terms relavent to derivatives wrt sigma are:
        -log(sqrt(v*pi)*s) -(1/2)*(v + 1)*log(1 + (1/v)*((y-f)/(s))^2))

        $$\frac{dp(y_{i}|f_{i})}{d\sigma} = -\frac{1}{\sigma} + \frac{(1+v)(y_{i}-f_{i})^2}{\sigma^3 v(1 + \frac{1}{v}(\frac{(y_{i} - f_{i})}{\sigma^2})^2)}$$
        """
        assert y.shape == f.shape
        e = y - f
        dlik_dsigma = ( - (1/self.sigma) +
                        ((1+self.v)*(e**2))/((self.sigma**3)*self.v*(1 + ((e**2) / ((self.sigma**2)*self.v)) ) )
                      )
        #dlik_dsigma = (((self.v + 1)*(e**2))/((e**2) + self.v*(self.sigma**2))) - 1
        return dlik_dsigma

    def dlik_df_dstd(self, y, f, extra_data=None):
        """
        Gradient of the dlik_df w.r.t sigma parameter (standard deviation)

        $$\frac{d}{d\sigma}(\frac{dp(y_{i}|f_{i})}{df}) = \frac{-2\sigma v(v + 1)(y_{i}-f_{i})}{(y_{i}-f_{i})^2 + \sigma^2 v)^2}$$
        """
        assert y.shape == f.shape
        e = y - f
        dlik_grad_dsigma = ((-2*self.sigma*self.v*(self.v + 1)*e) #2 might not want to be here?
                            / ((self.v*(self.sigma**2) + e**2)**2)
                           )
        return dlik_grad_dsigma

    def d2lik_d2f_dstd(self, y, f, extra_data=None):
        """
        Gradient of the hessian (d2lik_d2f) w.r.t sigma parameter (standard deviation)

        $$\frac{d}{d\sigma}(\frac{d^{2}p(y_{i}|f_{i})}{d^{2}f}) = \frac{2\sigma v(v + 1)(\sigma^2 v - 3(y-f)^2)}{((y-f)^2 + \sigma^2 v)^3}$$
        """
        assert y.shape == f.shape
        e = y - f
        dlik_hess_dsigma = (  (2*self.sigma*self.v*(self.v + 1)*((self.sigma**2)*self.v - 3*(e**2))) /
                              ((e**2 + (self.sigma**2)*self.v)**3)
                           )
        #dlik_hess_dsigma = ( 2*(self.v + 1)*self.v*(self.sigma**2)*((e**2) + (self.v*(self.sigma**2)) - 4*(e**2))
                             #/ ((e**2 + (self.sigma**2)*self.v)**3) )
        return dlik_hess_dsigma

    def _gradients(self, y, f, extra_data=None):
        #must be listed in same order as 'get_param_names'
        derivs = ([self.lik_dstd(y, f, extra_data=extra_data)],
                  [self.dlik_df_dstd(y, f, extra_data=extra_data)],
                  [self.d2lik_d2f_dstd(y, f, extra_data=extra_data)]
                 ) # lists as we might learn many parameters
        # ensure we have gradients for every parameter we want to optimize
        assert len(derivs[0]) == len(self._get_param_names())
        assert len(derivs[1]) == len(self._get_param_names())
        assert len(derivs[2]) == len(self._get_param_names())
        return derivs

    def predictive_values(self, mu, var):
        """
        Compute  mean, and conficence interval (percentiles 5 and 95) of the prediction

        Need to find what the variance is at the latent points for a student t*normal p(y*|f*)p(f*)
        (((g((v+1)/2))/(g(v/2)*s*sqrt(v*pi)))*(1+(1/v)*((y-f)/s)^2)^(-(v+1)/2))
        *((1/(s*sqrt(2*pi)))*exp(-(1/(2*(s^2)))*((y-f)^2)))
        """

        #We want the variance around test points y which comes from int p(y*|f*)p(f*) df*
        #Var(y*) = Var(E[y*|f*]) + E[Var(y*|f*)]
        #Since we are given f* (mu) which is our mean (expected) value of y*|f* then the variance is the variance around this
        #Which was also given to us as (var)
        #We also need to know the expected variance of y* around samples f*, this is the variance of the student t distribution
        #However the variance of the student t distribution is not dependent on f, only on sigma and the degrees of freedom
        true_var = var + self.variance

        #Now we have an analytical solution for the variances of the distribution p(y*|f*)p(f*) around our test points but we now
        #need the 95 and 5 percentiles.
        #FIXME: Hack, just pretend p(y*|f*)p(f*) is a gaussian and use the gaussian's percentiles
        p_025 = mu - 2.*true_var
        p_975 = mu + 2.*true_var

        return mu, np.nan*mu, p_025, p_975

    def sample_predicted_values(self, mu, var):
        """ Experimental sample approches and numerical integration """
        #p_025 = stats.t.ppf(.025, mu)
        #p_975 = stats.t.ppf(.975, mu)

        num_test_points = mu.shape[0]
        #Each mu is the latent point f* at the test point x*,
        #and the var is the gaussian variance at this point
        #Take lots of samples from this, so we have lots of possible values
        #for latent point f* for each test point x* weighted by how likely we were to pick it
        print "Taking %d samples of f*".format(num_test_points)
        num_f_samples = 10
        num_y_samples = 10
        student_t_means = np.random.normal(loc=mu, scale=np.sqrt(var), size=(num_test_points, num_f_samples))
        print "Student t means shape: ", student_t_means.shape

        #Now we have lots of f*, lets work out the likelihood of getting this by sampling
        #from a student t centred on this point, sample many points from this distribution
        #centred on f*
        #for test_point, f in enumerate(student_t_means):
            #print test_point
            #print f.shape
            #student_t_samples = stats.t.rvs(self.v, loc=f[:,None],
                                            #scale=self.sigma,
                                            #size=(num_f_samples, num_y_samples))
            #print student_t_samples.shape

        student_t_samples = stats.t.rvs(self.v, loc=student_t_means[:, None],
                                        scale=self.sigma,
                                        size=(num_test_points, num_y_samples, num_f_samples))
        student_t_samples = np.reshape(student_t_samples,
                                       (num_test_points, num_y_samples*num_f_samples))

        #Now take the 97.5 and 0.25 percentile of these points
        p_025 = stats.scoreatpercentile(student_t_samples, .025, axis=1)[:, None]
        p_975 = stats.scoreatpercentile(student_t_samples, .975, axis=1)[:, None]

        ##Alernenately we could sample from int p(y|f*)p(f*|x*) df*
        def t_gaussian(f, mu, var):
            return (((gamma((self.v+1)*0.5)) / (gamma(self.v*0.5)*self.sigma*np.sqrt(self.v*np.pi))) * ((1+(1/self.v)*(((mu-f)/self.sigma)**2))**(-(self.v+1)*0.5))
                    * ((1/(np.sqrt(2*np.pi*var)))*np.exp(-(1/(2*var)) *((mu-f)**2)))
                    )

        def t_gauss_int(mu, var):
            print "Mu: ", mu
            print "var: ", var
            result = integrate.quad(t_gaussian, 0.025, 0.975, args=(mu, var))
            print "Result: ", result
            return result[0]

        vec_t_gauss_int = np.vectorize(t_gauss_int)

        p = vec_t_gauss_int(mu, var)
        p_025 = mu - p
        p_975 = mu + p
        return mu, np.nan*mu, p_025, p_975


class weibull_survival(likelihood_function):
    """Weibull t likelihood distribution for survival analysis with censoring
        For nomanclature see Bayesian Survival Analysis

    Laplace:
    Needs functions to calculate
    ln p(yi|fi)
    dln p(yi|fi)_dfi
    d2ln p(yi|fi)_d2fifj
    """
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self.log_concave = True # Or false?

    def link_function(self, y, f, extra_data=None):
        """
        link_function $\ln p(y|f)$, i.e. log likelihood

        $$\ln p(y|f) = v_{i}(\ln \alpha + (\alpha - 1)\ln y_{i} + f_{i}) - y_{i}^{\alpha}\exp(f_{i})$$

        :y: time of event data
        :f: latent variables f
        :extra_data: the censoring indicator, 1 for censored, 0 for not
        :returns: float(likelihood evaluated for this point)

        """
        y = np.squeeze(y)
        f = np.squeeze(f)
        assert y.shape == f.shape

        v = extra_data
        objective = v*(np.log(self.shape) + (self.shape - 1)*np.log(y) + f) - (y**self.shape)*np.exp(f)  # FIXME: CHECK THIS WITH BOOK, wheres scale?
        return np.sum(objective)

    def dlik_df(self, y, f, extra_data=None):
        """
        Gradient of the link function at y, given f w.r.t f

        $$\frac{d}{df} \ln p(y_{i}|f_{i}) = v_{i} - y_{i}\exp(f_{i})

        :y: data
        :f: latent variables f
        :extra_data: the censoring indicator, 1 for censored, 0 for not
        :returns: gradient of likelihood evaluated at points

        """
        y = np.squeeze(y)
        f = np.squeeze(f)
        assert y.shape == f.shape

        v = extra_data
        grad = v - (y**self.shape)*np.exp(f)
        return np.squeeze(grad)

    def d2lik_d2f(self, y, f, extra_data=None):
        """
        Hessian at this point (if we are only looking at the link function not the prior) the hessian will be 0 unless i == j
        i.e. second derivative link_function at y given f f_j  w.r.t f and f_j

        Will return diagonal of hessian, since every where else it is 0

        $$\frac{d^{2}p(y_{i}|f_{i})}{df^{2}} = \frac{(v + 1)(y - f)}{v \sigma^{2} + (y_{i} - f_{i})^{2}}$$

        :y: data
        :f: latent variables f
        :extra_data: extra_data which is not used hessian
        :returns: array which is diagonal of covariance matrix (second derivative of likelihood evaluated at points)
        """
        y = np.squeeze(y)
        f = np.squeeze(f)
        assert y.shape == f.shape

        hess = (y**self.shape)*np.exp(f)
        return np.squeeze(hess)
