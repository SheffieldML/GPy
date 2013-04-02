from scipy.special import gammaln, gamma
from scipy import integrate
import numpy as np
from GPy.likelihoods.likelihood_functions import likelihood_function
from scipy import stats

class student_t(likelihood_function):
    """Student t likelihood distribution
    For nomanclature see Bayesian Data Analysis 2003 p576

    $$\ln p(y_{i}|f_{i}) = \ln \Gamma(\frac{v+1}{2}) - \ln \Gamma(\frac{v}{2})\sqrt{v \pi}\sigma - \frac{v+1}{2}\ln (1 + \frac{1}{v}\left(\frac{y_{i} - f_{i}}{\sigma}\right)^2$$

    Laplace:
    Needs functions to calculate
    ln p(yi|fi)
    dln p(yi|fi)_dfi
    d2ln p(yi|fi)_d2fifj
    """
    def __init__(self, deg_free, sigma=2):
        self.v = deg_free
        self.sigma = sigma

        #FIXME: This should be in the superclass
        self.log_concave = False

    @property
    def variance(self):
        return (self.v / float(self.v - 2)) * (self.sigma**2)

    def link_function(self, y, f):
        """link_function $\ln p(y|f)$
        $$\ln p(y_{i}|f_{i}) = \ln \Gamma(\frac{v+1}{2}) - \ln \Gamma(\frac{v}{2})\sqrt{v \pi}\sigma - \frac{v+1}{2}\ln (1 + \frac{1}{v}\left(\frac{y_{i} - f_{i}}{\sigma}\right)^2$$

        :y: data
        :f: latent variables f
        :returns: float(likelihood evaluated for this point)

        """
        assert y.shape == f.shape
        e = y - f
        objective = (gammaln((self.v + 1) * 0.5)
                     - gammaln(self.v * 0.5)
                     + np.log(self.sigma * np.sqrt(self.v * np.pi))
                     - (self.v + 1) * 0.5
                     * np.log(1 + ((e**2 / self.sigma**2) / self.v))
                     )
        return np.sum(objective)

    def link_grad(self, y, f):
        """
        Gradient of the link function at y, given f w.r.t f

        $$\frac{d}{df}p(y_{i}|f_{i}) = \frac{(v + 1)(y - f)}{v \sigma^{2} + (y_{i} - f_{i})^{2}}$$

        :y: data
        :f: latent variables f
        :returns: gradient of likelihood evaluated at points

        """
        assert y.shape == f.shape
        e = y - f
        grad = ((self.v + 1) * e) / (self.v * (self.sigma**2) + (e**2))
        return grad

    def link_hess(self, y, f):
        """
        Hessian at this point (if we are only looking at the link function not the prior) the hessian will be 0 unless i == j
        i.e. second derivative link_function at y given f f_j  w.r.t f and f_j

        Will return diaganol of hessian, since every where else it is 0

        $$\frac{d^{2}p(y_{i}|f_{i})}{df^{2}} = \frac{(v + 1)(y - f)}{v \sigma^{2} + (y_{i} - f_{i})^{2}}$$

        :y: data
        :f: latent variables f
        :returns: array which is diagonal of covariance matrix (second derivative of likelihood evaluated at points)
        """
        assert y.shape == f.shape
        e = y - f
        #hess = ((self.v + 1) * e) / ((((self.sigma**2) * self.v) + e**2)**2)
        hess = ((self.v + 1)*(e**2 - self.v*(self.sigma**2))) / ((((self.sigma**2)*self.v) + e**2)**2)
        return hess

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

        student_t_samples = stats.t.rvs(self.v, loc=student_t_means[:,None],
                                        scale=self.sigma,
                                        size=(num_test_points, num_y_samples, num_f_samples))
        student_t_samples = np.reshape(student_t_samples,
                                       (num_test_points, num_y_samples*num_f_samples))

        #Now take the 97.5 and 0.25 percentile of these points
        p_025 = stats.scoreatpercentile(student_t_samples, .025, axis=1)[:, None]
        p_975 = stats.scoreatpercentile(student_t_samples, .975, axis=1)[:, None]

        p_025 = 1+p_025
        p_975 = 1+p_975

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
        import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
