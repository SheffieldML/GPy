from scipy.special import gammaln
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
        Compute  mean, and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mean = np.exp(mu)
        p_025 = stats.t.ppf(.025, mean)
        p_975 = stats.t.ppf(.975, mean)

        return mean, np.nan*mean, p_025, p_975
