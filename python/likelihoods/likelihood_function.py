import GPy
from scipy.special import gamma, gammaln

class student_t(GPy.likelihoods.likelihood_function):
    """Student t likelihood distribution
    For nomanclature see Bayesian Data Analysis 2003 p576

    $$\ln(\frac{\Gamma(\frac{(v+1)}{2})}{\Gamma(\sqrt(v \pi \Gamma(\frac{v}{2}))})+ \ln(1+\frac{(y_i-f_i)^2}{\sigma v})^{-\frac{(v+1)}{2}}$$
    TODO:Double check this

    Laplace:
    Needs functions to calculate
    ln p(yi|fi)
    dln p(yi|fi)_dfi
    d2ln p(yi|fi)_d2fi
    """
    def __init__(self, deg_free, sigma=1):
        self.v = deg_free
        self.sigma = 1

    def link_function(self, y_i, f_i):
        """link_function $\ln p(y_i|f_i)$
        $$\ln \Gamma(\frac{v+1}{2}) - \ln \Gamma(\frac{v}{2}) - \ln \frac{v \pi \sigma}{2} - \frac{v+1}{2}\ln (1 + \frac{(y_{i} - f_{i})^{2}}{v\sigma})$$
        TODO: Double check this

        :y_i: datum number i
        :f_i: latent variable f_i
        :returns: float(likelihood evaluated for this point)

        """
        e = y_i - f_i
        return gammaln((v+1)*0.5) - gammaln(v*0.5) - np.ln(v*np.pi*sigma)*0.5 - (v+1)*0.5*np.ln(1 + ((e/sigma)**2)/v) #Check the /v!

    def link_grad(self, y_i, f_i):
        """gradient of the link function at y_i, given f_i w.r.t f_i

        derivative of log((gamma((v+1)/2)/gamma(sqrt(v*pi*gamma(v/2))))*(1+(t^2)/(a*v))^((-(v+1))/2)) with respect to t
        $$\frac{(y_i - f_i)(v + 1)}{\sigma v (y_{i} - f_{i})^{2}}$$
        TODO: Double check this

        :y_i: datum number i
        :f_i: latent variable f_i
        :returns: float(gradient of likelihood evaluated at this point)

        """
        pass

    def link_hess(self, y_i, f_i, f_j):
        """hessian at this point (the hessian will be 0 unless i == j)
        i.e. second derivative w.r.t f_i and f_j

        second derivative of

        :y_i: @todo
        :f_i: @todo
        :f_j: @todo
        :returns: @todo

        """
        if f_i =
        pass

