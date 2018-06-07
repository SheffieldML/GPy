# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf, derivLogCdfNormal, logCdfNormal
from . import link_functions
from .likelihood import Likelihood

class Bernoulli(Likelihood):
    """
    Bernoulli likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

    .. Note::
        Y takes values in either {-1, 1} or {0, 1}.
        link function should have the domain [0, 1], e.g. probit (default) or Heaviside

    .. See also::
        likelihood.py, for the parent class
    """
    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Probit()

        super(Bernoulli, self).__init__(gp_link, 'Bernoulli')

        if isinstance(gp_link , (link_functions.Heaviside, link_functions.Probit)):
            self.log_concave = True

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Bernoulli, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.Bernoulli"
        return input_dict

    def _preprocess_values(self, Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.

        ..Note:: Binary classification algorithm works better with classes {-1, 1}
        """
        Y_prep = Y.copy()
        Y1 = Y[Y.flatten()==1].size
        Y2 = Y[Y.flatten()==0].size
        assert Y1 + Y2 == Y.size, 'Bernoulli likelihood is meant to be used only with outputs in {0, 1}.'
        Y_prep[Y.flatten() == 0] = -1
        return Y_prep

    def moments_match_ep(self, Y_i, tau_i, v_i, Y_metadata_i=None):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        if Y_i == 1:
            sign = 1.
        elif Y_i == 0 or Y_i == -1:
            sign = -1
        else:
            raise ValueError("bad value for Bernoulli observation (0, 1)")
        if isinstance(self.gp_link, link_functions.Probit):
            z = sign*v_i/np.sqrt(tau_i**2 + tau_i)
            phi_div_Phi = derivLogCdfNormal(z)
            log_Z_hat = logCdfNormal(z)

            mu_hat = v_i/tau_i + sign*phi_div_Phi/np.sqrt(tau_i**2 + tau_i)
            sigma2_hat = 1./tau_i - (phi_div_Phi/(tau_i**2+tau_i))*(z+phi_div_Phi)

        elif isinstance(self.gp_link, link_functions.Heaviside):
            z = sign*v_i/np.sqrt(tau_i)
            phi_div_Phi = derivLogCdfNormal(z)
            log_Z_hat = logCdfNormal(z)
            mu_hat = v_i/tau_i + sign*phi_div_Phi/np.sqrt(tau_i)
            sigma2_hat = (1. - a*phi_div_Phi - np.square(phi_div_Phi))/tau_i
        else:
            #TODO: do we want to revert to numerical quadrature here?
            raise ValueError("Exact moment matching not available for link {}".format(self.gp_link.__name__))

        # TODO: Output log_Z_hat instead of Z_hat (needs to be change in all others likelihoods)
        return np.exp(log_Z_hat), mu_hat, sigma2_hat

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Probit):

            if gh_points is None:
                gh_x, gh_w = self._gh_points()
            else:
                gh_x, gh_w = gh_points


            gh_w = gh_w / np.sqrt(np.pi)
            shape = m.shape
            m,v,Y = m.flatten(), v.flatten(), Y.flatten()
            Ysign = np.where(Y==1,1,-1)
            X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + (m*Ysign)[:,None]
            p = std_norm_cdf(X)
            p = np.clip(p, 1e-9, 1.-1e-9) # for numerical stability
            N = std_norm_pdf(X)
            F = np.log(p).dot(gh_w)
            NoverP = N/p
            dF_dm = (NoverP*Ysign[:,None]).dot(gh_w)
            dF_dv = -0.5*(NoverP**2 + NoverP*X).dot(gh_w)
            return F.reshape(*shape), dF_dm.reshape(*shape), dF_dv.reshape(*shape), None
        else:
            raise NotImplementedError


    def predictive_mean(self, mu, variance, Y_metadata=None):

        if isinstance(self.gp_link, link_functions.Probit):
            return std_norm_cdf(mu/np.sqrt(1+variance))

        elif isinstance(self.gp_link, link_functions.Heaviside):
            return std_norm_cdf(mu/np.sqrt(variance))

        else:
            raise NotImplementedError

    def predictive_variance(self, mu, variance, pred_mean, Y_metadata=None):

        if isinstance(self.gp_link, link_functions.Heaviside):
            return 0.
        else:
            return np.nan

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Likelihood function given inverse link of f.

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: likelihood evaluated for this point
        :rtype: float

        .. Note:
            Each y_i must be in {0, 1}
        """
        #objective = (inv_link_f**y) * ((1.-inv_link_f)**(1.-y))
        return np.where(y==1, inv_link_f, 1.-inv_link_f)

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood function given inverse link of f.

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = y_{i}\\log\\lambda(f_{i}) + (1-y_{i})\\log (1-f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: log likelihood evaluated at points inverse link of f.
        :rtype: float
        """
        #objective = y*np.log(inv_link_f) + (1.-y)*np.log(inv_link_f)
        p = np.where(y==1, inv_link_f, 1.-inv_link_f)
        return np.log(np.clip(p, 1e-9 ,np.inf))

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given inverse link of f w.r.t inverse link of f.

        .. math::
            \\frac{d\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{y_{i}}{\\lambda(f_{i})} - \\frac{(1 - y_{i})}{(1 - \\lambda(f_{i}))}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: gradient of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array
        """
        #grad = (y/inv_link_f) - (1.-y)/(1-inv_link_f)
        #grad = np.where(y, 1./inv_link_f, -1./(1-inv_link_f))
        ff = np.clip(inv_link_f, 1e-9, 1-1e-9)
        denom = np.where(y==1, ff, -(1-ff))
        return 1./denom

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given inv_link_f, w.r.t inv_link_f the hessian will be 0 unless i == j
        i.e. second derivative logpdf at y given inverse link of f_i and inverse link of f_j  w.r.t inverse link of f_i and inverse link of f_j.


        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{-y_{i}}{\\lambda(f)^{2}} - \\frac{(1-y_{i})}{(1-\\lambda(f))^{2}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        #d2logpdf_dlink2 = -y/(inv_link_f**2) - (1-y)/((1-inv_link_f)**2)
        #d2logpdf_dlink2 = np.where(y, -1./np.square(inv_link_f), -1./np.square(1.-inv_link_f))
        arg = np.where(y==1, inv_link_f, 1.-inv_link_f)
        ret =  -1./np.square(np.clip(arg, 1e-9, 1e9))
        if np.any(np.isinf(ret)):
            stop
        return ret

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given inverse link of f w.r.t inverse link of f

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{2y_{i}}{\\lambda(f)^{3}} - \\frac{2(1-y_{i}}{(1-\\lambda(f))^{3}}

        :param inv_link_f: latent variables passed through inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: third derivative of log likelihood evaluated at points inverse_link(f)
        :rtype: Nx1 array
        """
        assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
        #d3logpdf_dlink3 = 2*(y/(inv_link_f**3) - (1-y)/((1-inv_link_f)**3))
        state = np.seterr(divide='ignore')
        # TODO check y \in {0, 1} or {-1, 1}
        d3logpdf_dlink3 = np.where(y==1, 2./(inv_link_f**3), -2./((1.-inv_link_f)**3))
        np.seterr(**state)
        return d3logpdf_dlink3

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        """
        Get the "quantiles" of the binary labels (Bernoulli draws). all the
        quantiles must be either 0 or 1, since those are the only values the
        draw can take!
        """
        p = self.predictive_mean(mu, var)
        return [np.asarray(p>(q/100.), dtype=np.int32) for q in quantiles]

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        ns = np.ones_like(gp, dtype=int)
        Ysim = np.random.binomial(ns, self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        return np.zeros(self.size)
