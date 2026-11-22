# Copyright (c) 2026 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Skewed observation likelihoods built as a probit (Gaussian CDF) times a
symmetric base density — the O'Hagan / Azzalini construction.

Used historically for Formula 1 lap-time modelling, where residuals are
asymmetric: cars are more often slower than a latent "pace" than faster.
"""

import numpy as np
from scipy import special, stats

from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from ..util.gaussian_special import d2_inv_mills_ratio, d_inv_mills_ratio, inv_mills_ratio
from . import link_functions
from .likelihood import Likelihood


class SkewGaussian(Likelihood):
    """
    Skew-normal (O'Hagan & Leonard, 1976; Azzalini) observation noise.

    .. math::
        p(y\\mid f) = \\frac{2}{\\sigma}\\,
        \\phi\\!\\left(\\frac{y-f}{\\sigma}\\right)
        \\Phi\\!\\left(\\alpha\\frac{y-f}{\\sigma}\\right)

    where :math:`\\phi` and :math:`\\Phi` are the standard normal PDF and CDF.
    When :math:`\\alpha=0` this reduces to a Gaussian with variance
    :math:`\\sigma^2`. Positive :math:`\\alpha` puts more mass on
    :math:`y>f` (right skew) — the usual choice for lap times.

    :param variance: noise variance :math:`\\sigma^2` (positive)
    :param alpha: skewness parameter (unconstrained; 0 recovers Gaussian)
    """

    def __init__(self, gp_link=None, variance=1.0, alpha=1.0, name="SkewGaussian"):
        if gp_link is None:
            gp_link = link_functions.Identity()
        if not isinstance(gp_link, link_functions.Identity):
            print(
                "Warning, Exact inference is not implemented for non-identity link functions; "
                "ensure Laplace (or EP) inference is used"
            )
        super(SkewGaussian, self).__init__(gp_link, name=name)
        self.variance = Param("variance", variance, Logexp())
        self.alpha = Param("alpha", alpha)
        self.link_parameter(self.variance)
        self.link_parameter(self.alpha)
        self.log_concave = False

    def to_dict(self):
        input_dict = super(SkewGaussian, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.SkewGaussian"
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["alpha"] = self.alpha.values.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(likelihood_class, input_dict):
        import copy

        input_dict = copy.deepcopy(input_dict)
        gp_link_dict = input_dict.pop("gp_link_dict")
        import GPy

        gp_link = GPy.likelihoods.link_functions.GPTransformation.from_dict(gp_link_dict)
        variance = np.asarray(input_dict.pop("variance")).item()
        alpha = np.asarray(input_dict.pop("alpha")).item()
        name = input_dict.pop("name", "SkewGaussian")
        return likelihood_class(gp_link=gp_link, variance=variance, alpha=alpha, name=name)

    def update_gradients(self, grads):
        self.variance.gradient = grads[0]
        self.alpha.gradient = grads[1]

    def _residual(self, link_f, y):
        return y - link_f

    def _sigma(self):
        return np.sqrt(self.variance.item())

    def _alpha(self):
        return self.alpha.item()

    def pdf_link(self, link_f, y, Y_metadata=None):
        return np.exp(self.logpdf_link(link_f, y, Y_metadata=Y_metadata))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        z = e / sigma
        delta = self._alpha() * z
        return (
            np.log(2.0)
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * z**2
            + special.log_ndtr(delta)
        )

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        return e / self.variance - (alpha / sigma) * lam

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        return -1.0 / self.variance + (alpha / sigma) ** 2 * d_inv_mills_ratio(delta, lam)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        return (alpha / sigma) ** 3 * d2_inv_mills_ratio(delta, lam)

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        s2 = self.variance.item()
        sigma = np.sqrt(s2)
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        return -0.5 / s2 + 0.5 * (e**2) / (s2**2) - lam * delta / (2.0 * s2)

    def dlogpdf_dlink_dvar(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        s2 = self.variance.item()
        sigma = np.sqrt(s2)
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        dlam = d_inv_mills_ratio(delta, lam)
        # d/dσ² [e/σ² - (α/σ) λ] ; a=α/σ, da/dσ²=-a/(2σ²), dδ/dσ²=-δ/(2σ²)
        a = alpha / sigma
        return -e / (s2**2) + (a / (2.0 * s2)) * (lam + delta * dlam)

    def d2logpdf_dlink2_dvar(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        s2 = self.variance.item()
        sigma = np.sqrt(s2)
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        dlam = d_inv_mills_ratio(delta, lam)
        d2lam = d2_inv_mills_ratio(delta, lam)
        # d² = -1/σ² + (α/σ)² λ' ; d/dσ² of that
        a2 = (alpha / sigma) ** 2
        return 1.0 / (s2**2) - (a2 / s2) * dlam - (a2 * delta * d2lam) / (2.0 * s2)

    def dlogpdf_link_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        delta = self._alpha() * e / sigma
        return inv_mills_ratio(delta) * (e / sigma)

    def dlogpdf_dlink_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        alpha = self._alpha()
        delta = alpha * e / sigma
        lam = inv_mills_ratio(delta)
        dlam = d_inv_mills_ratio(delta, lam)
        # d/dα [e/σ² - (α/σ) λ(α e/σ)]
        return -(1.0 / sigma) * lam - (alpha / sigma) * dlam * (e / sigma)

    def d2logpdf_dlink2_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        sigma = self._sigma()
        alpha = self._alpha()
        z = e / sigma
        delta = alpha * z
        lam = inv_mills_ratio(delta)
        dlam = d_inv_mills_ratio(delta, lam)
        d2lam = d2_inv_mills_ratio(delta, lam)
        # d² = -1/σ² + (α/σ)² λ' ; d/dα of that
        return (2.0 * alpha / (sigma**2)) * dlam + (alpha / sigma) ** 2 * d2lam * z

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.dlogpdf_link_dvar(f, y, Y_metadata=Y_metadata),
                self.dlogpdf_link_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.dlogpdf_dlink_dvar(f, y, Y_metadata=Y_metadata),
                self.dlogpdf_dlink_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.d2logpdf_dlink2_dvar(f, y, Y_metadata=Y_metadata),
                self.d2logpdf_dlink2_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        # Skew-normal mean: ξ + σ δ √(2/π) with δ = α/√(1+α²), ξ = mu under identity
        alpha = self._alpha()
        scale = self._sigma()
        delta = alpha / np.sqrt(1.0 + alpha**2)
        return self.gp_link.transf(mu) + scale * delta * np.sqrt(2.0 / np.pi)

    def predictive_variance(self, mu, variance, predictive_mean=None, Y_metadata=None):
        alpha = self._alpha()
        scale2 = self.variance.item()
        delta = alpha / np.sqrt(1.0 + alpha**2)
        # Var = σ² (1 - 2δ²/π) + Var(f)
        noise = scale2 * (1.0 - 2.0 * delta**2 / np.pi)
        return variance + noise

    def conditional_mean(self, gp):
        alpha = self._alpha()
        scale = self._sigma()
        delta = alpha / np.sqrt(1.0 + alpha**2)
        return self.gp_link.transf(gp) + scale * delta * np.sqrt(2.0 / np.pi)

    def conditional_variance(self, gp):
        alpha = self._alpha()
        scale2 = self.variance.item()
        delta = alpha / np.sqrt(1.0 + alpha**2)
        return np.ones_like(gp) * (scale2 * (1.0 - 2.0 * delta**2 / np.pi))

    def samples(self, gp, Y_metadata=None):
        orig_shape = gp.shape
        loc = self.gp_link.transf(gp.flatten())
        samples = stats.skewnorm.rvs(self._alpha(), loc=loc, scale=self._sigma())
        return np.asarray(samples).reshape(orig_shape)


class SkewExponential(Likelihood):
    """
    Skew-Laplace (probit × bilateral exponential) observation noise.

    The O'Hagan / Azzalini product needs a base density that is symmetric about
    zero. The natural exponential-family choice is the Laplace (double
    exponential) density :math:`(\\lambda/2)\\exp(-\\lambda|e|)`, giving

    .. math::
        p(y\\mid f) = \\lambda\\,
        \\exp\\!\\bigl(-\\lambda\\,|y-f|\\bigr)
        \\Phi\\!\\bigl(\\alpha(y-f)\\bigr).

    When :math:`\\alpha=0` this is Laplace with rate :math:`\\lambda`. Positive
    :math:`\\alpha` again favours :math:`y>f`.

    :param rate: Laplace rate :math:`\\lambda` (positive)
    :param alpha: skewness parameter (unconstrained)
    """

    def __init__(self, gp_link=None, rate=1.0, alpha=1.0, name="SkewExponential"):
        if gp_link is None:
            gp_link = link_functions.Identity()
        if not isinstance(gp_link, link_functions.Identity):
            print(
                "Warning, Exact inference is not implemented for non-identity link functions; "
                "ensure Laplace (or EP) inference is used"
            )
        super(SkewExponential, self).__init__(gp_link, name=name)
        self.rate = Param("rate", rate, Logexp())
        self.alpha = Param("alpha", alpha)
        self.link_parameter(self.rate)
        self.link_parameter(self.alpha)
        self.log_concave = False

    def to_dict(self):
        input_dict = super(SkewExponential, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.SkewExponential"
        input_dict["rate"] = self.rate.values.tolist()
        input_dict["alpha"] = self.alpha.values.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(likelihood_class, input_dict):
        import copy

        input_dict = copy.deepcopy(input_dict)
        gp_link_dict = input_dict.pop("gp_link_dict")
        import GPy

        gp_link = GPy.likelihoods.link_functions.GPTransformation.from_dict(gp_link_dict)
        rate = np.asarray(input_dict.pop("rate")).item()
        alpha = np.asarray(input_dict.pop("alpha")).item()
        name = input_dict.pop("name", "SkewExponential")
        return likelihood_class(gp_link=gp_link, rate=rate, alpha=alpha, name=name)

    def update_gradients(self, grads):
        self.rate.gradient = grads[0]
        self.alpha.gradient = grads[1]

    def _alpha(self):
        return self.alpha.item()

    def _residual(self, link_f, y):
        return y - link_f

    def pdf_link(self, link_f, y, Y_metadata=None):
        return np.exp(self.logpdf_link(link_f, y, Y_metadata=Y_metadata))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        rate = self.rate.item()
        alpha = self._alpha()
        return np.log(rate) - rate * np.abs(e) + special.log_ndtr(alpha * e)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        rate = self.rate.item()
        alpha = self._alpha()
        delta = alpha * e
        lam = inv_mills_ratio(delta)
        # d/df [ -λ|e| + log Φ(α e) ] with de/df = -1
        return rate * np.sign(e) - alpha * lam

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        alpha = self._alpha()
        delta = alpha * e
        # |e| contributes a Dirac at 0 that we omit; elsewhere second deriv is 0
        return (alpha**2) * d_inv_mills_ratio(delta)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        alpha = self._alpha()
        delta = alpha * e
        return (alpha**3) * d2_inv_mills_ratio(delta)

    def dlogpdf_link_drate(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        return 1.0 / self.rate.item() - np.abs(e)

    def dlogpdf_dlink_drate(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        return np.sign(e)

    def d2logpdf_dlink2_drate(self, link_f, y, Y_metadata=None):
        return np.zeros_like(link_f, dtype=float)

    def dlogpdf_link_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        return inv_mills_ratio(self._alpha() * e) * e

    def dlogpdf_dlink_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        alpha = self._alpha()
        delta = alpha * e
        lam = inv_mills_ratio(delta)
        dlam = d_inv_mills_ratio(delta, lam)
        # d/dα [λ_rate sgn(e) - α λ(α e)]
        return -lam - alpha * dlam * e

    def d2logpdf_dlink2_dalpha(self, link_f, y, Y_metadata=None):
        e = self._residual(link_f, y)
        alpha = self._alpha()
        delta = alpha * e
        dlam = d_inv_mills_ratio(delta)
        d2lam = d2_inv_mills_ratio(delta)
        # d² = α² λ' ; d/dα = 2α λ' + α² λ'' e
        return 2.0 * alpha * dlam + (alpha**2) * d2lam * e

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.dlogpdf_link_drate(f, y, Y_metadata=Y_metadata),
                self.dlogpdf_link_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.dlogpdf_dlink_drate(f, y, Y_metadata=Y_metadata),
                self.dlogpdf_dlink_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        return np.array(
            (
                self.d2logpdf_dlink2_drate(f, y, Y_metadata=Y_metadata),
                self.d2logpdf_dlink2_dalpha(f, y, Y_metadata=Y_metadata),
            )
        )

    def samples(self, gp, Y_metadata=None):
        """
        Sample via the Azzalini conditioning representation:
        draw Laplace Z0 and independent N(0,1) Z1; accept Z0 if Z1 < α Z0.
        """
        orig_shape = gp.shape
        n = int(np.prod(orig_shape))
        loc = self.gp_link.transf(gp.flatten())
        rate = self.rate.item()
        alpha = self._alpha()
        out = np.empty(n)
        filled = 0
        # Rejection sampling; expected acceptance is 1/2
        while filled < n:
            need = n - filled
            z0 = stats.laplace.rvs(scale=1.0 / rate, size=need * 2)
            z1 = np.random.standard_normal(size=z0.shape)
            keep = z1 < alpha * z0
            taken = z0[keep]
            take_n = min(need, taken.size)
            out[filled : filled + take_n] = taken[:take_n]
            filled += take_n
        return (out + loc).reshape(orig_shape)
