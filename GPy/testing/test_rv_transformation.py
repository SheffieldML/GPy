# Written by Ilias Bilionis
"""
Test if hyperparameters in models are properly transformed.
"""

import pytest
import numpy as np
import scipy.stats as st
import GPy


class TestModel(GPy.core.Model):
    """
    A simple GPy model with one parameter.
    """

    def __init__(self, theta=1.0):
        super(TestModel, self).__init__("test_model")
        theta = GPy.core.Param("theta", theta)
        self.link_parameter(theta)

    def log_likelihood(self):
        return 0.0


class TestRVTransformation:
    def _test_trans(self, trans):
        m = TestModel()
        prior = GPy.priors.LogGaussian(0.5, 0.1)
        m.theta.set_prior(prior)
        m.theta.unconstrain()
        m.theta.constrain(trans)
        # The PDF of the transformed variables
        p_phi = lambda phi: np.exp(-m._objective_grads(phi)[0])
        # To the empirical PDF of:
        theta_s = prior.rvs(1e5)
        phi_s = trans.finv(theta_s)
        # which is essentially a kernel density estimation
        kde = st.gaussian_kde(phi_s)
        # We will compare the PDF here:
        phi = np.linspace(phi_s.min(), phi_s.max(), 100)
        # The transformed PDF of phi should be this:
        pdf_phi = np.array([p_phi(p) for p in phi])
        # UNCOMMENT TO SEE GRAPHICAL COMPARISON
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.hist(phi_s, normed=True, bins=100, alpha=0.25, label='Histogram')
        # ax.plot(phi, kde(phi), '--', linewidth=2, label='Kernel Density Estimation')
        # ax.plot(phi, pdf_phi, ':', linewidth=2, label='Transformed PDF')
        # ax.set_xlabel(r'transformed $\theta$', fontsize=16)
        # ax.set_ylabel('PDF', fontsize=16)
        # plt.legend(loc='best')
        # plt.show(block=True)
        # END OF PLOT
        # The following test cannot be very accurate
        assert np.linalg.norm(pdf_phi - kde(phi)) / np.linalg.norm(kde(phi)) <= 1e-1

    def _test_grad(self, trans):
        np.random.seed(1234)
        m = TestModel(np.random.uniform(0.5, 1.5, 20))
        prior = GPy.priors.LogGaussian(0.5, 0.1)
        m.theta.set_prior(prior)
        m.theta.constrain(trans)
        m.randomize()
        print(m)
        assert m.checkgrad(1)

    def test_Logexp(self):
        self._test_trans(GPy.constraints.Logexp())

    @pytest.mark.skip(
        "Gradient not checking right, @jameshensman what is going on here?"
    )
    def test_Logexp_grad(self):
        self._test_grad(GPy.constraints.Logexp())

    def test_Exponent(self):
        self._test_trans(GPy.constraints.Exponent())

    @pytest.mark.skip(
        "Gradient not checking right, @jameshensman what is going on here?"
    )
    def test_Exponent_grad(self):
        self._test_grad(GPy.constraints.Exponent())
