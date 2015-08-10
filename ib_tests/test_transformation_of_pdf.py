"""
Test the transformation of a PDF.

Author:
    Ilias Bilionis

Date:
    8/4/2015

"""


import sys
import os
# Make sure we load the GP that is here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print 'trying'
import GPy
print 'done'
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate


class TestModel(GPy.core.Model):
    def __init__(self):
        GPy.core.Model.__init__(self, 'test_model')
        theta = GPy.core.Param('theta', 1.)
        self.link_parameter(theta)

    def log_likelihood(self):
        return 0.


if __name__ == '__main__':
    m = TestModel()
    prior = GPy.priors.LogGaussian(0., .9)
    m.theta.set_prior(prior)

    # The following should return the PDF in terms of the transformed quantities
    p_phi = lambda(phi): np.exp(-m._objective_grads(phi)[0])

    # Let's look at the transformation phi = log(exp(theta - 1))
    trans = GPy.constraints.Exponent()
    m.theta.constrain(trans)
    # Plot the transformed probability density
    phi = np.linspace(-8, 8, 100)
    fig, ax = plt.subplots()
    # Let's draw some samples of theta and transform them so that we see
    # which one is right
    theta_s = prior.rvs(10000)
    # Transform it to the new variables
    phi_s = trans.finv(theta_s)
    # And draw their histogram
    ax.hist(phi_s, normed=True, bins=100, alpha=0.25, label='Empirical')
    # This is to be compared to the PDF of the model expressed in terms of these new
    # variables
    ax.plot(phi, [p_phi(p) for p in phi], label='Transformed PDF', linewidth=2)
    ax.set_xlim(-3, 10)
    ax.set_xlabel(r'transformed $\theta$', fontsize=16)
    ax.set_ylabel('PDF', fontsize=16)
    plt.legend(loc='best')
    # Now let's test the gradients
    m.checkgrad(verbose=True)
    # And show the plot
    plt.show(block=True)
