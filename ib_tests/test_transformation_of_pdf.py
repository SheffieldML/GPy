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
import GPy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate


if __name__ == '__main__':
    p_theta = st.lognorm(.9)

    # Plot the PDF of theta
    fig, ax = plt.subplots()
    theta = np.linspace(0.0001, 8., 100)
    ax.plot(theta, p_theta.pdf(theta), linewidth=2, label='True PDF')
    ax.set_xlabel(r'$\theta$', fontsize=16)
    ax.set_ylabel(r'$p(\theta)$', fontsize=16)
    
    # Now let's look at the transformation phi = log(exp(theta - 1))
    t = GPy.constraints.Logexp()
    t.plot()
    # Plot the transformed probability density
    phi = np.linspace(-8, 8, 100)
    fig, ax = plt.subplots()
    ax.plot(phi, p_theta.pdf(t.f(phi)) * t.jacobianfactor(t.f(phi)), linewidth=2,
            label='Transformed PDF')
    # Now find the normalization constant for the naive transformation of the
    # PDF
    p_phi_prop = lambda(phi): p_theta.pdf(t.f(phi))
    c = integrate.quad(p_phi_prop, -np.inf, np.inf)[0]
    p_phi = lambda(phi): p_phi_prop(phi) / c
    ax.plot(phi, p_phi(phi), '--', linewidth=2, label='Naively transformed PDF')
    # Now let's draw some samples of theta and transform them so that we see
    # which one is right
    theta_s = p_theta.rvs(100000)
    phi_s = t.finv(theta_s)
    ax.hist(phi_s, normed=True, bins=100, alpha=0.25, label='Empirical')
    ax.set_xlim(-3, 10)
    ax.set_xlabel(r'transformed $\theta$', fontsize=16)
    ax.set_ylabel('PDF', fontsize=16)
    plt.legend(loc='best')
    plt.show(block=True)
