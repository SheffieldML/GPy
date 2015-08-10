"""
Test the regression we get with the new transformations.

Author:
    Ilias Bilionis

Date:
    3/8/2015

"""


import sys
import os
# Make sure we load the GP that is here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GPy
import matplotlib.pyplot as plt
import numpy as np
import triangle


if __name__ == '__main__':
    m = GPy.examples.regression.olympic_marathon_men(optimize=True)
    plt.show(block=True)
    print m
    mcmc = GPy.inference.mcmc.samplers.Metropolis_Hastings(m)
    mcmc.sample(Ntotal=100000, Nburn=10000, Nthin=100, tune_interval=1000, tune_throughout=True)
    samples = np.array(mcmc.chains[-1])
    fig = triangle.corner(samples)
    m.plot()
    fig = plt.figure()
    for i in xrange(samples.shape[1]):
        ax = fig.add_subplot(samples.shape[1], 1, i + 1)
        ax.plot(samples[:, i], linewidth=1.5)
    plt.show(block=True)

