# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Simple Gaussian Processes classification 1D
probit likelihood
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()

pb.close('all')

# Inputs
N = 30
X1 = np.random.normal(5,2,N/2)
X2 = np.random.normal(10,2,N/2)
X = np.hstack([X1,X2])[:,None]

# Output
Y = np.hstack([np.ones(N/2),np.repeat(-1,N/2)])[:,None]

# Kernel object
kernel = GPy.kern.rbf(1)

# Likelihood object
distribution = GPy.likelihoods.likelihood_functions.probit()
likelihood = GPy.likelihoods.EP(Y,distribution)

# Model definition
m = GPy.models.GP(X,kernel,likelihood=likelihood)

# Model constraints
m.ensure_default_constraints()

# Optimize model
"""
EPEM runs a loop that consists of two steps:
1) EP likelihood approximation:
    m.update_likelihood_approximation()
2) Parameters optimization:
    m.optimize()
"""
m.EPEM()

# Plot
pb.subplot(211)
m.plot_GP()
pb.subplot(212)
m.plot_output()

print(m)
