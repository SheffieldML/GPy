# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Simple Gaussian Processes classification 1D
Probit likelihood
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

# Outputs
Y = np.hstack([np.ones(N/2),np.repeat(-1,N/2)])[:,None]

# Kernel object
kernel = GPy.kern.rbf(1)

# Define likelihood
distribution = GPy.likelihoods.likelihood_functions.Probit()
likelihood_object = GPy.likelihoods.EP(Y,distribution)

# Model definition
m = GPy.models.GP(X,kernel,likelihood=likelihood_object)
m.ensure_default_constraints()
m.update_likelihood_approximation()
#m.checkgrad(verbose=1)
m.optimize()
print "Round 2"
m.update_likelihood_approximation()

#m.EPEM()
#m.plot()
#print(m)
