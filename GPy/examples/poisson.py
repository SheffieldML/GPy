# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Simple Gaussian Processes classification
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()

pb.close('all')
default_seed=10000

model_type='Full'
inducing=4
seed=default_seed
"""Simple 1D classification example.
:param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
:param seed : seed value for data generation (default is 4).
:type seed: int
:param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
:type inducing: int
"""

X = np.arange(0,100,5)[:,None]
F = np.round(np.sin(X/18.) + .1*X)
E = np.random.randint(-3,3,20)[:,None]
Y = F + E
pb.plot(X,F,'k-')
pb.plot(X,Y,'ro')
pb.figure()
likelihood = GPy.inference.likelihoods.poisson(Y,scale=6.)

m = GPy.models.GP(X,likelihood=likelihood)
#m = GPy.models.GP(data['X'],Y=likelihood.Y)

m.constrain_positive('var')
m.constrain_positive('len')
m.tie_param('lengthscale')
if not isinstance(m.likelihood,GPy.inference.likelihoods.gaussian):
    m.approximate_likelihood()
print m.checkgrad()
# Optimize and plot
m.optimize()
#m.em(plot_all=False) # EM algorithm
m.plot()

print(m)
