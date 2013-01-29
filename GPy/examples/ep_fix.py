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

model_type='Full'
inducing=4
"""Simple 1D classification example.
:param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
:param seed : seed value for data generation (default is 4).
:type seed: int
:param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
:type inducing: int
"""
data = GPy.util.datasets.toy_linear_1d_classification(seed=0)
likelihood = GPy.inference.likelihoods.probit(data['Y'][:, 0:1])

m = GPy.models.GP(data['X'],likelihood=likelihood)
#m = GPy.models.GP(data['X'],likelihood.Y)
m.ensure_default_constraints()

# Optimize and plot
if not isinstance(m.likelihood,GPy.inference.likelihoods.gaussian):
    m.approximate_likelihood()
#m.optimize()
m.EM()

print m.log_likelihood()
m.plot(samples=3)
print(m)
