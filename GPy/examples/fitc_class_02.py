import pylab as pb
import numpy as np
import GPy
pb.close('all')

seed=10000
"""Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

:param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
:param seed : seed value for data generation.
:type seed: int
:param inducing : number of inducing variables (only used for 'FITC' or 'DTC').
:type inducing: int
"""

data = GPy.util.datasets.crescent_data(seed=seed)

# Kernel object
kernel = GPy.kern.rbf(data['X'].shape[1]) + GPy.kern.white(data['X'].shape[1])

# Likelihood object
distribution = GPy.likelihoods.likelihood_functions.probit()
likelihood = GPy.likelihoods.EP(data['Y'],distribution)

sample = np.random.randint(0,data['X'].shape[0],10)
Z = data['X'][sample,:]
# create sparse GP EP model
#m = GPy.models.sparse_GP(data['X'],likelihood=likelihood,kernel=kernel,Z=Z)
m = GPy.models.generalized_FITC(data['X'],likelihood=likelihood,kernel=kernel,Z=Z)
m.ensure_default_constraints()
m.set('len',10.)

m.update_likelihood_approximation()

# optimize
m.optimize()
print(m)

# plot
m.plot()
fitc = m

pb.figure()
# Kernel object
kernel = GPy.kern.rbf(data['X'].shape[1]) + GPy.kern.white(data['X'].shape[1])

# Likelihood object
distribution = GPy.likelihoods.likelihood_functions.probit()
likelihood = GPy.likelihoods.EP(data['Y'],distribution)

sample = np.random.randint(0,data['X'].shape[0],10)
Z = data['X'][sample,:]
# create sparse GP EP model
m = GPy.models.sparse_GP(data['X'],likelihood=likelihood,kernel=kernel,Z=Z)
m.ensure_default_constraints()
m.set('len',10.)

m.update_likelihood_approximation()

# optimize
m.optimize()
print(m)

# plot
m.plot()
variational = m
