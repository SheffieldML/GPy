import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

"""
Simple 1D classification example
:param seed : seed value for data generation (default is 4).
:type seed: int
"""
seed=10000

#data = GPy.util.datasets.toy_linear_1d_classification(seed=seed)
#X = data['X']
#Y = data['Y'][:, 0:1]
#Y[Y == -1] = 0



X = np.vstack((np.random.uniform(0,10,(10,1)),np.random.uniform(7,17,(10,1)),np.random.uniform(15,25,(10,1))))
Y = np.vstack((np.zeros((10,1)),np.ones((10,1)),np.zeros((10,1))))

# Kernel object
kernel = GPy.kern.rbf(1) + GPy.kern.white(1)

# Likelihood object
distribution = GPy.likelihoods.likelihood_functions.probit()
likelihood = GPy.likelihoods.EP(Y,distribution)

Z = np.random.uniform(X.min(),X.max(),(10,1))
#Z = np.array([0,20])[:,None]
print Z

# Model definition
m = GPy.models.generalized_FITC(X,likelihood=likelihood,kernel=kernel,Z=Z,normalize_X=False)
m.set('len',2.)

m.ensure_default_constraints()
# Optimize
#m.constrain_fixed('iip')
m.update_likelihood_approximation()
print m.checkgrad(verbose=1)
# Parameters optimization:
#m.optimize()
m.pseudo_EM() #FIXME

# Plot
pb.subplot(211)
m.plot_f()
pb.subplot(212)
m.plot()
print(m)
