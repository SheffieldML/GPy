import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

N = 400
M = 10
# sample inputs and outputs
X = np.random.uniform(-3.,3.,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05

"""Run a 1D example of a sparse GP regression."""
"""
rbf =  GPy.kern.rbf(1)
noise = GPy.kern.white(1)
kernel = rbf + noise
Z = np.random.uniform(-3,3,(M,1))
likelihood = GPy.likelihoods.Gaussian(Y)
m = GPy.models.sparse_GP(X, likelihood, kernel, Z)
m.scale_factor=10000
m.constrain_positive('(variance|lengthscale|precision)')
m.checkgrad(verbose=1)
m.optimize('tnc', messages = 1)
pb.figure()
m.plot()

variational = m
"""

# construct kernel
rbf =  GPy.kern.rbf(1)
noise = GPy.kern.white(1)
kernel = rbf + noise
#Z = np.random.uniform(-3,3,(M,1))
Z = variational.Z
likelihood = GPy.likelihoods.Gaussian(Y)
# create simple GP model
m = GPy.models.generalized_FITC(X, likelihood, kernel, Z=Z)
m.constrain_positive('(variance|lengthscale|precision)')
#m.constrain_fixed('iip')
m.checkgrad(verbose=1)
m.optimize('tnc', messages = 1)
#pb.figure()
#m.plot()
"""
Xnew = X.copy().flatten()
Xnew.sort()
Xnew = Xnew[:,None]
mean,var,low,up = m.predict(Xnew)
GPy.util.plot.gpplot(Xnew,mean,low,up)
fitc = m
"""
