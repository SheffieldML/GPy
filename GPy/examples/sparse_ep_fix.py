# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
"""
Sparse Gaussian Processes regression with an RBF kernel
"""
import pylab as pb
import numpy as np
import GPy
np.random.seed(2)
pb.ion()
N = 500
M = 5

pb.close('all')
######################################
## 1 dimensional example

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(N,1))
#Y = np.sin(X)+np.random.randn(N,1)*0.05
F = np.sin(X)+np.random.randn(N,1)*0.05
Y = np.ones([F.shape[0],1])
Y[F<0] = -1
likelihood = GPy.inference.likelihoods.probit(Y)

# construct kernel
rbf =  GPy.kern.rbf(1)
noise = GPy.kern.white(1)
kernel = rbf + noise

# create simple GP model
m = GPy.models.sparse_GP(X,Y=None, kernel=kernel, M=M,likelihood= likelihood)
#m = GPy.models.sparse_GP(X, Y, kernel, M=M)

# contrain all parameters to be positive
m.ensure_default_constraints()
if not isinstance(m.likelihood,GPy.inference.likelihoods.gaussian):
    m.approximate_likelihood()
print m.checkgrad()
#check gradient FIXME unit test please
# optimize and plot
#m.optimize('tnc', messages = 1)
m.EM()
m.plot(samples=3,full_cov=False)
# print(m)

