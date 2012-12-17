# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
"""
Sparse Gaussian Processes regression with an RBF kernel, 
using the uncollapsed sparse GP (where the distribution of the 
inducing points is explicitley represented)
"""
import pylab as pb
import numpy as np
import GPy
np.random.seed(2)
pb.ion()
N = 500
M = 20

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05

kernel = GPy.kern.rbf(1) + GPy.kern.white(1)

# create simple GP model
m = GPy.models.uncollapsed_sparse_GP(X, Y, kernel=kernel, M=M)#, X_uncertainty=np.zeros_like(X)+0.01)

# contrain all parameters to be positive
m.ensure_default_constraints()
m.checkgrad()
# optimize and plot
m.plot()
