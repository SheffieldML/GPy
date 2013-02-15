# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
import GPy
np.random.seed(123344)

N = 10
M = 3
Q = 2
D = 4
#generate GPLVM-like data
X = np.random.rand(N, Q)
k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
K = k.K(X)
Y = np.random.multivariate_normal(np.zeros(N),K,D).T

k = GPy.kern.linear(Q, ARD = True) + GPy.kern.white(Q)
# k = GPy.kern.rbf(Q) + GPy.kern.rbf(Q) + GPy.kern.white(Q)
# k = GPy.kern.rbf(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
# k = GPy.kern.rbf(Q, ARD = False)  + GPy.kern.white(Q, 0.00001)

m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
m.constrain_positive('(rbf|bias|noise|white|S)')
# m.constrain_fixed('S', 1)

# pb.figure()
# m.plot()
# pb.title('PCA initialisation')
# pb.figure()
# m.optimize(messages = 1)
# m.plot()
# pb.title('After optimisation')
m.ensure_default_constraints()
m.randomize()
m.checkgrad(verbose = 1)
