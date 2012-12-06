# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import GPy
np.random.seed(1)
print "sparse GPLVM with RBF kernel"

N = 100
M = 4
Q = 1
D = 2
#generate GPLVM-like data
X = np.random.rand(N, Q)
k = GPy.kern.rbf(Q, 1.0, 2.0) + GPy.kern.white(Q, 0.00001)
K = k.K(X)
Y = np.random.multivariate_normal(np.zeros(N),K,D).T

m = GPy.models.sparse_GPLVM(Y, Q, M=M)
m.constrain_positive('(rbf|bias|noise)')
m.constrain_bounded('white', 1e-3, 0.1)
# m.plot()

pb.figure()
m.plot()
pb.title('PCA initialisation')
pb.figure()
m.optimize(messages = 1)
m.plot()
pb.title('After optimisation')
