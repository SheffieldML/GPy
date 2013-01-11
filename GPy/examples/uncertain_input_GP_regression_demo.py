# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')


# sample inputs and outputs
S = np.ones((20,1))
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X)+np.random.randn(20,1)*0.05

k = GPy.kern.rbf(1) + GPy.kern.white(1)

# create simple GP model
m = GPy.models.sparse_GP_regression(X,Y,X_uncertainty=S,kernel=k)

# contrain all parameters to be positive
m.constrain_positive('(variance|prec)')

# optimize and plot
m.optimize('tnc', max_f_eval = 1000, messages=1)
m.plot()
print(m)
