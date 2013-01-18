# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Simple Gaussian Processes regression with an RBF kernel
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')


######################################
## 1 dimensional example

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X)+np.random.randn(20,1)*0.05

# define kernel
ker = GPy.kern.rbf(1,ARD=False) + GPy.kern.white(1)

# create simple GP model
m = GPy.models.GP_regression(X,Y,ker)

# contrain all parameters to be positive
m.constrain_positive('')

# optimize and plot
m.optimize('tnc', max_f_eval = 1000)
m.plot()
print(m)


######################################
## 2 dimensional example

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(40,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(40,1)*0.05

# define kernel
ker = GPy.kern.rbf(2,ARD=True) + GPy.kern.white(2)

# create simple GP model
m = GPy.models.GP_regression(X,Y,ker)

# contrain all parameters to be positive
m.constrain_positive('')
# optimize and plot
pb.figure()
m.optimize('tnc', max_f_eval = 1000)
m.plot()
print(m)


