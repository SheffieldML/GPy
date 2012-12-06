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

######################################
## 1 dimensional example

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05

# construct kernel
rbf =  GPy.kern.rbf(1)
noise = GPy.kern.white(1)
kernel = rbf + noise

# create simple GP model
m1 = GPy.models.sparse_GP_regression(X, Y, kernel, M=M)

# contrain all parameters to be positive
m1.constrain_positive('(variance|lengthscale|precision)')
#m1.constrain_positive('(variance|lengthscale)')
#m1.constrain_fixed('prec',10.)


#check gradient FIXME unit test please
m1.checkgrad()
# optimize and plot
m1.optimize('tnc', messages = 1)
m1.plot()
# print(m1)

######################################
## 2 dimensional example

# # sample inputs and outputs
# X = np.random.uniform(-3.,3.,(N,2))
# Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(N,1)*0.05

# # construct kernel
# rbf =  GPy.kern.rbf(2)
# noise = GPy.kern.white(2)
# kernel = rbf + noise

# # create simple GP model
# m2 = GPy.models.sparse_GP_regression(X,Y,kernel, M = 50)
# create simple GP model

# # contrain all parameters to be positive (but not inducing inputs)
# m2.constrain_positive('(variance|lengthscale|precision)')

# #check gradient FIXME unit test please
# m2.checkgrad()

# # optimize and plot
# pb.figure()
# m2.optimize('tnc', messages = 1)
# m2.plot()
# print(m2)
