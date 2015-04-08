import GPy
import numpy as np
import matplotlib.pyplot as plt

import GPy.models.state_space_new as SS_new

#X = np.linspace(0, 10, 2000)[:, None]
#Y = np.sin(X) + np.random.randn(*X.shape)*0.1

# Need to run these lines when X and Y are imported ->
X.shape = (X.shape[0],1)
Y.shape = (Y.shape[0],1)
# Need to run these lines when X and Y are imported <-

## Generation of minimal example data ->
#X = np.random.rand(3)
#sort_index = np.argsort(X)
#X = X[sort_index]; X.shape = (X.shape[0],1)
#Y = np.sin(10*X) + np.random.randn(*X.shape)*0.1
## Generation of minimal example data <-

#plt.figure()
#plt.plot( X, Y)
#plt.show()

#kernel = GPy.kern.Matern32(X.shape[1])
#m = GPy.models.StateSpace(X,Y, kernel)
#
#print m
##
#m.optimize(optimizer='bfgs',messages=True)
##
#print m

kernel1 = GPy.kern.Matern32(X.shape[1])
m1  = GPy.models.GPRegression(X,Y, kernel1)

print m1
m1.optimize(optimizer='bfgs',messages=True)

print m1

kernel2 = GPy.kern.Matern32(X.shape[1])
m2  = SS_new.StateSpace(X,Y, kernel2)

print m2

m2.optimize(optimizer='bfgs',messages=True)

print m2

