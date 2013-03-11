# The detailed explanations of the commands used in this file can be found in the tutorial section

import pylab as pb
pb.ion()
import numpy as np
import GPy

X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05

kernel = GPy.kern.rbf(D=1, variance=1., lengthscale=1.)

m = GPy.models.GP_regression(X,Y,kernel)

print m
m.plot()

m.constrain_positive('')

m.unconstrain('')                            # Required to remove the previous constrains
m.constrain_positive('rbf_variance')
m.constrain_bounded('lengthscale',1.,10. )
m.constrain_fixed('noise',0.0025)

m.optimize()

m.optimize_restarts(Nrestarts = 10)

###########################
#  2-dimensional example  #
###########################

import pylab as pb
pb.ion()
import numpy as np
import GPy

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(50,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

# define kernel
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.white(2)

# create simple GP model
m = GPy.models.GP_regression(X,Y,ker)

# contrain all parameters to be positive
m.constrain_positive('')

# optimize and plot
pb.figure()
m.optimize('tnc', max_f_eval = 1000)

m.plot()
print(m)
