# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import scipy as sp
import pdb, sys, pickle
import matplotlib.pylab as plt
import GPy
np.random.seed(1)

N = 100
# sample inputs and outputs
X = np.random.uniform(-np.pi,np.pi,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05
# Y += np.abs(Y.min()) + 0.5
Z = np.exp(Y)# Y**(1/3.0)

# rescaling targets?
Zmax = Z.max()
Zmin = Z.min()
Z = (Z-Zmin)/(Zmax-Zmin) - 0.5

m = GPy.models.warpedGP(X, Z, warping_terms = 2)
m.constrain_positive('(tanh_a|tanh_b|tanh_d|rbf|white|bias)')
m.randomize()
plt.figure()
plt.xlabel('predicted f(Z)')
plt.ylabel('actual f(Z)')
plt.plot(m.Y, Y, 'o', alpha = 0.5, label = 'before training')
m.optimize(messages = True)
plt.plot(m.Y, Y, 'o', alpha = 0.5, label = 'after training')
plt.legend(loc = 0)
m.plot_warping()
plt.figure()
plt.title('warped GP fit')
m.plot()

m1 = GPy.models.GP_regression(X, Z)
m1.constrain_positive('(rbf|white|bias)')
m1.randomize()
m1.optimize(messages = True)
plt.figure()
plt.title('GP fit')
m1.plot()
