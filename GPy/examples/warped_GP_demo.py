# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import scipy as sp
import pdb, sys, pickle
import matplotlib.pylab as plt
import GPy
np.random.seed(2)

N = 120
# sample inputs and outputs
X = np.random.uniform(-np.pi,np.pi,(N,1))
Y = np.sin(X)+np.random.randn(N,1)*0.05
Y += np.abs(Y.min()) + 0.5
Z = np.exp(Y)#Y**(1/3.0)
Zmax = Z.max()
Zmin = Z.min()
Z = (Z-Zmin)/(Zmax-Zmin) - 0.5
train = range(X.shape[0])[:100]
test = range(X.shape[0])[100:]

kernel = GPy.kern.rbf(1) + GPy.kern.bias(1)
m = GPy.models.warpedGP(X[train], Z[train], kernel=kernel, warping_terms = 2)
m.constrain_positive('(tanh_a|tanh_b|rbf|noise|bias)')
m.constrain_fixed('tanh_d', 1.0)
m.randomize()
plt.figure()
plt.xlabel('predicted f(Z)')
plt.ylabel('actual f(Z)')
plt.plot(m.likelihood.Y, Y[train], 'o', alpha = 0.5, label = 'before training')
m.optimize(messages = True)
# m.optimize_restarts(4, parallel = True, messages = True)
plt.plot(m.likelihood.Y, Y[train], 'o', alpha = 0.5, label = 'after training')
plt.legend(loc = 0)
m.plot_warping()
plt.figure()
plt.title('warped GP fit')
m.plot()
m.optimize(messages=1)
plt.figure(); plt.plot(m.predict(X[test])[0].flatten(), Y[test].flatten(), 'x'); plt.title('prediction in unwarped space')
m.predict_in_warped_space = True
plt.figure(); plt.plot(m.predict(X[test])[0].flatten(), Z[test].flatten(), 'x'); plt.title('prediction in warped space')

m1 = GPy.models.GP_regression(X[train], Z[train])
m1.constrain_positive('(rbf|noise|bias)')
m1.randomize()
m1.optimize(messages = True)
plt.figure()
plt.title('GP fit')
m1.plot()
