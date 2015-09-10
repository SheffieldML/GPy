# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
try:
    from matplotlib import pyplot as pb
except:
    pass
import GPy
pb.ion()
pb.close('all')

X1 = np.arange(3)[:,None]
X2 = np.arange(4)[:,None]
I1 = np.zeros_like(X1)
I2 = np.ones_like(X2)

_X = np.vstack([ X1, X2 ])
_I = np.vstack([ I1, I2 ])

X = np.hstack([ _X, _I ])

Y1 = np.sin(X1/8.)
Y2 = np.cos(X2/8.)

Bias = GPy.kern.Bias(1,active_dims=[0])
Coreg = GPy.kern.Coregionalize(1,2,active_dims=[1])
K = Bias.prod(Coreg,name='X')

#K.coregion.W = 0
#print K.coregion.W
#print Bias.K(_X,_X)
#print K.K(X,X)
#pb.matshow(K.K(X,X))

Mlist = [GPy.kern.Matern32(1,lengthscale=20.,name="Mat")]
kern = GPy.util.multioutput.LCM(input_dim=1,num_outputs=2,kernels_list=Mlist,name='H')
kern.B.W = 0
kern.B.kappa = 1.
#kern.B.W.fix()
#kern.B.kappa.fix()
#m = GPy.models.GPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2], kernel=kern)


Z1 = np.array([1.5,2.5])[:,None]

m = GPy.models.SparseGPCoregionalizedRegression(X_list=[X1], Y_list=[Y1], Z_list = [Z1], kernel=kern)
#m.optimize()
m.checkgrad(verbose=1)

"""
fig = pb.figure()
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
slices = GPy.util.multioutput.get_slices([Y1,Y2])
m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],ax=ax0)
#m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],ax=ax1)
"""



"""

X1 = 100 * np.random.rand(100)[:,None]
X2 = 100 * np.random.rand(100)[:,None]
#X1.sort()
#X2.sort()

Y1 = np.sin(X1/10.) + np.random.rand(100)[:,None]
Y2 = np.cos(X2/10.) + np.random.rand(100)[:,None]




Mlist = [GPy.kern.Matern32(1,lengthscale=20.,name="Mat")]
kern = GPy.util.multioutput.LCM(input_dim=1,num_outputs=12,kernels_list=Mlist,name='H')


m = GPy.models.GPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2], kernel=kern)
m.optimize()

fig = pb.figure()
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
slices = GPy.util.multioutput.get_slices([Y1,Y2])
m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],ax=ax0)
m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],ax=ax1)

"""
