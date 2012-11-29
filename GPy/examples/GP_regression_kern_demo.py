# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Simple one-dimensional Gaussian Processes with assorted kernel functions
"""
import pylab as pb
import numpy as np
import GPy

# sample inputs and outputs
D = 1
X = np.random.randn(10,D)*2
X = np.linspace(-1.5,1.5,5)[:,None]
X = np.append(X,[[5]],0)
Y = np.sin(np.pi*X/2) #+np.random.randn(X.shape[0],1)*0.05

models = [GPy.models.GP_regression(X,Y, k) for k in (GPy.kern.rbf(D), GPy.kern.Matern52(D), GPy.kern.Matern32(D), GPy.kern.exponential(D), GPy.kern.linear(D) + GPy.kern.white(D),  GPy.kern.bias(D) + GPy.kern.white(D))]

pb.figure(figsize=(12,8))
for i,m in enumerate(models):
    m.constrain_positive('')
    m.optimize()
    pb.subplot(3,2,i+1)
    m.plot()
    #pb.title(m.kern.parts[0].name)

GPy.util.plot.align_subplots(3,2,(-3,6),(-2.5,2.5))

pb.show()


