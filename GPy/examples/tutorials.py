# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Code of Tutorials
"""

import pylab as pb
pb.ion()
import numpy as np
import GPy

def tuto_GP_regression():
    """The detailed explanations of the commands used in this file can be found in the tutorial section"""

    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X) + np.random.randn(20,1)*0.05

    kernel = GPy.kern.rbf(input_dim=1, variance=1., lengthscale=1.)

    m = GPy.models.GPRegression(X, Y, kernel)

    print m
    m.plot()

    m.ensure_default_constraints() 
    m.constrain_positive('')

    m.unconstrain('')               # may be used to remove the previous constrains
    m.constrain_positive('.*rbf_variance')
    m.constrain_bounded('.*lengthscale',1.,10. )
    m.constrain_fixed('.*noise',0.0025)

    m.optimize()

    m.optimize_restarts(num_restarts = 10)

    #######################################################
    #######################################################
    # sample inputs and outputs
    X = np.random.uniform(-3.,3.,(50,2))
    Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

    # define kernel
    ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.white(2)

    # create simple GP model
    m = GPy.models.GPRegression(X, Y, ker)

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize and plot
    m.optimize('tnc', max_f_eval = 1000)
    m.plot()
    print(m)
    return(m)

def tuto_kernel_overview():
    """The detailed explanations of the commands used in this file can be found in the tutorial section"""
    ker1 = GPy.kern.rbf(1)  # Equivalent to ker1 = GPy.kern.rbf(input_dim=1, variance=1., lengthscale=1.)
    ker2 = GPy.kern.rbf(input_dim=1, variance = .75, lengthscale=2.)
    ker3 = GPy.kern.rbf(1, .5, .5)
    
    print ker2

    ker1.plot()
    ker2.plot()
    ker3.plot()

    k1 = GPy.kern.rbf(1,1.,2.)
    k2 = GPy.kern.Matern32(1, 0.5, 0.2)

    # Product of kernels
    k_prod = k1.prod(k2)                        # By default, tensor=False
    k_prodtens = k1.prod(k2,tensor=True)

    # Sum of kernels
    k_add = k1.add(k2)                          # By default, tensor=False
    k_addtens = k1.add(k2,tensor=True)    
    
    k1 = GPy.kern.rbf(1,1.,2)
    k2 = GPy.kern.periodic_Matern52(1,variance=1e3, lengthscale=1, period = 1.5, lower=-5., upper = 5)

    k = k1 * k2  # equivalent to k = k1.prod(k2)
    print k

    # Simulate sample paths
    X = np.linspace(-5,5,501)[:,None]
    Y = np.random.multivariate_normal(np.zeros(501),k.K(X),1)

    k1 = GPy.kern.rbf(1)
    k2 = GPy.kern.Matern32(1)
    k3 = GPy.kern.white(1)

    k = k1 + k2 + k3
    print k

    k.constrain_positive('.*var')
    k.constrain_fixed(np.array([1]),1.75)
    k.tie_params('.*len')
    k.unconstrain('white')
    k.constrain_bounded('white',lower=1e-5,upper=.5)
    print k
    
    k_cst = GPy.kern.bias(1,variance=1.)
    k_mat = GPy.kern.Matern52(1,variance=1., lengthscale=3)
    Kanova = (k_cst + k_mat).prod(k_cst + k_mat,tensor=True)
    print Kanova

    # sample inputs and outputs
    X = np.random.uniform(-3.,3.,(40,2))
    Y = 0.5*X[:,:1] + 0.5*X[:,1:] + 2*np.sin(X[:,:1]) * np.sin(X[:,1:])

    # Create GP regression model
    m = GPy.models.GPRegression(X, Y, Kanova)
    fig = pb.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    m.plot(ax=ax)
   
    pb.figure(figsize=(20,3))
    pb.subplots_adjust(wspace=0.5)
    axs = pb.subplot(1,5,1)
    m.plot(ax=axs)
    pb.subplot(1,5,2)
    pb.ylabel("=   ",rotation='horizontal',fontsize='30')
    axs = pb.subplot(1,5,3)
    m.plot(ax=axs, which_parts=[False,True,False,False])
    pb.ylabel("cst          +",rotation='horizontal',fontsize='30')
    axs = pb.subplot(1,5,4)
    m.plot(ax=axs, which_parts=[False,False,True,False])
    pb.ylabel("+   ",rotation='horizontal',fontsize='30')
    axs = pb.subplot(1,5,5)
    pb.ylabel("+   ",rotation='horizontal',fontsize='30')
    m.plot(ax=axs, which_parts=[False,False,False,True])

    m.ensure_default_constraints()
    return(m)


def model_interaction():
    X = np.random.randn(20,1)
    Y = np.sin(X) + np.random.randn(*X.shape)*0.01 + 5.
    k = GPy.kern.rbf(1) + GPy.kern.bias(1)
    m = GPy.models.GPRegression(X, Y, kernel=k)
    m.ensure_default_constraints()
    return m

