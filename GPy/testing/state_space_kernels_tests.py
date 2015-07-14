# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:02:25 2015

@author: alex
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import GPy.models.state_space_new as SS_new
from test_periodic_kernel import generate_sine_data 
from test_periodic_kernel import generate_linear_data
from test_periodic_kernel import generate_brownian_data

def test_matern32(X=None,Y=None):
    """
    Test Matern 32 Covariance Function
    """
    np.random.seed(234) # seed the random number generator
       
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    matern32_kernel = GPy.kern.Matern32(1,active_dims=[0,])
    
    kernel1 = matern32_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Matern32 Kernel Model')
    plt.show()
    
    sde_Matern32_kernel = GPy.kern.sde_Matern32(1,active_dims=[0,])
    
    kernel2 = sde_Matern32_kernel
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Matern32 Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_matern52(X=None,Y=None):
    """
    Test Matern 52 Covariance Function
    """  
    np.random.seed(234) # seed the random number generator
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    matern52_kernel = GPy.kern.Matern52(1,active_dims=[0,])
    
    kernel1 = matern52_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X,x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Matern52 Kernel Model')
    plt.show()
    
    sde_Matern52_kernel = GPy.kern.sde_Matern52(1,active_dims=[0,])
    
    kernel2 = sde_Matern52_kernel
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Matern52 Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_RBF(X=None,Y=None):
    """
    Test  RBF Covariance Function
    """  
    np.random.seed(234) # seed the random number generator
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    Gaussian_kernel = GPy.kern.RBF(1,active_dims=[0,])
    
    kernel1 = Gaussian_kernel
    kernel1.lengthscale = 1.0
    kernel1.variance = 1.0
    
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular RBF Kernel Model')
    plt.show()
    
    sde_RBF_kernel = GPy.kern.sde_RBF(1,active_dims=[0,])
    
    kernel2 = sde_RBF_kernel
    kernel2.lengthscale = 1.0
    kernel2.variance = 1.0
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space RBF Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y
    
def test_kernel_multiplication(X=None,Y=None):
    """
    Test State-Space multiplication of kernels
    """
    #np.random.seed(234) # seed the random number generator !!! Error occured
    np.random.seed(234)
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    matern32_kernel = GPy.kern.Matern32(1,active_dims=[0,])
    matern52_kernel = GPy.kern.Matern52(1,active_dims=[0,])
    
    kernel1 = matern32_kernel*matern52_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.mul.Mat32.variance.constrain_fixed(1.0)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Matern52*Matern32 multiplication Kernel Model')
    plt.show()
    
    sde_Matern32_kernel = GPy.kern.sde_Matern32(1,active_dims=[0,])
    sde_Matern52_kernel = GPy.kern.sde_Matern52(1,active_dims=[0,])
    
    kernel2 = sde_Matern32_kernel*sde_Matern52_kernel
    #kernel2.prod.    
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.mul.Mat32.variance.constrain_fixed(1.0)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Matern52*Matern32 multiplication Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y
    
def test_periodic(X=None,Y=None):
    """
    Test regular periodic covariance and State-Space representation.
    """
    np.random.seed(235) # seed the random number generator
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    periodic_kernel = GPy.kern.StdPeriodic(1,active_dims=[0,])
    #quasi_periodic_kernel = GPy.kern.Matern32(1) + periodic_kernel
    kernel1 = periodic_kernel
    kernel1.lengthscales.constrain_bounded(0.25, 1000)
    kernel1.wavelengths.constrain_bounded(0.15, 100)
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r')
    plt.plot( X, x_quant_reg[1], '--r')
    plt.title('Regular Periodic Kernel Model')
    plt.show()
    
    periodic_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
    
    
    kernel2 = periodic_kernel
    kernel2.lengthscales.constrain_bounded(0.25, 1000)
    kernel2.wavelengths.constrain_bounded(0.15, 100)
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Periodic Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_add_and_periodic(X=None,Y=None):
    """
    Test regular periodic covariance and State-Space representation.
    """
    np.random.seed(234) # seed the random number generator
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    periodic_kernel = GPy.kern.StdPeriodic(1,active_dims=[0,])
    add_periodic_kernel = GPy.kern.Matern32(1) + periodic_kernel
    kernel1 = add_periodic_kernel
    kernel1.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
    kernel1.std_periodic.wavelengths.constrain_bounded(0.15, 100)
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r')
    plt.plot( X, x_quant_reg[0], '--r')
    plt.plot( X, x_quant_reg[1], '--r')
    plt.title('Sum of Matern32 and Regular Periodic Kernel Model')
    plt.show()
    
    periodic_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
    add_periodic_kernel = GPy.kern.sde_Matern32(1) + periodic_kernel
    
    kernel2 = add_periodic_kernel
    kernel2.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
    kernel2.std_periodic.wavelengths.constrain_bounded(0.15, 100)
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r')
    plt.plot( X, x_quant_ss[0], '--r')
    plt.plot( X, x_quant_ss[1], '--r')
    plt.title('State-Space of Sum of Matern32 and Regular Periodic')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_quasi_periodic(X=None,Y=None):
    """
    Test regular periodic covariance and State-Space representation.
    """
    np.random.seed(234) # seed the random number generator
    
    if (X is None) or (Y is None):
        # Sine data ->
        (X,Y) = generate_sine_data(x_points=None, sin_period=2.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=300, x_interval = (0, 20), random=True)
        # Sine data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    periodic_kernel = GPy.kern.StdPeriodic(1,active_dims=[0,])
    quasi_periodic_kernel = GPy.kern.Matern32(1) * periodic_kernel
    kernel1 = quasi_periodic_kernel
    #kernel1.Mat32.variance = 2
    kernel1.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
    kernel1.std_periodic.wavelengths.constrain_bounded(0.15, 100)
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    
    m1.optimize(optimizer='bfgs',messages=True )
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular quasi-periodic (with Matern32)')
    plt.show()
    
    periodic_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
    quasi_periodic_kernel = GPy.kern.sde_Matern32(1) * periodic_kernel
    
    kernel2 = quasi_periodic_kernel
    #kernel2.Mat32.variance = 2
    kernel2.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
    kernel2.std_periodic.wavelengths.constrain_bounded(0.15, 100)
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space quasi-periodic (with Matern32)')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2) 
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )  
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_linear(X=None,Y=None):
    """
    Test Linear Covariance Function (same as Bayesian linear regression)
    """
    
    np.random.seed(234) # seed the random number generator
       
    if (X is None) or (Y is None):
        # Linear data ->
        (X,Y) = generate_linear_data(x_points=None, tangent=2.0, add_term=20.0, noise_var=2.0,
                    plot = False, points_num=300, x_interval = (0, 20), random=True)
                        
        # Linear data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    linear_kernel = GPy.kern.Linear(1, active_dims=[0,]) + GPy.kern.Bias(1, active_dims=[0,]) #+\
                    #GPy.kern.White(1,active_dims=[0,])
    kernel1 = linear_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    #import pdb; pdb.set_trace()
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Linear Kernel Model')
    plt.show()
    
    sde_Linear_kernel = GPy.kern.sde_Linear(1,X,active_dims=[0,]) + GPy.kern.sde_Bias(1, active_dims=[0,]) #+\
                        #GPy.kern.sde_White(1,active_dims=[0,])
    
    kernel2 = sde_Linear_kernel
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Linear Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y
    
def test_Brownian(X=None,Y=None):
    """
    Test Brownian Covariance Function.
    """
    
    np.random.seed(234) # seed the random number generator
       
    if (X is None) or (Y is None):
        # Brownian data ->
        (X,Y) = generate_brownian_data(x_points=None, kernel_var=2.0, noise_var = 0.1,
                    plot = False, points_num=300, x_interval = (0, 20), random=True)
                    
        # Brownian data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1) 
    
    brownian_kernel = GPy.kern.Brownian()
    
    kernel1 = brownian_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    #import pdb; pdb.set_trace()
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Brownian Kernel Model')
    plt.show()
    
    sde_brownian_kernel = GPy.kern.sde_Brownian()
    
    kernel2 = sde_brownian_kernel
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Brownian Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y

def test_exponential(X=None,Y=None):
    """
    Test Exponential Covariance Function.
    """

    np.random.seed(234) # seed the random number generator
       
    if (X is None) or (Y is None):
        # Linear data ->
        (X,Y) = generate_linear_data(x_points=None, tangent=0.0, add_term=20.0, noise_var=2.0,
                    plot = False, points_num=300, x_interval = (0, 20), random=True)
                        
        # Linear data <-                
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)    
    
    exp_kernel = GPy.kern.Exponential(1, active_dims=[0,])
    kernel1 = exp_kernel
  
    m1  = GPy.models.GPRegression(X,Y, kernel1)
    print(m1)
    #import pdb; pdb.set_trace()
    m1.optimize(optimizer='bfgs',messages=True)
    x_pred_reg = m1.predict(X)
    x_quant_reg = m1.predict_quantiles(X)
    
    plt.figure(1)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_reg[0], '-r' )
    plt.plot( X, x_quant_reg[0], '--r' )
    plt.plot( X, x_quant_reg[1], '--r' )
    plt.title('Regular Exponential Kernel Model')
    plt.show()
    
    sde_exp_kernel = GPy.kern.sde_Exponential(1, active_dims=[0,])
    
    kernel2 = sde_exp_kernel
    
    m2  = SS_new.StateSpace(X,Y, kernel2)
    print(m2)
    
    m2.optimize(optimizer='bfgs',messages=True) # max_iters=5
    x_pred_ss = m2.predict(X)
    x_quant_ss = m2.predict_quantiles(X)
    
    plt.figure(2)  
    plt.plot( X, Y, 'b.' )
    plt.plot( X, x_pred_ss[0], '-r' )
    plt.plot( X, x_quant_ss[0], '--r' )
    plt.plot( X, x_quant_ss[1], '--r' )
    plt.title('State-Space Exponential Kernel Model')
    plt.show()
    
    print(m1)
    print(m1.objective_function_gradients())
    print(m2)
    print(m2.objective_function_gradients())
    print("Maximum absolute diff in prediction: %s" % ( np.max(np.abs(x_pred_reg[0]-x_pred_ss[0])), ) )    
    print("Maximum absolute diff in predicted variance: %s" % ( np.max(np.abs(x_pred_reg[1]-x_pred_ss[1])), ) )
    
    return X,Y
    
#def compare_matlab_matrices():
#    """
#
#    """
#    loading_folder = '/home/agrigori/Programming/python/my_utils/'    
#    
#    import scipy as sp
#    matlab_F = sp.io.loadmat(loading_folder + 'matlab_F.mat')['F']
#    matlab_L = sp.io.loadmat(loading_folder + 'matlab_L.mat')['L']    
#    matlab_Qc = sp.io.loadmat(loading_folder + 'matlab_Qc.mat')['Qc']
#    matlab_H = sp.io.loadmat(loading_folder + 'matlab_H.mat')['H']    
#    matlab_Pinf = sp.io.loadmat(loading_folder + 'matlab_Pinf.mat')['Pinf']     
#    
#    matlab_dF = sp.io.loadmat(loading_folder + 'matlab_dF.mat')['dF']
#    matlab_dQc = sp.io.loadmat(loading_folder + 'matlab_dQc.mat')['dQc']    
#    matlab_dPinf = sp.io.loadmat(loading_folder + 'matlab_dPinf.mat')['dPinf']
#    
#    matlab_T = sp.io.loadmat(loading_folder + 'matlab_T.mat')['T']  
#    
#    sde_RBF_kernel = GPy.kern.sde_RBF(1,active_dims=[0,])
#    
#    kernel2 = sde_RBF_kernel
#    kernel2.variance = 0016.0
#    kernel2.lengthscale = 133.0
#    
#    
#    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0,T) = sde_RBF_kernel.sde()
#    
#    print('F max diff:', np.max(np.abs((F - matlab_F))) )
#    print('L max diff:', np.max(np.abs((L - matlab_L))) )
#    print('Qc max diff:', np.max(np.abs((Qc - matlab_Qc))) )
#    print('H max diff:', np.max(np.abs((H - matlab_H))) )
#    print('Pinf max diff:', np.max(np.abs((Pinf - matlab_Pinf))) )
#    
#    print('dF max diff:', np.max(np.abs((dF - matlab_dF))) )
#    print('dQc max diff:', np.max(np.abs((dQc - matlab_dQc))) )
#    print('dPinf max diff:', np.max(np.abs((dPinf - matlab_dPinf))) )
#    
#    print('T max diff:', np.max(np.abs((T - matlab_T))) )
    
    
    
    
if __name__ == '__main__':
    #X,Y = test_periodic() #(X,Y)
    #X,Y = test_add_and_periodic() #(X,Y) #(X=None,Y=None)
    #X,Y = test_quasi_periodic()
    #X,Y = test_matern32()
    #X,Y = test_matern52()
    #X,Y = test_kernel_multiplication() # badd
    #X,Y = test_linear()
    #test_Brownian()
    #test_exponential()
    X,Y = test_RBF()