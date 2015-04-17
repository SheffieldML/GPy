# -*- coding: utf-8 -*-
"""
Alexander Grigorevskiy, 2015
"""

import numpy as np
import scipy as sp
from scipy import linalg
import GPy
import matplotlib.pyplot as plt

def normalize(D):
    """
    Function which mormalizes the data to zero mean unit variance.
    D - column vector, or column-wise matrix
    """
    
    
    n_rows = D.shape[0]    
    means = np.nanmean(D, axis= 0)    
    tmp = D - np.tile( means, (n_rows,1) ) # temporary result. Data with substracted mean                                                                         
  
    stds = np.nanstd(tmp,axis=0, ddof=1 ) # one degree of freadom as matlab default
  
    result = np.divide( tmp, stds ) 
    
    return (result,means,stds)
    
def generate_data(func_type_num, gd_plot=False):
    """
    Function generates data.
    """
    
    func_types = [ 'linear', 'sin' ]
    func_type = func_types[func_type_num]   

    
    time_int = np.array((0, 20)) # time interval
    points_num = 300
    
    t_points = np.random.rand(points_num) * ( time_int[1] - time_int[0] ) + time_int[0]
    t_points = np.sort( t_points )
    
    t_regular_points = np.linspace(time_int[0], time_int[1], num=points_num )    
    
    noise_sigma = 1
    # linear function ->
    def linear_function(tt):
        a = 2; b= 1
        ff = lambda tt:  a*tt + b
        return ff(tt)
    # linear function <-

    # sin function ->
    def sin_function(tt):
        a = 2; b= 10; # a-period
        ff = lambda tt:  b * np.sin( 2*np.pi/a * tt )
        return ff(tt)
    # sin function <-

    if func_type == 'linear':
        xx_func = linear_function( t_points ) + np.random.randn( len(t_points) ) * noise_sigma
        xx_reg_func = linear_function( t_regular_points ) 
       
    elif func_type == 'sin':
        xx_func = sin_function( t_points ) + np.random.randn( len(t_points) ) * noise_sigma
        xx_reg_func = sin_function( t_regular_points )
         
    if gd_plot:
        plt.figure(1)  
        plt.plot( t_points, xx_func, 'b.' )
        plt.plot( t_regular_points, xx_reg_func, '-b' )
        plt.title('Generated Samples and Underlying Function')
        plt.show()
        #plt.close()
    return  (t_points,  xx_func)

def generate_data2D(func_type_num, gd_plot=False):
    """
    Function generates data.
    """
    
    func_type = 'sin'

    
    time_int = np.array((0, 20)) # time interval
    points_num = 300
    
    t_points_x = np.random.rand(points_num) * ( time_int[1] - time_int[0] ) + time_int[0]
    t_points_y = np.random.rand(points_num) * ( time_int[1] - time_int[0] ) + time_int[0]
    
    t_regular_points_x = np.linspace(time_int[0], time_int[1], num=points_num )    
    t_regular_points_y = np.linspace(time_int[0], time_int[1], num=points_num )
    
    noise_sigma = 1
   
    # sin function ->
    def sin_function(tt1,tt2):
        a1 = 2; a2=8; b= 10; # a - wavelengths
        ff = lambda tt1, tt2:  b * np.sin( 2*np.pi/a1 * tt1 + 2*np.pi/a2 * tt2)
        return ff(tt1,tt2)
    # sin function <-

   
      
    if func_type == 'sin':
        xx_func = sin_function( t_points_x, t_points_y) + np.random.randn( len(t_points_x) ) * noise_sigma
        xx_reg_func = sin_function( t_regular_points_x, t_regular_points_y)
         
#    if gd_plot:
#        plt.figure(1)
#        xx,yy = np.meshgrid(t_regular_points_x, t_regular_points_x)
#        
#        plt.title('Generated Samples and Underlying Function')
#        plt.show()
#        #plt.close()
    return  (t_points_x, t_points_y, xx_func)
    
def test_simple():
    """
    Simple test
    """
    
    (xx, yy)  =  generate_data( 1, gd_plot=True )
    xx = xx[:, np.newaxis ]; 
    yy = yy[:, np.newaxis ];  
    
    input_dim = 1
    xx1 = xx # np.hstack( (xx, xx**2))    # possible input data transformation
    
    (xx_norm, xx_mean, xx_std) = normalize( xx1) # normalize in place   
    (yy_norm, yy_mean, yy_std) = normalize( yy ) # normalize in place
   

    #periodic_kernel = GPy.kern.Linear(input_dim)
    periodic_kernel = GPy.kern.StdPeriodic(input_dim)
    
    KK = periodic_kernel
 
   # Show sample paths ->    
    Cov = KK.K( xx_norm, xx_norm); # KK.plot();GPy.kern.periodic_Matern52
    #plt.figure(1)
    plt.matshow(Cov); plt.colorbar()
    plt.title('Covariance BEFORE optimization')
    #plt.show()        
    
    zz = np.random.multivariate_normal( np.zeros(xx.shape[0]), Cov, 5 )
    plt.figure(2)  
    for i in range(5):
        plt.plot(xx_norm,zz[i,:])
    plt.title('Covariance sample paths')
    plt.show()
    # Show sample paths <-
    
    gpy_reg = GPy.models.GPRegression( xx_norm, yy, KK )
    print "Before Optimization ->\n"
    print gpy_reg
    gpy_reg.plot()
    
    gpy_reg.optimize_restarts(num_restarts = 3, robust=True, parallel=False, num_processes=4)    
    
    print "After Optimization ->\n"
    print gpy_reg
    gpy_reg.plot()
     
    return gpy_reg
    
def test_2D_sine():
    """
    Test function of 2D argument.
    """
    
    (xx, yy,zz)  =  generate_data2D( 1, gd_plot=True )
    xx = xx[:, np.newaxis ]; 
    yy = yy[:, np.newaxis ];  
    zz = zz[:, np.newaxis ];  
    
    input_dim = 1
    xx1 = np.hstack( (xx, yy))    # possible input data transformation
    
    (xx_norm, xx_mean, xx_std) = normalize( xx1) # normalize in place   
    (yy_norm, yy_mean, yy_std) = normalize( yy ) # normalize in place
   

    #periodic_kernel = GPy.kern.Linear(input_dim)
    periodic_kernel = GPy.kern.StdPeriodic(input_dim,active_dims=[0,])
    
    KK = periodic_kernel
 
   # Show sample paths ->    
    Cov = KK.K( xx_norm, xx_norm); # KK.plot();GPy.kern.periodic_Matern52
    #plt.figure(1)
    plt.matshow(Cov); plt.colorbar()
    plt.title('Covariance BEFORE optimization')
    #plt.show()        
    
    zz = np.random.multivariate_normal( np.zeros(xx.shape[0]), Cov, 5 )
    plt.figure(2)  
    for i in range(5):
        plt.plot(xx_norm,zz[i,:])
    plt.title('Covariance sample paths')
    plt.show()
    # Show sample paths <-
    
    gpy_reg = GPy.models.GPRegression( xx_norm, yy, KK )
    print "Before Optimization ->\n"
    print gpy_reg
    gpy_reg.plot()
    gpy_reg.optimize_restarts(num_restarts = 3, robust=True, parallel=False, num_processes=4)    
    
    print "After Optimization ->\n"
    print gpy_reg
    gpy_reg.plot()
    
    return gpy_reg
    
    
if __name__ == '__main__':   
    reg = test_simple()
    #reg = test_2D_sine()