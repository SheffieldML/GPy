# -*- coding: utf-8 -*-
"""
Testing state space related functions.
"""
import unittest
import numpy as np
import GPy
import GPy.models.state_space_model as SS_model
from .state_space_main_tests import generate_x_points, generate_sine_data, \
    generate_linear_data, generate_brownian_data, generate_linear_plus_sin


class StateSpaceKernelsTests(np.testing.TestCase):
    def setUp(self):
        pass
    
    def run_for_model(self, X, Y, ss_kernel, kalman_filter_type = 'regular',
                      use_cython=False, check_gradients=True, 
                      optimize = True, predict_X=None, 
                      compare_with_GP=True, gp_kernel=None, 
                      mean_compare_decimal=10, var_compare_decimal=7):
                          
        m1  = SS_model.StateSpace(X,Y, ss_kernel, 
                                kalman_filter_type=kalman_filter_type,
                                use_cython=use_cython)
        
        if check_gradients:
            self.assertTrue(m1.checkgrad())
        
        #import pdb; pdb.set_trace()
        
        if optimize:
            m1.optimize(optimizer='bfgs')
        
        if compare_with_GP and (predict_X is None):
            predict_X = X
            
        if (predict_X is not None):
            x_pred_reg_1 = m1.predict(predict_X)
            x_quant_reg_1 = m1.predict_quantiles(predict_X)
        
        if compare_with_GP:
            m2  = GPy.models.GPRegression(X,Y, gp_kernel)
            m2.optimize(optimizer='bfgs')
            #print(m2)
            
            x_pred_reg_2 = m2.predict(predict_X)
            x_quant_reg_2 = m2.predict_quantiles(predict_X)
        
            # Test values
            #print np.max(np.abs(x_pred_reg_1[0]-x_pred_reg_2[0]))
            np.testing.assert_almost_equal(np.max(np.abs(x_pred_reg_1[0]- \
                                x_pred_reg_2[0])), 0, decimal=mean_compare_decimal)
            
            # Test variances
            #print np.max(np.abs(x_pred_reg_1[1]-x_pred_reg_2[1]))
            
            np.testing.assert_almost_equal(np.max(np.abs(x_pred_reg_1[1]- \
                                x_pred_reg_2[1])), 0, decimal=var_compare_decimal)
            
    def test_Matern32_kernel(self,):
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Matern32(1,active_dims=[0,])
        gp_kernel = GPy.kern.Matern32(1,active_dims=[0,])
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True,
                           predict_X=X,
                           compare_with_GP=True,
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=10, var_compare_decimal=7)
                      
    def test_Matern52_kernel(self,):
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Matern52(1,active_dims=[0,])
        gp_kernel = GPy.kern.Matern52(1,active_dims=[0,])
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True, 
                           optimize = True, predict_X=X, 
                           compare_with_GP=True, gp_kernel=gp_kernel, 
                           mean_compare_decimal=8, var_compare_decimal=7)
                      
    def test_RBF_kernel(self,):
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_RBF(1,active_dims=[0,])
        gp_kernel = GPy.kern.RBF(1,active_dims=[0,])
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True,
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=1, var_compare_decimal=1)
                      
    def test_periodic_kernel(self,):
        np.random.seed(322) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
        ss_kernel.lengthscales.constrain_bounded(0.25, 1000)
        ss_kernel.wavelengths.constrain_bounded(0.15, 100)
    
        gp_kernel = GPy.kern.StdPeriodic(1,active_dims=[0,])
        gp_kernel.lengthscales.constrain_bounded(0.25, 1000)
        gp_kernel.wavelengths.constrain_bounded(0.15, 100)        
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True,
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=4, var_compare_decimal=4)                      
                            
    def test_quasi_periodic_kernel(self,):
        np.random.seed(329) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Matern32(1)*GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
        ss_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
        ss_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)
    
        gp_kernel = GPy.kern.Matern32(1)*GPy.kern.StdPeriodic(1,active_dims=[0,])
        gp_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
        gp_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)        
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True,
                            predict_X=X, 
                            gp_kernel=gp_kernel, 
                            mean_compare_decimal=1, var_compare_decimal=2)   

    def test_linear_kernel(self,):
        
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_linear_data(x_points=None, tangent=2.0, add_term=20.0, noise_var=2.0,
                    plot = False, points_num=50, x_interval = (0, 20), random=True)
                    
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Linear(1,X,active_dims=[0,]) + GPy.kern.sde_Bias(1, active_dims=[0,])
        gp_kernel = GPy.kern.Linear(1, active_dims=[0,]) + GPy.kern.Bias(1, active_dims=[0,])
        
        self.run_for_model(X, Y, ss_kernel, check_gradients= False, 
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=5, var_compare_decimal=5)

    def test_brownian_kernel(self,):
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_brownian_data(x_points=None, kernel_var=2.0, noise_var = 0.1,
                    plot = False, points_num=50, x_interval = (0, 20), random=True)
                    
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Brownian()
        gp_kernel = GPy.kern.Brownian()
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True, 
                            predict_X=X, 
                            gp_kernel=gp_kernel, 
                            mean_compare_decimal=10, var_compare_decimal=7)
                      
    def test_exponential_kernel(self,):
        np.random.seed(234) # seed the random number generator
        (X,Y) = generate_linear_data(x_points=None, tangent=1.0, add_term=20.0, noise_var=2.0,
                    plot = False, points_num=50, x_interval = (0, 20), random=True)
                    
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        ss_kernel = GPy.kern.sde_Exponential(1, active_dims=[0,])
        gp_kernel = GPy.kern.Exponential(1, active_dims=[0,])
        
        self.run_for_model(X, Y, ss_kernel, check_gradients=True, 
                      predict_X=X, 
                      gp_kernel=gp_kernel, 
                      mean_compare_decimal=5, var_compare_decimal=6)                                                      

    def test_kernel_addition(self,):
        np.random.seed(329) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
                        
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        def get_new_kernels():
            ss_kernel = GPy.kern.sde_Matern32(1) + GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
            ss_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
            ss_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)
        
            gp_kernel = GPy.kern.Matern32(1) + GPy.kern.StdPeriodic(1,active_dims=[0,])
            gp_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
            gp_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)
            
            return ss_kernel, gp_kernel
        
        # Cython is available only with svd.
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=True, check_gradients=True,
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=0, var_compare_decimal=-1)
        
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'regular',
                           use_cython=False, check_gradients=True,
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=4, var_compare_decimal=3)
                           
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=False, check_gradients=True,
                           predict_X=X, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=0, var_compare_decimal=-1)
        
        
        
    def test_kernel_multiplication(self,):
        np.random.seed(329) # seed the random number generator
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=10.0, noise_var=2.0,
                        plot = False, points_num=50, x_interval = (0, 20), random=True)
                        
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1)
        
        def get_new_kernels():
            ss_kernel = GPy.kern.sde_Matern32(1)*GPy.kern.sde_Matern52(1)
            gp_kernel = GPy.kern.Matern32(1)*GPy.kern.sde_Matern52(1)
        
            return ss_kernel, gp_kernel
        
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=True, check_gradients=True,
                            predict_X=X, 
                            gp_kernel=gp_kernel, 
                            mean_compare_decimal=-1, var_compare_decimal=0)  
        
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'regular',
                           use_cython=False, check_gradients=True,
                            predict_X=X, 
                            gp_kernel=gp_kernel, 
                            mean_compare_decimal=-1, var_compare_decimal=0)
                            
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X, Y, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=False, check_gradients=True,
                            predict_X=X, 
                            gp_kernel=gp_kernel, 
                            mean_compare_decimal=-1, var_compare_decimal=0)

    def test_forecast(self,):
        """
        Test time series forecasting.
        """
        
        # Generate data ->
        np.random.seed(339) # seed the random number generator
        #import pdb; pdb.set_trace()
        (X,Y) = generate_sine_data(x_points=None, sin_period=5.0, sin_ampl=5.0, noise_var=2.0,
                        plot = False, points_num=100, x_interval = (0, 40), random=True)
                        
        (X1,Y1) = generate_linear_data(x_points=X, tangent=1.0, add_term=20.0, noise_var=0.0,
                    plot = False, points_num=100, x_interval = (0, 40), random=True)
                        
        Y = Y + Y1

        X_train = X[X <= 20]
        Y_train = Y[X <= 20]        
        X_test = X[X > 20]
        Y_test = Y[X > 20]
        
        X.shape = (X.shape[0],1); Y.shape = (Y.shape[0],1) 
        X_train.shape = (X_train.shape[0],1); Y_train.shape = (Y_train.shape[0],1) 
        X_test.shape = (X_test.shape[0],1); Y_test.shape = (Y_test.shape[0],1) 
        # Generate data <-
        
        #import pdb; pdb.set_trace()
        
        def get_new_kernels():
            periodic_kernel = GPy.kern.StdPeriodic(1,active_dims=[0,])
            gp_kernel = GPy.kern.Linear(1, active_dims=[0,]) + GPy.kern.Bias(1, active_dims=[0,]) + periodic_kernel
            gp_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
            gp_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)
        
            periodic_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
            ss_kernel = GPy.kern.sde_Linear(1,X,active_dims=[0,]) + \
                GPy.kern.sde_Bias(1, active_dims=[0,]) + periodic_kernel
    
            ss_kernel.std_periodic.lengthscales.constrain_bounded(0.25, 1000)
            ss_kernel.std_periodic.wavelengths.constrain_bounded(0.15, 100)
            
            return ss_kernel, gp_kernel
        
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X_train, Y_train, ss_kernel, kalman_filter_type = 'regular',
                           use_cython=False, check_gradients=True,
                           predict_X=X_test, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=0, var_compare_decimal=0)
        
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X_train, Y_train, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=False, check_gradients=False,
                           predict_X=X_test, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=0, var_compare_decimal=-1)
                           
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(X_train, Y_train, ss_kernel, kalman_filter_type = 'svd',
                           use_cython=True, check_gradients=False,
                           predict_X=X_test, 
                           gp_kernel=gp_kernel, 
                           mean_compare_decimal=0, var_compare_decimal=-1) 
        
if __name__ == "__main__":
    print("Running state-space inference tests...")
    unittest.main()