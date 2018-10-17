# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class MFtests(unittest.TestCase):
    def test_simple_mean_function(self):
        """
        The simplest possible mean function. No parameters, just a simple Sinusoid.
        """
        #create  simple mean function
        mf = GPy.core.Mapping(1,1)
        mf.f = np.sin
        mf.update_gradients = lambda a,b: None

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape)

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(-1,10,50).reshape(-1,1)
        
        Y = 3-np.abs((X-6))
        Y += .5*np.cos(3*X) + 0.3*np.random.randn(*X.shape) 

        mf = GPy.mappings.PiecewiseLinear(1, 1, [-1,1], [9,2])

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function_composition(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape) + 3*X

        mf = GPy.mappings.Compound(GPy.mappings.Linear(1,1), 
                                   GPy.mappings.Kernel(1, 1, np.random.normal(0,1,(1,1)), 
                                                       GPy.kern.RBF(1))
                                   )

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_parametric_mean_function_additive(self):
        """
        A linear mean function with parameters that we'll learn alongside the kernel
        """

        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape) + 3*X

        mf = GPy.mappings.Additive(GPy.mappings.Constant(1,1,3),
               GPy.mappings.Additive(GPy.mappings.MLP(1,1),
                     GPy.mappings.Identity(1,1)
                           )
                        )

        k =GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())

    def test_svgp_mean_function(self):

        # an instance of the SVIGOP with a men function
        X = np.linspace(0,10,500).reshape(-1,1)
        Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape)
        Y = np.where(Y>0, 1,0) # make aclassificatino problem

        mf = GPy.mappings.Linear(1,1)
        Z = np.linspace(0,10,50).reshape(-1,1)
        lik = GPy.likelihoods.Bernoulli()
        k =GPy.kern.RBF(1) + GPy.kern.White(1, 1e-4)
        m = GPy.core.SVGP(X, Y,Z=Z, kernel=k, likelihood=lik, mean_function=mf)
        self.assertTrue(m.checkgrad())



