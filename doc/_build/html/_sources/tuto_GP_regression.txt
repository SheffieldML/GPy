*************************************
Gaussian process regression tutorial
*************************************

We will see in this tutorial the basics for building a 1 dimensional and a 2 dimensional Gaussian process regression model, also known as a kriging model. The code shown in this tutorial can be obtained at GPy/examples/tutorials.py, or by running ``GPy.examples.tutorials.tuto_GP_regression()``.

We first import the libraries we will need: ::

    import pylab as pb
    pb.ion()
    import numpy as np
    import GPy

1-dimensional model
===================

For this toy example, we assume we have the following inputs and outputs::

    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X) + np.random.randn(20,1)*0.05

Note that the observations Y include some noise.

The first step is to define the covariance kernel we want to use for the model. We choose here a kernel based on Gaussian kernel (i.e. rbf or square exponential)::

    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

The parameter ``input_dim`` stands for the dimension of the input space. The parameters ``variance`` and ``lengthscale`` are optional. Many other kernels are implemented such as:

* linear (:py:class:`~GPy.kern.Linear`)
* exponential kernel (:py:class:`GPy.kern.Exponential`)
* Matern 3/2 (:py:class:`GPy.kern.Matern32`)
* Matern 5/2 (:py:class:`GPy.kern.Matern52`)
* spline (:py:class:`GPy.kern.Spline`)
* and many others...

The inputs required for building the model are the observations and the kernel::

    m = GPy.models.GPRegression(X,Y,kernel)

By default, some observation noise is added to the modle. The functions ``print`` and ``plot`` give an insight of the model we have just build. The code::

    print m
    m.plot()

gives the following output: ::

  Name                 : GP regression
  Log-likelihood       : -22.8178418808
  Number of Parameters : 3
  Parameters:
    GP_regression.           |  Value  |  Constraint  |  Prior  |  Tied to
    rbf.variance             |    1.0  |     +ve      |         |         
    rbf.lengthscale          |    1.0  |     +ve      |         |         
    Gaussian_noise.variance  |    1.0  |     +ve      |         |         
  
.. figure::  Figures/tuto_GP_regression_m1.png
    :align:   center
    :height: 350px

    GP regression model before optimization of the parameters. The shaded region corresponds to ~95% confidence intervals (ie +/- 2 standard deviation).

The default values of the kernel parameters may not be relevant for
the current data (for example, the confidence intervals seems too wide
on the previous figure). A common approach is to find the values of
the parameters that maximize the likelihood of the data. It as easy as
calling ``m.optimize`` in GPy::

  m.optimize()

If we want to perform some restarts to try to improve the result of the optimization, we can use the ``optimize_restart`` function::

    m.optimize_restarts(num_restarts = 10)

Once again, we can use ``print(m)`` and ``m.plot()`` to look at the resulting model  resulting model::

  Name                 : GP regression
  Log-likelihood       : 11.947469082
  Number of Parameters : 3
  Parameters:
    GP_regression.           |       Value        |  Constraint  |  Prior  |  Tied to
    rbf.variance             |     0.74229417323  |     +ve      |         |         
    rbf.lengthscale          |     1.43020495724  |     +ve      |         |         
    Gaussian_noise.variance  |  0.00325654460991  |     +ve      |         |         
  
.. figure::  Figures/tuto_GP_regression_m2.png
    :align:   center
    :height: 350px

    GP regression model after optimization of the parameters.


2-dimensional example
=====================

Here is a 2 dimensional example::

    import pylab as pb
    pb.ion()
    import numpy as np
    import GPy

    # sample inputs and outputs
    X = np.random.uniform(-3.,3.,(50,2))
    Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

    # define kernel
    ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

    # create simple GP model
    m = GPy.models.GPRegression(X,Y,ker)

    # optimize and plot
    m.optimize(max_f_eval = 1000)
    m.plot()
    print(m)

The flag ``ARD=True`` in the definition of the Matern kernel specifies that we want one lengthscale parameter per dimension (ie the GP is not isotropic). The output of the last two lines is::

  Name                 : GP regression
  Log-likelihood       : 26.787156248
  Number of Parameters : 5
  Parameters:
    GP_regression.           |        Value        |  Constraint  |  Prior  |  Tied to
    add.Mat52.variance       |     0.385463739076  |     +ve      |         |         
    add.Mat52.lengthscale    |               (2,)  |     +ve      |         |         
    add.white.variance       |  0.000835329608514  |     +ve      |         |         
    Gaussian_noise.variance  |  0.000835329608514  |     +ve      |         |         

If you want to see the ``ARD`` parameters explicitly print them
directly::

  >>> print m.add.Mat52.lengthscale
    Index  |  GP_regression.add.Mat52.lengthscale  |  Constraint  |   Prior   |  Tied to
     [0]   |                            1.9575587  |     +ve      |           |    N/A    
     [1]   |                            1.9689948  |     +ve      |           |    N/A    
  
.. figure::  Figures/tuto_GP_regression_m3.png
    :align:   center
    :height: 350px

    Contour plot of the best predictor (posterior mean).
