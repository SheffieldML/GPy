
*************************************
Gaussian process regression tutorial
*************************************

We will see in this tutorial the basics for building a 1 dimensional and a 2 dimensional Gaussian process model, also known as a kriging model.

We first import the libraries we will need: ::

    import pylab as pb
    pb.ion()
    import numpy as np
    import GPy

1 dimensional model
===================

For this toy example, we assume we have the following inputs and outputs::

    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X) + np.random.randn(20,1)*0.05

Note that the observations Y include some noise.

The first step is to define the covariance kernel we want to use for the model. We choose here a kernel based on Gaussian kernel (i.e. rbf or square exponential) plus some white noise::

    Gaussian = GPy.kern.rbf(D=1)
    noise = GPy.kern.white(D=1)
    kernel = Gaussian + noise

The parameter D stands for the dimension of the input space. Note that many other kernels are implemented such as:

* linear (``GPy.kern.linear``)
* exponential kernel (``GPy.kern.exponential``)
* Matern 3/2 (``GPy.kern.Matern32``)
* Matern 5/2 (``GPy.kern.Matern52``)
* spline (``GPy.kern.spline``)
* and many others...

The inputs required for building the model are the observations and the kernel::

    m = GPy.models.GP_regression(X,Y,kernel)

The functions ``print`` and ``plot`` can help us understand the model we have just build::

    print m
    m.plot()

The default values of the kernel parameters may not be relevant for the current data (for example, the confidence intervals seems too wide on the previous figure). A common approach is find the values of the parameters that maximize the likelihood of the data. There are two steps for doing that with GPy:

* Constrain the parameters of the kernel to ensure the kernel will always be a valid covariance structure (For example, we don\'t want some variances to be negative!).
* Run the optimization

There are various ways to constrain the parameters of the kernel. The most basic is to constrain all the parameters to be positive::

    m.constrain_positive('')

but it is also possible to set a range on to constrain one parameter to be fixed. The parameter of ``m.constrain_positive`` is a regular expression that matches the name of the parameters to be constrained (as seen in ``print m``). For example, if we want the variance to be positive, the lengthscale to be in [1,10] and the noise variance to be fixed we can write::

    #m.unconstrain('')                            # Required if the model has been previously constrained
    m.constrain_positive('rbf_variance')
    m.constrain_bounded('lengthscale',1.,10. )
    m.constrain_fixed('white',0.0025)

Once the constrains have bee imposed, the model can be optimized::

    m.optimize()

If we want to perform some restarts to try to improve the result of the optimization, we can use the optimize_restart function::

    m.optimize_restarts(Nrestarts = 10)
    m.plot()
    print(m)

2 dimensional example
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
    ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.white(2)

    # create simple GP model
    m = GPy.models.GP_regression(X,Y,ker)

    # contrain all parameters to be positive
    m.constrain_positive('')

    # optimize and plot
    pb.figure()
    m.optimize('tnc', max_f_eval = 1000)

    m.plot()
    print(m)

The flag ``ARD=True`` in the definition of the Matern kernel specifies that we want one lengthscale parameter per dimension (ie the GP is not isotropic).
