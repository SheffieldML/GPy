.. _interacting_with_models:

*************************************
Interacting with models
*************************************

The GPy model class has a set of features which are 
designed to make it simple to explore the parameter 
space of the model. By default, the scipy optimisers 
are used to fit GPy models (via model.optimize()), 
for which we provide mechanisms for 'free' optimisation: 
GPy can ensure that naturally positive parameters 
(such as variances) remain positive. But these mechanisms 
are much more powerful than simple reparameterisation, 
as we shall see. 

Along this tutorial we'll use a sparse GP regression model 
as example. This example can be in ``GPy.examples.regression``.  
All of the examples included in GPy return an instance 
of a model class, and therefore they can be called in 
the following way: ::

    import numpy as np
    import pylab as pb
    pb.ion()
    import GPy
    m = GPy.examples.regression.sparse_GP_regression_1D()

Examining the model using print
===============================
To see the current state of the model parameters, 
and the model's (marginal) likelihood just print the model ::

    print m

The first thing displayed on the screen is the log-likelihood 
value of the model with its current parameters. Below the 
log-likelihood, a table with all the model's parameters 
is shown. For each parameter, the table contains the name 
of the parameter, the current value, and in case there are 
defined: constraints, ties and prior distrbutions associated. ::

  Name                 : sparse gp
  Log-likelihood       : 588.947189413
  Number of Parameters : 8
  Parameters:
    sparse_gp.               |       Value        |  Constraint  |  Prior  |  Tied to
    inducing inputs          |            (5, 1)  |              |         |         
    rbf.variance             |     1.91644016819  |     +ve      |         |         
    rbf.lengthscale          |     2.62103621347  |     +ve      |         |         
    Gaussian_noise.variance  |  0.00269870373421  |     +ve      |         |           

In this case the kernel parameters (``rbf.variance``, 
``rbf.lengthscale``) as well as 
the likelihood noise parameter (``Gaussian_noise.variance``), are constrained 
to be positive, while the inducing inputs have no
constraints associated. Also there are no ties or prior defined.

You can also print all subparts of the model, by printing the
subcomponents individually::

  print m.rbf

This will print the details of this particular parameter handle::

    rbf.         |      Value      |  Constraint  |  Prior  |  Tied to
    variance     |  1.91644016819  |     +ve      |         |         
    lengthscale  |  2.62103621347  |     +ve      |         |         

When you want to get a closer look into
multivalue parameters, print them directly::

  print m.inducing_indputs

  Index  |  sparse_gp.inducing_inputs  |  Constraint  |   Prior   |  Tied to
  [0 0]  |                  2.7189499  |              |           |    N/A    
  [1 0]  |                 0.02006533  |              |           |    N/A    
  [2 0]  |                 -1.5299386  |              |           |    N/A    
  [3 0]  |                 -2.7001675  |              |           |    N/A    
  [4 0]  |                  1.4654162  |              |           |    N/A    

Interacting with Parameters:
=======================
The preferred way of interacting with parameters is to act on the
parameter handle itself.
Interacting with parameter handles is simple. The names, printed by `print m`
are accessible interactively and programatically. For example try to
set kernels (`rbf`) `lengthscale` to `.2` and print the result::

  m.rbf.lengthscale = .2
  print m

You should see this::

  Name                 : sparse gp
  Log-likelihood       : 588.947189413
  Number of Parameters : 8
  Parameters:
    sparse_gp.               |       Value        |  Constraint  |  Prior  |  Tied to
    inducing inputs          |            (5, 1)  |              |         |         
    rbf.variance             |     1.91644016819  |     +ve      |         |         
    rbf.lengthscale          |              0.2  |     +ve      |         |         
    Gaussian_noise.variance  |  0.00269870373421  |     +ve      |         |           

This will already have updated the model's inner state, so you can
plot it or see the changes in the posterior `m.posterior` of the model.

Regular expressions
----------------
The model's parameters can also be accessed through regular
expressions, by 'indexing' the model with a regular expression,
matching the parameter name. Through indexing by regular expression,
you can only retrieve leafs of the hierarchy, and you can retrieve the
values matched by calling `values()` on the returned object::

  >>> print m['.*var']
    Index  |       sparse_gp.rbf.variance        |  Constraint  |    Prior     |  Tied to
     [0]   |                          2.1500132  |              |              |    N/A    
    -----  |  sparse_gp.Gaussian_noise.variance  |  ----------  |  ----------  |  -------
     [0]   |                       0.0024268215  |              |              |    N/A    
  >>> print m['.*var'].values()
  [ 2.1500132   0.00242682]
  >>> print m['rbf']
     Index  |   sparse_gp.rbf.variance    |  Constraint  |    Prior     |  Tied to
     [0]   |                  2.1500132  |              |              |    N/A    
    -----  |  sparse_gp.rbf.lengthscale  |  ----------  |  ----------  |  -------
     [0]   |                  2.6782803  |              |              |    N/A    
  
There is access to setting parameters by regular expression,
as well. Here are a few examples of how to set parameters by regular expression::

  >>> m['.*var'] = .1
  >>> print m['.*var']
    Index  |       sparse_gp.rbf.variance        |  Constraint  |    Prior     |  Tied to
     [0]   |                                0.1  |              |              |    N/A    
    -----  |  sparse_gp.Gaussian_noise.variance  |  ----------  |  ----------  |  -------
     [0]   |                                0.1  |              |              |    N/A    
  >>> m['.*var'] = [.1, .2]
  >>> print m['.*var']
    Index  |       sparse_gp.rbf.variance        |  Constraint  |    Prior     |  Tied to
     [0]   |                                0.1  |              |              |    N/A    
    -----  |  sparse_gp.Gaussian_noise.variance  |  ----------  |  ----------  |  -------
     [0]   |                                0.2  |              |              |    N/A    
  
The fact that only leaf nodes can be accesses we can print all
parameters in a flattened view, by printing the regular expression
match of matching all objects::

  >>> print m['']
    Index  |      sparse_gp.inducing_inputs      |  Constraint  |    Prior     |  Tied to
    [0 0]  |                         -2.6716041  |              |              |    N/A    
    [1 0]  |                         -1.4665111  |              |              |    N/A    
    [2 0]  |                       -0.031010293  |              |              |    N/A    
    [3 0]  |                          1.4563711  |              |              |    N/A    
    [4 0]  |                          2.6803046  |              |              |    N/A    
    -----  |       sparse_gp.rbf.variance        |  ----------  |  ----------  |  -------
     [0]   |                                0.1  |              |              |    N/A    
    -----  |      sparse_gp.rbf.lengthscale      |  ----------  |  ----------  |  -------
     [0]   |                          2.6782803  |              |              |    N/A    
    -----  |  sparse_gp.Gaussian_noise.variance  |  ----------  |  ----------  |  -------
     [0]   |                                0.2  |              |              |    N/A    

Setting and fetching parameters `parameter_array`
------------------------------------------
Another way to interact with the model's parameters is through the
`parameter_array`. The Parameter array holds all the parameters of the
model in one place and is editable. It can be accessed through
indexing the model for example you can set all the parameters through
this mechanism::

  >>> new_params = np.r_[[-4,-2,0,2,4], [.5,2], [.3]]
  >>> print new_params
  array([-4. , -2. ,  0. ,  2. ,  4. ,  0.5,  2. ,  0.3])
  >>> m[:] = new_params
  >>> print m
  Name                 : sparse gp
  Log-likelihood       : -147.561160209
  Number of Parameters : 8
  Parameters:
    sparse_gp.               |  Value   |  Constraint  |  Prior  |  Tied to
    inducing inputs          |  (5, 1)  |              |         |         
    rbf.variance             |     0.5  |     +sq      |         |         
    rbf.lengthscale          |     2.0  |     +ve      |         |         
    Gaussian_noise.variance  |     0.3  |     +sq      |         |         
 
Parameters themselves (leafs of the hierarchy) can be indexed and used
the same way as numpy arrays. First let us set a slice of the
`inducing_inputs`::

  >>> m.inducing_inputs[2:, 0] = [1,3,5]
  >>> print m.inducing_indputs
    Index  |  sparse_gp.inducing_inputs  |  Constraint  |   Prior   |  Tied to
    [0 0]  |                         -4  |              |           |    N/A    
    [1 0]  |                         -2  |              |           |    N/A    
    [2 0]  |                          1  |              |           |    N/A    
    [3 0]  |                          3  |              |           |    N/A    
    [4 0]  |                          5  |              |           |    N/A    

Or you use the parameters as normal numpy arrays for calculations::

  >>> precision = 1./m.Gaussian_noise.variance
  array([ 3.33333333])

Getting the model's log likelihood
=============================================
Appart form the printing the model,  the marginal 
log-likelihood can be obtained by using the function
``log_likelihood()``. Also, the log-likelihood gradients
wrt. each parameter can be obtained with the funcion
``_log_likelihood_gradients()``. ::

    >>> m.log_likelihood()
    array([-152.83377316])

If you want to ensure the log likelihood as a float, call `float()`
around it::

  >>> float(m.log_likelihood())
  -152.83377316356177

Getting the model parameter's gradients
============================
The gradients of a model can shed light on understanding the
(possibly hard) optimization process. The gradients of each parameter
handle can be accessed through their `gradient` field.::

  >>> print m.gradient
  [   5.51170031    9.71735112   -4.20282106   -3.45667035   -1.58828165
   -2.11549358   12.40292787 -627.75467803]
  >>> print m.rbf.gradient
  [ -2.11549358  12.40292787]
  >>> m.optimize()
  >>> print m.gradient
  [ -5.98046560e-04  -3.64576085e-04   1.98005930e-04   3.43381219e-04
  -6.85685104e-04  -1.28800748e-05   1.08552429e-03   2.74058081e-01]

Adjusting the model's constraints
================================
When we initially call the example, it was optimized and hence the
log-likelihood gradients were close to zero. However, since
we have been changing the parameters, the gradients are far from zero now.
Next we are going to show how to optimize the model setting different 
restrictions on the parameters. 

Once a constraint has been set on a parameter, it is possible to remove
it with the command ``unconstrain()``, which can be called on any
parameter handle of the model. The methods `constrain()` and
`unconstrain()` return the indices which were actually unconstrained,
relative to the parameter handle the method was called on. This is
particularly handy for reporting which parameters where reconstrained,
when reconstraining a parameter, which was already constrained::

	>>> m.rbf.variance.unconstrain()
	array([0])
	>>>m.unconstrain()
	array([6, 7])

The parameter handles come with default constraints, so you will
rarely be needing to adjust the constraints of a model. In the rare
cases of needing to adjust the constraints of a model, or in need of
fixing some parameters, you can do so with the functions
``constrain_{positive|negative|bounded|fixed}()``.::

    m['.*var'].constrain_positive()

For convenience, GPy also provides a catch all function 
which ensures that anything which appears to require 
positivity is constrianed appropriately::

    m.ensure_default_constraints()

Fixing parameters
=================
Parameters values can be fixed using ``constrain_fixed()``. 
For example we can define the first inducing input to be 
fixed on zero: ::

    m.constrain_fixed('iip_0',0)
	
Bounding parameters
===================
Defining bounding constraints is an easily task in GPy too,
it only requires to use the function ``constrain_bounded()``.
For example, lets bound inducing inputs 2 and 3 to have
values between -4 and -1: ::

    m.constrain_bounded('iip_(1|2)',-4,-1)

Tying Parameters
================
The values of two or more parameters can be tied together,
so that they share the same value during optimization.
The function to do so is ``tie_params()``. For the example
we are using, it doesn't make sense to tie parameters together,
however for the sake of the example we will tie the white noise
and the variance together. See `A kernel overview <tuto_kernel_overview.html>`_.
for a proper use of the tying capabilities.::

    m.tie_params('.*e_var')

Optimizing the model
====================
Once we have finished defining the constraints, 
we can now optimize the model with the function
``optimize``.::

    m.optimize()

We can print again the model and check the new results.
The table now shows that ``iip_0_0`` is fixed, ``iip_1_0`` 
and ``iip_2_0`` are bounded and the kernel parameters are constrained to
be positive. In addition the table now indicates that
white_variance and noise_variance are tied together.::

	Log-likelihood: 9.967e+01

  	     Name        |   Value   |  Constraints  |  Ties  |  Prior  
	------------------------------------------------------------------
	    iip_0_0      |  0.0000   |     Fixed     |        |         
	    iip_1_0      |  -2.8834  |   (-4, -1)    |        |         
	    iip_2_0      |  -1.9152  |   (-4, -1)    |        |         
	    iip_3_0      |  1.5034   |               |        |         
	    iip_4_0      |  -1.0162  |               |        |         
	 rbf_variance    |  0.0158   |     (+ve)     |        |         
	rbf_lengthscale  |  0.9760   |     (+ve)     |        |         
	white_variance   |  0.0049   |     (+ve)     |  (0)   |         
	noise_variance   |  0.0049   |     (+ve)     |  (0)   |         


Further Reading
===============
All of the mechansiams for dealing with parameters are baked right into GPy.core.model, from which all of the classes in GPy.models inherrit. To learn how to construct your own model, you might want to read :ref:`creating_new_models`. 

By deafult, GPy uses the scg optimizer. To use other optimisers, and to control the setting of those optimisers, as well as other funky features like automated restarts and diagnostics, you can read the optimization tutorial ??link??.
