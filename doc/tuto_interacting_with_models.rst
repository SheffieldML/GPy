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

	Log-likelihood: 6.309e+02

	     Name        |   Value   |  Constraints  |  Ties  |  Prior  
	------------------------------------------------------------------
	    iip_0_0      |  -1.4671  |               |        |         
	    iip_1_0      |  2.6378   |               |        |         
	    iip_2_0      |  -0.0396  |               |        |         
	    iip_3_0      |  -2.6372  |               |        |         
	    iip_4_0      |  1.4704   |               |        |         
	 rbf_variance    |  1.5672   |     (+ve)     |        |         
	rbf_lengthscale  |  2.5625   |     (+ve)     |        |         
	white_variance   |  0.0000   |     (+ve)     |        |         
	noise_variance   |  0.0022   |     (+ve)     |        |         

In this case the kernel parameters (``rbf_variance``, 
``rbf_lengthscale`` and ``white_variance``) as well as 
the noise parameter (``noise_variance``), are constrained 
to be positive, while the inducing inputs have not 
constraints associated. Also there are no ties or prior defined.

Setting and fetching parameters by name
=======================================
Another way to interact with the model's parameters is through
the functions ``_get_param_names()``, ``_get_params()`` and 
``_set_params()``.

``_get_param_names()`` returns a list of the parameters names ::

	['iip_0_0',
	 'iip_1_0',
	 'iip_2_0',
	 'iip_3_0',
	 'iip_4_0',
	 'rbf_variance',
	 'rbf_lengthscale',
	 'white_variance',
	 'noise_variance']

``_get_params()`` returns an array of the parameters values ::

	array([ -1.46705227e+00,   2.63782176e+00,  -3.96422982e-02,
		-2.63715255e+00,   1.47038653e+00,   1.56724596e+00,
		 2.56248679e+00,   2.20963633e-10,   2.18379922e-03])

``_set_params()`` takes an array as input and substitutes 
the current values of the parameters for those of the array. For example,
we can define a new array of values and change the parameters as follows: ::

	new_params = np.array([1.,2.,3.,4.,1.,1.,1.,1.,1.])
	m._set_params(new_params)

If we call the function ``_get_params()`` again, we will obtain the new
parameters we have just set.

Parameters can be also set by name using dictionary notations. For example,
let's change the lengthscale to .5: ::

	m['rbf_lengthscale'] = .5

Here, the matching accepts a regular expression and therefore all parameters matching that regular expression are set to the given value. In this case rather 
than passing as second output a single value, we can also 
use a list of arrays. For example, lets change the inducing 
inputs: ::

	m['iip'] = np.arange(-5,0)

Getting the model's likelihood and gradients
=============================================
Appart form the printing the model,  the marginal 
log-likelihood can be obtained by using the function
``log_likelihood()``. Also, the log-likelihood gradients
wrt. each parameter can be obtained with the funcion
``_log_likelihood_gradients()``. ::

    m.log_likelihood()
    -791.15371409346153

    m._log_likelihood_gradients()
    array([  7.08278455e-03,   1.37118783e+01,   2.66948031e+00,
             3.50184014e+00,   7.08278455e-03,  -1.43501702e+02,
	     6.10662266e+01,  -2.18472649e+02,   2.14663691e+02])

Removing the model's constraints
================================
When we initially call the example, it was optimized and hence the
log-likelihood gradients were close to zero. However, since
we have been changing the parameters, the gradients are far from zero now.
Next we are going to show how to optimize the model setting different 
restrictions on the parameters. 

Once a constrain has been set on a parameter, it is possible to remove it
with the command ``unconstrain()``, and
just as the previous matching commands, it also accepts regular expression.
In this case we will remove all the constraints: ::

	m.unconstrain('')

Constraining and optimising the model
=====================================
A requisite needed for some parameters, such as variances,
is to be positive. This is constraint is easily set 
with the function ``constrain_positive()``. Regular expressions
are also accepted. ::

    m.constrain_positive('.*var')

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
