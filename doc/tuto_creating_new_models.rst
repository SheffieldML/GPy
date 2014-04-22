.. _creating_new_models:

*******************
Creating new Models
*******************

In GPy all models inherit from the base class :py:class:`~GPy.core.parameterized.Parameterized`. :py:class:`~GPy.core.parameterized.Parameterized` is a class which allows for parameterization of objects. All it holds is functionality for tying, bounding and fixing of parameters. It also provides the functionality of searching and manipulating parameters by regular expression syntax. See :py:class:`~GPy.core.parameterized.Parameterized` for more information. 

The :py:class:`~GPy.core.model.Model` class provides parameter introspection, objective function and optimization.

In order to fully use all functionality of
:py:class:`~GPy.core.model.Model` some methods need to be implemented
/ overridden. And the model needs to be  told its parameters, such
that it can provide optimized parameter distribution and handling. 
In order to explain the functionality of those methods
we will use a wrapper to the numpy ``rosen`` function, which holds
input parameters :math:`\mathbf{X}`. Where
:math:`\mathbf{X}\in\mathbb{R}^{N\times 1}`.

Obligatory methods
==================

:py:meth:`~GPy.core.model.Model.__init__` :
	Initialize the model with the given parameters. These need to
	be added to the model by calling
	`self.add_parameter(<param>)`, where param needs to be a
	parameter handle (See parameterized_ for details).::
	
		self.X = GPy.core.Param("input", X)
		self.add_parameter(self.X)
		
:py:meth:`~GPy.core.model.Model.log_likelihood` :
	Returns the log-likelihood of the new model. For our example
	this is just the call to ``rosen`` and as we want to minimize
	it, we need to negate the objective.::

		return -scipy.optimize.rosen(self.X)

:py:meth:`~GPy.core.model.Model.parameters_changed` :
    Updates the internal state of the model and sets the gradient of
    each parameter handle in the hierarchy with respect to the
    log_likelihod. Thus here we need to put the negative derivative of
    the rosenbrock function:

 		self.X.gradient = -scipy.optimize.rosen_der(self.X)


Optional methods
================

Currently none.
