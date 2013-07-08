.. _creating_new_models:

*******************
Creating new Models
*******************

In GPy all models inherit from the base class :py:class:`~GPy.core.parameterized.Parameterized`. :py:class:`~GPy.core.parameterized.Parameterized` is a class which allows for parameterization of objects. All it holds is functionality for tying, bounding and fixing of parameters. It also provides the functionality of searching and manipulating parameters by regular expression syntax. See :py:class:`~GPy.core.parameterized.Parameterized` for more information. 

The :py:class:`~GPy.core.model.Model` class provides parameter introspection, objective function and optimization.

In order to fully use all functionality of :py:class:`~GPy.core.model.Model` some methods need to be implemented / overridden. In order to explain the functionality of those methods we will use a wrapper to the numpy ``rosen`` function, which holds input parameters :math:`\mathbf{X}`. Where :math:`\mathbf{X}\in\mathbb{R}^{N\times 1}`.

Obligatory methods
==================

:py:meth:`~GPy.core.model.Model.__init__` :
	Initialize the model with the given parameters. In our example we have to store shape information of :math:`\mathbf X` and the parameters themselves::
	
		self.X = X
		self.num_inputs = self.X.shape[0]
		assert self.X.ndim == 1, only vector inputs allowed

:py:meth:`~GPy.core.model.Model._get_params` : 
    Return parameters of the model as a flattened numpy array-like. So, in our example we have to return the input parameters::
    
    	return self.X.flatten()

:py:meth:`~GPy.core.model.Model._set_params` : 
    Set parameters, which have been fetched through :py:meth:`~GPy.core.model.Model._get_params`. In other words, "invert" the functionality of :py:meth:`~GPy.core.model.Model._get_params`::

    	self.X = params[:self.num_inputs*self.input_dim].reshape(self.num_inputs)

:py:meth:`~GPy.core.model.Model.log_likelihood` :
	Returns the log-likelihood of the new model. For our example this is just the call to ``rosen``::

		return scipy.optimize.rosen(self.X)

:py:meth:`~GPy.core.model.Model._log_likelihood_gradients` :
	Returns the gradients with respect to all parameters::

		return scipy.optimize.rosen_der(self.X)


Optional methods
================

If you want some special functionality please provide the following methods:

Using the pickle functionality
------------------------------

To be able to use the pickle functionality ``m.pickle(<path>)`` the methods ``getstate(self)`` and ``setstate(self, state)`` have to be provided. The convention for a ``state`` in ``GPy`` is a list of all parameters, which are needed to restore the model. All classes provided in ``GPy`` follow this convention, thus you can just append to the state of the inherited class and call the inherited class' ``setstate`` with the appropriate state.

:py:meth:`~GPy.core.model.Model.getstate` :
	This method returns a state of the model, following the memento pattern. As we are inheriting from :py:class:`~GPy.core.model.Model`, we have to return the state of Model as well. In out example we have `X` and `num_inputs` as state::

		return Model.getstate(self) + [self.X, self.num_inputs]

:py:meth:`~GPy.core.model.Model.setstate` :
	This method restores this model with the given ``state``::

		self.num_inputs = state.pop()
		self.X = state.pop()
		return Model.setstate(self, state)