.. _parameterized:

*******************
Parameterization handling
*******************

Parameterization in GPy is done through so called parameter handles. The parameter handles are handles to parameters of a model of any kind. A parameter handle can be constrained, fixed, randomized and others. All parameters in GPy have a name, with which they can be accessed in the model. The most common way of accesssing a parameter programmatically though, is by variable name. 

Parameter handles
==============

A parameter handle in GPy is a handle on a parameter, as the name suggests. A parameter can be constrained, fixed, randomized and more (See e.g. `working with models`). This gives the freedom to the model to handle parameter distribution and model updates as efficiently as possible. All parameter handles share a common memory space, which is just a flat numpy array, stored in the highest parent of a model hierarchy.
In the following we will introduce and elucidate the different parameter handles which exist in GPy.

:py:class:`~GPy.core.parameterization.parameterized.Parameterized`
==========

A parameterized object itself holds parameter handles and is just a summarization of the parameters below. It can use those parameters to change the internal state of the model and GPy ensures those parameters to allways hold the right value when in an optimization routine or any other update.

:py:class:`~GPy.core.parameterization.param.Param`
===========

The lowest level of parameter is a numpy array. This Param class inherits all functionality of a numpy array and can simply be used as if it where a numpy array. These parameters can be accessed in the same way as a numpy array is indexed.
