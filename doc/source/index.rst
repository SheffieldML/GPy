GPy - A Gaussian Process (GP) framework in Python
=================================================

Introduction
------------

`GPy <http://sheffieldml.github.io/GPy/>`_ is a Gaussian Process (GP) framework written in Python, from the Sheffield machine learning group.

The `GPy homepage <http://sheffieldml.github.io/GPy/>`_ contains tutorials for users and further information on the project, including installation instructions.
This documentation is mostly aimed at developers interacting closely with the code-base.

The code can be found on our `Github project page <https://github.com/SheffieldML/GPy>`_. It is open source and provided under the BSD license.

Installation
------------

For developers
--------------

- `Writing new models <tuto_creating_new_models.html>`_
- `Writing new kernels <tuto_creating_new_kernels.html>`_
- `Write a new plotting routine using gpy_plot <tuto_plotting.html>`_
- `Parameterization handles <tuto_parameterized.html>`_

API Documentation
-----------------

.. toctree::
   :maxdepth: 1

   GPy.models
   GPy.kern
   GPy.likelihoods
   GPy.mappings
   GPy.examples
   GPy.util
   GPy.plotting.gpy_plot
   GPy.plotting.matplot_dep
   GPy.core
   GPy.core.parameterization
   GPy.inference.optimization
   GPy.inference.latent_function_inference
   GPy.inference.mcmc
	      
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

