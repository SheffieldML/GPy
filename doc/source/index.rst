GPy - A Gaussian Process (GP) framework in Python
=================================================

Introduction
------------

`GPy <http://sheffieldml.github.io/GPy/>`_ is a Gaussian Process (GP) framework written in Python, from the Sheffield machine learning group. It includes support for basic GP regression, multiple output GPs (using coregionalization), various noise models, sparse GPs, non-parametric regression and latent variables.

The `GPy homepage <http://sheffieldml.github.io/GPy/>`_ contains tutorials for users and further information on the project, including installation instructions.

The documentation hosted here is mostly aimed at developers interacting closely with the code-base.

Source Code
-----------

The code can be found on our `Github project page <https://github.com/SheffieldML/GPy>`_. It is open source and provided under the BSD license.

Installation
------------

Installation instructions can currently be found on our `Github project page <https://github.com/SheffieldML/GPy>`_.

Tutorials
---------

Several tutorials have been developed in the form of `Jupyter Notebooks <https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb>`_. 

.. toctree::
   :maxdepth: 1
   :caption: For developers

   tuto_creating_new_models
   tuto_creating_new_kernels
   tuto_plotting
   tuto_parameterized

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   GPy.core
   GPy.core.parameterization
   GPy.models
   GPy.kern
   GPy.likelihoods
   GPy.mappings
   GPy.examples
   GPy.util
   GPy.plotting.gpy_plot
   GPy.plotting.matplot_dep
   GPy.inference.optimization
   GPy.inference.latent_function_inference
   GPy.inference.mcmc

