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

Architecture
------------

GPy is a big, powerful package, with many features. The concept of how to use GPy in general terms is roughly as follows. A model (:py:class:`GPy.models`) is created - this is at the heart of GPy from a user perspective. A kernel (:py:class:`GPy.kern`), data and, usually, a representation of noise are assigned to the model. Specific models require, or can make use of, additional information. The kernel and noise are controlled by hyperparameters - calling the optimize (:py:class:`GPy.core.gp.GP.optimize`) method against the model invokes an iterative process which seeks optimal hyperparameter values. The model object can be used to make plots and predictions (:py:class:`GPy.core.gp.GP.predict`).

.. graphviz::

   digraph GPy_Arch {
      
      rankdir=LR
      node[shape="rectangle" style="rounded,filled" fontname="Arial"]
      edge [color="#006699" len=2.5]

      Data->Model
      Hyperparameters->Kernel
      Hyperparameters->Noise
      Kernel->Model
      Noise->Model
      
      Model->Optimize
      Optimize->Hyperparameters
      
      Model->Predict
      Model->Plot
      
      Optimize [shape="ellipse"]
      Predict [shape="ellipse"]
      Plot [shape="ellipse"]
      
      subgraph cluster_0 {
         Data
         Kernel
         Noise
      }

   }

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
   GPy.plotting
   GPy.inference.optimization
   GPy.inference.latent_function_inference
   GPy.inference.mcmc

