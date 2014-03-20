#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.4.6'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'GPy',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "james.hensman@gmail.com",
      description = ("The Gaussian Process Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "http://sheffieldml.github.com/GPy/",
      packages = ["GPy.models", "GPy.inference.optimization", "GPy.inference", "GPy.inference.latent_function_inference", "GPy.likelihoods", "GPy.mappings", "GPy.examples", "GPy.core.parameterization", "GPy.core", "GPy.testing", "GPy", "GPy.util", "GPy.kern", "GPy.kern._src.psi_comp", "GPy.kern._src", "GPy.plotting.matplot_dep.latent_space_visualizations.controllers", "GPy.plotting.matplot_dep.latent_space_visualizations", "GPy.plotting.matplot_dep", "GPy.plotting"],
      package_dir={'GPy': 'GPy'},
      package_data = {'GPy': ['GPy/examples']},
      py_modules = ['GPy.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.6', 'scipy>=0.9','matplotlib>=1.1', 'nose'],
      extras_require = {
        'docs':['Sphinx', 'ipython'],
      },
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      )
