#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.2'

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
      packages = ['GPy', 'GPy.core', 'GPy.kern', 'GPy.util', 'GPy.models', 'GPy.inference', 'GPy.examples', 'GPy.likelihoods', 'GPy.testing'],
      package_dir={'GPy': 'GPy'},
      package_data = {'GPy': ['GPy/examples']},
      py_modules = ['GPy.__init__'],
      long_description=read('README.md'),
      #ext_modules =  [Extension(name = 'GPy.kern.lfmUpsilonf2py',
      #          sources = ['GPy/kern/src/lfmUpsilonf2py.f90'])],
      install_requires=['sympy', 'numpy>=1.6', 'scipy>=0.9','matplotlib>=1.1', 'nose'],
      extras_require = {
        'docs':['Sphinx', 'ipython'],
      },
      #setup_requires=['sphinx'],
      #cmdclass = {'build_sphinx': BuildDoc},
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      )
