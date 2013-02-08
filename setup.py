#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from numpy.distutils.core import Extension, setup
#from sphinx.setup_command import BuildDoc

# Version number
version = '0.1.3'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'GPy',
      version = version,
      author = 'James Hensman, Nicolo Fusi, Ricardo Andrade, Nicolas Durrande, Alan Saul, Neil D. Lawrence',
      author_email = "james.hensman@gmail.com",
      description = ("The Gaussian Process Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "http://ml.sheffield.ac.uk/GPy/",
      packages = ['GPy', 'GPy.core', 'GPy.kern', 'GPy.util', 'GPy.models', 'GPy.inference', 'GPy.examples', 'GPy.likelihoods'],
      package_dir={'GPy': 'GPy'},
      package_data = {'GPy': ['GPy/examples']},
      py_modules = ['GPy.__init__'],
      long_description=read('README.md'),
      #ext_modules =  [Extension(name = 'GPy.kern.lfmUpsilonf2py',
      #          sources = ['GPy/kern/src/lfmUpsilonf2py.f90'])],
      install_requires=['sympy', 'numpy>=1.6', 'scipy>=0.9','matplotlib>=1.1'],
      extras_require = {
        'docs':['Sphinx', 'ipython'],
      },
      #setup_requires=['sphinx'],
      #cmdclass = {'build_sphinx': BuildDoc},
      classifiers=[
      "Development Status :: 1 - Alpha",
      "Topic :: Machine Learning",
      "License :: OSI Approved :: BSD License"],
      )
