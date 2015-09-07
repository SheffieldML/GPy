#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, Extension
import numpy as np

# Version number
version = '0.6.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#Mac OS X Clang doesn't support OpenMP th the current time.
#This detects if we are building on a Mac
def ismac():
    platform = sys.platform
    ismac = False
    if platform[:6] == 'darwin':
        ismac = True
    return ismac

if ismac():
    compile_flags = [ '-O3', ]
    link_args = ['']
else:
    compile_flags = [ '-fopenmp', '-O3', ]
    link_args = ['-lgomp']

ext_mods = [Extension(name='GPy.kern._src.stationary_cython',
                      sources=['GPy/kern/_src/stationary_cython.c','GPy/kern/_src/stationary_utils.c'],
                      include_dirs=[np.get_include()],
                      extra_compile_args=compile_flags,
                      extra_link_args = link_args),
            Extension(name='GPy.util.choleskies_cython',
                      sources=['GPy/util/choleskies_cython.c'],
                      include_dirs=[np.get_include()],
                      extra_link_args = link_args,
                      extra_compile_args=compile_flags),
            Extension(name='GPy.util.linalg_cython',
                      sources=['GPy/util/linalg_cython.c'],
                      include_dirs=[np.get_include()],
                      extra_compile_args=compile_flags),
            Extension(name='GPy.kern._src.coregionalize_cython',
                      sources=['GPy/kern/_src/coregionalize_cython.c'],
                      include_dirs=[np.get_include()],
                      extra_compile_args=compile_flags)]

setup(name = 'GPy',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "james.hensman@gmail.com",
      description = ("The Gaussian Process Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "http://sheffieldml.github.com/GPy/",
      ext_modules = ext_mods,
      packages = ["GPy.models",
                  "GPy.inference.optimization",
                  "GPy.inference.mcmc",
                  "GPy.inference",
                  "GPy.inference.latent_function_inference",
                  "GPy.likelihoods", "GPy.mappings",
                  "GPy.examples", "GPy.core.parameterization",
                  "GPy.core", "GPy.testing",
                  "GPy", "GPy.util", "GPy.kern",
                  "GPy.kern._src.psi_comp", "GPy.kern._src",
                  "GPy.plotting.matplot_dep.latent_space_visualizations.controllers",
                  "GPy.plotting.matplot_dep.latent_space_visualizations",
                  "GPy.plotting.matplot_dep", "GPy.plotting"],
      package_dir={'GPy': 'GPy'},
      package_data = {'GPy': ['defaults.cfg', 'installation.cfg',
                              'util/data_resources.json',
                              'util/football_teams.json']},
      include_package_data = True,
      py_modules = ['GPy.__init__'],
      test_suite = 'GPy.testing',
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.16'],
      extras_require = {'docs':['matplotlib >=1.3','Sphinx','IPython']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
