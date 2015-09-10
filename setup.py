#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from setuptools import setup, Extension
import numpy as np


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def read_to_rst(fname):
    try:
        import pypandoc
        #print 'Warning in installation: For rst formatting in pypi, consider installing pypandoc for conversion'
        with open('README.rst', 'w') as f:
            f.write(pypandoc.convert('README.md', 'rst'))
    except:
        return read(fname)

version_dummy = {}
exec(read('GPy/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy

#Mac OS X Clang doesn't support OpenMP at the current time.
#This detects if we are building on a Mac
def ismac():
    return sys.platform[:6] == 'darwin'

if ismac():
    compile_flags = [ '-O3', ]
    link_args = []
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
      version = __version__,
      author = read('AUTHORS.txt'),
      author_email = "gpy.authors@gmail.com",
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
                              'util/football_teams.json',
                              ]},
      include_package_data = True,
      py_modules = ['GPy.__init__'],
      test_suite = 'GPy.testing',
      long_description=read_to_rst('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.16', 'six'],
      extras_require = {'docs':['matplotlib >=1.3','Sphinx','IPython']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
