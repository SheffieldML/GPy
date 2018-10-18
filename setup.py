#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Copyright (c) 2014, James Hensman, Max Zwiessele
# Copyright (c) 2015, Max Zwiessele
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of GPy nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

from __future__ import print_function

import codecs
import os
import sys
from glob import glob

from setuptools import find_packages, setup
from setuptools.extension import Extension


def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()


def read_to_rst(fname):
    try:
        import pypandoc
        rstname = "{}.{}".format(os.path.splitext(fname)[0], 'rst')
        pypandoc.convert(read(fname), 'rst', format='md', outputfile=rstname)
        with open(rstname, 'r') as f:
            rststr = f.read()
        return rststr
        # return read(rstname)
    except ImportError:
        return read(fname)


desc = """

Please refer to the github homepage for detailed instructions on installation 
and usage.

"""

version_dummy = {}
exec(read('src/GPy/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy


# Mac OS X Clang doesn't support OpenMP at the current time.
# This detects if we are building on a Mac
def ismac():
    return sys.platform[:6] == 'darwin'


if ismac():
    compile_flags = ['-O3']
    link_args = []
else:
    compile_flags = ['-fopenmp', '-O3']
    link_args = ['-lgomp']

try:
    from Cython.Build import build_ext as _build_ext

    USE_CYTHON = True
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

    USE_CYTHON = False


# See https://stackoverflow.com/a/21621689/3622042
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())


ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension('GPy.kern.src.stationary_cython',
              sources=['src/GPy/kern/src/stationary_cython' + ext,
                       'src/GPy/kern/src/stationary_utils.c'],
              include_dirs=['.'],
              extra_compile_args=compile_flags,
              extra_link_args=link_args),
    Extension('GPy.util.choleskies_cython',
              sources=['src/GPy/util/choleskies_cython' + ext],
              include_dirs=['.'],
              extra_compile_args=compile_flags,
              extra_link_args=link_args),
    Extension('GPy.util.linalg_cython',
              sources=['src/GPy/util/linalg_cython' + ext],
              include_dirs=['.'],
              extra_compile_args=compile_flags,
              extra_link_args=link_args),
    Extension('GPy.kern.src.coregionalize_cython',
              sources=['src/GPy/kern/src/coregionalize_cython' + ext],
              include_dirs=['.'],
              extra_compile_args=compile_flags,
              extra_link_args=link_args),
    Extension('GPy.models.state_space_cython',
              sources=['src/GPy/models/state_space_cython' + ext],
              include_dirs=['.'],
              extra_compile_args=compile_flags,
              extra_link_args=link_args)
]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules)


def find_py_modules(pattern):
    return [os.path.splitext(os.path.basename(p))[0] for p in glob(pattern)]


setup(
        name='GPy',
        version=__version__,
        author=read_to_rst('AUTHORS.txt'),
        author_email="gpy.authors@gmail.com",
        description="The Gaussian Process Toolbox",
        long_description=desc,
        license="BSD 3-clause",
        keywords="machine-learning gaussian-processes kernels",
        url="http://sheffieldml.github.com/GPy/",
        download_url='https://github.com/SheffieldML/GPy/',
        cmdclass={
            'build_ext': build_ext
        },
        ext_modules=ext_modules,
        packages=find_packages('src'),
        package_dir={'': 'src'},
        include_package_data=True,
        py_modules=find_py_modules('src/*.py'),
        test_suite='GPy.testing',
        setup_requires=[
            # Setuptools 18.0 properly handles Cython extensions.
            'setuptools>=18.0',
            'numpy>=1.7'
        ],
        install_requires=[
            'numpy>=1.7',
            'scipy>=0.16',
            'six',
            'paramz>=0.9.0'
        ],
        extras_require={
            'docs': [
                'sphinx'
            ],
            'optional': [
                'mpi4py',
                'ipython>=4.0.0',
            ],
            'plotting': [
                'matplotlib >= 1.3',
                'plotly >= 1.8.6'
            ],
            'notebook': [
                'jupyter_client >= 4.0.6',
                'ipywidgets >= 4.0.3',
                'ipykernel >= 4.1.0',
                'notebook >= 4.0.5',
            ]
        },
        classifiers=[
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Framework :: IPython',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ],
        zip_safe=False
)

# Check config files and settings:
local_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          'src', 'GPy', 'installation.cfg'))
home = os.getenv('HOME') or os.getenv('USERPROFILE')
user_file = os.path.join(home, '.config', 'GPy', 'user.cfg')

print("")
try:
    if not os.path.exists(user_file):
        # Does an old config exist?
        old_user_file = os.path.join(home, '.gpy_user.cfg')
        if os.path.exists(old_user_file):
            # Move it to new location:
            print("GPy: Found old config file, moving to new location "
                  "{}".format(user_file))
            if not os.path.exists(os.path.dirname(user_file)):
                os.makedirs(os.path.dirname(user_file))
            os.rename(old_user_file, user_file)
        else:
            # No config file exists, save informative stub to user config
            # folder:
            print("GPy: Saving user configuration file to {}".format(user_file))
            if not os.path.exists(os.path.dirname(user_file)):
                os.makedirs(os.path.dirname(user_file))
            with open(user_file, 'w') as f:
                with open(local_file, 'r') as l:
                    tmp = l.read()
                    f.write(tmp)
    else:
        print("GPy: User configuration file at location {}".format(user_file))
except:
    print("GPy: Could not write user configuration file {}".format(user_file))
