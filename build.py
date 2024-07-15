import sys
import numpy as np
from setuptools import setup, Extension


def ismac():
    return sys.platform[:6] == "darwin"

if ismac():
    compile_flags = ["-O3"]
    link_args = []
else:
    compile_flags = ["-fopenmp", "-O3"]
    link_args = ["-lgomp"]

extensions = [
    Extension(
        name="GPy.kern.src.stationary_cython",
        sources=[
            "GPy/kern/src/stationary_cython.pyx",
            "GPy/kern/src/stationary_utils.c",
        ],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=compile_flags,
        extra_link_args=link_args,
    ),
    Extension(
        name="GPy.util.choleskies_cython",
        sources=["GPy/util/choleskies_cython.pyx"],
        include_dirs=[np.get_include(), "."],
        extra_link_args=link_args,
        extra_compile_args=compile_flags,
    ),
    Extension(
        name="GPy.util.linalg_cython",
        sources=["GPy/util/linalg_cython.pyx"],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=compile_flags,
        extra_link_args=link_args,
    ),
    Extension(
        name="GPy.kern.src.coregionalize_cython",
        sources=["GPy/kern/src/coregionalize_cython.pyx"],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=compile_flags,
        extra_link_args=link_args,
    ),
    Extension(
        name="GPy.models.state_space_cython",
        sources=["GPy/models/state_space_cython.pyx"],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=compile_flags,
        extra_link_args=link_args,
    ),
]

def build(setup_kwargs):
    """Needed for the poetry building interface."""
    setup_kwargs.update({
        'ext_modules': extensions,
        # 'include_dirs': [np.get_include()],
    })
