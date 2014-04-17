# Copyright (c) 2013, 2014 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernel import Kernel
from linear import Linear
from mlp import MLP
#from rbf import RBF
# TODO need to fix this in a config file.
try:
    import sympy as sym
    sympy_available=True
except ImportError:
    sympy_available=False

if sympy_available:
    # These are likelihoods that rely on symbolic.
    from symbolic import Symbolic
