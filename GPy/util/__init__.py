# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import linalg
import misc
import plot
import squashers
import Tango
import warping_functions
import datasets
import mocap
import visualize
import decorators
import classification
import latent_space_visualizations

try:
    import sympy
    _sympy_available = True
    del sympy
except ImportError as e:
    _sympy_available = False

if _sympy_available:
    import symbolic

import netpbmfile
