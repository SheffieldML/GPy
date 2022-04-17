# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Introduction
^^^^^^^^^^^^

The examples in this package usually depend on `pods <https://github.com/sods/ods>`_ so make sure 
you have that installed before running examples. The easiest way to do this is to run `pip install pods`. `pods` enables access to 3rd party data required for most of the examples. 

The examples are executable and self-contained workflows in that they have their own source data, create their own models, kernels and other objects as needed, execute optimisation as required, and display output.

Viewing the source code of each model will clarify the steps taken in its execution, and may provide inspiration for developing of user-specific applications of `GPy`.

"""
from . import classification
from . import regression
from . import dimensionality_reduction
from . import non_gaussian
