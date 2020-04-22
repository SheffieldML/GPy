"""

In terms of Gaussian Processes, a kernel is a function that specifies the degree of similarity between variables given their relative positions in parameter space. If known variables *x* and *x'* are close together then observed variables *y* and *y'* may also be similar, depending on the kernel function and its parameters.

.. inheritance-diagram:: GPy.kern.src.kern.Kern
   :top-classes: GPy.core.parameterization.parameterized.Parameterized

"""

from . import psi_comp
