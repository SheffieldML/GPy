# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
(Hyper-)Parameter domains defined for :py:mod:`~GPy.core.priors` and :py:mod:`~GPy.kern`.
These domains specify the legitimate realm of the parameters to live in.

:const:`~GPy.core.domains._REAL` :
    real domain, all values in the real numbers are allowed

:const:`~GPy.core.domains._POSITIVE`:
    positive domain, only positive real values are allowed

:const:`~GPy.core.domains._NEGATIVE`:
    same as :const:`~GPy.core.domains._POSITIVE`, but only negative values are allowed

:const:`~GPy.core.domains._BOUNDED`:
    only values within the bounded range are allowed,
    the bounds are specified withing the object with the bounded range
"""

_REAL = 'real'
_POSITIVE = "positive"
_NEGATIVE = 'negative'
_BOUNDED = 'bounded'
