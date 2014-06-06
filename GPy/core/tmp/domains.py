'''
Created on 4 Jun 2013

@author: maxz

(Hyper-)Parameter domains defined for :py:mod:`~GPy.core.priors` and :py:mod:`~GPy.kern`.
These domains specify the legitimate realm of the parameters to live in.

:const:`~GPy.core.domains.REAL` :
    real domain, all values in the real numbers are allowed

:const:`~GPy.core.domains.POSITIVE`:
    positive domain, only positive real values are allowed
    
:const:`~GPy.core.domains.NEGATIVE`:
    same as :const:`~GPy.core.domains.POSITIVE`, but only negative values are allowed
    
:const:`~GPy.core.domains.BOUNDED`:
    only values within the bounded range are allowed,
    the bounds are specified withing the object with the bounded range
'''

REAL = 'real'
POSITIVE = "positive"
NEGATIVE = 'negative'
BOUNDED = 'bounded'
