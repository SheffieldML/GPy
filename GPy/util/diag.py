'''
.. module:: GPy.util.diag

.. moduleauthor:: Max Zwiessele <ibinbei@gmail.com>

'''
__updated__ = '2013-12-03'

import numpy as np

def view(A, offset=0):
    """
    Get a view on the diagonal elements of a 2D array.
    
    This is actually a view (!) on the diagonal of the array, so you can 
    in-place adjust the view.
    
    :param :class:`ndarray` A: 2 dimensional numpy array
    :param int offset: view offset to give back (negative entries allowed)
    :rtype: :class:`ndarray` view of diag(A)
    
    >>> import numpy as np
    >>> X = np.arange(9).reshape(3,3)
    >>> view(X)
    array([0, 4, 8])
    >>> d = view(X)
    >>> d += 2
    >>> view(X)
    array([ 2,  6, 10])
    >>> view(X, offset=-1)
    array([3, 7])
    >>> subtract(X, 3, offset=-1)
    array([[ 2,  1,  2],
           [ 0,  6,  5],
           [ 6,  4, 10]])
    """
    from numpy.lib.stride_tricks import as_strided
    assert A.ndim == 2, "only implemented for 2 dimensions"
    assert A.shape[0] == A.shape[1], "attempting to get the view of non-square matrix?!" 
    if offset > 0:
        return as_strided(A[0, offset:], shape=(A.shape[0] - offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    elif offset < 0:
        return as_strided(A[-offset:, 0], shape=(A.shape[0] + offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    else:
        return as_strided(A, shape=(A.shape[0], ), strides=((A.shape[0]+1)*A.itemsize, ))

def _diag_ufunc(A,b,offset,func):
    dA = view(A, offset); func(dA,b,dA)
    return A

def times(A, b, offset=0):
    """
    Times the view of A with b in place (!).
    Returns modified A 
    Broadcasting is allowed, thus b can be scalar.
    
    if offset is not zero, make sure b is of right shape!
    
    :param ndarray A: 2 dimensional array
    :param ndarray-like b: either one dimensional or scalar
    :param int offset: same as in view.
    :rtype: view of A, which is adjusted inplace
    """
    return _diag_ufunc(A, b, offset, np.multiply)
multiply = times

def divide(A, b, offset=0):
    """
    Divide the view of A by b in place (!).
    Returns modified A 
    Broadcasting is allowed, thus b can be scalar.
    
    if offset is not zero, make sure b is of right shape!
    
    :param ndarray A: 2 dimensional array
    :param ndarray-like b: either one dimensional or scalar
    :param int offset: same as in view.
    :rtype: view of A, which is adjusted inplace
    """
    return _diag_ufunc(A, b, offset, np.divide)

def add(A, b, offset=0):
    """
    Add b to the view of A in place (!).
    Returns modified A.
    Broadcasting is allowed, thus b can be scalar.
    
    if offset is not zero, make sure b is of right shape!
    
    :param ndarray A: 2 dimensional array
    :param ndarray-like b: either one dimensional or scalar
    :param int offset: same as in view.
    :rtype: view of A, which is adjusted inplace
    """
    return _diag_ufunc(A, b, offset, np.add)

def subtract(A, b, offset=0):
    """
    Subtract b from the view of A in place (!).
    Returns modified A.
    Broadcasting is allowed, thus b can be scalar.
    
    if offset is not zero, make sure b is of right shape!
    
    :param ndarray A: 2 dimensional array
    :param ndarray-like b: either one dimensional or scalar
    :param int offset: same as in view.
    :rtype: view of A, which is adjusted inplace
    """
    return _diag_ufunc(A, b, offset, np.subtract)
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()