"""
.. module:: GPy.util.subarray_and_sorting

.. moduleauthor:: Max Zwiessele <ibinbei@gmail.com>

"""
__updated__ = "2014-05-21"


def common_subarrays(X, axis=0):
    """
    Find common subarrays of 2 dimensional X, where axis is the axis to apply the search over.
    Common subarrays are returned as a dictionary of <subarray, [index]> pairs, where
    the subarray is a tuple representing the subarray and the index is the index
    for the subarray in X, where index is the index to the remaining axis.

    :param :class:`np.ndarray` X: 2d array to check for common subarrays in
    :param int axis: axis to apply subarray detection over.
        When the index is 0, compare rows -- columns, otherwise.

    Examples:
    =========

    In a 2d array:
    >>> import numpy as np
    >>> X = np.zeros((3,6), dtype=bool)
    >>> X[[1,1,1],[0,4,5]] = 1; X[1:,[2,3]] = 1
    >>> X
    array([[False, False, False, False, False, False],
           [ True, False,  True,  True,  True,  True],
           [False, False,  True,  True, False, False]], dtype=bool)
    >>> d = common_subarrays(X,axis=1)
    >>> len(d)
    3
    >>> X[:, d[tuple(X[:,0])]]
    array([[False, False, False],
           [ True,  True,  True],
           [False, False, False]], dtype=bool)
    >>> d[tuple(X[:,4])] == d[tuple(X[:,0])] == [0, 4, 5]
    True
    >>> d[tuple(X[:,1])]
    [1]
    """
    from collections import defaultdict
    from itertools import count
    from operator import iadd

    assert X.ndim == 2 and axis in (0, 1), "Only implemented for 2D arrays"
    subarrays = defaultdict(list)
    cnt = count()

    def accumulate(x, s, c):
        t = tuple(x)
        col = next(c)
        iadd(s[t], [col])
        return None

    if axis == 0:
        [accumulate(x, subarrays, cnt) for x in X]
    else:
        [accumulate(x, subarrays, cnt) for x in X.T]
    return subarrays


if __name__ == "__main__":
    import doctest

    doctest.testmod()
