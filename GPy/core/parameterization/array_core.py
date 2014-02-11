# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

__updated__ = '2013-12-16'

import numpy as np
from parameter_core import Observable

class ListArray(np.ndarray):
    """
    ndarray which can be stored in lists and checked if it is in.
    WARNING: This overrides the functionality of x==y!!!
    Use numpy.equal(x,y) for element-wise equality testing.
    """
    def __new__(cls, input_array):
        obj = np.asanyarray(input_array).view(cls)
        return obj
    #def __eq__(self, other):
    #    return other is self

class ParamList(list):

    def __contains__(self, other):
        for el in self:
            if el is other:
                return True
        return False

    pass
class C(np.ndarray):
    __array_priority__ = 1.
    def __new__(cls, array):
        obj = array.view(cls)
        return obj
    #def __array_finalize__(self, obj):
    #    #print 'finalize'
    #    return obj
    def __array_prepare__(self, out_arr, context):
        #print 'prepare'
        while type(out_arr) is C:
            out_arr = out_arr.base
        return out_arr
    def __array_wrap__(self, out_arr, context):
        #print 'wrap', type(self), type(out_arr), context
        while type(out_arr) is C:
            out_arr = out_arr.base
        return out_arr

class ObservableArray(ListArray, Observable):
    """
    An ndarray which reports changes to its observers.
    The observers can add themselves with a callable, which
    will be called every time this array changes. The callable
    takes exactly one argument, which is this array itself.
    """
    __array_priority__ = -1 # Never give back ObservableArray
    def __new__(cls, input_array):
        cls.__name__ = "ObservableArray\n     "
        obj = super(ObservableArray, cls).__new__(cls, input_array).view(cls)
        obj._observers_ = {}
        return obj
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._observers_ = getattr(obj, '_observers_', None)
    def __setitem__(self, s, val, update=True):
        super(ObservableArray, self).__setitem__(s, val)
        if update:
            self._notify_observers()
#         if self.ndim:
#             if not np.all(np.equal(self[s], val)):
#                 super(ObservableArray, self).__setitem__(s, val)
#                 if update:
#                     self._notify_observers()
#         else:
#             if not np.all(np.equal(self, val)):
#                 super(ObservableArray, self).__setitem__(Ellipsis, val)
#                 if update:
#                     self._notify_observers()
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))
    def __setslice__(self, start, stop, val):
        return self.__setitem__(slice(start, stop), val)
    def __copy__(self, *args):
        return ObservableArray(self.view(np.ndarray).copy())
    def copy(self, *args):
        return self.__copy__(*args)
