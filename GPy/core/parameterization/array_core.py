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
    def __eq__(self, other):
        return other is self

class ObservableArray(ListArray, Observable):
    """
    An ndarray which reports changes to its observers.
    The observers can add themselves with a callable, which
    will be called every time this array changes. The callable
    takes exactly one argument, which is this array itself.
    """
    __array_priority__ = 0 # Never give back Param
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
        if self.ndim:
            if not np.all(np.equal(self[s], val)):
                super(ObservableArray, self).__setitem__(s, val)
                if update:
                    self._notify_observers()
        else:
            if not np.all(np.equal(self, val)):
                super(ObservableArray, self).__setitem__(Ellipsis, val)
                if update:
                    self._notify_observers()
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))
    def __setslice__(self, start, stop, val):
        return self.__setitem__(slice(start, stop), val)  
    def __copy__(self, *args):
        return ObservableArray(self.base.base.copy(*args))
    def copy(self, *args):
        return self.__copy__(*args)
