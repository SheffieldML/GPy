# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

__updated__ = '2014-03-21'

import numpy as np
from parameter_core import Observable

class ObsAr(np.ndarray, Observable):
    """
    An ndarray which reports changes to its observers.
    The observers can add themselves with a callable, which
    will be called every time this array changes. The callable
    takes exactly one argument, which is this array itself.
    """
    __array_priority__ = -1 # Never give back ObsAr
    def __new__(cls, input_array, *a, **kw):
        if not isinstance(input_array, ObsAr):
            obj = np.atleast_1d(np.require(input_array, dtype=np.float64, requirements=['W', 'C'])).view(cls)
        else: obj = input_array
        #cls.__name__ = "ObsAr" # because of fixed printing of `array` in np printing
        super(ObsAr, obj).__init__(*a, **kw)
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._observer_callables_ = getattr(obj, '_observer_callables_', None)

    def __array_wrap__(self, out_arr, context=None):
        return out_arr.view(np.ndarray)

    def __reduce__(self):
        func, args, state = np.ndarray.__reduce__(self)
        return func, args, (state, Observable._getstate(self))

    def __setstate__(self, state):
        np.ndarray.__setstate__(self, state[0])
        Observable._setstate(self, state[1])

    def _s_not_empty(self, s):
        # this checks whether there is something picked by this slice.
        return True
        # TODO:  disarmed, for performance increase,
        if not isinstance(s, (list,tuple,np.ndarray)):
            return True
        if isinstance(s, (list,tuple)):
            return len(s)!=0
        if isinstance(s, np.ndarray):
            if s.dtype is bool:
                return np.all(s)
            else:
                return s.size != 0

    def __setitem__(self, s, val):
        if self._s_not_empty(s):
            super(ObsAr, self).__setitem__(s, val)
            self.notify_observers()

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        return self.__setitem__(slice(start, stop), val)

    def __copy__(self, *args):
        return ObsAr(self.view(np.ndarray).copy())

    def copy(self, *args):
        return self.__copy__(*args)

    def __ilshift__(self, *args, **kwargs):
        r = np.ndarray.__ilshift__(self, *args, **kwargs)
        self.notify_observers()
        return r

    def __irshift__(self, *args, **kwargs):
        r = np.ndarray.__irshift__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __ixor__(self, *args, **kwargs):
        r = np.ndarray.__ixor__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __ipow__(self, *args, **kwargs):
        r = np.ndarray.__ipow__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __ifloordiv__(self, *args, **kwargs):
        r = np.ndarray.__ifloordiv__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __isub__(self, *args, **kwargs):
        r = np.ndarray.__isub__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __ior__(self, *args, **kwargs):
        r = np.ndarray.__ior__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __itruediv__(self, *args, **kwargs):
        r = np.ndarray.__itruediv__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __idiv__(self, *args, **kwargs):
        r = np.ndarray.__idiv__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __iand__(self, *args, **kwargs):
        r = np.ndarray.__iand__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __imod__(self, *args, **kwargs):
        r = np.ndarray.__imod__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __iadd__(self, *args, **kwargs):
        r = np.ndarray.__iadd__(self, *args, **kwargs)
        self.notify_observers()
        return r


    def __imul__(self, *args, **kwargs):
        r = np.ndarray.__imul__(self, *args, **kwargs)
        self.notify_observers()
        return r


#     def __rrshift__(self, *args, **kwargs):
#         r = np.ndarray.__rrshift__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __ror__(self, *args, **kwargs):
#         r =  np.ndarray.__ror__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rxor__(self, *args, **kwargs):
#         r = np.ndarray.__rxor__(self, *args, **kwargs)
#         self.notify_observers()
#         return r



#     def __rdivmod__(self, *args, **kwargs):
#         r = np.ndarray.__rdivmod__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __radd__(self, *args, **kwargs):
#         r = np.ndarray.__radd__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rdiv__(self, *args, **kwargs):
#         r = np.ndarray.__rdiv__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rtruediv__(self, *args, **kwargs):
#         r = np.ndarray.__rtruediv__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rshift__(self, *args, **kwargs):
#         r = np.ndarray.__rshift__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rmul__(self, *args, **kwargs):
#         r = np.ndarray.__rmul__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rpow__(self, *args, **kwargs):
#         r = np.ndarray.__rpow__(self, *args, **kwargs)
#         self.notify_observers()
#         return r


#     def __rsub__(self, *args, **kwargs):
#         r = np.ndarray.__rsub__(self, *args, **kwargs)
#         self.notify_observers()
#         return r

#     def __rfloordiv__(self, *args, **kwargs):
#         r = np.ndarray.__rfloordiv__(self, *args, **kwargs)
#         self.notify_observers()
#         return r

