'''
Created on 27 Feb 2014

@author: maxz
'''

from collections import defaultdict

def intarray_default_factory():
    import numpy as np
    return np.int_([])

class IntArrayDict(defaultdict):
    def __init__(self, default_factory=None):
        """
        Default will be self._default, if not set otherwise
        """
        defaultdict.__init__(self, intarray_default_factory)

class ArrayList(list):
    """
    List to store ndarray-likes in.
    It will look for 'is' instead of calling __eq__ on each element.
    """
    def __contains__(self, other):
        for el in self:
            if el is other:
                return True
        return False

    def index(self, item):
        index = 0
        for el in self:
            if el is item:
                return index
            index += 1
        raise ValueError, "{} is not in list".format(item)
    pass

class ObservablesList(object):
    def __init__(self):
        self._poc = []

    def remove(self, value):
        return self._poc.remove(value)


    def __delitem__(self, ind):
        return self._poc.__delitem__(ind)


    def __setitem__(self, ind, item):
        return self._poc.__setitem__(ind, item)


    def __getitem__(self, ind):
        return self._poc.__getitem__(ind)


    def __repr__(self):
        return self._poc.__repr__()


    def append(self, obj):
        return self._poc.append(obj)


    def index(self, value):
        return self._poc.index(value)


    def extend(self, iterable):
        return self._poc.extend(iterable)


    def __str__(self):
        return self._poc.__str__()


    def __iter__(self):
        return self._poc.__iter__()


    def insert(self, index, obj):
        return self._poc.insert(index, obj)


    def __len__(self):
        return self._poc.__len__()

    def __deepcopy__(self, memo):
        s = ObservablesList()
        import copy
        s._poc = copy.deepcopy(self._poc, memo)
        return s

    def __getstate__(self):
        from ...util.caching import Cacher
        obs = []
        for p, o, c in self:
            if (getattr(o, c.__name__, None) is not None 
                and not isinstance(o, Cacher)):
                obs.append((p,o,c.__name__))
        return obs

    def __setstate__(self, state):
        self._poc = []
        for p, o, c in state:
            self._poc.append((p,o,getattr(o, c)))

    pass
