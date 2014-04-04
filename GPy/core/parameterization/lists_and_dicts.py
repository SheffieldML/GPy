'''
Created on 27 Feb 2014

@author: maxz
'''

from collections import defaultdict
import weakref

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

    def __getitem__(self, ind):
        p,o,c = self._poc[ind]
        return p, o(), c

    def remove(self, priority, observable, callble):
        """
        """
        self.flush()
        for i in range(len(self) - 1, -1, -1):
            p,o,c = self[i]
            if priority==p and observable==o and callble==c:
                del self._poc[i]

    def __repr__(self):
        return self._poc.__repr__()

    def add(self, priority, observable, callble):
        ins = 0
        for pr, _, _ in self:
            if priority > pr:
                break
            ins += 1
        self._poc.insert(ins, (priority, weakref.ref(observable), callble))
        
    def __str__(self):
        ret = []
        curr_p = None
        for p, o, c in self:
            curr = ''
            if curr_p != p:
                pre = "{!s}: ".format(p)
                curr_pre = pre
            else: curr_pre = " "*len(pre)
            curr_p = p
            curr += curr_pre
            ret.append(curr + ", ".join(map(repr, [o,c])))
        return '\n'.join(ret)

    def flush(self):
        self._poc = [(p,o,c) for p,o,c in self._poc if o() is not None]

    def __iter__(self):
        self.flush()
        for p, o, c in self._poc:
            if o() is not None:
                yield p, o(), c 

    def __len__(self):
        self.flush()
        return self._poc.__len__()

    def __deepcopy__(self, memo):
        self.flush()
        s = ObservablesList()
        import copy
        s._poc = copy.deepcopy(self._poc, memo)
        return s

    def __getstate__(self):
        self.flush()
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
            self.add(p,o,getattr(o, c))

    pass
