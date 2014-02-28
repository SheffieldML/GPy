'''
Created on 27 Feb 2014

@author: maxz
'''

from collections import defaultdict
class DefaultArrayDict(defaultdict):
    def __init__(self):
        """
        Default will be self._default, if not set otherwise
        """
        defaultdict.__init__(self, self.default_factory)

class SetDict(DefaultArrayDict):
    def default_factory(self):
        return set()

class IntArrayDict(DefaultArrayDict):
    def default_factory(self):
        import numpy as np
        return np.int_([])

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

    pass
