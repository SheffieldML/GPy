'''
Created on Oct 2, 2013

@author: maxzwiessele
'''
import numpy
from numpy.lib.function_base import vectorize
from param import Param
from collections import defaultdict

class ParamDict(defaultdict):
    def __init__(self):
        """
        Default will be self._default, if not set otherwise
        """
        defaultdict.__init__(self, self.default_factory)
        
    def __getitem__(self, key):
        try:
            return defaultdict.__getitem__(self, key)
        except KeyError:
            for a in self.iterkeys():
                if numpy.all(a==key) and a._parent_index_==key._parent_index_:
                    return defaultdict.__getitem__(self, a)
            raise        
        
    def __contains__(self, key):
        if defaultdict.__contains__(self, key):
            return True
        for a in self.iterkeys():
            if numpy.all(a==key) and a._parent_index_==key._parent_index_:
                return True
        return False

    def __setitem__(self, key, value):
        if isinstance(key, Param):
            for a in self.iterkeys():
                if numpy.all(a==key) and a._parent_index_==key._parent_index_:
                    return super(ParamDict, self).__setitem__(a, value)
        defaultdict.__setitem__(self, key, value)

class SetDict(ParamDict):
    def default_factory(self):
        return set()

class IntArrayDict(ParamDict):
    def default_factory(self):
        return numpy.int_([])

class ParameterIndexOperations(object):
    '''
    Index operations for storing param index _properties
    This class enables index with slices retrieved from object.__getitem__ calls.
    Adding an index will add the selected indexes by the slice of an indexarray
    indexing a shape shaped array to the flattened index array. Remove will
    remove the selected slice indices from the flattened array.
    You can give an offset to set an offset for the given indices in the
    index array, for multi-param handling.
    '''
    def __init__(self):
        self._properties = ParamDict()
        #self._reverse = collections.defaultdict(list)
        
    def __getstate__(self):
        return self._properties#, self._reverse
        
    def __setstate__(self, state):
        self._properties = state[0]
        # self._reverse = state[1]

    def iteritems(self):
        return self._properties.iteritems()
    
    def properties(self):
        return self._properties.keys()

    def iter_properties(self):
        return self._properties.iterkeys()
    
    def clear(self):
        self._properties.clear()
    
    def size(self):
        return reduce(lambda a,b: a+b.size, self.iterindices(), 0)    
    
    def iterindices(self):
        return self._properties.itervalues()
    
    def indices(self):
        return self._properties.values()

    def properties_for(self, index):
        return vectorize(lambda i: [prop for prop in self.iter_properties() if i in self._properties[prop]], otypes=[list])(index)
        
    def add(self, prop, indices):
        try:
            self._properties[prop] = combine_indices(self._properties[prop], indices)
        except KeyError: 
            self._properties[prop] = indices
    
    def remove(self, prop, indices):
        if prop in self._properties:
            diff = remove_indices(self[prop], indices)
            removed = numpy.intersect1d(self[prop], indices, True)
            if not index_empty(diff):
                self._properties[prop] = diff
            else:
                del self._properties[prop]
            return removed.astype(int)
        return numpy.array([]).astype(int)
    def __getitem__(self, prop):
        return self._properties[prop]
       
def combine_indices(arr1, arr2):
    return numpy.union1d(arr1, arr2)

def remove_indices(arr, to_remove):
    return numpy.setdiff1d(arr, to_remove, True)

def index_empty(index):
    return numpy.size(index) == 0 



