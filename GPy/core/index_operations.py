'''
Created on Oct 2, 2013

@author: maxzwiessele
'''
import numpy
from numpy.lib.function_base import vectorize
from parameter import Param

class ParamDict(dict):
    def __getitem__(self, key):
        try:
            return super(ParamDict, self).__getitem__(key)
        except KeyError:
            for a in self.iterkeys():
                if numpy.all(a==key) and a._parent_index_==key._parent_index_:
                    return super(ParamDict, self).__getitem__(a)
            raise        
        
    def __contains__(self, key):
        if super(ParamDict, self).__contains__(key):
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
            raise KeyError, key
        super(ParamDict, self).__setitem__(key, value)
        

class ParameterIndexOperations(object):
    '''
    Index operations for storing parameter index _properties
    This class enables index with slices retrieved from object.__getitem__ calls.
    Adding an index will add the selected indexes by the slice of an indexarray
    indexing a shape shaped array to the flattened index array. Remove will
    remove the selected slice indices from the flattened array.
    You can give an offset to set an offset for the given indices in the
    index array, for multi-parameter handling.
    '''
    def __init__(self):
        self._properties = ParamDict()
        #self._reverse = collections.defaultdict(list)
        
    def __getstate__(self):
        return self._properties, self._reverse
        
    def __setstate__(self, state):
        self._properties = state[0]
        self._reverse = state[1]

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
#         already_seen = dict()
#         for ni in index:
#             if ni not in already_seen:
#                 already_seen[ni] = [prop for prop in self.iter_properties() if ni in self._properties[prop]] 
#             yield already_seen[ni]
        return vectorize(lambda i: [prop for prop in self.iter_properties() if i in self._properties[prop]], otypes=[list])(index)
        
    def add(self, prop, indices, shape, offset=False):
        ind = create_raveled_indices(indices, shape, offset)
        #[self._reverse[i].__add__(prop) for i in ind]
        try:
            self._properties[prop] = combine_indices(self._properties[prop], ind)
        except KeyError: 
#         for a in self.properties(): 
#             if numpy.all(a==prop) and a._parent_index_ == prop._parent_index_:
#                 self._properties[a] = combine_indices(self._properties[a], ind)
#                 return
            self._properties[prop] = ind
    
    def remove(self, prop, indices, shape, offset=False):
        if prop in self._properties:
            ind = create_raveled_indices(indices, shape, offset)
            diff = remove_indices(self[prop], ind)
            removed = numpy.intersect1d(self[prop], ind, True)
            if not index_empty(diff):
                self._properties[prop] = diff
            else:
                del self._properties[prop]
            #[self._reverse[i].remove(prop) for i in removed if prop in self._reverse[i]] 
            return removed.astype(int)
#         else:
#             for a in self.properties(): 
#                 if numpy.all(a==prop) and a._parent_index_ == prop._parent_index_:
#                     ind = create_raveled_indices(indices, shape, offset)
#                     diff = remove_indices(self[a], ind)
#                     removed = numpy.intersect1d(self[a], ind, True)
#                     if not index_empty(diff):
#                         self._properties[a] = diff
#                     else:
#                         del self._properties[a]
#                     [self._reverse[i].remove(a) for i in removed if a in self._reverse[i]] 
#                     return removed.astype(int)
        return numpy.array([]).astype(int)
    def __getitem__(self, prop):
        return self._properties[prop]
       
class TieIndexOperations(object):
    def __init__(self, params):
        self.params = params
        self.tied_from = ParameterIndexOperations()
        self.tied_to = ParameterIndexOperations()
    def add(self, tied_from, tied_to):
        self.tied_from.add(tied_to, tied_from._current_slice_, tied_from._realshape_, self.params._offset(tied_from))
        self.tied_to.add(tied_to, tied_to._current_slice_, tied_to._realshape_, self.params._offset(tied_to))
    def remove(self, tied_from, tied_to):
        self.tied_from.remove(tied_to, tied_from._current_slice_, tied_from._realshape_, self.params._offset(tied_from))
        self.tied_to.remove(tied_to, tied_to._current_slice_, tied_to._realshape_, self.params._offset(tied_to))
    def from_to_for(self, index):
        return self.tied_from.properties_for(index), self.tied_to.properties_for(index)
    def iter_from_to_indices(self):
        for k, f in self.tied_from.iteritems():
            yield f, self.tied_to[k]
    def iter_to_indices(self):
        return self.tied_to.iterindices()
    def iter_from_indices(self):
        return self.tied_from.iterindices()
    def iter_from_items(self):
        for f, i in self.tied_from.iteritems():
            yield f, i
    def iter_properties(self):
        return self.tied_from.iter_properties()
    def properties(self):
        return self.tied_from.properties()
    def from_to_indices(self, param):
        return self.tied_from[param], self.tied_to[param]
    
def create_raveled_indices(index, shape, offset=0):
    if isinstance(index, (tuple, list)): i = [slice(None)] + list(index)
    else: i = [slice(None), index]
    ind = numpy.array(numpy.ravel_multi_index(numpy.indices(shape)[i], shape)).flat + numpy.int_(offset)
    return ind

def combine_indices(arr1, arr2):
    return numpy.union1d(arr1, arr2)

def remove_indices(arr, to_remove):
    return numpy.setdiff1d(arr, to_remove, True)

def index_empty(index):
    return numpy.size(index) == 0 



