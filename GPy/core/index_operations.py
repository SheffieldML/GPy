'''
Created on Oct 2, 2013

@author: maxzwiessele
'''
import numpy
import itertools

class ConstraintIndexOperations(object):
    '''
    Index operations for storing parameter index _properties
    This class enables index with slices retrieved from object.__getitem__ calls.
    Adding an index will add the selected indexes by the slice of an indexarray
    indexing a shape shaped array to the flattened index array. Remove will
    remove the selected slice indices from the flattened array.
    You can give an offset to set an offset for the given indices in the
    index array, for multiparameter handling.
    
    
    '''
    def __init__(self):
        self._properties = {}
        
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
        
    def add(self, prop, indices, shape, offset=False):
        ind = create_raveled_indices(indices, shape, offset)
        if prop in self._properties:
            self._properties[prop] = combine_indices(self._properties[prop], ind)
            return 
        for a in self.properties(): 
            if numpy.all(a==prop) and a._parent_index == prop._parent_index:
                self._properties[a] = combine_indices(self._properties[a], ind)
                return
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
                return removed.astype(int)
        else:
            for a in self.properties(): 
                if numpy.all(a==prop) and a._parent_index == prop._parent_index:
                    ind = create_raveled_indices(indices, shape, offset)
                    diff = remove_indices(self[a], ind)
                    removed = numpy.intersect1d(self[a], ind, True)
                    if not index_empty(diff):
                        self._properties[a] = diff
                    else:
                        del self._properties[a]
                    return removed.astype(int)
        return numpy.array([]).astype(int)
    def __getitem__(self, prop):
        return self._properties[prop]
       
class TieIndexOperations(object):
    def __init__(self, params):
        self.params = params
        self.tied_from = ConstraintIndexOperations()
        self.tied_to = ConstraintIndexOperations()
    def add(self, tied_from, tied_to):
        self.tied_from.add(tied_to, tied_from._current_slice, tied_from._realshape, self.params._offset(tied_from))
        self.tied_to.add(tied_to, tied_to._current_slice, tied_to._realshape, self.params._offset(tied_to))
    def remove(self, tied_from, tied_to):
        self.tied_from.remove(tied_to, tied_from._current_slice, tied_from._realshape, self.params._offset(tied_from))
        self.tied_to.remove(tied_to, tied_to._current_slice, tied_to._realshape, self.params._offset(tied_to))
    def iter_from_to_indices(self):
        for k, f in self.tied_from.iteritems():
            yield f, self.tied_to[k]
    def iter_from_items(self):
        for f, i in self.tied_from.iteritems():
            yield f, i
    def iter_properties(self):
        return self.tied_from.iter_properties()
    def properties(self):
        return self.tied_from.properties()
    def from_to_indices(self, param):
        return self.tied_from[param], self.tied_to[param]
    
def create_raveled_indices(index, shape, offset=False):
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



