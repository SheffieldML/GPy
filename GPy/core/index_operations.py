'''
Created on Oct 2, 2013

@author: maxzwiessele
'''
import numpy

class ParameterIndexOperations(object):
    '''
    Index operations for storing parameter index _properties
    This class enables indexing with slices retrieved from object.__getitem__ calls.
    
    :param _shape: _shape of parameter, handled by this index restriction class
    '''
    

    def __init__(self, param):
        self._properties = {}
        self._shape = param.shape
    
    def iteritems(self):
        for prop, indices in self._properties.iteritems():
            yield prop, numpy.unravel_index(indices, self._shape)
    
    def keys(self):
        return self._properties.keys()

    def items(self):
        return self._properties.items()
    
    def indices(self, prop):
        """
        get indices for prop prop.
        these indices can be used as X[indices], which will be a flattened array of
        all restricted elements
        """
        return numpy.unravel_index(self._properties[prop], self._shape)
    
    def add(self, prop, indices):
        ind = self.create_raveled_indices(indices)
        if prop in self._properties:
            self._properties[prop] = numpy.union1d(self._properties[prop], ind)
        else:
            self._properties[prop] = ind
    
    def remove(self, prop, indices):
        if prop in self._properties:
            ind = self.create_raveled_indices(indices)
            diff = numpy.setdiff1d(self._properties[prop], ind, True)
            if numpy.size(diff):
                self._properties[prop] = diff
            else:
                del self._properties[prop] 
            
    def create_raveled_indices(self, indices):
        if isinstance(indices, (tuple, list)):
            i = [slice(None)] + list(indices)
        else:
            i = [slice(None), indices]
        return numpy.array(numpy.ravel_multi_index(numpy.indices(self._shape)[i], self._shape)).flatten()
