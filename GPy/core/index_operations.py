'''
Created on Oct 2, 2013

@author: maxzwiessele
'''
import numpy

class ParameterIndexOperations(object):
    '''
    Index operations for storing parameter index restrictions
    This class enables indexing with slices retrieved from object.__getitem__ calls.
    
    :param shape: shape of parameter, handled by this index restriction class
    '''
    

    def __init__(self, shape):
        self.restrictions = {}
        self.shape = shape
    
    def get_restriction_indices(self, restriction):
        """
        get indices for restriction restriction.
        these indices can be used as X[indices], which will be a flattened array of
        all restricted elements
        """
        return numpy.unravel_index(self.restrictions[restriction], self.shape)
    
    def add_restriction(self, restriction, indices):
        ind = self._create_raveled_indices(indices)
        if restriction in self.restrictions:
            self.restrictions[restriction] = numpy.union1d(self.restrictions[restriction], ind)
        else:
            self.restrictions[restriction] = ind
    
    def remove_restriction(self, restriction, indices):
        if restriction in self.restrictions:
            ind = self._create_raveled_indices(indices)
            diff = numpy.setdiff1d(self.restrictions[restriction], ind, True)
            if numpy.size(diff):
                self.restrictions[restriction] = diff
            else:
                del self.restrictions[restriction] 
            
    def _create_raveled_indices(self, indices):
        if isinstance(indices, (tuple, list)):
            i = [slice(None)] + list(indices)
        else:
            i = [slice(None), indices]
        return numpy.array(numpy.ravel_multi_index(numpy.indices(self.shape)[i], self.shape)).flatten()
