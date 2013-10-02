'''
Created on 4 Sep 2013

@author: maxz
'''
import re
import itertools
import numpy
from GPy.core.transformations import Logexp
_index_re = re.compile('(?:_(\d+))+')  # pattern match for indices

def translate_param_names_to_parameters(param_names):
    """
    Naive translation from _get_param_names return to Parameters object.
    Assumptions:
     - array indices are at the and matching _\d+_\d+...
     - names are in order and names match field names
    """


class Parameters(object):
    def __init__(self, parameterlist, prefix=None, *args, **kwargs):
        self._params = parameterlist
        sizes = numpy.cumsum([0] + self.sizes)
        self._param_slices = itertools.starmap(lambda start,stop: slice(start, stop), zip(sizes, sizes[1:]))
        for p in parameterlist:
            self.__setattr__(p.name, p)
        
    def grep_param_names(self, regexp):
        """
        Wrapper for parameterized.grep_param_names
        """
        pass
    
    def _get_params(self):
        return numpy.hstack([x._get_params() for x in self._params])
    
    def _set_params(self, params):
        [p._set_params(params[s]) for s in self._param_slices]
    
    def _get_params_transformed(self):
        return numpy.hstack([x._get_params_transformed() for x in self._params])
    
    @property
    def names(self):
        return [x.name for x in self._params]

    @property
    def sizes(self):
        return [x.size for x in self._params]
    @property
    def constraints(self):
        return [x.constraints for x in self._params]
    @property
    def shapes(self):
        return [x.shape for x in self._params]
    @property
    def _constrs(self):
        return [x._constr for x in self._params]
    @property
    def _descs(self):
        return [x._desc for x in self._params]
    
    def __str__(self, header=True):
        nl = max([len(str(x)) for x in self.names + ["Name"]])
        sl = max([len(str(x)) for x in self._descs + ["Value"]])
        cl = max([len(str(x)) if x else 0 for x in self._constrs  + ["Constraint"]])
        format_spec = "  {{self.name:^{0}s}}  |  {{self._desc:^{1}s}}  |  {{self._constr:^{2}s}}  ".format(nl, sl, cl)
        if header:
            header = "  {{0:^{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  ".format(nl, sl, cl).format("Name", "Value", "Constraint")
            header += '\n' + '-'*len(header)
            return '\n'.join([header]+[x.__str__(format_spec=format_spec) for x in self._params])
        return '\n'.join([x.__str__(format_spec=format_spec) for x in self._params])
    pass

class ParameterIndexing(object):
    def __init__(self, corresponding_param):
        self.properties = {}
        self.param = corresponding_param
    def add(self, prop, s):
        if prop in self.properties.keys():
            self.properties[prop] = self.combine_indices(self.properties[prop], s) 
        else:
            self.properties[prop] = [numpy.r_[st] for st in s]
    def combine_indices(self, s1, s2):
        return [numpy.union1d(numpy.r_[ar1], numpy.r_[ar2]) for ar1, ar2 in itertools.izip_longest(s1, s2)]

class Parameter(object):
    tied_to = []  # list of parameters this parameter is tied to
    fixed = False  # if this parameter is fixed
    
    def __init__(self, name, value, constraint=None, *args, **kwargs):
        self.name = name
        self.constraints = ParameterIndexing(self)
        self._value = value
        self._current_slice = slice(None)

        for name in dir(value):
            if not hasattr(self, name):
                self.__setattr__(name, value.__getattribute__(name))
    
    @property
    def value(self):
        return self._value[self._current_slice]
    @value.setter
    def value(self, value):
        self._value[self._current_slice] = value
    @property
    def size(self):
        return self.value.size
    @property
    def shape(self):
        return self.value.shape
    @property
    def _desc(self):
        if self.size <= 1:
            return "%f"%self.value
        else:
            return self.shape
    @property
    def _constr(self):
        return ' '.join([str(c) if c else '' for c in self.constraints.properties.keys()])

    def _set_params(self, param):
        self.value.flat = param

    def _get_params(self):
        return self.value.flat
    
    def _get_params_transformed(self):
        params = self.value.copy()
        import ipdb;ipdb.set_trace()
        return  
    
    def constrain_positive(self):
        import ipdb;ipdb.set_trace()
        self.constraints.add(Logexp(), self._current_slice)
        self._current_slice = slice(None)
    
    def __getitem__(self, s):
        try:
            self.value[s]
            self._current_slice = s#[s if s else slice(s2) for s,s2 in itertools.izip_longest([s], self.shape, fillvalue=None)]
            return self
        except IndexError as i:
            raise i
    
    def __setitem__(self, s, value):
        try:
            self.value[s] = value
            return self
        except IndexError as i:
            raise i

        
    def __repr__(self, *args, **kwargs):
        view = repr(self.value)
        self._current_slice = slice(None)
        return view
        
    def __str__(self, format_spec=None):
        if format_spec is None:
            return str(self.value)
        return format_spec.format(self=self)
    
    
    
if __name__ == '__main__':
    X = numpy.random.randn(3,2)
    p = Parameter("X", X)
    p2 = Parameter("Y", numpy.random.randn(3,1))
    p3 = Parameter("rbf_variance", numpy.random.rand(1))
    p4 = Parameter("rbf_lengthscale", numpy.random.rand(2))
    params = Parameters([p,p2,p3,p4])
#     params.X[5].constrain_positive()
    #params.X[1,1].constrain_positive()
    