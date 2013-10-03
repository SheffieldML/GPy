'''
Created on 4 Sep 2013

@author: maxz
'''
import re
import itertools
import numpy
from GPy.core.transformations import Logexp, NegativeLogexp
from GPy.core.index_operations import ParameterIndexOperations
from types import FunctionType
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
        return numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params(), self._params)), dtype=numpy.float64, count=sum(self.sizes))
    
    def _set_params(self, params):
        [p._set_params(params[s]) for s in self._param_slices]
    
    def _get_params_transformed(self):
        return numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params_transformed(), self._params)), dtype=numpy.float64, count=sum(self.sizes))
    
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


class Parameter(numpy.ndarray):
    tied_to = []  # list of parameters this parameter is tied to
    fixed = False  # if this parameter is fixed
    __array_priority__ = 3.0
    
    def __new__(cls, name, input_array, info=None):
        obj = numpy.array(input_array).view(cls)
        obj.name = name
        obj._current_slice = slice(None)
        obj._realshape = input_array.shape
#         def attribute_func(value, name):
#             value_func = self.value.__getattribute__(name)
#             def f(*args, **kwargs):
#                 with self.slicing():
#                     raise AttributeError("This is a parameter view, use self.value for array view")
#             try:
#                 f.__doc__ = value_func.__doc__
#             except AttributeError:
#                 # no docstring present
#                 pass
#             return f
#           
#         for name in dir(value):
#             if not hasattr(self, name):
#                 self.__setattr__(name, attribute_func(value, name))#value.__getattribute__(name))
        obj.constraints = ParameterIndexOperations(obj)        
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self._realshape = getattr(obj, '_realshape', None)
        self.constraints = getattr(obj, 'constraints', None)
        self._current_slice = getattr(obj, '_current_slice', None)

    def __array_wrap__(self, out_arr, context=None):
        return numpy.ndarray.__array_wrap__(self, out_arr, context)
    
#     def __init__(self, name, value, constraint=None, *args, **kwargs):
#         self.constraints = ParameterIndexOperations(self)
#         
#         self._value = value
#         self._current_slice = slice(None)
#         def attribute_func(value, name):
#             value_func = self.value.__getattribute__(name)
#             def f(*args, **kwargs):
#                 with self.slicing():
#                     raise AttributeError("This is a parameter view, use self.value for array view")
#             try:
#                 f.__doc__ = value_func.__doc__
#             except AttributeError:
#                 # no docstring present
#                 pass
#             return f
#            
#         for name in dir(value):
#             if not hasattr(self, name):
#                 self.__setattr__(name, attribute_func(value, name))#value.__getattribute__(name))
#         super(Parameter, self).__init__(value, *args, **kwargs)
    
    @property
    def value(self):
        return self#self.base[self._current_slice]
    
#     @value.setter
#     def value(self, value):
#         self.base[self._current_slice] = value

#     @property
#     def value(self):
#         return self._value[self._current_slice]
#     @value.setter
#     def value(self, value):
#         self._value[self._current_slice] = value
#     @property
#     def size(self):
#         return self.value.size
#     @property
#     def shape(self):
#         return self.value.shape
#     @property
#     def realshape(self):
#         return self._value.shape
    @property
    def _desc(self):
        if self.size <= 1:
            return "%f"%self.value
        else:
            return self.shape
    @property
    def _constr(self):
        return ' '.join([str(c) if c else '' for c in self.constraints.keys()])

    def _set_params(self, param):
        with self.slicing():
            self.value.flat = param

    def _get_params(self):
        with self.slicing():
            return self.value.flat
    
    def _get_params_transformed(self):
        with self.slicing():
            params = self.value.copy()
        
    def constrain(self, constraint):
        with self.slicing():
            self.constraints.add(constraint, self._current_slice)
    
    def constrain_positive(self):
        self.constrain(Logexp())

    def constrain_negative(self):
        self.constrain(NegativeLogexp())

    def unconstrain(self, constraints=None):
        with self.slicing():
            if constraints is None:
                constraints = self.constraints.keys()
            elif not isinstance(constraints, (tuple, list, numpy.ndarray)):
                constraints = [constraints]
            for constr in constraints:
                self.constraints.remove(constr, self._current_slice)

    def unconstrain_positive(self):
        self.unconstrain(Logexp())
        
        
    def __getitem__(self, s, *args, **kwargs):
        #self._current_slice = s
#         import ipdb;ipdb.set_trace()
        new_arr = numpy.ndarray.__getitem__(self, s, *args, **kwargs)
        new_arr = new_arr.view(self.__class__)
        new_arr._current_slice = s
        return new_arr
#     def __getitem__(self, s):
#         try:
#             self.value[s]
#             self._current_slice = s#[s if s else slice(s2) for s,s2 in itertools.izip_longest([s], self.shape, fillvalue=None)]
#             return self
#         except IndexError as i:
#             self._current_slice = slice(None)
#             raise i
#     
#     def __setitem__(self, s, value):
#         try:
#             self.value[s] = value
#             self._current_slice = slice(None)
#             return self
#         except IndexError as i:
#             raise i
# 
#         
    def __repr__(self, *args, **kwargs):
        view = str(self.value)
        return view
        
    def __str__(self, format_spec=None):
        #with self.slicing():
            if format_spec is None:
                constr_matrix = numpy.empty(self._realshape, dtype=object)
                constr_matrix[:] = ''
                for constr, indices in self.constraints.iteritems():
                    constr_matrix[indices] = numpy.vectorize(lambda x: " ".join([x,str(constr)]) if x else str(constr))(constr_matrix[indices])
                constr_matrix = constr_matrix.astype(numpy.string_)[self._current_slice]
                p = numpy.get_printoptions()['precision']
                constr = constr_matrix.flat
                ind = numpy.array(list(itertools.product(*itertools.imap(range, self._realshape))))[self.constraints.create_raveled_indices(self._current_slice),...]
                c_name, x_name, i_name = "Constraint", "Value", "Index"
                lc = max(reduce(lambda a,b: max(a, len(b)), constr_matrix.flat, 0), len(c_name))
                lx = max(reduce(lambda a,b: max(a, len("{x:=.{0}G}".format(p,x=b))), self.value.flat, 0), len(x_name))
                li = max(reduce(lambda a,b: max(a, len(str(b))), ind, 0), len(i_name))
                header = "  {i:^{3}s}  |  {x:^{1}s}  |  {c:^{0}s}".format(lc,lx,p,li, x=x_name, c=c_name, i=i_name)
                return "\n".join([header]+["  {i:^{3}s}  |  {x: >{1}.{2}G}  |  {c:^{0}s}".format(lc,lx,p,li, x=x, c=constr.next(), i=i) for i,x in itertools.izip(ind,self.value.flat)])
            return format_spec.format(self=self)
    
    import contextlib    
    @contextlib.contextmanager
    def slicing(self, *args, **kwargs):
        try:
            yield 
        finally:
            self._current_slice = slice(None)
    del contextlib
    
if __name__ == '__main__':
    X = numpy.random.randn(2,4)
    p = Parameter("X", X)
    p2 = Parameter("Y", numpy.random.randn(3,1))
    p3 = Parameter("rbf_variance", numpy.random.rand(1))
    p4 = Parameter("rbf_lengthscale", numpy.random.rand(2))
    params = Parameters([p,p2,p3,p4])
    params.X[1].constrain_positive()
    print params.X
    #params.X[1,1].constrain_positive()
    