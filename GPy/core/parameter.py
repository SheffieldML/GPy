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
from re import compile, _pattern_type
_index_re = re.compile('(?:_(\d+))+')  # pattern match for indices

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
######

def translate_param_names_to_parameters(param_names):
    """
    Naive translation from _get_param_names return to Parameterized object.
    Assumptions:
     - array indices are at the and matching _\d+_\d+...
     - names are in order and names match field names
    """


class Parameterized(object):
    def __init__(self, parameterlist, prefix=None, *args, **kwargs):
        self._params = []
        for p in parameterlist:
            if isinstance(p, Parameterized):
                self._params.extend(p._params)
            else:
                self._params.append(p)
        sizes = numpy.cumsum([0] + self.sizes)
        self._param_slices = itertools.starmap(lambda start,stop: slice(start, stop), zip(sizes, sizes[1:]))
        for p in self._params:
            p._parent = self
            self.__setattr__(p.name, p)
    
    def _get_params(self):
        return numpy.hstack([x._get_params() for x in self._params])#numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params(), self._params)), dtype=numpy.float64, count=sum(self.sizes))
    
    def _set_params(self, params):
        [p._set_params(params[s]) for p,s in itertools.izip(self._params,self._param_slices)]
    
    def _get_params_transformed(self):
        return numpy.hstack([x._get_params_transformed() for x in self._params])
        return numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params_transformed(), self._params)), dtype=numpy.float64, count=self.num_params_transformed)
    
    def _set_params_transformed(self, params):
        current_index = 0
        for p in self._params:
            s = p.num_params_transformed
            p._set_params_transformed(params[current_index:current_index+s])
            current_index += s
        self._handle_ties()
        
    def _handle_ties(self):
        [p._handle_ties() for p in self._params]
                
    def __getitem__(self, name):
        return self.__getattr__(name)
        
    def grep_param_names(self, regexp):
        if not isinstance(regexp, _pattern_type):
            regexp = compile(regexp)
        paramlist = [param for param in self._params if regexp.search(param.name) is not None]
        return paramlist

    def tie_params(self, regexp):
        paramlist = self.grep_param_names(regexp)
        

    def __getattr__(self, name, *args, **kwargs):
        if name in self.__dict__:
            return object.__getattribute__(self, name, *args, **kwargs)
        else:
            paramlist = self.grep_param_names(name)
            if len(paramlist) < 1:
                raise AttributeError("'{:s}' object has no attribute '{:s}'".format(str(self.__class__.__name__), name))
            if len(paramlist) == 1:
                return paramlist[-1]
            return ParamConcatenation(paramlist)    
        
    @property
    def names(self):
        return [x.name for x in self._params]
    @property
    def size(self):
        return sum(self.sizes)
    @property
    def sizes(self):
        return [x.size for x in self._params]
    @property
    def sizes_transformed(self):
        return [x.num_params_transformed() for x in self._params]
    @property
    def num_params_transformed(self):
        return reduce(lambda a,b: a+b.num_params_transformed, self._params, 0)
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
        format_spec = "  \033[1m{{self.name:^{0}s}}\033[0;0m  |  {{self._desc:^{1}s}}  |  {{self._constr:^{2}s}}  ".format(nl, sl, cl)
        if header:
            header = "  {{0:^{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  ".format(nl, sl, cl).format("Name", "Value", "Constraint")
            header += '\n' + '-'*len(header)
            return '\n'.join([header]+[x.__str__(format_spec=format_spec) for x in self._params])
        return '\n'.join([x.__str__(format_spec=format_spec) for x in self._params])
    pass

class Param(numpy.ndarray):
    """
    Parameter object for GPy models.
    
    - constraining singular entries: 
      - param[1:2,1:2].constrain{_..}() to keep the shape
      - param[1,1,...].constrain{_..}(), which tells numpy to keep the shapes
      - param[[1],[1]] to trigger advanced indexin, which keeps shapes
      
      this is because of numpy's ndarray functionality:
      it gives back a float, when indexing singular entries
      we want a Param object though, with the right shape
      workarounds:
      
    - slice paramters and constrain the slices.
    
    - unconstrain slices
    
    """
    fixed = False  # if this parameter is fixed
    __array_priority__ = -1 # Only give back a param if slicing
    
    def __new__(cls, name, input_array, constraints=None, ties=None):
        obj = numpy.atleast_1d(numpy.array(input_array)).view(cls)
        obj.name = name
        obj._current_slice = slice(None)
        obj._realshape = obj.shape
        #obj.parameters = parameters
        if constraints is None:
            obj.constraints = ParameterIndexOperations(obj)
        else:
            obj.constraints = constraints
        if ties is None:
            obj.ties = ParameterIndexOperations(obj)
        else:
            obj.ties = ties
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self._realshape = getattr(obj, '_realshape', None)
        self.constraints = getattr(obj, 'constraints', None)
        self._parent = getattr(obj, '_parent', None)
        self.ties = getattr(obj, 'ties', None)
        self._current_slice = getattr(obj, '_current_slice', None)
    
    @property
    def value(self):
        return self#self.base[self._current_slice]
    
    @property
    def _desc(self):
        if self.size <= 1:
            return "%f"%self.value
        else:
            return self.shape
    @property
    def _constr(self):
        return ' '.join([str(c) if c else '' for c in self.constraints.properties()])

    def _set_params(self, param):
        self.value.flat = param

    def _get_params(self):
        return self.value.flat
    
    @property
    def num_params_transformed(self):
        return self.size - self.ties.size()
    
    def _get_params_transformed(self):
        params = self.base.flatten()
        [numpy.put(params, indices, constr.finv(params[indices])) for constr, indices in self.constraints.iteritems()]
        return numpy.delete(params, reduce(numpy.union1d, self.ties.iterindices(), numpy.array([],dtype=self.dtype)))

    def _set_params_transformed(self, params):
        numpy.put(self.base, numpy.setdiff1d(numpy.r_[:self.size],reduce(numpy.union1d, self.ties.iterindices(), numpy.array([],dtype=self.dtype))), params)
        [numpy.put(self.base, indices, constraint.f(self.base.flat[indices])) for constraint, indices in self.constraints.iteritems()]
        
    def _handle_ties(self):
        # handle all tied values
        for tie, indices in self.ties.iteritems(): 
            self.base.flat[indices] = tie.flat
                
    def tie_to(self, param):
        assert isinstance(param, Param), "Argument {1} not of type {0}".format(Param,param.__class__)
        try:
            rav_index = self.ties.create_raveled_indices(self._current_slice)
            self.base.flat[rav_index] = param.flat
            self.ties.add(param, self._current_slice)
            self._handle_ties()
        except ValueError:
            raise ValueError("Trying to tie params with shape {} to params with shape {}".format(self.shape, param.shape))
            
    def untie(self, *params):
        if len(params) == 0:
            params = self.ties.properties()
        for p in self.ties.properties():
            if any((p == x).all() for x in params):
                self.ties.remove(p, self._current_slice)
    
    def constrain(self, transform):
        self.constraints.add(transform, self._current_slice)
        self[:] = transform.initialize(self)
    
    def constrain_positive(self):
        self.constrain(Logexp())

    def constrain_negative(self):
        self.constrain(NegativeLogexp())

    def unconstrain(self, transforms=None):
        if transforms is None:
            transforms = self.constraints.properties()
        elif not isinstance(transforms, (tuple, list, numpy.ndarray)):
            transforms = [transforms]
        for constr in transforms:
            self.constraints.remove(constr, self._current_slice)

    def unconstrain_positive(self):
        self.unconstrain(Logexp())
    
    def unconstrain_negative(self):
        self.unconstrain(NegativeLogexp())
        
    def __getitem__(self, s, *args, **kwargs):
        new_arr = numpy.ndarray.__getitem__(self, s, *args, **kwargs)
        try:
            new_arr._current_slice = s
        except AttributeError:
            # returning 0d array or float, double etc:
            pass
        return new_arr

    def __setitem__(self, *args, **kwargs):
        numpy.ndarray.__setitem__(self, *args, **kwargs)
        self._parent._handle_ties()

    def __repr__(self, *args, **kwargs):
        return super(Param, self).__repr__(*args, **kwargs)
        view = repr(self.value)
        return view
    
    def _constr_matrix_str(self):
        constr_matrix = numpy.empty(self._realshape, dtype=object) # we need the whole constraints matrix
        constr_matrix[:] = ''
        for constr, indices in self.constraints.iteritems(): # put in all the constraints:
            constr_matrix[indices] = numpy.vectorize(lambda x:" ".join([x, str(constr)]) if x else str(constr))(constr_matrix[indices])
        return constr_matrix.astype(numpy.string_)[self._current_slice] # and get the slice we did before
    def _ties_matrix_str(self):
        ties_matr = numpy.empty(self._realshape, dtype=object) # we need the whole constraints matrix
        ties_matr[:] = ''
        for tie, indices in self.ties.iteritems(): # go through all ties:
            tie_cycle = itertools.cycle(tie._indices())
            ties_matr[indices] = numpy.vectorize(lambda x:" ".join([x, str(tie.name) + str(str(tie_cycle.next()))]) if x else str(tie.name)+str(str(tie_cycle.next())), otypes=[str])(ties_matr[indices])
        return ties_matr.astype(numpy.string_)[self._current_slice] # and get the slice we did before
    def _indices(self):
        return numpy.array(list(itertools.product(*itertools.imap(range, self._realshape))))[self.constraints.create_raveled_indices(self._current_slice),...] # find out which indices to print
    def _max_len_names(self, constr_matrix, header):
        return max(reduce(lambda a, b:max(a, len(b)), constr_matrix.flat, 0), len(header))
    def _max_len_values(self):
        return max(reduce(lambda a, b:max(a, len("{x:=.{0}G}".format(__precision__, x=b))), self.value.flat, 0), len(self.name))
    def _max_len_index(self, ind):
        return max(reduce(lambda a, b:max(a, len(str(b))), ind, 0), len(__index_name__))
    def __str__(self, format_spec=None, constr_matrix=None, indices=None, ties=None, lc=None, lx=None, li=None, lt=None):
        if format_spec is None:
            if indices is None:
                indices = self._indices() 
            if constr_matrix is None:
                constr_matrix = self._constr_matrix_str()
            if ties is None:
                ties = self._ties_matrix_str()
            if lc is None:
                lc = self._max_len_names(constr_matrix, __constraints_name__)
            if lx is None:
                lx = self._max_len_values()
            if li is None:
                li = self._max_len_index(indices)
            if lt is None:
                lt = self._max_len_names(ties, __tie_name__)
            constr = constr_matrix.flat
            ties = ties.flat
            header = "  {i:^{2}s}  |  \033[1m{x:^{1}s}\033[0;0m  |  {c:^{0}s}  |  {t:^{3}s}".format(lc,lx,li,lt, x=self.name, c=__constraints_name__, i=__index_name__, t=__tie_name__) # nice header for printing
            return "\n".join([header]+["  {i:^{3}s}  |  {x: >{1}.{2}G}  |  {c:^{0}s}  |  {t:^{4}}  ".format(lc,lx,__precision__,li,lt, x=x, c=constr.next(), t=ties.next(), i=i) for i,x in itertools.izip(indices,self.value.flat)]) # return all the constraints with right indices
        return format_spec.format(self=self)
        
class ParamConcatenation(object):
    def __init__(self, params):
        self.params = params
    def __getitem__(self, s):
        raise AttributeError("Cannot index a concatenation of parameters!")
    def constrain(self, constraint):
        [param.constrain(constraint) for param in self.params]
    def constrain_positive(self):
        [param.constrain_positive() for param in self.params]
    def constrain_negative(self):
        [param.constrain_negative() for param in self.params]
    def unconstrain(self, constraints=None):
        [param.unconstrain(constraints) for param in self.params]
    def unconstrain_negative(self):
        [param.unconstrain_negative() for param in self.params]
    def __str__(self, *args, **kwargs):
        constr_matrices = [p._constr_matrix_str() for p in self.params]
        ties_matrices = [p._ties_matrix_str() for p in self.params]
        indices = [p._indices() for p in self.params]
        lc = max([p._max_len_names(cm, __constraints_name__) for p, cm in itertools.izip(self.params, constr_matrices)])
        lx = max([p._max_len_values() for p in self.params])
        li = max([p._max_len_index(i) for p, i in itertools.izip(self.params, indices)])
        lt = max([p._max_len_names(tm, __tie_name__) for p, tm in itertools.izip(self.params, ties_matrices)])
        return "\n".join([p.__str__(None, cm, i, tm, lc, lx, li, lt) for p, cm, i, tm in itertools.izip(self.params,constr_matrices,indices,ties_matrices)])    
    def __repr__(self):
        return self.__str__()
    
if __name__ == '__main__':
    X = numpy.random.randn(100,8)
    p = Param("X", X)
    p1 = Param("X_variance", numpy.random.rand(*X.shape))
    p2 = Param("Y", numpy.random.randn(3,1))
    p3 = Param("rbf_variance", numpy.random.rand(1))
    p4 = Param("rbf_lengthscale", numpy.random.rand(2))
    params = Parameterized([p,p1,p2,p3,p4])
    params.rbf.constrain_positive()
    params[".*variance"].constrain_positive()
    params.rbf_l.tie_to(params.rbf_va)
    pt = numpy.array(params._get_params_transformed())
    ptr = numpy.random.randn(*pt.shape)
#     params.X.tie_to(params.rbf_v)