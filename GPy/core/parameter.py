'''
Created on 4 Sep 2013

@author: maxz
'''
import re
import itertools
import numpy
from GPy.core.transformations import Logexp, NegativeLogexp
from GPy.core.index_operations import ConstraintIndexOperations,\
    create_raveled_indices, index_empty, TieIndexOperations
from types import FunctionType
from re import compile, _pattern_type
_index_re = re.compile('(?:_(\d+))+')  # pattern match for indices

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
__fixed__ = "fixed"
######

class Parameterized(object):
    """
    Parameterized class
    
    Say m is a handle to a parameterized class.

    Printing parameters:
    
        >>> print m
        # prints a nice summary over all parameters
        >>> print m.name
        # prints all the parameters which start with name
    
    Getting and setting parameters:
        
        Two ways to get parameters:
            
            - m.name regular expression matches all parameters beginning with name
            - m['name'] regular expression matches all parameters with name
    
    Handling of constraining, fixing and tieing parameters:
        
        
        
    """
    def __init__(self, parameterlist, prefix=None, *args, **kwargs):
        self._init = True
        self._params = []
        for p in parameterlist:
            if isinstance(p, Parameterized):
                self._params.extend(p._params)
            else:
                self._params.append(p)
        self._constraints = ConstraintIndexOperations()
        self._ties = TieIndexOperations(self)
        self._fixes = ConstraintIndexOperations
        self._ties_fixes = None
        self._connect_parameters()
        self._init = False
    def _connect_parameters(self):
        sizes = numpy.cumsum([0] + self.parameter_sizes)
        self._param_slices = [slice(start, stop) for start,stop in zip(sizes, sizes[1:])]
        for i, p in enumerate(self._params):
            p._parent = self
            p._parent_index = i
            not_unique = []
            if p.name in self.__dict__:
                not_unique.append(p.name)
                del self.__dict__[p.name]
            elif not (p.name in not_unique):
                self.__dict__[p.name] = p

    #===========================================================================
    # Optimization handles:
    #===========================================================================
    def _get_params(self):
        return numpy.hstack([x._get_params() for x in self._params])#numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params(), self._params)), dtype=numpy.float64, count=sum(self.parameter_sizes))    
    def _set_params(self, params):
        [p._set_params(params[s]) for p,s in itertools.izip(self._params,self._param_slices)]
    def _get_params_transformed(self):
        p = self._get_params()
        [numpy.put(p, ind, c.finv(p[ind])) for c,ind in self._constraints.iteritems() if c is not __fixed__]
        if self._ties_fixes is not None:
            return p[self._ties_fixes]
        return p
    def _set_params_transformed(self, p):
        if self._ties_fixes is not None: tmp = self._get_params(); tmp[self._ties_fixes] = p; p = tmp; del tmp
        [numpy.put(p, ind, c.f(p[ind])) for c,ind in self._constraints.iteritems() if c is not __fixed__]
        [numpy.put(p, f, p[t]) for f,t in self._ties.iter_from_to_indices()]
        self._set_params(p)
    def _handle_ties(self):
        if not self._init:
            self._set_params_transformed(self._get_params_transformed())
    #===========================================================================
    # Index Handling
    #===========================================================================
    def _backtranslate_index(self, param, ind):
        ind = ind-self._offset(param)
        ind = ind[ind >= 0]
        internal_offset = (numpy.arange(param._realsize).reshape(param._realshape)[param._current_slice]).flat[0]
        ind = ind[ind < param.size + internal_offset]
        return ind - internal_offset
    #===========================================================================
    # Handle ties:
    #===========================================================================
    def _add_tie(self, param, tied_to):
        try:
            param[...] = tied_to
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
        self._ties.add(param, tied_to)
        if self._ties_fixes is None: self._ties_fixes = numpy.ones(self.parameter_size, dtype=bool)
        f = create_raveled_indices(param._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[f] = False
    def _remove_tie(self, param, *params):
        if len(params) == 0:
            params = self._ties.properties()
        for p in self._ties.properties():
            if any((p == x).all() for x in params):
                ind = create_raveled_indices(p._current_slice, param._realshape, self._offset(param))
                self._ties.remove(param, p)
                self._ties_fixes[ind] = True
        if numpy.all(self._ties_fixes): self._ties_fixes = None
    def _ties_iter_items(self, param):
        for tied_to, ind in self._ties.iter_from_items():
            ind = self._backtranslate_index(param, ind)
            if not index_empty(ind):
                yield tied_to, ind
    def _ties_iter(self, param):
        for constr, _ in self._ties_iter_items(param):
            yield constr
    def _ties_iter_indices(self, param):
        for _, ind in self._ties_iter_items(param):
            yield ind
    #===========================================================================
    # Fixing parameters:
    #===========================================================================
    def _fix(self, param):
        reconstrained = self._remove_constrain(param, None)
        if any(reconstrained):
            # to print whole params: m = str(param[self._backtranslate_index(param, reconstrained)])
            m = param.name
            if param._realsize > 1:
                m += str("".join(map(str,param._indices()[self._backtranslate_index(param, reconstrained)])))
            print "Reconstrained parameters:\n{}".format(m)
        self._constraints.add(__fixed__, param._current_slice, param._realshape, self._offset(param))
        if self._ties_fixes is None: self._ties_fixes = numpy.ones(self.parameter_size, dtype=bool)
        f = create_raveled_indices(param._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[f] = False
    def _unfix(self, param):
        self._remove_constrain(param, __fixed__)
        ind = create_raveled_indices(p._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[ind] = True
        if numpy.all(self._ties_fixes): self._ties_fixes = None
    #===========================================================================
    # Constraint Handling:
    #===========================================================================
    def constrain(self, regexp, constraint):
        self[regexp].constrain(constraint)
    def _offset(self, param):
        # offset = reduce(lambda a, b:a + (b.stop - b.start), self._param_slices[:param._parent_index], 0)
        return self._param_slices[param._parent_index].start 
    def _add_constrain(self, param, transform):
        reconstrained = self._remove_constrain(param, None)
        self._constraints.add(transform, param._current_slice, param._realshape, self._offset(param))
        if any(reconstrained):
            # to print whole params: m = str(param[self._backtranslate_index(param, reconstrained)])
            m = param.name + str("".join(map(str,param._indices()[self._backtranslate_index(param, reconstrained)])))
            print "Reconstrained parameters:\n{}".format(m)
    def _remove_constrain(self, param, transforms):
        if transforms is None:
            transforms = self._constraints.properties()
        elif not isinstance(transforms, (tuple, list, numpy.ndarray)):
            transforms = [transforms]
        removed_indices = numpy.array([]).astype(int)
        for constr in transforms:
            removed = self._constraints.remove(constr, param._current_slice, param._realshape, self._offset(param))
            removed_indices = numpy.union1d(removed_indices, removed)
        return removed_indices
    def _constraints_iter_items(self, param):
        for constr, ind in self._constraints.iteritems():
            ind = self._backtranslate_index(param, ind)
            if not index_empty(ind):
                yield constr, ind
    def _constraints_iter(self, param):
        for constr, _ in self._constraints_iter_items(param):
            yield constr
    def _contraints_iter_indices(self, param):
        for _, ind in self._constraints_iter_items(param):
            yield ind
    def _constraint_indices(self, param, constraint):
        return self._backtranslate_index(param, self._constraints[constraint])
    #===========================================================================
    # Get/set parameters:
    #===========================================================================
    def grep_param_names(self, regexp):
        if not isinstance(regexp, _pattern_type): regexp = compile(regexp)
        paramlist = [param for param in self._params if regexp.match(param.name) is not None]
        return paramlist
    def __getitem__(self, name, paramlist=None):
        if paramlist is None:
            paramlist = self.grep_param_names(name)
        if len(paramlist) < 1: raise AttributeError, name
        if len(paramlist) == 1: return paramlist[-1]
        return ParamConcatenation(paramlist)  
    def __setitem__(self, name, value, paramlist=None):
        try: param = self.__getitem__(name, paramlist)
        except AttributeError as a: raise a
        param[...] = value
    def __getattr__(self, name, *args, **kwargs):
        return self.__getitem__(name)
    def __setattr__(self, name, val):
        if hasattr(self, "_params"):
            paramlist = self.grep_param_names(name)
            if len(paramlist) > 1: object.__setattr__(self, name, val); return# raise AttributeError("Non-unique params identified {}".format([p.name for p in paramlist]))
            if len(paramlist) == 1: self.__setitem__(name, val, paramlist); return
        object.__setattr__(self, name, val);
    #===========================================================================
    # Printing:        
    #===========================================================================
    @property
    def names(self):
        return [x.name for x in self._params]
    @property
    def parameter_size(self):
        return sum(self.parameter_sizes)
    @property
    def parameter_sizes(self):
        return [x.size for x in self._params]
    @property
    def parameter_size_transformed(self):
        return sum(self._ties_fixes)
    @property
    def parameter_shapes(self):
        return [x.shape for x in self._params]
    @property
    def _constrs(self):
        return [p._constr for p in self._params]
    @property
    def _descs(self):
        return [x._desc for x in self._params]
    @property
    def _ts(self):
        return [x._t for x in self._params]
    def __str__(self, header=True):
        nl = max([len(str(x)) for x in self.names + ["Name"]])
        sl = max([len(str(x)) for x in self._descs + ["Value"]])
        cl = max([len(str(x)) if x else 0 for x in self._constrs  + ["Constraint"]])
        tl = max([len(str(x)) if x else 0 for x in self._ts + ["Tied to"]])
        format_spec = "  \033[1m{{p.name:^{0}s}}\033[0;0m  |  {{p._desc:^{1}s}}  |  {{p._constr:^{2}s}}  |  {{p._t:^{2}s}}".format(nl, sl, cl)
        to_print = [format_spec.format(p=p) for p in self._params]
        sep = '-'*len(to_print[0])
        if header:
            header = "  {{0:^{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  |  {{3:^{3}s}}".format(nl, sl, cl, tl).format("Name", "Value", "Constraint", "Tied to")
            header += '\n' + sep
            to_print.insert(0, header)
        return '\n'.format(sep).join(to_print)
    pass

class Param(numpy.ndarray):
    """
    Parameter object for GPy models.
          
    - slice paramters and constrain the slices.
    
    - unconstrain slices
    """
    fixed = False  # if this parameter is fixed
    __array_priority__ = -1. # Allways give back Param
    def __new__(cls, name, input_array):
        obj = numpy.atleast_1d(numpy.array(input_array)).view(cls)
        obj.name = name
        obj._parent = None
        obj._parent_index = None
        obj._current_slice = slice(None)
        obj._realshape = obj.shape
        obj._realsize = obj.size
        return obj    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self._current_slice = getattr(obj, '_current_slice', None)
        self._parent = getattr(obj, '_parent', None)
        self._parent_index = getattr(obj, '_parent_index', None)
        self._realshape = getattr(obj, '_realshape', None)
        self._realsize = getattr(obj, '_realsize', None)
    #===========================================================================
    # get/set parameters
    #===========================================================================
    def _set_params(self, param):
        self.flat = param
    def _get_params(self):
        return self.flat
    #===========================================================================
    # Fixing Parameters:
    #===========================================================================
    def constrain_fixed(self):
        self._parent._fix(self)
    def unconstrain_fixed(self):
        self._parent._unfix(self)
    #===========================================================================
    # Constrain operations -> done
    #===========================================================================
    def constrain(self, transform):
        self._parent._add_constrain(self, transform)
        self[...] = transform.initialize(self)
    def constrain_positive(self):
        self.constrain(Logexp())
    def constrain_negative(self):
        self.constrain(NegativeLogexp())
    def unconstrain(self, transforms=None):
        self._parent._remove_constrain(self, transforms)
    def unconstrain_positive(self):
        self.unconstrain(Logexp())
    def unconstrain_negative(self):
        self.unconstrain(NegativeLogexp())
    #===========================================================================
    # Tying operations -> done
    #===========================================================================
    def tie_to(self, param):
        assert isinstance(param, Param), "Argument {1} not of type {0}".format(Param,param.__class__)
        try:
            self[...] = param
            self._parent._add_tie(self, param)
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
    def untie(self, *params):
        if len(params) == 0:
            params = self._parent._ties.properties()
        for p in self._parent._ties.properties():
            if any((p == x).all() for x in params):
                self._parent._remove_tie(self, params)
    #===========================================================================
    # Array operations -> done
    #===========================================================================
    def __getitem__(self, s, *args, **kwargs):
        if not isinstance(s, tuple):
            s = (s,)
        if not( Ellipsis in s):
            s = (s + (Ellipsis,))
        new_arr = numpy.ndarray.__getitem__(self, s, *args, **kwargs)
        try: new_arr._current_slice = s
        except AttributeError: pass# returning 0d array or float, double etc
        return new_arr
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))
    def __setitem__(self, *args, **kwargs):
        numpy.ndarray.__setitem__(self, *args, **kwargs)
        self._parent._handle_ties()
    #===========================================================================
    # Printing -> 
    #===========================================================================
    @property
    def _desc(self):
        if self.size <= 1: return "%f"%self
        else: return self.shape
    @property
    def _constr(self):
        return ' '.join(map(lambda c: str(c[0]) if len(c[1])==self._realsize else "{"+str(c[0])+"}", self._parent._constraints_iter_items(self)))
    @property
    def _t(self):
        # indices one by one: "".join(map(str,c[0]._indices()))
        return ' '.join(map(lambda c: c[0].name if len(c[1])==self._realsize else c[0].name+"[...]", self._parent._ties_iter_items(self)))
    def round(self, decimals=0, out=None):
        view = super(Param, self).round(decimals, out).view(Param)
        view.__array_finalize__(self)
        return view
    round.__doc__ = numpy.round.__doc__
    def __repr__(self, *args, **kwargs):
        return "\033[1m{x:s}\033[0;0m:\n".format(x=self.name)+super(Param, self).__repr__(*args,**kwargs)
    def _constr_matrix_str(self):
        constr_matrix = numpy.empty(self._realsize, dtype=object) # we need the whole constraints matrix
        constr_matrix[:] = ''
        for constr, indices in self._parent._constraints_iter_items(self): # put in all the constraints:
            cstr = ""+str(constr)+""
            constr_matrix[indices] = numpy.vectorize(lambda x:" ".join([x, cstr]) if x else cstr, otypes=[str])(constr_matrix[indices])
        return constr_matrix.astype(numpy.string_).reshape(self._realshape)[self._current_slice].flatten() # and get the slice we did before
    def _ties_matrix_str(self):
        ties_matr = numpy.empty(self._realsize, dtype=object) # we need the whole constraints matrix
        ties_matr[:] = ''
        for tie, indices in self._parent._ties_iter_items(self): # go through all ties:
            tie_cycle = itertools.cycle(tie._indices()) if tie._realsize > 1 else itertools.repeat('')
            ties_matr[indices] = numpy.vectorize(lambda x:" ".join([x, str(tie.name) + str(str(tie_cycle.next()))]) if x else str(tie.name)+str(str(tie_cycle.next())), otypes=[str])(ties_matr[indices])
        return ties_matr.astype(numpy.string_).reshape(*(self._realshape+(-1,)))[self._current_slice] # and get the slice we did before
    def _indices(self):
        flat_indices = numpy.array(list(itertools.product(*itertools.imap(range, self._realshape)))).reshape(self._realshape + (-1,))
        return flat_indices[self._current_slice].reshape(self.size, -1) # find out which indices to print
    def _max_len_names(self, constr_matrix, header):
        return max(reduce(lambda a, b:max(a, len(b)), constr_matrix.flat, 0), len(header))
    def _max_len_values(self):
        return max(reduce(lambda a, b:max(a, len("{x:=.{0}G}".format(__precision__, x=b))), self.flat, 0), len(self.name))
    def _max_len_index(self, ind):
        return max(reduce(lambda a, b:max(a, len(str(b))), ind, 0), len(__index_name__))
    def __str__(self, constr_matrix=None, indices=None, ties=None, lc=None, lx=None, li=None, lt=None):
        try:
            if indices is None: indices = self._indices() 
            if constr_matrix is None: constr_matrix = self._constr_matrix_str()
            if ties is None: ties = self._ties_matrix_str()
            if lc is None: lc = self._max_len_names(constr_matrix, __constraints_name__)
            if lx is None: lx = self._max_len_values()
            if li is None: li = self._max_len_index(indices)
            if lt is None: lt = self._max_len_names(ties, __tie_name__)
            constr = constr_matrix.flat
            ties = ties.flat
            header = "  {i:^{2}s}  |  \033[1m{x:^{1}s}\033[0;0m  |  {c:^{0}s}  |  {t:^{3}s}".format(lc,lx,li,lt, x=self.name, c=__constraints_name__, i=__index_name__, t=__tie_name__) # nice header for printing
            return "\n".join([header]+["  {i:^{3}s}  |  {x: >{1}.{2}G}  |  {c:^{0}s}  |  {t:^{4}}  ".format(lc,lx,__precision__,li,lt, x=x, c=constr.next(), t=ties.next(), i=i) for i,x in itertools.izip(indices,self.flat)]) # return all the constraints with right indices
        except: return super(Param, self).__str__()
class ParamConcatenation(object):
    def __init__(self, params):
        """
        Parameter concatenation for convienience of printing regular expression matched arrays
        you can index this concatenation as if it was the flattened concatenation
        of all the parameters it contains, same for setting parameters 
        """
        self.params = params
        self._param_sizes = [p.size for p in self.params]
        startstops = numpy.cumsum([0] + self._param_sizes)
        self._param_slices = [slice(start, stop) for start,stop in zip(startstops, startstops[1:])]
    def __getitem__(self, s):
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True; 
        params = [p.flatten()[ind[ps]] for p,ps in zip(self.params, self._param_slices) if numpy.any(p.flat[ind[ps]])]
        if len(params)==1: return params[0]
        return ParamConcatenation(params)
    def __setitem__(self, s, val):
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True; 
        vals = self._vals(); vals[s] = val; del val
        [numpy.place(p, ind[ps], vals[ps]) for p, ps in zip(self.params, self._param_slices)]
    def _vals(self):
        return numpy.hstack([p._get_params() for p in self.params])
    def constrain(self, constraint):
        [param.constrain(constraint) for param in self.params]
    def constrain_positive(self):
        [param.constrain_positive() for param in self.params]
    def constrain_fixed(self):
        [param.constrain_fixed() for param in self.params]
    def constrain_negative(self):
        [param.constrain_negative() for param in self.params]
    def unconstrain(self, constraints=None):
        [param.unconstrain(constraints) for param in self.params]
    def unconstrain_negative(self):
        [param.unconstrain_negative() for param in self.params]
    def unconstrain_positive(self):
        [param.unconstrain_positive() for param in self.params]
    def unconstrain_fixed(self):
        [param.unconstrain_fixed() for param in self.params]
    def __str__(self, *args, **kwargs):
        constr_matrices = [p._constr_matrix_str() for p in self.params]
        ties_matrices = [p._ties_matrix_str() for p in self.params]
        indices = [p._indices() for p in self.params]
        lc = max([p._max_len_names(cm, __constraints_name__) for p, cm in itertools.izip(self.params, constr_matrices)])
        lx = max([p._max_len_values() for p in self.params])
        li = max([p._max_len_index(i) for p, i in itertools.izip(self.params, indices)])
        lt = max([p._max_len_names(tm, __tie_name__) for p, tm in itertools.izip(self.params, ties_matrices)])
        strings = [p.__str__(cm, i, tm, lc, lx, li, lt) for p, cm, i, tm in itertools.izip(self.params,constr_matrices,indices,ties_matrices)]
        return "\n{}\n".format(" -"+"- | -".join(['-'*l for l in [li,lx,lc,lt]])).join(strings)    
    def __repr__(self):
        return "\n".join(map(repr,self.params))
    
if __name__ == '__main__':
    X = numpy.random.randn(4,2)
    p = Param("q_mean", X)
    p1 = Param("q_variance", numpy.random.rand(*p.shape))
    p2 = Param("Y", numpy.random.randn(p.shape[0],1))
    p3 = Param("rbf_variance", numpy.random.rand())
    p4 = Param("rbf_lengthscale", numpy.random.rand(2))
    m = Parameterized([p,p1,p2,p3,p4])
    m[".*variance"].constrain_positive()
    m.rbf.constrain_positive()
    m.q_v.tie_to(m.rbf_v)
#     m.rbf_l.tie_to(m.rbf_va)
    # pt = numpy.array(params._get_params_transformed())
    # ptr = numpy.random.randn(*pt.shape)
#     params.X.tie_to(params.rbf_v)