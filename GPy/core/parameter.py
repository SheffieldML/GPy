'''
Created on 4 Sep 2013

@author: maxz
'''
import itertools
import numpy
from transformations import Logexp, NegativeLogexp
from GPy.core.transformations import Logistic

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
######

class Param(numpy.ndarray):
    """
    Parameter object for GPy models.

    You can add/remove constraints by calling the constrain on the parameter itself, e.g:
    
        - self[:,1].constrain_positive()
        - self[0].tie_to(other)
        - self.untie()
        - self[:3,:].unconstrain()
        - self[1].fix()
        
    Fixing parameters will fix them to the value they are right now. If you change
    the fixed value, it will be fixed to the new value!
    
    See :py:class:`GPy.core.parameterized.Parameterized` for more details.
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
    def constrain_fixed(self, warning=True):
        """
        Constrain this paramter to be fixed to the current value it carries.
        
        :param warning: print a warning for overwriting constraints.
        """
        self._parent._fix(self,warning)
    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        self._parent._unfix(self)
    #===========================================================================
    # Constrain operations -> done
    #===========================================================================
    def constrain(self, transform, warning=True):
        """
        :param transform: the :py:class:`GPy.core.transformations.Transformation`
                          to constrain the this parameter to.
        :param warning: print a warning if re-constraining parameters.
        
        Constrain the parameter to the given
        :py:class:`GPy.core.transformations.Transformation`.
        """
        self._parent._add_constrain(self, transform, warning)
        self[...] = transform.initialize(self)
    def constrain_positive(self, warning=True):
        """
        :param warning: print a warning if re-constraining parameters.
        
        Constrain this parameter to the default positive constraint.
        """
        self.constrain(Logexp(), warning)
    def constrain_negative(self, warning=True):
        """
        :param warning: print a warning if re-constraining parameters.
        
        Constrain this parameter to the default negative constraint.
        """
        self.constrain(NegativeLogexp(), warning)
    def constrain_bounded(self, lower, upper, warning=True):
        """
        :param lower, upper: the limits to bound this parameter to
        :param warning: print a warning if re-constraining parameters.
        
        Constrain this parameter to lie within the given range.
        """
        self.constrain(Logistic(lower, upper), warning)
    def unconstrain(self, *transforms):
        """
        :param transforms: The transformations to unconstrain from.
        
        remove all :py:class:`GPy.core.transformations.Transformation` 
        transformats of this parameter object.
        """
        self._parent._remove_constrain(self, *transforms)
    def unconstrain_positive(self):
        """
        Remove positive constraint of this parameter. 
        """
        self.unconstrain(Logexp())
    def unconstrain_negative(self):
        """
        Remove negative constraint of this parameter. 
        """
        self.unconstrain(NegativeLogexp())
    def unconstrain_bounded(self, lower, upper):
        """
        :param lower, upper: the limits to unbound this parameter from
        
        Remove (lower, upper) bounded constrain from this parameter/
        """
        self.unconstrain(Logistic(lower, upper))
    #===========================================================================
    # Tying operations -> done
    #===========================================================================
    def tie_to(self, param):
        """
        :param param: the parameter object to tie this parameter to.
        
        Tie this parameter to the given parameter.
        Broadcasting is allowed, so you can tie a whole dimension to
        one parameter:  self[:,0].tie_to(other), where other is a one-value
        parameter.
        """
        assert isinstance(param, Param), "Argument {1} not of type {0}".format(Param,param.__class__)
        try:
            self[...] = param
            self._parent._add_tie(self, param)
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
    def untie(self, *params):
        """
        :param params: parameters to untie from

        remove ties to the paramters given.
        """
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
    # Printing -> done
    #===========================================================================
    @property
    def _desc(self):
        if self.size <= 1: return "%f"%self
        else: return str(self.shape)
    @property
    def _constr(self):
        return ' '.join(map(lambda c: str(c[0]) if len(c[1])==self._realsize else "{"+str(c[0])+"}", self._parent._constraints_iter_items(self)))
    @property
    def _t(self):
        # indices one by one: "".join(map(str,c[0]._indices()))
        def decide(c):
            if c[0]._realsize > 1:
                return c[0].name + "".join(map(str,c[0]._indices()))
            else:
                return c[0].name
        return ' '.join(map(lambda c: decide(c), self._parent._ties_iter_items(self)))
    def round(self, decimals=0, out=None):
        view = super(Param, self).round(decimals, out).view(Param)
        view.__array_finalize__(self)
        return view
    round.__doc__ = numpy.round.__doc__
    def __repr__(self, *args, **kwargs):
        return "\033[1m{x:s}\033[0;0m:\n".format(x=self.name)+super(Param, self).__repr__(*args,**kwargs)
    def _constr_matrix_str(self):
        # create a matrix, which shows the constraints of all indices
        constr_matrix = numpy.empty(self._realsize, dtype=object) # we need the whole constraints matrix
        constr_matrix[:] = ''
        for constr, indices in self._parent._constraints_iter_items(self): # put in all the constraints:
            cstr = ""+str(constr)+""
            constr_matrix[indices] = numpy.vectorize(lambda x:" ".join([x, cstr]) if x else cstr, otypes=[str])(constr_matrix[indices])
        return constr_matrix.astype(numpy.string_).reshape(self._realshape)[self._current_slice].flatten() # and get the slice we did before
    def _ties_matrix_str(self):
        # create a matrix, which shows the ties of all indices
        ties_matr = numpy.empty(self._realsize, dtype=object) # we need the whole constraints matrix
        ties_matr[:] = ''
        for tie, indices in self._parent._ties_iter_items(self): # go through all ties
            tie_cycle = itertools.cycle(tie._indices()) if tie._realsize > 1 else itertools.repeat('')
            ties_matr[indices] = numpy.vectorize(lambda x:" ".join([x, str(tie.name) + str(str(tie_cycle.next()))]) if x else str(tie.name)+str(str(tie_cycle.next())), otypes=[str])(ties_matr[indices])
        return ties_matr.astype(numpy.string_).reshape(*(self._realshape+(-1,)))[self._current_slice] # and get the slice we did before
    def _indices(self):
        # get a int-array containing all indices in the first axis.
        flat_indices = numpy.array(list(itertools.product(*itertools.imap(range, self._realshape)))).reshape(self._realshape + (-1,))
        return flat_indices[self._current_slice].reshape(self.size, -1) # find out which indices to print
    def _max_len_names(self, constr_matrix, header):
        return max(reduce(lambda a, b:max(a, len(b)), constr_matrix.flat, 0), len(header))
    def _max_len_values(self):
        return max(reduce(lambda a, b:max(a, len("{x:=.{0}G}".format(__precision__, x=b))), self.flat, 0), len(self.name))
    def _max_len_index(self, ind):
        return max(reduce(lambda a, b:max(a, len(str(b))), ind, 0), len(__index_name__))
    def __str__(self, constr_matrix=None, indices=None, ties=None, lc=None, lx=None, li=None, lt=None):
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
        #except: return super(Param, self).__str__()

class ParamConcatenation(object):
    def __init__(self, params):
        """
        Parameter concatenation for convienience of printing regular expression matched arrays
        you can index this concatenation as if it was the flattened concatenation
        of all the parameters it contains, same for setting parameters (Broadcasting enabled).

        See :py:class:`GPy.core.parameter.Param` for more details on constraining.
        """
        self.params = params
        self._param_sizes = [p.size for p in self.params]
        startstops = numpy.cumsum([0] + self._param_sizes)
        self._param_slices = [slice(start, stop) for start,stop in zip(startstops, startstops[1:])]
    #===========================================================================
    # Get/set items, enable broadcasting
    #===========================================================================
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
    #===========================================================================
    # parameter operations:
    #===========================================================================
    def constrain(self, constraint, warning=True):
        [param.constrain(constraint) for param in self.params]
    constrain.__doc__ = Param.constrain.__doc__
    def constrain_positive(self, warning=True):
        [param.constrain_positive(warning) for param in self.params]
    constrain_positive.__doc__ = Param.constrain_positive.__doc__
    def constrain_fixed(self, warning=True):
        [param.constrain_fixed(warning) for param in self.params]
    constrain_fixed.__doc__ = Param.constrain_fixed.__doc__
    def constrain_negative(self, warning=True):
        [param.constrain_negative(warning) for param in self.params]
    constrain_negative.__doc__ = Param.constrain_negative.__doc__
    def constrain_bounded(self, lower, upper, warning=True):
        [param.constrain_bounded(lower, upper, warning) for param in self.params]
    constrain_bounded.__doc__ = Param.constrain_bounded.__doc__
    def unconstrain(self, constraints=None):
        [param.unconstrain(constraints) for param in self.params]
    unconstrain.__doc__ = Param.unconstrain.__doc__
    def unconstrain_negative(self):
        [param.unconstrain_negative() for param in self.params]
    unconstrain_negative.__doc__ = Param.unconstrain_negative.__doc__
    def unconstrain_positive(self):
        [param.unconstrain_positive() for param in self.params]
    unconstrain_positive.__doc__ = Param.unconstrain_positive.__doc__
    def unconstrain_fixed(self):
        [param.unconstrain_fixed() for param in self.params]
    unconstrain_fixed.__doc__ = Param.unconstrain_fixed.__doc__
    def unconstrain_bounded(self, lower, upper):
        [param.unconstrain_bounded(lower, upper) for param in self.params]
    unconstrain_bounded.__doc__ = Param.unconstrain_bounded.__doc__
    __lt__ = lambda self, val: self._vals()<val
    __le__ = lambda self, val: self._vals()<=val
    __eq__ = lambda self, val: self._vals()==val
    __ne__ = lambda self, val: self._vals()!=val
    __gt__ = lambda self, val: self._vals()>val
    __ge__ = lambda self, val: self._vals()>=val
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
    from GPy.core.parameterized import Parameterized
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