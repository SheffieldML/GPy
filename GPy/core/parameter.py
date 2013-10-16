'''
Created on 4 Sep 2013

@author: maxz
'''
import itertools
import numpy
from transformations import Logexp, NegativeLogexp
from GPy.core.transformations import Logistic
import collections

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
######

class Param(numpy.ndarray):
    """
    Parameter object for GPy models.

    :param name:        name of the parameter to be printed
    :param input_array: array which this parameter handles
    :param gradient:    callable with one argument, which is the model of this parameter

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
    __array_priority__ = -numpy.inf # Never give back Param
    def __new__(cls, name, input_array, gradient):
        obj = numpy.atleast_1d(numpy.array(input_array)).view(cls)
        obj._name_ = name
        obj._parent_ = None
        obj._parent_index_ = None
        obj._gradient_ = gradient
        obj._current_slice_ = [slice(obj.shape[0])]
        obj._realshape_ = obj.shape
        obj._realsize_ = obj.size
        obj._realndim_ = obj.ndim
        obj._flat_indices_ = None
        return obj    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._name_ = getattr(obj, '_name_', None)
        self._current_slice_ = getattr(obj, '_current_slice_', None)
        self._parent_ = getattr(obj, '_parent_', None)
        self._parent_index_ = getattr(obj, '_parent_index_', None)
        self._flat_indices_ = getattr(obj, '_flat_indices_', None)
        self._gradient_ = getattr(obj, '_gradient_', None)
        self._realshape_ = getattr(obj, '_realshape_', None)
        self._realsize_ = getattr(obj, '_realsize_', None)
        self._realndim_ = getattr(obj, '_realndim_', None)
    def __array_wrap__(self, out_arr, context=None):
        return out_arr.view(numpy.ndarray)
    #===========================================================================
    # Pickling operations
    #===========================================================================
    def __reduce__(self):
        func, args, state = super(Param, self).__reduce__()
        return func, args, (state, 
                            (self._name_,
                             self._parent_,
                             self._parent_index_,
                             self._gradient_,
                             self._current_slice_,
                             self._realshape_,
                             self._realsize_,
                             self._realndim_,
                             )
                            )
    def __setstate__(self, state):
        super(Param, self).__setstate__(state[0])
        state = list(state[1])
        self._realndim_ = state.pop()
        self._realsize_ = state.pop()
        self._realshape_ = state.pop()
        self._current_slice_ = state.pop()
        self._parent_index_ = state.pop()
        self._gradient_ = state.pop()
        self._parent_ = state.pop()
        self._name_ = state.pop()
    #===========================================================================
    # get/set parameters
    #===========================================================================
    def _set_params(self, param):
        self.flat = param
    def _get_params(self):
        return self.flat
    @property
    def name(self):
        """
        Name of this parameter. 
        This can be a callable without parameters. The callable will be called
        every time the name property is accessed.
        """
        if callable(self._name_):
            return self._name_()
        return self._name_
    @name.setter
    def name(self, new_name):
        from_name = self.name
        self._name_ = new_name
        self._parent_._name_changed(self, from_name)
    #===========================================================================
    # Fixing Parameters:
    #===========================================================================
    def constrain_fixed(self, warning=True):
        """
        Constrain this paramter to be fixed to the current value it carries.
        
        :param warning: print a warning for overwriting constraints.
        """
        self._parent_._fix(self,warning)
    fix = constrain_fixed
    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        self._parent_._unfix(self)
    unfix = unconstrain_fixed
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
        self[...] = transform.initialize(self)
        self._parent_._add_constrain(self, transform, warning)
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
        self._parent_._remove_constrain(self, *transforms)
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
            self._parent_._add_tie(self, param)
            if self.base is None: # this happens when indexing created a copy of the array
                self._parent_._handle_ties() 
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
    def untie(self, *params):
        """
        :param params: parameters to untie from

        remove ties to parameters given.
        """
        if len(params) == 0:
            params = self._parent_._ties_.properties()
        self._parent_._remove_tie(self, params)
    #===========================================================================
    # Prior Operations
    #===========================================================================
    def set_prior(self, prior):
        """
        :param prior: prior to be set for this parameter

        Set prior for this parameter.
        """
        if not hasattr(self._parent_, '_set_prior'):
            raise AttributeError("Parent of type {} does not support priors".format(self._parent_.__class__))
        self._parent_._set_prior(self, prior)
    def unset_prior(self, *priors):
        """
        :param priors: priors to remove from this parameter
        
        Remove all priors from this parameter
        """
        self._parent_._remove_prior(self, *priors)
    #===========================================================================
    # Array operations -> done
    #===========================================================================
    def __getitem__(self, s, *args, **kwargs):
        if not isinstance(s, tuple):
            s = (s,)
        if not reduce(lambda a,b: a or numpy.any(b is Ellipsis), s, False):
            s += (Ellipsis,)
        new_arr = numpy.ndarray.__getitem__(self, s, *args, **kwargs)
        try: new_arr._current_slice_ = s
        except AttributeError: pass# returning 0d array or float, double etc
        return new_arr
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))
    def __setitem__(self, *args, **kwargs):
        numpy.ndarray.__setitem__(self, *args, **kwargs)
        self._parent_.parameters_changed()
    #===========================================================================
    # Index Operations:
    #===========================================================================
    def _internal_offset(self):
        internal_offset = 0
        extended_realshape = numpy.cumprod((1,) + self._realshape_[:0:-1])[::-1]
        for i, si in enumerate(self._current_slice_[:self._realndim_]):
            if numpy.all(si == Ellipsis):
                continue
            if isinstance(si, slice):
                a = si.indices(self._realshape_[i])[0] 
            elif isinstance(si, (list,numpy.ndarray,tuple)):
                a = si[0]
            else: a = si
            if a<0:
                a = self._realshape_[i]+a
            internal_offset += a * extended_realshape[i]
        return internal_offset
    def _raveled_index(self):
        # return an index array on the raveled array, which is formed by the current_slice
        # of this object
        extended_realshape = numpy.cumprod((1,) + self._realshape_[:0:-1])[::-1]
        ind = self._indices() 
        if ind.ndim < 2: ind=ind[:,None]
        return numpy.apply_along_axis(lambda x: numpy.sum(extended_realshape*x), 1, ind)
    def _expand_index(self):
        # this calculates the full indexing arrays from the slicing objects given by get_item for _real..._ attributes
        # it basically translates slices to their respective index arrays and turns negative indices around
        # it tells you in the second return argument if it has only seen arrays as indices
        def f(a):
            a, b = a
            if a not in (slice(None), Ellipsis):
                if isinstance(a, slice):
                    start, stop, step = a.indices(b)
                    return numpy.r_[start:stop:step]
                elif isinstance(a, (list,numpy.ndarray,tuple)):
                    a = numpy.asarray(a, dtype=int)
                    a[a<0] = b + a[a<0]
                elif a<0:
                    a = b+a
                return numpy.r_[a]
            return numpy.r_[:b]
        return itertools.imap(f, itertools.izip_longest(self._current_slice_[:self._realndim_], self._realshape_, fillvalue=slice(None)))
    #===========================================================================
    # Printing -> done
    #===========================================================================
    @property
    def _desc(self):
        if self.size <= 1: return "%f"%self
        else: return str(self.shape)
    @property
    def _constr(self):
        return ' '.join(map(lambda c: str(c[0]) if c[1].size==self._realsize_ else "{"+str(c[0])+"}", self._parent_._constraints_iter_items(self)))
    @property
    def _t(self):
        # indices one by one: "".join(map(str,c[0]._indices()))
        def decide(c):
            if c[0]._realsize_ > 1 and not c[0].size==self.size:
                return c[0].name + "".join(map(str,c[0]._indices()))
            else:
                return c[0].name
        return ' '.join(map(lambda c: decide(c), self._parent_._ties_iter_items(self)))
    def round(self, decimals=0, out=None):
        view = super(Param, self).round(decimals, out).view(Param)
        view.__array_finalize__(self)
        return view
    round.__doc__ = numpy.round.__doc__
    def __repr__(self, *args, **kwargs):
        return "\033[1m{x:s}\033[0;0m:\n".format(x=self.name)+super(Param, self).__repr__(*args,**kwargs)
#     def _constr_matrix_str(self):
        # create an iterator, which shows the constraints of all indices
#         cons_turnaround = collections.defaultdict(list)
#         for c, index in self._parent_._constraints_iter_items(self):
#             for i in index:
#                 cons_turnaround[i] += [str(c)]
#         offset = self._internal_offset()
#         return [" ".join(cons_turnaround[i]) for i in xrange(offset, offset+self.size)]
#         self._str_dummy_ = numpy.empty(self._realshape_, dtype=object)
#         constr_matrix = self._str_dummy_ # we need the whole constraints matrix
#         constr_matrix[:] = ''
#         for constr, indices in self._parent_._constraints_iter_items(self): # put in all the constraints:
#             cstr = ""+str(constr)+""
#             constr_matrix[indices] = numpy.vectorize(lambda x:" ".join([x, cstr]) if x else cstr, otypes=[str])(constr_matrix[indices])
#         return constr_matrix.astype(numpy.string_).reshape(self._realshape_)[self._current_slice_].flatten() # and get the slice we did before
#     def _ties_matrix_str(self):
        # create an iterator, which shows the ties of all indices
#         ties_turnaround = collections.defaultdict(list)
#         for tie, index in self._parent_._ties_iter_items(self):
#             for i in index:
#                 ties_turnaround[i] += [str(tie)]
#         offset = self._internal_offset()
#         return [" ".join(ties_turnaround[i]) for i in xrange(offset, offset+self.size)]
#         for i in xrange(self.size):
#             yield " ".join(ties_turnaround[i])
#         if self._str_dummy_ is None:
#             self._str_dummy_ = numpy.empty(self._realshape_, dtype=object)
#         ties_matr = self._str_dummy_; ties_matr[:] = ''
#         for tie, indices in self._parent_._ties_iter_items(self): # go through all ties
#             tie_cycle = itertools.cycle(tie._indices()) if tie._realsize_ > 1 else itertools.repeat('')
#             ties_matr[indices] = numpy.vectorize(lambda x:" ".join([x, str(tie.name) + str(str(tie_cycle.next()))]) if x else str(tie.name)+str(str(tie_cycle.next())), otypes=[str])(ties_matr[indices])
#         return ties_matr.astype(numpy.string_).reshape(*(self._realshape_+(-1,)))[self._current_slice_] # and get the slice we did before
        
#     def _in_index(self, i, index):
#         if isinstance(index, slice):
#             start,stop,step = index.indices()
#             return i>=start and i<stop and step%i==0
#         elif index.dtype in (numpy.bool, numpy.bool_):
#             return index[]
    def _ties_for(self, index):
        return self._parent_._ties_for(self, index)
    def _constraints_for(self, index):
        return self._parent_._constraints_for(self, index)
    def _indices(self):
        # get a int-array containing all indices in the first axis.
        if isinstance(self._current_slice_, (tuple, list)):
            clean_curr_slice = [s for s in self._current_slice_ if s != Ellipsis]
            if (all(isinstance(n, (numpy.ndarray, list, tuple)) for n in clean_curr_slice) 
                and len(set(map(len,clean_curr_slice))) <= 1):
                return numpy.fromiter(itertools.izip(*self._expand_index()),
                    dtype=[('',int)]*self._realndim_,count=self.size).view((int, self._realndim_)) 
        return numpy.fromiter(itertools.product(*self._expand_index()),
                 dtype=[('',int)]*self._realndim_,count=self.size).view((int, self._realndim_))
    def _max_len_names(self, gen, header):
        return reduce(lambda a, b:max(a, len(b)), gen, len(header))
    def _max_len_values(self):
        return reduce(lambda a, b:max(a, len("{x:=.{0}G}".format(__precision__, x=b))), self.flat, len(self.name))
    def _max_len_index(self, ind):
        return reduce(lambda a, b:max(a, len(str(b))), ind, len(__index_name__))
    def _short(self):
        if self._realsize_ < 2:
            return self.name
        ind = self._indices()
        if self.size > 4: indstr = ','.join(map(str,ind[:2])) + "..." + ','.join(map(str,ind[-2:])) 
        else: indstr = ','.join(map(str,ind)) 
        return self.name+'['+indstr+']'
    def __str__(self, constr_matrix=None, indices=None, ties=None, lc=None, lx=None, li=None, lt=None):
        if indices is None: indices = self._indices()
        ravi = self._raveled_index()
        if constr_matrix is None: constr_matrix = self._constraints_for(ravi)
        if ties is None: ties = self._ties_for(ravi)
        if lc is None: lc = self._max_len_names(constr_matrix, __constraints_name__)
        if lx is None: lx = self._max_len_values()
        if li is None: li = self._max_len_index(self._indices())
        if lt is None: lt = self._max_len_names(ties[0], __tie_name__)
        header = "  {i:^{2}s}  |  \033[1m{x:^{1}s}\033[0;0m  |  {c:^{0}s}  |  {t:^{3}s}".format(lc,lx,li,lt, x=self.name, c=__constraints_name__, i=__index_name__, t=__tie_name__) # nice header for printing
        return "\n".join([header]+["  {i!s:^{3}s}  |  {x: >{1}.{2}G}  |  {c:^{0}s}  |  {t:^{4}}  ".format(lc,lx,__precision__,li,lt, x=x, c=" ".join(map(str,c)), t=" ".join([tie._short() for tie in t]), i=i) for i,x,c,t in itertools.izip(indices,self.flat,constr_matrix,ties[0])]) # return all the constraints with right indices
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
    fix = constrain_fixed
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
    unfix = unconstrain_fixed
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
    #X = numpy.random.randn(2,3,1,5,2,4,3)
    X = numpy.random.randn(800,1e4)
    p = Param("q_mean", X, None)
    p1 = Param("q_variance", numpy.random.rand(*p.shape), None)
    p2 = Param("Y", numpy.random.randn(p.shape[0],1), None)
    p3 = Param("rbf_variance", numpy.random.rand(), None)
    p4 = Param("rbf_lengthscale", numpy.random.rand(2), None)
    m = Parameterized()
    m.set_as_parameters(p,p1,p2,p3,p4)
    #print m.q_v[3:5,[1,4,5]]
    m[".*variance"].constrain_positive()
    m.rbf.constrain_positive()
    #m.q_v.tie_to(m.rbf_v)
#     m.rbf_l.tie_to(m.rbf_va)
    # pt = numpy.array(params._get_params_transformed())
    # ptr = numpy.random.randn(*pt.shape)
#     params.X.tie_to(params.rbf_v)