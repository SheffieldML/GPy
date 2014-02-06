# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import itertools
import numpy
from parameter_core import Constrainable, adjust_name_for_printing
from array_core import ObservableArray

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
__print_threshold__ = 5
######      

class Float(numpy.float64, Constrainable):
    def __init__(self, f, base):
        super(Float,self).__init__(f)
        self._base = base
        
        
class Param(ObservableArray, Constrainable):
    """
    Parameter object for GPy models.

    :param name:        name of the parameter to be printed
    :param input_array: array which this parameter handles
    
    You can add/remove constraints by calling constrain on the parameter itself, e.g:
    
        - self[:,1].constrain_positive()
        - self[0].tie_to(other)
        - self.untie()
        - self[:3,:].unconstrain()
        - self[1].fix()
        
    Fixing parameters will fix them to the value they are right now. If you change
    the fixed value, it will be fixed to the new value!
    
    See :py:class:`GPy.core.parameterized.Parameterized` for more details on constraining etc.

    This ndarray can be stored in lists and checked if it is in.

    >>> import numpy as np
    >>> x = np.random.normal(size=(10,3))
    >>> x in [[1], x, [3]]
    True
    
    WARNING: This overrides the functionality of x==y!!!
    Use numpy.equal(x,y) for element-wise equality testing.
    """
    __array_priority__ = 0 # Never give back Param
    _fixes_ = None
    def __new__(cls, name, input_array, *args, **kwargs):
        obj = numpy.atleast_1d(super(Param, cls).__new__(cls, input_array=input_array))
        obj._current_slice_ = (slice(obj.shape[0]),)
        obj._realshape_ = obj.shape
        obj._realsize_ = obj.size
        obj._realndim_ = obj.ndim
        obj._updated_ = False
        from index_operations import SetDict
        obj._tied_to_me_ = SetDict()
        obj._tied_to_ = []
        obj._original_ = True
        obj.gradient = None
        return obj

    def __init__(self, name, input_array):
        super(Param, self).__init__(name=name)
        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        super(Param, self).__array_finalize__(obj)
        self._direct_parent_ = getattr(obj, '_direct_parent_', None)
        self._parent_index_ = getattr(obj, '_parent_index_', None)
        self._highest_parent_ = getattr(obj, '_highest_parent_', None)
        self._current_slice_ = getattr(obj, '_current_slice_', None)
        self._tied_to_me_ = getattr(obj, '_tied_to_me_', None)
        self._tied_to_ = getattr(obj, '_tied_to_', None)
        self._realshape_ = getattr(obj, '_realshape_', None)
        self._realsize_ = getattr(obj, '_realsize_', None)
        self._realndim_ = getattr(obj, '_realndim_', None)
        self._updated_ = getattr(obj, '_updated_', None)
        self._original_ = getattr(obj, '_original_', None)
        self._name = getattr(obj, 'name', None)
        self.gradient = getattr(obj, 'gradient', None)
        
    def __array_wrap__(self, out_arr, context=None):
        return out_arr.view(numpy.ndarray)
    #===========================================================================
    # Pickling operations
    #===========================================================================
    def __reduce_ex__(self):
        func, args, state = super(Param, self).__reduce__()
        return func, args, (state, 
                            (self.name,
                             self._direct_parent_,
                             self._parent_index_,
                             self._highest_parent_,
                             self._current_slice_,
                             self._realshape_,
                             self._realsize_,
                             self._realndim_,
                             self._tied_to_me_,
                             self._tied_to_,
                             self._updated_,
                            )
                            )

    def __setstate__(self, state):
        super(Param, self).__setstate__(state[0])
        state = list(state[1])
        self._updated_ = state.pop()
        self._tied_to_ = state.pop()
        self._tied_to_me_ = state.pop()
        self._realndim_ = state.pop()
        self._realsize_ = state.pop()
        self._realshape_ = state.pop()
        self._current_slice_ = state.pop()
        self._highest_parent_ = state.pop()
        self._parent_index_ = state.pop()
        self._direct_parent_ = state.pop()
        self.name = state.pop()

    #===========================================================================
    # get/set parameters
    #===========================================================================

    def _set_params(self, param, update=True):
        self.flat = param
        self._notify_tied_parameters()
        self._notify_observers()
        
    def _get_params(self):
        return self.flat
#     @property
#     def name(self):
#         """
#         Name of this parameter. 
#         This can be a callable without parameters. The callable will be called
#         every time the name property is accessed.
#         """
#         if callable(self.name):
#             return self.name()
#         return self.name
#     @name.setter
#     def name(self, new_name):
#         from_name = self.name
#         self.name = new_name
#         self._direct_parent_._name_changed(self, from_name)
    @property
    def _parameters_(self):
        return []
    def _connect_highest_parent(self, highest_parent):
        self._highest_parent_ = highest_parent
    def _collect_gradient(self, target):
        target[:] = self.gradient.flat
    #===========================================================================
    # Fixing Parameters:
    #===========================================================================
    def constrain_fixed(self, warning=True):
        """
        Constrain this paramter to be fixed to the current value it carries.
        
        :param warning: print a warning for overwriting constraints.
        """
        self._highest_parent_._fix(self,warning)
    fix = constrain_fixed
    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        self._highest_parent_._unfix(self)
    unfix = unconstrain_fixed
    #===========================================================================
    # Tying operations -> bugged, TODO
    #===========================================================================
    def tie_to(self, param):
        """
        :param param: the parameter object to tie this parameter to. 
                      Can be ParamConcatenation (retrieved by regexp search)
        
        Tie this parameter to the given parameter.
        Broadcasting is not allowed, but you can tie a whole dimension to
        one parameter:  self[:,0].tie_to(other), where other is a one-value
        parameter.
        
        Note: For now only one parameter can have ties, so all of a parameter
              will be removed, when re-tieing!
        """
        #Note: this method will tie to the parameter which is the last in 
        #      the chain of ties. Thus, if you tie to a tied parameter,
        #      this tie will be created to the parameter the param is tied
        #      to.

        assert isinstance(param, Param), "Argument {1} not of type {0}".format(Param,param.__class__)
        param = numpy.atleast_1d(param)
        if param.size != 1:
            raise NotImplementedError, "Broadcast tying is not implemented yet"
        try:
            if self._original_: 
                self[:] = param
            else: # this happens when indexing created a copy of the array
                self._direct_parent_._get_original(self)[self._current_slice_] = param
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
        if param is self:
            raise RuntimeError, 'Cyclic tieing is not allowed'
#         if len(param._tied_to_) > 0:
#             if (self._direct_parent_._get_original(self) is param._direct_parent_._get_original(param)
#                 and len(set(self._raveled_index())&set(param._tied_to_[0]._raveled_index()))!=0):
#                 raise RuntimeError, 'Cyclic tieing is not allowed'
#             self.tie_to(param._tied_to_[0])
#             return
        if not param in self._direct_parent_._get_original(self)._tied_to_:
            self._direct_parent_._get_original(self)._tied_to_ += [param]
        param._add_tie_listener(self)
        self._highest_parent_._set_fixed(self)
        cs = self._highest_parent_._constraints_for(param, param._raveled_index())
        for cs in self._highest_parent_._constraints_for(param, param._raveled_index()):
            [self.constrain(c, warning=False) for c in cs]
#         for t in self._tied_to_me_.keys():
#             if t is not self:
#                 t.untie(self)
#                 t.tie_to(param)

    def untie(self, *ties):
        """
        remove all ties.
        """
        [t._direct_parent_._get_original(t)._remove_tie_listener(self) for t in self._tied_to_]
        new_ties = []
        for t in self._direct_parent_._get_original(self)._tied_to_:
            for tied in t._tied_to_me_.keys():
                if t._parent_index_ is tied._parent_index_:
                    new_ties.append(tied)
        self._direct_parent_._get_original(self)._tied_to_ = new_ties
        self._direct_parent_._get_original(self)._highest_parent_._set_unfixed(self)
#         self._direct_parent_._remove_tie(self, *params)
    def _notify_tied_parameters(self):
        for tied, ind in self._tied_to_me_.iteritems():
            tied._on_tied_parameter_changed(self.base, list(ind))
    def _add_tie_listener(self, tied_to_me):
        for t in self._tied_to_me_.keys():
            if tied_to_me._parent_index_ is t._parent_index_:
                t_rav_i = t._raveled_index()
                tr_rav_i = tied_to_me._raveled_index()
                new_index = list(set(t_rav_i) | set(tr_rav_i))
                tmp = t._direct_parent_._get_original(t)[numpy.unravel_index(new_index,t._realshape_)]
                self._tied_to_me_[tmp] = self._tied_to_me_[t] | set(self._raveled_index())
                del self._tied_to_me_[t]
                return
        self._tied_to_me_[tied_to_me] = set(self._raveled_index())
    def _remove_tie_listener(self, to_remove):
        for t in self._tied_to_me_.keys():
            if t._parent_index_ == to_remove._parent_index_:
                t_rav_i = t._raveled_index()
                tr_rav_i = to_remove._raveled_index()
                import ipdb;ipdb.set_trace()
                new_index = list(set(t_rav_i) - set(tr_rav_i))
                if new_index:
                    tmp = t._direct_parent_._get_original(t)[numpy.unravel_index(new_index,t._realshape_)]
                    self._tied_to_me_[tmp] = self._tied_to_me_[t]
                    del self._tied_to_me_[t]
                    if len(self._tied_to_me_[tmp]) == 0:
                        del self._tied_to_me_[tmp]
                else:
                    del self._tied_to_me_[t]
    def _on_tied_parameter_changed(self, val, ind):
        if not self._updated_: #not fast_array_equal(self, val[ind]):
            val = numpy.atleast_1d(val)
            self._updated_ = True
            if self._original_:
                self.__setitem__(slice(None), val[ind], update=False)
            else: # this happens when indexing created a copy of the array
                self._direct_parent_._get_original(self).__setitem__(self._current_slice_, val[ind], update=False)
            self._notify_tied_parameters()
            self._updated_ = False
    #===========================================================================
    # Prior Operations
    #===========================================================================
    def set_prior(self, prior):
        """
        :param prior: prior to be set for this parameter

        Set prior for this parameter.
        """
        if not hasattr(self._highest_parent_, '_set_prior'):
            raise AttributeError("Parent of type {} does not support priors".format(self._highest_parent_.__class__))
        self._highest_parent_._set_prior(self, prior)
    def unset_prior(self, *priors):
        """
        :param priors: priors to remove from this parameter
        
        Remove all priors from this parameter
        """
        self._highest_parent_._remove_prior(self, *priors)
    #===========================================================================
    # Array operations -> done
    #===========================================================================
    def __getitem__(self, s, *args, **kwargs):
        if not isinstance(s, tuple):
            s = (s,)
        if not reduce(lambda a,b: a or numpy.any(b is Ellipsis), s, False) and len(s) <= self.ndim:
            s += (Ellipsis,)
        new_arr = super(Param, self).__getitem__(s, *args, **kwargs)
        try: new_arr._current_slice_ = s; new_arr._original_ = self.base is new_arr.base
        except AttributeError: pass# returning 0d array or float, double etc
        return new_arr
    def __setitem__(self, s, val, update=True):
        super(Param, self).__setitem__(s, val, update=update)
        self._notify_tied_parameters()
        if update:
            self._highest_parent_.parameters_changed()
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
    def _raveled_index(self, slice_index=None):
        # return an index array on the raveled array, which is formed by the current_slice
        # of this object
        extended_realshape = numpy.cumprod((1,) + self._realshape_[:0:-1])[::-1]
        ind = self._indices(slice_index)
        if ind.ndim < 2: ind=ind[:,None]
        return numpy.asarray(numpy.apply_along_axis(lambda x: numpy.sum(extended_realshape*x), 1, ind), dtype=int)
    def _expand_index(self, slice_index=None):
        # this calculates the full indexing arrays from the slicing objects given by get_item for _real..._ attributes
        # it basically translates slices to their respective index arrays and turns negative indices around
        # it tells you in the second return argument if it has only seen arrays as indices
        if slice_index is None:
            slice_index = self._current_slice_
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
        return itertools.imap(f, itertools.izip_longest(slice_index[:self._realndim_], self._realshape_, fillvalue=slice(self.size)))
    #===========================================================================
    # Convienience
    #===========================================================================
    @property
    def is_fixed(self):
        return self._highest_parent_._is_fixed(self)
    def round(self, decimals=0, out=None):
        view = super(Param, self).round(decimals, out).view(Param)
        view.__array_finalize__(self)
        return view
    def _has_fixes(self):
        return False
    round.__doc__ = numpy.round.__doc__
    def _get_original(self, param):
        return self
    #===========================================================================
    # Printing -> done
    #===========================================================================
    @property
    def _description_str(self):
        if self.size <= 1: return ["%f"%self]
        else: return [str(self.shape)]
    def _parameter_names(self, add_name):
        return [self.name]
    @property
    def flattened_parameters(self):
        return [self]
    @property
    def parameter_shapes(self):
        return [self.shape]
    @property
    def _constraints_str(self):
        return [' '.join(map(lambda c: str(c[0]) if c[1].size==self._realsize_ else "{"+str(c[0])+"}", self._highest_parent_._constraints_iter_items(self)))]
    @property
    def _ties_str(self):
        return [t._short() for t in self._tied_to_] or ['']
    @property
    def name_hirarchical(self):
        if self.has_parent():
            return self._direct_parent_.hirarchy_name()+adjust_name_for_printing(self.name)
        return adjust_name_for_printing(self.name)
    def __repr__(self, *args, **kwargs):
        name = "\033[1m{x:s}\033[0;0m:\n".format(
                            x=self.name_hirarchical)
        return name + super(Param, self).__repr__(*args,**kwargs)
    def _ties_for(self, rav_index):
        #size = sum(p.size for p in self._tied_to_)
        ties = numpy.empty(shape=(len(self._tied_to_), numpy.size(rav_index)), dtype=Param)
        for i, tied_to in enumerate(self._tied_to_):
            for t, ind in tied_to._tied_to_me_.iteritems():
                if t._parent_index_ == self._parent_index_:
                    matches = numpy.where(rav_index[:,None] == t._raveled_index()[None, :])
                    tt_rav_index = tied_to._raveled_index()
                    ind_rav_matches = numpy.where(tt_rav_index == numpy.array(list(ind)))[0]
                    if len(ind) != 1: ties[i, matches[0][ind_rav_matches]] = numpy.take(tt_rav_index, matches[1], mode='wrap')[ind_rav_matches]
                    else: ties[i, matches[0]] = numpy.take(tt_rav_index, matches[1], mode='wrap')
        return map(lambda a: sum(a,[]), zip(*[[[tie.flatten()] if tx!=None else [] for tx in t] for t,tie in zip(ties,self._tied_to_)]))
    def _constraints_for(self, rav_index):
        return self._highest_parent_._constraints_for(self, rav_index)
    def _indices(self, slice_index=None):
        # get a int-array containing all indices in the first axis.
        if slice_index is None:
            slice_index = self._current_slice_
        if isinstance(slice_index, (tuple, list)):
            clean_curr_slice = [s for s in slice_index if numpy.any(s != Ellipsis)]
            if (all(isinstance(n, (numpy.ndarray, list, tuple)) for n in clean_curr_slice) 
                and len(set(map(len,clean_curr_slice))) <= 1):
                return numpy.fromiter(itertools.izip(*clean_curr_slice),
                    dtype=[('',int)]*self._realndim_,count=len(clean_curr_slice[0])).view((int, self._realndim_))
        expanded_index = list(self._expand_index(slice_index))
        return numpy.fromiter(itertools.product(*expanded_index),
                 dtype=[('',int)]*self._realndim_,count=reduce(lambda a,b: a*b.size,expanded_index,1)).view((int, self._realndim_))
    def _max_len_names(self, gen, header):
        return reduce(lambda a, b:max(a, len(b)), gen, len(header))
    def _max_len_values(self):
        return reduce(lambda a, b:max(a, len("{x:=.{0}g}".format(__precision__, x=b))), self.flat, len(self.name_hirarchical))
    def _max_len_index(self, ind):
        return reduce(lambda a, b:max(a, len(str(b))), ind, len(__index_name__))
    def _short(self):
        # short string to print
        name = self._direct_parent_.hirarchy_name() + adjust_name_for_printing(self.name)
        if self._realsize_ < 2:
            return name
        ind = self._indices()
        if ind.size > 4: indstr = ','.join(map(str,ind[:2])) + "..." + ','.join(map(str,ind[-2:])) 
        else: indstr = ','.join(map(str,ind))
        return name+'['+indstr+']'
    def __str__(self, constr_matrix=None, indices=None, ties=None, lc=None, lx=None, li=None, lt=None):
        filter_ = self._current_slice_
        vals = self.flat
        if indices is None: indices = self._indices(filter_)
        ravi = self._raveled_index(filter_)
        if constr_matrix is None: constr_matrix = self._constraints_for(ravi)
        if ties is None: ties = self._ties_for(ravi)
        ties = [' '.join(map(lambda x: x._short(), t)) for t in ties]
        if lc is None: lc = self._max_len_names(constr_matrix, __constraints_name__)
        if lx is None: lx = self._max_len_values()
        if li is None: li = self._max_len_index(indices)
        if lt is None: lt = self._max_len_names(ties, __tie_name__)
        header = "  {i:^{2}s}  |  \033[1m{x:^{1}s}\033[0;0m  |  {c:^{0}s}  |  {t:^{3}s}".format(lc,lx,li,lt, x=self.name_hirarchical, c=__constraints_name__, i=__index_name__, t=__tie_name__) # nice header for printing
        if not ties: ties = itertools.cycle([''])
        return "\n".join([header]+["  {i!s:^{3}s}  |  {x: >{1}.{2}g}  |  {c:^{0}s}  |  {t:^{4}s}  ".format(lc,lx,__precision__,li,lt, x=x, c=" ".join(map(str,c)), t=(t or ''), i=i) for i,x,c,t in itertools.izip(indices,vals,constr_matrix,ties)]) # return all the constraints with right indices
        #except: return super(Param, self).__str__()

class ParamConcatenation(object):
    def __init__(self, params):
        """
        Parameter concatenation for convienience of printing regular expression matched arrays
        you can index this concatenation as if it was the flattened concatenation
        of all the parameters it contains, same for setting parameters (Broadcasting enabled).

        See :py:class:`GPy.core.parameter.Param` for more details on constraining.
        """
        #self.params = params
        self.params = []
        for p in params:
            for p in p.flattened_parameters:
                if p not in self.params:
                    self.params.append(p)           
        self._param_sizes = [p.size for p in self.params]
        startstops = numpy.cumsum([0] + self._param_sizes)
        self._param_slices_ = [slice(start, stop) for start,stop in zip(startstops, startstops[1:])]
    #===========================================================================
    # Get/set items, enable broadcasting
    #===========================================================================
    def __getitem__(self, s):
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True; 
        params = [p._get_params()[ind[ps]] for p,ps in zip(self.params, self._param_slices_) if numpy.any(p._get_params()[ind[ps]])]
        if len(params)==1: return params[0]
        return ParamConcatenation(params)
    def __setitem__(self, s, val, update=True):
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True; 
        vals = self._vals(); vals[s] = val; del val
        [numpy.place(p, ind[ps], vals[ps]) and p._notify_tied_parameters() 
         for p, ps in zip(self.params, self._param_slices_)]
        if update:
            self.params[0]._highest_parent_.parameters_changed()
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
    def unconstrain(self, *constraints):
        [param.unconstrain(*constraints) for param in self.params]
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
    def untie(self, *ties):
        [param.untie(*ties) for param in self.params]
    __lt__ = lambda self, val: self._vals()<val
    __le__ = lambda self, val: self._vals()<=val
    __eq__ = lambda self, val: self._vals()==val
    __ne__ = lambda self, val: self._vals()!=val
    __gt__ = lambda self, val: self._vals()>val
    __ge__ = lambda self, val: self._vals()>=val
    def __str__(self, *args, **kwargs):
        def f(p):
            ind = p._raveled_index()
            return p._constraints_for(ind), p._ties_for(ind)
        params = self.params
        constr_matrices, ties_matrices = zip(*map(f, params))
        indices = [p._indices() for p in params]
        lc = max([p._max_len_names(cm, __constraints_name__) for p, cm in itertools.izip(params, constr_matrices)])
        lx = max([p._max_len_values() for p in params])
        li = max([p._max_len_index(i) for p, i in itertools.izip(params, indices)])
        lt = max([p._max_len_names(tm, __tie_name__) for p, tm in itertools.izip(params, ties_matrices)])
        strings = [p.__str__(cm, i, tm, lc, lx, li, lt) for p, cm, i, tm in itertools.izip(params,constr_matrices,indices,ties_matrices)]
        return "\n".join(strings)
        return "\n{}\n".format(" -"+"- | -".join(['-'*l for l in [li,lx,lc,lt]])).join(strings)
    def __repr__(self):
        return "\n".join(map(repr,self.params))
    
if __name__ == '__main__':
    

    from GPy.core.parameterized import Parameterized
    from GPy.core.parameter import Param

    #X = numpy.random.randn(2,3,1,5,2,4,3)
    X = numpy.random.randn(3,2)
    print "random done"
    p = Param("q_mean", X)
    p1 = Param("q_variance", numpy.random.rand(*p.shape))
    p2 = Param("Y", numpy.random.randn(p.shape[0],1))
    
    p3 = Param("variance", numpy.random.rand())
    p4 = Param("lengthscale", numpy.random.rand(2))
    
    m = Parameterized()
    rbf = Parameterized(name='rbf')
    
    rbf.add_parameter(p3,p4)
    m.add_parameter(p,p1,rbf)
    
    print "setting params"
    #print m.q_v[3:5,[1,4,5]]
    print "constraining variance"
    #m[".*variance"].constrain_positive()
    #print "constraining rbf"
    #m.rbf_l.constrain_positive()
    #m.q_variance[1,[0,5,11,19,2]].tie_to(m.rbf_v)
    #m.rbf_v.tie_to(m.rbf_l[0])
    #m.rbf_l[0].tie_to(m.rbf_l[1])
    #m.q_v.tie_to(m.rbf_v)
#     m.rbf_l.tie_to(m.rbf_va)
    # pt = numpy.array(params._get_params_transformed())
    # ptr = numpy.random.randn(*pt.shape)
#     params.X.tie_to(params.rbf_v)
