# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import itertools
import numpy
from parameter_core import Constrainable, Gradcheckable, Indexable, Parentable, adjust_name_for_printing
from array_core import ObservableArray, ParamList

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__priors_name__ = "Prior"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
__print_threshold__ = 5
######

class Param(Constrainable, ObservableArray, Gradcheckable):
    """
    Parameter object for GPy models.

    :param str name:           name of the parameter to be printed
    :param input_array:        array which this parameter handles
    :type input_array:         numpy.ndarray
    :param default_constraint: The default constraint for this parameter
    :type default_constraint:

    You can add/remove constraints by calling constrain on the parameter itself, e.g:

        - self[:,1].constrain_positive()
        - self[0].tie_to(other)
        - self.untie()
        - self[:3,:].unconstrain()
        - self[1].fix()

    Fixing parameters will fix them to the value they are right now. If you change
    the fixed value, it will be fixed to the new value!

    See :py:class:`GPy.core.parameterized.Parameterized` for more details on constraining etc.

    """
    __array_priority__ = -1 # Never give back Param
    _fixes_ = None
    _parameters_ = []
    def __new__(cls, name, input_array, default_constraint=None):
        obj = numpy.atleast_1d(super(Param, cls).__new__(cls, input_array=input_array))
        cls.__name__ = "Param"
        obj._current_slice_ = (slice(obj.shape[0]),)
        obj._realshape_ = obj.shape
        obj._realsize_ = obj.size
        obj._realndim_ = obj.ndim
        obj._updated_ = False
        from index_operations import SetDict
        obj._tied_to_me_ = SetDict()
        obj._tied_to_ = []
        obj._original_ = True
        obj._gradient_ = None
        return obj

    def __init__(self, name, input_array, default_constraint=None, *a, **kw):
        super(Param, self).__init__(name=name, default_constraint=default_constraint, *a, **kw)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        super(Param, self).__array_finalize__(obj)
        self._direct_parent_ = getattr(obj, '_direct_parent_', None)
        self._parent_index_ = getattr(obj, '_parent_index_', None)
        self._default_constraint_ = getattr(obj, '_default_constraint_', None)
        self._current_slice_ = getattr(obj, '_current_slice_', None)
        self._tied_to_me_ = getattr(obj, '_tied_to_me_', None)
        self._tied_to_ = getattr(obj, '_tied_to_', None)
        self._realshape_ = getattr(obj, '_realshape_', None)
        self._realsize_ = getattr(obj, '_realsize_', None)
        self._realndim_ = getattr(obj, '_realndim_', None)
        self._updated_ = getattr(obj, '_updated_', None)
        self._original_ = getattr(obj, '_original_', None)
        self._name = getattr(obj, 'name', None)
        self._gradient_ = getattr(obj, '_gradient_', None)
        self.constraints = getattr(obj, 'constraints', None)
        self.priors = getattr(obj, 'priors', None)


    @property
    def gradient(self):
        if self._gradient_ is None:
            self._gradient_ = numpy.zeros(self._realshape_)
        return self._gradient_[self._current_slice_]
    @gradient.setter
    def gradient(self, val):
        self.gradient[:] = val
        
    #===========================================================================
    # Pickling operations
    #===========================================================================
    def __reduce_ex__(self):
        func, args, state = super(Param, self).__reduce__()
        return func, args, (state,
                            (self.name,
                             self._direct_parent_,
                             self._parent_index_,
                             self._default_constraint_,
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
        self._default_constraint_ = state.pop()
        self._parent_index_ = state.pop()
        self._direct_parent_ = state.pop()
        self.name = state.pop()
    
    def copy(self, *args):
        constr = self.constraints.copy()
        priors = self.priors.copy()
        p = Param(self.name, self.view(numpy.ndarray).copy(), self._default_constraint_)
        p.constraints = constr
        p.priors = priors
        return p
    #===========================================================================
    # get/set parameters
    #===========================================================================
    def _set_params(self, param, update=True):
        self.flat = param

    def _get_params(self):
        return self.flat

    def _collect_gradient(self, target):
        target += self.gradient.flat

    def _set_gradient(self, g):
        self.gradient = g.reshape(self._realshape_)

    #===========================================================================
    # Array operations -> done
    #===========================================================================
    def __getitem__(self, s, *args, **kwargs):
        if not isinstance(s, tuple):
            s = (s,)
        if not reduce(lambda a, b: a or numpy.any(b is Ellipsis), s, False) and len(s) <= self.ndim:
            s += (Ellipsis,)
        new_arr = super(Param, self).__getitem__(s, *args, **kwargs)
        try: new_arr._current_slice_ = s; new_arr._original_ = self.base is new_arr.base
        except AttributeError: pass  # returning 0d array or float, double etc
        return new_arr
    def __setitem__(self, s, val):
        super(Param, self).__setitem__(s, val)
        if self.has_parent():
            self._direct_parent_._notify_parameters_changed()
        #self._notify_observers()
        
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
            if a < 0:
                a = self._realshape_[i] + a
            internal_offset += a * extended_realshape[i]
        return internal_offset
    
    def _raveled_index(self, slice_index=None):
        # return an index array on the raveled array, which is formed by the current_slice
        # of this object
        extended_realshape = numpy.cumprod((1,) + self._realshape_[:0:-1])[::-1]
        ind = self._indices(slice_index)
        if ind.ndim < 2: ind = ind[:, None]
        return numpy.asarray(numpy.apply_along_axis(lambda x: numpy.sum(extended_realshape * x), 1, ind), dtype=int)
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
                elif isinstance(a, (list, numpy.ndarray, tuple)):
                    a = numpy.asarray(a, dtype=int)
                    a[a < 0] = b + a[a < 0]
                elif a < 0:
                    a = b + a
                return numpy.r_[a]
            return numpy.r_[:b]
        return itertools.imap(f, itertools.izip_longest(slice_index[:self._realndim_], self._realshape_, fillvalue=slice(self.size)))

    #===========================================================================
    # Convenience
    #===========================================================================
    @property
    def is_fixed(self):
        return self._highest_parent_._is_fixed(self)
    #def round(self, decimals=0, out=None):
    #    view = super(Param, self).round(decimals, out).view(Param)
    #    view.__array_finalize__(self)
    #    return view
    #round.__doc__ = numpy.round.__doc__
    def _get_original(self, param):
        return self

    #===========================================================================
    # Printing -> done
    #===========================================================================
    @property
    def _description_str(self):
        if self.size <= 1: return ["%f" % self]
        else: return [str(self.shape)]
    def parameter_names(self, add_self=False, adjust_for_printing=False):
        if adjust_for_printing:
            return [adjust_name_for_printing(self.name)]
        return [self.name]
    @property
    def flattened_parameters(self):
        return [self]
    @property
    def parameter_shapes(self):
        return [self.shape]
    @property
    def _constraints_str(self):
        return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.constraints.iteritems()))]
    @property
    def _priors_str(self):
        return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.priors.iteritems()))]
    @property
    def _ties_str(self):
        return [t._short() for t in self._tied_to_] or ['']
    def __repr__(self, *args, **kwargs):
        name = "\033[1m{x:s}\033[0;0m:\n".format(
                            x=self.hirarchy_name())
        return name + super(Param, self).__repr__(*args, **kwargs)
    def _ties_for(self, rav_index):
        # size = sum(p.size for p in self._tied_to_)
        ties = numpy.empty(shape=(len(self._tied_to_), numpy.size(rav_index)), dtype=Param)
        for i, tied_to in enumerate(self._tied_to_):
            for t, ind in tied_to._tied_to_me_.iteritems():
                if t._parent_index_ == self._parent_index_:
                    matches = numpy.where(rav_index[:, None] == t._raveled_index()[None, :])
                    tt_rav_index = tied_to._raveled_index()
                    ind_rav_matches = numpy.where(tt_rav_index == numpy.array(list(ind)))[0]
                    if len(ind) != 1: ties[i, matches[0][ind_rav_matches]] = numpy.take(tt_rav_index, matches[1], mode='wrap')[ind_rav_matches]
                    else: ties[i, matches[0]] = numpy.take(tt_rav_index, matches[1], mode='wrap')
        return map(lambda a: sum(a, []), zip(*[[[tie.flatten()] if tx != None else [] for tx in t] for t, tie in zip(ties, self._tied_to_)]))
    def _indices(self, slice_index=None):
        # get a int-array containing all indices in the first axis.
        if slice_index is None:
            slice_index = self._current_slice_
        if isinstance(slice_index, (tuple, list)):
            clean_curr_slice = [s for s in slice_index if numpy.any(s != Ellipsis)]
            for i in range(self._realndim_-len(clean_curr_slice)):
                i+=len(clean_curr_slice)
                clean_curr_slice += range(self._realshape_[i])
            if (all(isinstance(n, (numpy.ndarray, list, tuple)) for n in clean_curr_slice)
                and len(set(map(len, clean_curr_slice))) <= 1):
                return numpy.fromiter(itertools.izip(*clean_curr_slice),
                    dtype=[('', int)] * self._realndim_, count=len(clean_curr_slice[0])).view((int, self._realndim_))
        expanded_index = list(self._expand_index(slice_index))
        return numpy.fromiter(itertools.product(*expanded_index),
                 dtype=[('', int)] * self._realndim_, count=reduce(lambda a, b: a * b.size, expanded_index, 1)).view((int, self._realndim_))
    def _max_len_names(self, gen, header):
        gen = map(lambda x: " ".join(map(str, x)), gen)
        return reduce(lambda a, b:max(a, len(b)), gen, len(header))
    def _max_len_values(self):
        return reduce(lambda a, b:max(a, len("{x:=.{0}g}".format(__precision__, x=b))), self.flat, len(self.hirarchy_name()))
    def _max_len_index(self, ind):
        return reduce(lambda a, b:max(a, len(str(b))), ind, len(__index_name__))
    def _short(self):
        # short string to print
        name = self.hirarchy_name()
        if self._realsize_ < 2:
            return name
        ind = self._indices()
        if ind.size > 4: indstr = ','.join(map(str, ind[:2])) + "..." + ','.join(map(str, ind[-2:]))
        else: indstr = ','.join(map(str, ind))
        return name + '[' + indstr + ']'
    def __str__(self, constr_matrix=None, indices=None, prirs=None, ties=None, lc=None, lx=None, li=None, lp=None, lt=None, only_name=False):
        filter_ = self._current_slice_
        vals = self.flat
        if indices is None: indices = self._indices(filter_)
        ravi = self._raveled_index(filter_)
        if constr_matrix is None: constr_matrix = self.constraints.properties_for(ravi)
        if prirs is None: prirs = self.priors.properties_for(ravi)
        if ties is None: ties = self._ties_for(ravi)
        ties = [' '.join(map(lambda x: x._short(), t)) for t in ties]
        if lc is None: lc = self._max_len_names(constr_matrix, __constraints_name__)
        if lx is None: lx = self._max_len_values()
        if li is None: li = self._max_len_index(indices)
        if lt is None: lt = self._max_len_names(ties, __tie_name__)
        if lp is None: lp = self._max_len_names(prirs, __tie_name__)
        sep = '-'
        header_format = "  {i:{5}^{2}s}  |  \033[1m{x:{5}^{1}s}\033[0;0m  |  {c:{5}^{0}s}  |  {p:{5}^{4}s}  |  {t:{5}^{3}s}"
        if only_name: header = header_format.format(lc, lx, li, lt, lp, ' ', x=self.hirarchy_name(), c=sep*lc, i=sep*li, t=sep*lt, p=sep*lp)  # nice header for printing
        else: header = header_format.format(lc, lx, li, lt, lp, ' ', x=self.hirarchy_name(), c=__constraints_name__, i=__index_name__, t=__tie_name__, p=__priors_name__)  # nice header for printing
        if not ties: ties = itertools.cycle([''])
        return "\n".join([header] + ["  {i!s:^{3}s}  |  {x: >{1}.{2}g}  |  {c:^{0}s}  |  {p:^{5}s}  |  {t:^{4}s}  ".format(lc, lx, __precision__, li, lt, lp, x=x, c=" ".join(map(str, c)), p=" ".join(map(str, p)), t=(t or ''), i=i) for i, x, c, t, p in itertools.izip(indices, vals, constr_matrix, ties, prirs)])  # return all the constraints with right indices
        # except: return super(Param, self).__str__()

class ParamConcatenation(object):
    def __init__(self, params):
        """
        Parameter concatenation for convienience of printing regular expression matched arrays
        you can index this concatenation as if it was the flattened concatenation
        of all the parameters it contains, same for setting parameters (Broadcasting enabled).

        See :py:class:`GPy.core.parameter.Param` for more details on constraining.
        """
        # self.params = params
        self.params = ParamList([])
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
        if isinstance(val, ParamConcatenation):
            val = val._vals()
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True;
        vals = self._vals(); vals[s] = val; del val
        [numpy.place(p, ind[ps], vals[ps]) and update and p._notify_observers()
         for p, ps in zip(self.params, self._param_slices_)]
    def _vals(self):
        return numpy.hstack([p._get_params() for p in self.params])
    #===========================================================================
    # parameter operations:
    #===========================================================================
    def update_all_params(self):
        for p in self.params:
            p._notify_observers()

    def constrain(self, constraint, warning=True):
        [param.constrain(constraint, update=False) for param in self.params]
        self.update_all_params()
    constrain.__doc__ = Param.constrain.__doc__

    def constrain_positive(self, warning=True):
        [param.constrain_positive(warning, update=False) for param in self.params]
        self.update_all_params()
    constrain_positive.__doc__ = Param.constrain_positive.__doc__

    def constrain_fixed(self, warning=True):
        [param.constrain_fixed(warning) for param in self.params]
    constrain_fixed.__doc__ = Param.constrain_fixed.__doc__
    fix = constrain_fixed

    def constrain_negative(self, warning=True):
        [param.constrain_negative(warning, update=False) for param in self.params]
        self.update_all_params()
    constrain_negative.__doc__ = Param.constrain_negative.__doc__

    def constrain_bounded(self, lower, upper, warning=True):
        [param.constrain_bounded(lower, upper, warning, update=False) for param in self.params]
        self.update_all_params()
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

    def checkgrad(self, verbose=0, step=1e-6, tolerance=1e-3):
        return self.params[0]._highest_parent_._checkgrad(self, verbose, step, tolerance)
    #checkgrad.__doc__ = Gradcheckable.checkgrad.__doc__

    __lt__ = lambda self, val: self._vals() < val
    __le__ = lambda self, val: self._vals() <= val
    __eq__ = lambda self, val: self._vals() == val
    __ne__ = lambda self, val: self._vals() != val
    __gt__ = lambda self, val: self._vals() > val
    __ge__ = lambda self, val: self._vals() >= val
    def __str__(self, *args, **kwargs):
        def f(p):
            ind = p._raveled_index()
            return p.constraints.properties_for(ind), p._ties_for(ind), p.priors.properties_for(ind)
        params = self.params
        constr_matrices, ties_matrices, prior_matrices = zip(*map(f, params))
        indices = [p._indices() for p in params]
        lc = max([p._max_len_names(cm, __constraints_name__) for p, cm in itertools.izip(params, constr_matrices)])
        lx = max([p._max_len_values() for p in params])
        li = max([p._max_len_index(i) for p, i in itertools.izip(params, indices)])
        lt = max([p._max_len_names(tm, __tie_name__) for p, tm in itertools.izip(params, ties_matrices)])
        lp = max([p._max_len_names(pm, __constraints_name__) for p, pm in itertools.izip(params, prior_matrices)])
        strings = []
        start = True
        for p, cm, i, tm, pm in itertools.izip(params,constr_matrices,indices,ties_matrices,prior_matrices):
            strings.append(p.__str__(constr_matrix=cm, indices=i, prirs=pm, ties=tm, lc=lc, lx=lx, li=li, lp=lp, lt=lt, only_name=(1-start)))
            start = False
        return "\n".join(strings)
    def __repr__(self):
        return "\n".join(map(repr,self.params))
