# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import itertools
import numpy
np = numpy
from .parameter_core import Parameterizable, adjust_name_for_printing, Pickleable
from .observable_array import ObsAr
from functools import reduce

###### printing
__constraints_name__ = "Constraint"
__index_name__ = "Index"
__tie_name__ = "Tied to"
__priors_name__ = "Prior"
__precision__ = numpy.get_printoptions()['precision'] # numpy printing precision used, sublassing numpy ndarray after all
__print_threshold__ = 5
######

class Param(Parameterizable, ObsAr):
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

    Important Note:
    Multilevel indexing (e.g. self[:2][1:]) is not supported and might lead to unexpected behaviour.
    Try to index in one go, using boolean indexing or the numpy builtin
    np.index function.

    See :py:class:`GPy.core.parameterized.Parameterized` for more details on constraining etc.

    """
    __array_priority__ = -1 # Never give back Param
    _fixes_ = None
    parameters = []
    def __new__(cls, name, input_array, default_constraint=None):
        obj = numpy.atleast_1d(super(Param, cls).__new__(cls, input_array=input_array))
        obj._current_slice_ = (slice(obj.shape[0]),)
        obj._realshape_ = obj.shape
        obj._realsize_ = obj.size
        obj._realndim_ = obj.ndim
        obj._original_ = obj
        return obj

    def __init__(self, name, input_array, default_constraint=None, *a, **kw):
        self._in_init_ = True
        super(Param, self).__init__(name=name, default_constraint=default_constraint, *a, **kw)
        self._in_init_ = False

    def build_pydot(self,G):
        import pydot
        node = pydot.Node(id(self), shape='trapezium', label=self.name)#, fontcolor='white', color='white')
        G.add_node(node)
        for _, o, _ in self.observers:
            label = o.name if hasattr(o, 'name') else str(o)
            observed_node = pydot.Node(id(o), label=label)
            G.add_node(observed_node)
            edge = pydot.Edge(str(id(self)), str(id(o)), color='darkorange2', arrowhead='vee')
            G.add_edge(edge)

        return node

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        super(Param, self).__array_finalize__(obj)
        self._parent_ = getattr(obj, '_parent_', None)
        self._parent_index_ = getattr(obj, '_parent_index_', None)
        self._default_constraint_ = getattr(obj, '_default_constraint_', None)
        self._current_slice_ = getattr(obj, '_current_slice_', None)
        self._realshape_ = getattr(obj, '_realshape_', None)
        self._realsize_ = getattr(obj, '_realsize_', None)
        self._realndim_ = getattr(obj, '_realndim_', None)
        self._original_ = getattr(obj, '_original_', None)
        self._name = getattr(obj, '_name', None)
        self._gradient_array_ = getattr(obj, '_gradient_array_', None)
        self._update_on = getattr(obj, '_update_on', None)
        self.constraints = getattr(obj, 'constraints', None)
        self.priors = getattr(obj, 'priors', None)

    @property
    def param_array(self):
        """
        As we are a leaf, this just returns self
        """
        return self

    @property
    def values(self):
        """
        Return self as numpy array view
        """
        return self.view(np.ndarray)

    @property
    def gradient(self):
        """
        Return a view on the gradient, which is in the same shape as this parameter is.
        Note: this is not the real gradient array, it is just a view on it.

        To work on the real gradient array use: self.full_gradient
        """
        if getattr(self, '_gradient_array_', None) is None:
            self._gradient_array_ = numpy.empty(self._realshape_, dtype=numpy.float64)
        return self._gradient_array_#[self._current_slice_]

    @gradient.setter
    def gradient(self, val):
        self._gradient_array_[:] = val

    #===========================================================================
    # Array operations -> done
    #===========================================================================
    def __getitem__(self, s, *args, **kwargs):
        if not isinstance(s, tuple):
            s = (s,)
        #if not reduce(lambda a, b: a or numpy.any(b is Ellipsis), s, False) and len(s) <= self.ndim:
        #    s += (Ellipsis,)
        new_arr = super(Param, self).__getitem__(s, *args, **kwargs)
        try:
            new_arr._current_slice_ = s
            new_arr._gradient_array_ = self.gradient[s]
            new_arr._original_ = self._original_
        except AttributeError: pass  # returning 0d array or float, double etc
        return new_arr

    def _raveled_index(self, slice_index=None):
        # return an index array on the raveled array, which is formed by the current_slice
        # of this object
        extended_realshape = numpy.cumprod((1,) + self._realshape_[:0:-1])[::-1]
        ind = self._indices(slice_index)
        if ind.ndim < 2: ind = ind[:, None]
        return numpy.asarray(numpy.apply_along_axis(lambda x: numpy.sum(extended_realshape * x), 1, ind), dtype=int)

    def _raveled_index_for(self, obj):
        return self._raveled_index()

    #===========================================================================
    # Constrainable
    #===========================================================================
    def _ensure_fixes(self):
        if not self._has_fixes(): self._fixes_ = numpy.ones(self._realsize_, dtype=bool)

    #===========================================================================
    # Convenience
    #===========================================================================
    @property
    def is_fixed(self):
        from .transformations import __fixed__
        return self.constraints[__fixed__].size == self.size

    def _get_original(self, param):
        return self._original_

    #===========================================================================
    # Pickling and copying
    #===========================================================================
    def copy(self):
        return Parameterizable.copy(self, which=self)

    def __deepcopy__(self, memo):
        s = self.__new__(self.__class__, name=self.name, input_array=self.view(numpy.ndarray).copy())
        memo[id(self)] = s
        import copy
        Pickleable.__setstate__(s, copy.deepcopy(self.__getstate__(), memo))
        return s

    def _setup_observers(self):
        """
        Setup the default observers

        1: pass through to parent, if present
        """
        if self.has_parent():
            self.add_observer(self._parent_, self._parent_._pass_through_notify_observers, -np.inf)

    #===========================================================================
    # Printing -> done
    #===========================================================================
    @property
    def _description_str(self):
        if self.size <= 1:
            return [str(self.view(numpy.ndarray)[0])]
        else: return [str(self.shape)]
    def parameter_names(self, add_self=False, adjust_for_printing=False, recursive=True):
        # this is just overwrighting the parameterized calls to parameter names, in order to maintain OOP
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
    def num_params(self):
        return 0
    @property
    def _constraints_str(self):
        #py3 fix
        #return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.constraints.iteritems()))]
        return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.constraints.items()))]
    @property
    def _priors_str(self):
        #py3 fix
        #return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.priors.iteritems()))]
        return [' '.join(map(lambda c: str(c[0]) if c[1].size == self._realsize_ else "{" + str(c[0]) + "}", self.priors.items()))]
    @property
    def _ties_str(self):
        return ['']
    def _ties_for(self, ravi):
        return [['N/A']]*ravi.size
    def __repr__(self, *args, **kwargs):
        name = "\033[1m{x:s}\033[0;0m:\n".format(
                            x=self.hierarchy_name())
        return name + super(Param, self).__repr__(*args, **kwargs)
    def _indices(self, slice_index=None):
        # get a int-array containing all indices in the first axis.
        if slice_index is None:
            slice_index = self._current_slice_
        try:
            indices = np.indices(self._realshape_, dtype=int)
            indices = indices[(slice(None),)+slice_index]
            indices = np.rollaxis(indices, 0, indices.ndim).reshape(-1,self._realndim_)
            #print indices_
            #if not np.all(indices==indices__):
            #    import ipdb; ipdb.set_trace()
        except:
            indices = np.indices(self._realshape_, dtype=int)
            indices = indices[(slice(None),)+slice_index]
            indices = np.rollaxis(indices, 0, indices.ndim)
        return indices
    def _max_len_names(self, gen, header):
        gen = map(lambda x: " ".join(map(str, x)), gen)
        return reduce(lambda a, b:max(a, len(b)), gen, len(header))
    def _max_len_values(self):
        return reduce(lambda a, b:max(a, len("{x:=.{0}g}".format(__precision__, x=b))), self.flat, len(self.hierarchy_name()))
    def _max_len_index(self, ind):
        return reduce(lambda a, b:max(a, len(str(b))), ind, len(__index_name__))
    def _short(self):
        # short string to print
        name = self.hierarchy_name()
        if self._realsize_ < 2:
            return name
        ind = self._indices()
        if ind.size > 4: indstr = ','.join(map(str, ind[:2])) + "..." + ','.join(map(str, ind[-2:]))
        else: indstr = ','.join(map(str, ind))
        return name + '[' + indstr + ']'

    def _repr_html_(self, constr_matrix=None, indices=None, prirs=None, ties=None):
        """Representation of the parameter in html for notebook display."""
        filter_ = self._current_slice_
        vals = self.flat
        if indices is None: indices = self._indices(filter_)
        ravi = self._raveled_index(filter_)
        if constr_matrix is None: constr_matrix = self.constraints.properties_for(ravi)
        if prirs is None: prirs = self.priors.properties_for(ravi)
        if ties is None: ties = self._ties_for(ravi)
        ties = [' '.join(map(lambda x: x, t)) for t in ties]
        header_format = """
<tr>
  <th><b>{i}</b></th>
  <th><b>{x}</b></th>
  <th><b>{c}</b></th>
  <th><b>{p}</b></th>
  <th><b>{t}</b></th>
</tr>"""
        header = header_format.format(x=self.hierarchy_name(), c=__constraints_name__, i=__index_name__, t=__tie_name__, p=__priors_name__)  # nice header for printing
        if not ties: ties = itertools.cycle([''])
        return "\n".join(["""<style type="text/css">
.tg  {padding:2px 3px;word-break:normal;border-collapse:collapse;border-spacing:0;border-color:#DCDCDC;margin:0px auto;width:100%;}
.tg td{font-family:"Courier New", Courier, monospace !important;font-weight:bold;color:#444;background-color:#F7FDFA;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
.tg th{font-family:"Courier New", Courier, monospace !important;font-weight:normal;color:#fff;background-color:#26ADE4;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
.tg .tg-left{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:left;}
.tg .tg-right{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:right;}
</style>"""] + ['<table class="tg">'] + [header] + ["<tr><td class=tg-left>{i}</td><td  class=tg-right>{x}</td><td class=tg-left>{c}</td><td class=tg-left>{p}</td><td class=tg-left>{t}</td></tr>".format(x=x, c=" ".join(map(str, c)), p=" ".join(map(str, p)), t=(t or ''), i=i) for i, x, c, t, p in zip(indices, vals, constr_matrix, ties, prirs)] + ["</table>"])

    def __str__(self, constr_matrix=None, indices=None, prirs=None, ties=None, lc=None, lx=None, li=None, lp=None, lt=None, only_name=False):
        filter_ = self._current_slice_
        vals = self.flat
        if indices is None: indices = self._indices(filter_)
        ravi = self._raveled_index(filter_)
        if constr_matrix is None: constr_matrix = self.constraints.properties_for(ravi)
        if prirs is None: prirs = self.priors.properties_for(ravi)
        if ties is None: ties = self._ties_for(ravi)
        ties = [' '.join(map(lambda x: x, t)) for t in ties]
        if lc is None: lc = self._max_len_names(constr_matrix, __constraints_name__)
        if lx is None: lx = self._max_len_values()
        if li is None: li = self._max_len_index(indices)
        if lt is None: lt = self._max_len_names(ties, __tie_name__)
        if lp is None: lp = self._max_len_names(prirs, __tie_name__)
        sep = '-'
        header_format = "  {i:{5}^{2}s}  |  \033[1m{x:{5}^{1}s}\033[0;0m  |  {c:{5}^{0}s}  |  {p:{5}^{4}s}  |  {t:{5}^{3}s}"
        if only_name: header = header_format.format(lc, lx, li, lt, lp, ' ', x=self.hierarchy_name(), c=sep*lc, i=sep*li, t=sep*lt, p=sep*lp)  # nice header for printing
        else: header = header_format.format(lc, lx, li, lt, lp, ' ', x=self.hierarchy_name(), c=__constraints_name__, i=__index_name__, t=__tie_name__, p=__priors_name__)  # nice header for printing
        if not ties: ties = itertools.cycle([''])
        return "\n".join([header] + ["  {i!s:^{3}s}  |  {x: >{1}.{2}g}  |  {c:^{0}s}  |  {p:^{5}s}  |  {t:^{4}s}  ".format(lc, lx, __precision__, li, lt, lp, x=x, c=" ".join(map(str, c)), p=" ".join(map(str, p)), t=(t or ''), i=i) for i, x, c, t, p in zip(indices, vals, constr_matrix, ties, prirs)])  # return all the constraints with right indices
        # except: return super(Param, self).__str__()

class ParamConcatenation(object):
    def __init__(self, params):
        """
        Parameter concatenation for convenience of printing regular expression matched arrays
        you can index this concatenation as if it was the flattened concatenation
        of all the parameters it contains, same for setting parameters (Broadcasting enabled).

        See :py:class:`GPy.core.parameter.Param` for more details on constraining.
        """
        # self.params = params
        from .lists_and_dicts import ArrayList
        self.params = ArrayList([])
        for p in params:
            for p in p.flattened_parameters:
                if p not in self.params:
                    self.params.append(p)
        self._param_sizes = [p.size for p in self.params]
        startstops = numpy.cumsum([0] + self._param_sizes)
        self._param_slices_ = [slice(start, stop) for start,stop in zip(startstops, startstops[1:])]

        parents = dict()
        for p in self.params:
            if p.has_parent():
                parent = p._parent_
                level = 0
                while parent is not None:
                    if parent in parents:
                        parents[parent] = max(level, parents[parent])
                    else:
                        parents[parent] = level
                    level += 1
                    parent = parent._parent_
        import operator
        #py3 fix
        #self.parents = map(lambda x: x[0], sorted(parents.iteritems(), key=operator.itemgetter(1)))
        self.parents = map(lambda x: x[0], sorted(parents.items(), key=operator.itemgetter(1)))
    #===========================================================================
    # Get/set items, enable broadcasting
    #===========================================================================
    def __getitem__(self, s):
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True;
        params = [p.param_array.flat[ind[ps]] for p,ps in zip(self.params, self._param_slices_) if numpy.any(p.param_array.flat[ind[ps]])]
        if len(params)==1: return params[0]
        return ParamConcatenation(params)
    def __setitem__(self, s, val, update=True):
        if isinstance(val, ParamConcatenation):
            val = val.values()
        ind = numpy.zeros(sum(self._param_sizes), dtype=bool); ind[s] = True;
        vals = self.values(); vals[s] = val
        for p, ps in zip(self.params, self._param_slices_):
            p.flat[ind[ps]] = vals[ps]
        if update:
            self.update_all_params()
    def values(self):
        return numpy.hstack([p.param_array.flat for p in self.params])
    #===========================================================================
    # parameter operations:
    #===========================================================================
    def update_all_params(self):
        for par in self.parents:
            par.trigger_update(trigger_parent=False)

    def constrain(self, constraint, warning=True):
        [param.constrain(constraint, trigger_parent=False) for param in self.params]
        self.update_all_params()
    constrain.__doc__ = Param.constrain.__doc__

    def constrain_positive(self, warning=True):
        [param.constrain_positive(warning, trigger_parent=False) for param in self.params]
        self.update_all_params()
    constrain_positive.__doc__ = Param.constrain_positive.__doc__

    def constrain_fixed(self, value=None, warning=True, trigger_parent=True):
        [param.constrain_fixed(value, warning, trigger_parent) for param in self.params]
    constrain_fixed.__doc__ = Param.constrain_fixed.__doc__
    fix = constrain_fixed

    def constrain_negative(self, warning=True):
        [param.constrain_negative(warning, trigger_parent=False) for param in self.params]
        self.update_all_params()
    constrain_negative.__doc__ = Param.constrain_negative.__doc__

    def constrain_bounded(self, lower, upper, warning=True):
        [param.constrain_bounded(lower, upper, warning, trigger_parent=False) for param in self.params]
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

    __lt__ = lambda self, val: self.values() < val
    __le__ = lambda self, val: self.values() <= val
    __eq__ = lambda self, val: self.values() == val
    __ne__ = lambda self, val: self.values() != val
    __gt__ = lambda self, val: self.values() > val
    __ge__ = lambda self, val: self.values() >= val
    def __str__(self, *args, **kwargs):
        def f(p):
            ind = p._raveled_index()
            return p.constraints.properties_for(ind), p._ties_for(ind), p.priors.properties_for(ind)
        params = self.params
        constr_matrices, ties_matrices, prior_matrices = zip(*map(f, params))
        indices = [p._indices() for p in params]
        lc = max([p._max_len_names(cm, __constraints_name__) for p, cm in zip(params, constr_matrices)])
        lx = max([p._max_len_values() for p in params])
        li = max([p._max_len_index(i) for p, i in zip(params, indices)])
        lt = max([p._max_len_names(tm, __tie_name__) for p, tm in zip(params, ties_matrices)])
        lp = max([p._max_len_names(pm, __constraints_name__) for p, pm in zip(params, prior_matrices)])
        strings = []
        start = True
        for p, cm, i, tm, pm in zip(params,constr_matrices,indices,ties_matrices,prior_matrices):
            strings.append(p.__str__(constr_matrix=cm, indices=i, prirs=pm, ties=tm, lc=lc, lx=lx, li=li, lp=lp, lt=lt, only_name=(1-start)))
            start = False
        return "\n".join(strings)
    def __repr__(self):
        return "\n".join(map(repr,self.params))

    def __ilshift__(self, *args, **kwargs):
        self[:] = np.ndarray.__ilshift__(self.values(), *args, **kwargs)

    def __irshift__(self, *args, **kwargs):
        self[:] = np.ndarray.__irshift__(self.values(), *args, **kwargs)

    def __ixor__(self, *args, **kwargs):
        self[:] = np.ndarray.__ixor__(self.values(), *args, **kwargs)

    def __ipow__(self, *args, **kwargs):
        self[:] = np.ndarray.__ipow__(self.values(), *args, **kwargs)

    def __ifloordiv__(self, *args, **kwargs):
        self[:] = np.ndarray.__ifloordiv__(self.values(), *args, **kwargs)

    def __isub__(self, *args, **kwargs):
        self[:] = np.ndarray.__isub__(self.values(), *args, **kwargs)

    def __ior__(self, *args, **kwargs):
        self[:] = np.ndarray.__ior__(self.values(), *args, **kwargs)

    def __itruediv__(self, *args, **kwargs):
        self[:] = np.ndarray.__itruediv__(self.values(), *args, **kwargs)

    def __idiv__(self, *args, **kwargs):
        self[:] = np.ndarray.__idiv__(self.values(), *args, **kwargs)

    def __iand__(self, *args, **kwargs):
        self[:] = np.ndarray.__iand__(self.values(), *args, **kwargs)

    def __imod__(self, *args, **kwargs):
        self[:] = np.ndarray.__imod__(self.values(), *args, **kwargs)

    def __iadd__(self, *args, **kwargs):
        self[:] = np.ndarray.__iadd__(self.values(), *args, **kwargs)

    def __imul__(self, *args, **kwargs):
        self[:] = np.ndarray.__imul__(self.values(), *args, **kwargs)
