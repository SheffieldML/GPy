# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy; np = numpy
import copy
import cPickle
import itertools
from re import compile, _pattern_type
from param import ParamConcatenation, Param
from parameter_core import Constrainable, Pickleable, Observable, adjust_name_for_printing, Gradcheckable
from transformations import __fixed__, FIXED, UNFIXED
from array_core import ParamList

class Parameterized(Constrainable, Pickleable, Observable, Gradcheckable):
    """
    Parameterized class

    Say m is a handle to a parameterized class.

    Printing parameters:

        - print m:           prints a nice summary over all parameters
        - print m.name:      prints details for param with name 'name'
        - print m[regexp]: prints details for all the parameters
                             which match (!) regexp
        - print m['']:       prints details for all parameters

        Fields:

            Name:       The name of the param, can be renamed!
            Value:      Shape or value, if one-valued
            Constrain:  constraint of the param, curly "{c}" brackets indicate
                        some parameters are constrained by c. See detailed print
                        to get exact constraints.
            Tied_to:    which paramter it is tied to.

    Getting and setting parameters:

        Set all values in param to one:

            m.name.to.param = 1

    Handling of constraining, fixing and tieing parameters:

        You can constrain parameters by calling the constrain on the param itself, e.g:

            - m.name[:,1].constrain_positive()
            - m.name[0].tie_to(m.name[1])

        Fixing parameters will fix them to the value they are right now. If you change
        the parameters value, the param will be fixed to the new value!

        If you want to operate on all parameters use m[''] to wildcard select all paramters
        and concatenate them. Printing m[''] will result in printing of all parameters in detail.
    """
    def __init__(self, name=None):
        super(Parameterized, self).__init__(name=name)
        self._in_init_ = True
        self._parameters_ = ParamList()
        self.size = sum(p.size for p in self._parameters_)
        if not self._has_fixes():
            self._fixes_ = None
        self._param_slices_ = []
        self._connect_parameters()
        self._added_names_ = set()
        del self._in_init_

    def _has_fixes(self):
        return hasattr(self, "_fixes_") and self._fixes_ is not None

    def add_parameter(self, param, index=None):
        """
        :param parameters:  the parameters to add
        :type parameters:   list of or one :py:class:`GPy.core.param.Param`
        :param [index]:     index of where to put parameters


        Add all parameters to this param class, you can insert parameters
        at any given index using the :func:`list.insert` syntax
        """
        # if param.has_parent():
        #    raise AttributeError, "parameter {} already in another model, create new object (or copy) for adding".format(param._short())
        if param in self._parameters_ and index is not None:
            self.remove_parameter(param)
            self.add_parameter(param, index)
        elif param not in self._parameters_:
            # make sure the size is set
            if index is None:
                self.constraints.update(param.constraints, self.size)
                self._parameters_.append(param)
            else:
                start = sum(p.size for p in self._parameters_[:index])
                self.constraints.shift(start, param.size)
                self.constraints.update(param.constraints, start)
                self._parameters_.insert(index, param)
            self.size += param.size
        else:
            raise RuntimeError, """Parameter exists already added and no copy made"""
        self._connect_parameters()
        self._notify_parent_change()
        self._connect_fixes()


    def add_parameters(self, *parameters):
        """
        convenience method for adding several
        parameters without gradient specification
        """
        [self.add_parameter(p) for p in parameters]

    def remove_parameter(self, param):
        """
        :param param: param object to remove from being a parameter of this parameterized object.
        """
        if not param in self._parameters_:
            raise RuntimeError, "Parameter {} does not belong to this object, remove parameters directly from their respective parents".format(param._short())
        del self._parameters_[param._parent_index_]
        self.size -= param.size
        constr = param.constraints.copy()
        param.constraints.clear()
        param.constraints = constr
        param._direct_parent_ = None
        param._parent_index_ = None
        param._connect_fixes()
        param._notify_parent_change()
        pname = adjust_name_for_printing(param.name)
        if pname in self._added_names_:
            del self.__dict__[pname]
        self._connect_parameters()
        #self._notify_parent_change()
        self._connect_fixes()

    def _connect_parameters(self):
        # connect parameterlist to this parameterized object
        # This just sets up the right connection for the params objects
        # to be used as parameters
        # it also sets the constraints for each parameter to the constraints 
        # of their respective parents 
        if not hasattr(self, "_parameters_") or len(self._parameters_) < 1:
            # no parameters for this class
            return
        sizes = [0]
        self._param_slices_ = []
        for i, p in enumerate(self._parameters_):
            p._direct_parent_ = self
            p._parent_index_ = i
            not_unique = []
            sizes.append(p.size + sizes[-1])
            self._param_slices_.append(slice(sizes[-2], sizes[-1]))
            pname = adjust_name_for_printing(p.name)
            # and makes sure to not delete programmatically added parameters
            if pname in self.__dict__:
                if isinstance(self.__dict__[pname], (Parameterized, Param)):
                    if not p is self.__dict__[pname]:
                        not_unique.append(pname)
                        del self.__dict__[pname]
            elif not (pname in not_unique):
                self.__dict__[pname] = p
                self._added_names_.add(pname)

    #===========================================================================
    # Pickling operations
    #===========================================================================
    def pickle(self, f, protocol=-1):
        """
        :param f: either filename or open file object to write to.
                  if it is an open buffer, you have to make sure to close
                  it properly.
        :param protocol: pickling protocol to use, python-pickle for details.
        """
        if isinstance(f, str):
            with open(f, 'w') as f:
                cPickle.dump(self, f, protocol)
        else:
            cPickle.dump(self, f, protocol)
    def copy(self):
        """Returns a (deep) copy of the current model """
        # dc = dict()
        # for k, v in self.__dict__.iteritems():
            # if k not in ['_highest_parent_', '_direct_parent_']:
                # dc[k] = copy.deepcopy(v)

        # dc = copy.deepcopy(self.__dict__)
        # dc['_highest_parent_'] = None
        # dc['_direct_parent_'] = None
        # s = self.__class__.new()
        # s.__dict__ = dc
        return copy.deepcopy(self)
    def __getstate__(self):
        if self._has_get_set_state():
            return self._getstate()
        return self.__dict__
    def __setstate__(self, state):
        if self._has_get_set_state():
            self._setstate(state)  # set state
            # self._set_params(self._get_params()) # restore all values
            return
        self.__dict__ = state
    def _has_get_set_state(self):
        return '_getstate' in vars(self.__class__) and '_setstate' in vars(self.__class__)
    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        For inheriting from Parameterized:

        Allways append the state of the inherited object
        and call down to the inherited object in _setstate!!
        """
        return [
                self._fixes_,
                self.constraints,
                self._parameters_,
                self._name,
                self._added_names_,
                ]

    def _setstate(self, state):
        self._added_names_ = state.pop()
        self._name = state.pop()
        self._parameters_ = state.pop()
        self._connect_parameters()
        self.constraints = state.pop()
        self._fixes_ = state.pop()
        self.parameters_changed()
    #===========================================================================
    # Gradient control
    #===========================================================================
    def _transform_gradients(self, g):
        if self.has_parent():
            return g
        x = self._get_params()
        [numpy.put(g, i, g[i] * c.gradfactor(x[i])) for c, i in self.constraints.iteritems() if c != __fixed__]
        for p in self.flattened_parameters:
            for t, i in p._tied_to_me_.iteritems():
                g[self._offset_for(p) + numpy.array(list(i))] += g[self._raveled_index_for(t)]
        if self._has_fixes(): return g[self._fixes_]
        return g
    #===========================================================================
    # Optimization handles:
    #===========================================================================
    def _get_param_names(self):
        n = numpy.array([p.name_hirarchical + '[' + str(i) + ']' for p in self.flattened_parameters for i in p._indices()])
        return n
    def _get_param_names_transformed(self):
        n = self._get_param_names()
        if self._has_fixes():
            return n[self._fixes_]
        return n
    def _get_params(self):
        # don't overwrite this anymore!
        if not self.size:
            return np.empty(shape=(0,), dtype=np.float64)
        return numpy.hstack([x._get_params() for x in self._parameters_ if x.size > 0])

    def _set_params(self, params, update=True):
        # don't overwrite this anymore!
        [p._set_params(params[s], update=update) for p, s in itertools.izip(self._parameters_, self._param_slices_)]
        self.parameters_changed()
    def _get_params_transformed(self):
        # transformed parameters (apply transformation rules)
        p = self._get_params()
        [numpy.put(p, ind, c.finv(p[ind])) for c, ind in self.constraints.iteritems() if c != __fixed__]
        if self._has_fixes():
            return p[self._fixes_]
        return p
    def _set_params_transformed(self, p):
        # inverse apply transformations for parameters and set the resulting parameters
        self._set_params(self._untransform_params(p))
    def _untransform_params(self, p):
        p = p.copy()
        if self._has_fixes(): tmp = self._get_params(); tmp[self._fixes_] = p; p = tmp; del tmp
        [numpy.put(p, ind, c.f(p[ind])) for c, ind in self.constraints.iteritems() if c != __fixed__]
        return p
    def _name_changed(self, param, old_name):
        if hasattr(self, old_name) and old_name in self._added_names_:
            delattr(self, old_name)
            self._added_names_.remove(old_name)
        pname = adjust_name_for_printing(param.name)
        if pname not in self.__dict__:
            self._added_names_.add(pname)
            self.__dict__[pname] = param
    #===========================================================================
    # Indexable Handling
    #===========================================================================
    def _backtranslate_index(self, param, ind):
        # translate an index in parameterized indexing into the index of param
        ind = ind - self._offset_for(param)
        ind = ind[ind >= 0]
        internal_offset = param._internal_offset()
        ind = ind[ind < param.size + internal_offset]
        return ind
    def _offset_for(self, param):
        # get the offset in the parameterized index array for param
        if param.has_parent():
            if param._direct_parent_._get_original(param) in self._parameters_:
                return self._param_slices_[param._direct_parent_._get_original(param)._parent_index_].start
            return self._offset_for(param._direct_parent_) + param._direct_parent_._offset_for(param)
        return 0
    
    def _raveled_index_for(self, param):
        """
        get the raveled index for a param
        that is an int array, containing the indexes for the flattened
        param inside this parameterized logic.
        """
        if isinstance(param, ParamConcatenation):
            return numpy.hstack((self._raveled_index_for(p) for p in param.params))
        return param._raveled_index() + self._offset_for(param)
    
    def _raveled_index(self):
        """
        get the raveled index for this object,
        this is not in the global view of things!
        """
        return numpy.r_[:self.size]
    #===========================================================================
    # Fixing parameters:
    #===========================================================================
    def _fixes_for(self, param):
        if self._has_fixes():
            return self._fixes_[self._raveled_index_for(param)]
        return numpy.ones(self.size, dtype=bool)[self._raveled_index_for(param)]
    #===========================================================================
    # Convenience for fixed, tied checking of param:
    #===========================================================================
    def fixed_indices(self):
        return np.array([x.is_fixed for x in self._parameters_])
    def _is_fixed(self, param):
        # returns if the whole param is fixed
        if not self._has_fixes():
            return False
        return not self._fixes_[self._raveled_index_for(param)].any()
        # return not self._fixes_[self._offset_for(param): self._offset_for(param)+param._realsize_].any()
    @property
    def is_fixed(self):
        for p in self._parameters_:
            if not p.is_fixed: return False
        return True
    def _get_original(self, param):
        # if advanced indexing is activated it happens that the array is a copy
        # you can retrieve the original param through this method, by passing
        # the copy here
        return self._parameters_[param._parent_index_]
    def hirarchy_name(self):
        if self.has_parent():
            return self._direct_parent_.hirarchy_name() + adjust_name_for_printing(self.name) + "."
        return ''
    #===========================================================================
    # Constraint Handling:
    #===========================================================================
    #===========================================================================
    # def _add_constrain(self, param, transform, warning=True):
    #     rav_i = self._raveled_index_for(param)
    #     reconstrained = self._remove_constrain(param, index=rav_i) # remove constraints before
    #     # if removing constraints before adding new is not wanted, just delete the above line!
    #     self.constraints.add(transform, rav_i)
    #     param = self._get_original(param)
    #     if not (transform == __fixed__):
    #         param._set_params(transform.initialize(param._get_params()), update=False)
    #     if warning and any(reconstrained):
    #         # if you want to print the whole params object, which was reconstrained use:
    #         # m = str(param[self._backtranslate_index(param, reconstrained)])
    #         print "Warning: re-constraining parameters:\n{}".format(param._short())
    #     return rav_i
    # def _remove_constrain(self, param, *transforms, **kwargs):
    #     if not transforms:
    #         transforms = self.constraints.properties()
    #     removed_indices = numpy.array([]).astype(int)
    #     if "index" in kwargs: index = kwargs['index']
    #     else: index = self._raveled_index_for(param)
    #     for constr in transforms:
    #         removed = self.constraints.remove(constr, index)
    #         if constr is __fixed__:
    #             self._set_unfixed(removed)
    #         removed_indices = numpy.union1d(removed_indices, removed)
    #     return removed_indices
    #===========================================================================
    #===========================================================================
    # Get/set parameters:
    #===========================================================================
    def grep_param_names(self, regexp):
        """
        create a list of parameters, matching regular expression regexp
        """
        if not isinstance(regexp, _pattern_type): regexp = compile(regexp)
        found_params = []
        for p in self._parameters_:
            if regexp.match(p.name) is not None:
                found_params.append(p)
            if isinstance(p, Parameterized):
                found_params.extend(p.grep_param_names(regexp))
        return found_params
        return [param for param in self._parameters_ if regexp.match(param.name) is not None]
    def __getitem__(self, name, paramlist=None):
        if paramlist is None:
            paramlist = self.grep_param_names(name)
        if len(paramlist) < 1: raise AttributeError, name
        if len(paramlist) == 1:
            if isinstance(paramlist[-1], Parameterized):
                paramlist = paramlist[-1].flattened_parameters
                if len(paramlist) != 1:
                    return ParamConcatenation(paramlist)
            return paramlist[-1]
        return ParamConcatenation(paramlist)
    def __setitem__(self, name, value, paramlist=None):
        try: param = self.__getitem__(name, paramlist)
        except AttributeError as a: raise a
        param[:] = value
#     def __getattr__(self, name):
#         return self.__getitem__(name)
#     def __getattribute__(self, name):
#         #try:
#             return object.__getattribute__(self, name)
        # except AttributeError:
        #    _, a, tb = sys.exc_info()
        #    try:
        #        return self.__getitem__(name)
        #    except AttributeError:
        #        raise AttributeError, a.message, tb
    def __setattr__(self, name, val):
        # override the default behaviour, if setting a param, so broadcasting can by used
        if hasattr(self, "_parameters_"):
            paramlist = self.grep_param_names(name)
            if len(paramlist) == 1: self.__setitem__(name, val, paramlist); return
        object.__setattr__(self, name, val);
    #===========================================================================
    # Printing:
    #===========================================================================
    def _short(self):
        # short string to print
        if self.has_parent():
            return self._direct_parent_.hirarchy_name() + adjust_name_for_printing(self.name)
        else:
            return adjust_name_for_printing(self.name)
    #parameter_names = property(parameter_names, doc="Names for all parameters handled by this parameterization object -- will add hirarchy name entries for printing")
    def _collect_gradient(self, target):
        [p._collect_gradient(target[s]) for p, s in itertools.izip(self._parameters_, self._param_slices_)]
    @property
    def flattened_parameters(self):
        return [xi for x in self._parameters_ for xi in x.flattened_parameters]
    @property
    def _parameter_sizes_(self):
        return [x.size for x in self._parameters_]
    @property
    def size_transformed(self):
        if self._has_fixes():
            return sum(self._fixes_)
        return self.size
    @property
    def parameter_shapes(self):
        return [xi for x in self._parameters_ for xi in x.parameter_shapes]
    @property
    def _constraints_str(self):
        return [cs for p in self._parameters_ for cs in p._constraints_str]
    @property
    def _description_str(self):
        return [xi for x in self._parameters_ for xi in x._description_str]
    @property
    def _ties_str(self):
        return [','.join(x._ties_str) for x in self.flattened_parameters]
    def __str__(self, header=True):

        name = adjust_name_for_printing(self.name) + "."
        constrs = self._constraints_str; ts = self._ties_str
        desc = self._description_str; names = self.parameter_names()
        nl = max([len(str(x)) for x in names + [name]])
        sl = max([len(str(x)) for x in desc + ["Value"]])
        cl = max([len(str(x)) if x else 0 for x in constrs + ["Constraint"]])
        tl = max([len(str(x)) if x else 0 for x in ts + ["Tied to"]])
        format_spec = "  \033[1m{{name:<{0}s}}\033[0;0m  |  {{desc:^{1}s}}  |  {{const:^{2}s}}  |  {{t:^{3}s}}".format(nl, sl, cl, tl)
        to_print = []
        for n, d, c, t in itertools.izip(names, desc, constrs, ts):
            to_print.append(format_spec.format(name=n, desc=d, const=c, t=t))
        # to_print = [format_spec.format(p=p, const=c, t=t) if isinstance(p, Param) else p.__str__(header=False) for p, c, t in itertools.izip(self._parameters_, constrs, ts)]
        sep = '-' * (nl + sl + cl + tl + 8 * 2 + 3)
        if header:
            header = "  {{0:<{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  |  {{3:^{3}s}}".format(nl, sl, cl, tl).format(name, "Value", "Constraint", "Tied to")
            # header += '\n' + sep
            to_print.insert(0, header)
        return '\n'.format(sep).join(to_print)
    pass


