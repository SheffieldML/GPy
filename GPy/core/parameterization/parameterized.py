# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy; np = numpy
import cPickle
import itertools
from re import compile, _pattern_type
from param import ParamConcatenation
from parameter_core import Pickleable, Parameterizable, adjust_name_for_printing
from transformations import __fixed__
from lists_and_dicts import ArrayList

class ParametersChangedMeta(type):
    def __call__(self, *args, **kw):
        instance = super(ParametersChangedMeta, self).__call__(*args, **kw)
        instance.parameters_changed()
        return instance

class Parameterized(Parameterizable, Pickleable):
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
    #===========================================================================
    # Metaclass for parameters changed after init. 
    # This makes sure, that parameters changed will always be called after __init__
    # **Never** call parameters_changed() yourself 
    __metaclass__ = ParametersChangedMeta
    #===========================================================================
    def __init__(self, name=None, parameters=[], *a, **kw):
        super(Parameterized, self).__init__(name=name, *a, **kw)
        self._in_init_ = True
        self._parameters_ = ArrayList()
        self.size = sum(p.size for p in self._parameters_)
        self.add_observer(self, self._parameters_changed_notification, -100)
        if not self._has_fixes():
            self._fixes_ = None
        self._param_slices_ = []
        self._connect_parameters()
        del self._in_init_
        self.add_parameters(*parameters)

    def build_pydot(self, G=None):
        import pydot  # @UnresolvedImport
        iamroot = False
        if G is None:
            G = pydot.Dot(graph_type='digraph')
            iamroot=True
        node = pydot.Node(id(self), shape='record', label=self.name)
        G.add_node(node)
        for child in self._parameters_:
            child_node = child.build_pydot(G)
            G.add_edge(pydot.Edge(node, child_node))

        for o in self._observer_callables_.keys():
            label = o.name if hasattr(o, 'name') else str(o)
            observed_node = pydot.Node(id(o), label=label)
            G.add_node(observed_node)
            edge = pydot.Edge(str(id(self)), str(id(o)), color='darkorange2', arrowhead='vee')
            G.add_edge(edge)

        if iamroot:
            return G
        return node

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
                self.priors,
                self.constraints,
                self._parameters_,
                self._name,
                self._added_names_,
                ]

    def _setstate(self, state):
        self._added_names_ = state.pop()
        self._name = state.pop()
        self._parameters_ = state.pop()
        self.constraints = state.pop()
        self.priors = state.pop()
        self._fixes_ = state.pop()
        self._connect_parameters()
        self.parameters_changed()
    #===========================================================================
    # Override copy to handle programmatically added observers
    #===========================================================================
    def copy(self):
        c = super(Pickleable, self).copy()
        c.add_observer(c, c._parameters_changed_notification, -100)
        return c

    #===========================================================================
    # Gradient control
    #===========================================================================
    def _transform_gradients(self, g):
        if self.has_parent():
            return g
        [numpy.put(g, i, g[i] * c.gradfactor(self._param_array_[i])) for c, i in self.constraints.iteritems() if c != __fixed__]
        if self._has_fixes(): return g[self._fixes_]
        return g


    #===========================================================================
    # Indexable
    #===========================================================================
    def _offset_for(self, param):
        # get the offset in the parameterized index array for param
        if param.has_parent():
            if param._parent_._get_original(param) in self._parameters_:
                return self._param_slices_[param._parent_._get_original(param)._parent_index_].start
            return self._offset_for(param._parent_) + param._parent_._offset_for(param)
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
    # Convenience for fixed, tied checking of param:
    #===========================================================================
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
    
    #===========================================================================
    # Get/set parameters:
    #===========================================================================
    def grep_param_names(self, regexp):
        """
        create a list of parameters, matching regular expression regexp
        """
        if not isinstance(regexp, _pattern_type): regexp = compile(regexp)
        found_params = []
        for n, p in itertools.izip(self.parameter_names(False, False, True), self.flattened_parameters):
            if regexp.match(n) is not None:
                found_params.append(p)
        return found_params

    def __getitem__(self, name, paramlist=None):
        if isinstance(name, (int, slice, tuple, np.ndarray)):
            return self._param_array_[name]
        else:
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
        if isinstance(name, (slice, tuple, np.ndarray)):
            try:
                self._param_array_[name] = value
            except:
                raise ValueError, "Setting by slice or index only allowed with array-like"
            self._trigger_params_changed()
        else:
            try: param = self.__getitem__(name, paramlist)
            except: raise
            param[:] = value

    def __setattr__(self, name, val):
        # override the default behaviour, if setting a param, so broadcasting can by used        
        if hasattr(self, '_parameters_'):
            pnames = self.parameter_names(False, adjust_for_printing=True, recursive=False)
            if name in pnames: self._parameters_[pnames.index(name)][:] = val; return
        object.__setattr__(self, name, val);
    #===========================================================================
    # Printing:
    #===========================================================================
    def _short(self):
        return self.hierarchy_name()
    @property
    def flattened_parameters(self):
        return [xi for x in self._parameters_ for xi in x.flattened_parameters]
    @property
    def _parameter_sizes_(self):
        return [x.size for x in self._parameters_]
    @property
    def parameter_shapes(self):
        return [xi for x in self._parameters_ for xi in x.parameter_shapes]
    @property
    def _constraints_str(self):
        return [cs for p in self._parameters_ for cs in p._constraints_str]
    @property
    def _priors_str(self):
        return [cs for p in self._parameters_ for cs in p._priors_str]
    @property
    def _description_str(self):
        return [xi for x in self._parameters_ for xi in x._description_str]
    @property
    def _ties_str(self):
        return [','.join(x._ties_str) for x in self.flattened_parameters]
    def __str__(self, header=True):

        name = adjust_name_for_printing(self.name) + "."
        constrs = self._constraints_str; 
        ts = self._ties_str
        prirs = self._priors_str
        desc = self._description_str; names = self.parameter_names()
        nl = max([len(str(x)) for x in names + [name]])
        sl = max([len(str(x)) for x in desc + ["Value"]])
        cl = max([len(str(x)) if x else 0 for x in constrs + ["Constraint"]])
        tl = max([len(str(x)) if x else 0 for x in ts + ["Tied to"]])
        pl = max([len(str(x)) if x else 0 for x in prirs + ["Prior"]])
        format_spec = "  \033[1m{{name:<{0}s}}\033[0;0m  |  {{desc:>{1}s}}  |  {{const:^{2}s}}  |  {{pri:^{3}s}}  |  {{t:^{4}s}}".format(nl, sl, cl, pl, tl)
        to_print = []
        for n, d, c, t, p in itertools.izip(names, desc, constrs, ts, prirs):
            to_print.append(format_spec.format(name=n, desc=d, const=c, t=t, pri=p))
        # to_print = [format_spec.format(p=p, const=c, t=t) if isinstance(p, Param) else p.__str__(header=False) for p, c, t in itertools.izip(self._parameters_, constrs, ts)]
        sep = '-' * (nl + sl + cl + + pl + tl + 8 * 2 + 3)
        if header:
            header = "  {{0:<{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  |  {{3:^{3}s}}  |  {{4:^{4}s}}".format(nl, sl, cl, pl, tl).format(name, "Value", "Constraint", "Prior", "Tied to")
            # header += '\n' + sep
            to_print.insert(0, header)
        return '\n'.format(sep).join(to_print)
    pass


