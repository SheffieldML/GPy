# Copyright (c) 2014, Max Zwiessele, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import six # For metaclass support in Python 2 and 3 simultaneously
import numpy; np = numpy
import itertools
from re import compile, _pattern_type
from .param import ParamConcatenation
from .parameter_core import HierarchyError, Parameterizable, adjust_name_for_printing

import logging
from .index_operations import ParameterIndexOperationsView
logger = logging.getLogger("parameters changed meta")

class ParametersChangedMeta(type):
    def __call__(self, *args, **kw):
        self._in_init_ = True
        #import ipdb;ipdb.set_trace()
        self = super(ParametersChangedMeta, self).__call__(*args, **kw)
        logger.debug("finished init")
        self._in_init_ = False
        logger.debug("connecting parameters")
        self._highest_parent_._connect_parameters()
        #self._highest_parent_._notify_parent_change()
        self._highest_parent_._connect_fixes()
        logger.debug("calling parameters changed")
        self.parameters_changed()
        return self

@six.add_metaclass(ParametersChangedMeta)
class Parameterized(Parameterizable):
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
    #This is ignored in Python 3 -- you need to put the meta class in the function definition.
    #__metaclass__ = ParametersChangedMeta
    #The six module is used to support both Python 2 and 3 simultaneously
    #===========================================================================
    def __init__(self, name=None, parameters=[], *a, **kw):
        super(Parameterized, self).__init__(name=name, *a, **kw)
        self.size = sum(p.size for p in self.parameters)
        self.add_observer(self, self._parameters_changed_notification, -100)
        if not self._has_fixes():
            self._fixes_ = None
        self._param_slices_ = []
        #self._connect_parameters()
        self.link_parameters(*parameters)

    def build_pydot(self, G=None):
        import pydot  # @UnresolvedImport
        iamroot = False
        if G is None:
            G = pydot.Dot(graph_type='digraph', bgcolor=None)
            iamroot=True
        node = pydot.Node(id(self), shape='box', label=self.name)#, color='white')
        G.add_node(node)
        for child in self.parameters:
            child_node = child.build_pydot(G)
            G.add_edge(pydot.Edge(node, child_node))#, color='white'))

        for _, o, _ in self.observers:
            label = o.name if hasattr(o, 'name') else str(o)
            observed_node = pydot.Node(id(o), label=label)
            G.add_node(observed_node)
            edge = pydot.Edge(str(id(self)), str(id(o)), color='darkorange2', arrowhead='vee')
            G.add_edge(edge)

        if iamroot:
            return G
        return node

    #===========================================================================
    # Add remove parameters:
    #===========================================================================
    def link_parameter(self, param, index=None, _ignore_added_names=False):
        """
        :param parameters:  the parameters to add
        :type parameters:   list of or one :py:class:`GPy.core.param.Param`
        :param [index]:     index of where to put parameters

        :param bool _ignore_added_names: whether the name of the parameter overrides a possibly existing field

        Add all parameters to this param class, you can insert parameters
        at any given index using the :func:`list.insert` syntax
        """
        if param in self.parameters and index is not None:
            self.unlink_parameter(param)
            self.link_parameter(param, index)
        # elif param.has_parent():
        #    raise HierarchyError, "parameter {} already in another model ({}), create new object (or copy) for adding".format(param._short(), param._highest_parent_._short())
        elif param not in self.parameters:
            if param.has_parent():
                def visit(parent, self):
                    if parent is self:
                        raise HierarchyError("You cannot add a parameter twice into the hierarchy")
                param.traverse_parents(visit, self)
                param._parent_.unlink_parameter(param)
            # make sure the size is set
            if index is None:
                start = sum(p.size for p in self.parameters)
                self.constraints.shift_right(start, param.size)
                self.priors.shift_right(start, param.size)
                self.constraints.update(param.constraints, self.size)
                self.priors.update(param.priors, self.size)
                param._parent_ = self
                param._parent_index_ = len(self.parameters)
                self.parameters.append(param)
            else:
                start = sum(p.size for p in self.parameters[:index])
                self.constraints.shift_right(start, param.size)
                self.priors.shift_right(start, param.size)
                self.constraints.update(param.constraints, start)
                self.priors.update(param.priors, start)
                param._parent_ = self
                param._parent_index_ = index if index>=0 else len(self.parameters[:index])
                for p in self.parameters[index:]:
                    p._parent_index_ += 1
                self.parameters.insert(index, param)

            param.add_observer(self, self._pass_through_notify_observers, -np.inf)

            parent = self
            while parent is not None:
                parent.size += param.size
                parent = parent._parent_
            self._notify_parent_change()

            if not self._in_init_:
                #self._connect_parameters()
                #self._notify_parent_change()

                self._highest_parent_._connect_parameters(ignore_added_names=_ignore_added_names)
                self._highest_parent_._notify_parent_change()
                self._highest_parent_._connect_fixes()

        else:
            raise HierarchyError("""Parameter exists already, try making a copy""")


    def link_parameters(self, *parameters):
        """
        convenience method for adding several
        parameters without gradient specification
        """
        [self.link_parameter(p) for p in parameters]

    def unlink_parameter(self, param):
        """
        :param param: param object to remove from being a parameter of this parameterized object.
        """
        if not param in self.parameters:
            try:
                raise RuntimeError("{} does not belong to this object {}, remove parameters directly from their respective parents".format(param._short(), self.name))
            except AttributeError:
                raise RuntimeError("{} does not seem to be a parameter, remove parameters directly from their respective parents".format(str(param)))

        start = sum([p.size for p in self.parameters[:param._parent_index_]])
        self.size -= param.size
        del self.parameters[param._parent_index_]
        self._remove_parameter_name(param)


        param._disconnect_parent()
        param.remove_observer(self, self._pass_through_notify_observers)
        self.constraints.shift_left(start, param.size)

        self._connect_parameters()
        self._notify_parent_change()

        parent = self._parent_
        while parent is not None:
            parent.size -= param.size
            parent = parent._parent_

        self._highest_parent_._connect_parameters()
        self._highest_parent_._connect_fixes()
        self._highest_parent_._notify_parent_change()

    def add_parameter(self, *args, **kwargs):
        raise DeprecationWarning("add_parameter was renamed to link_parameter to avoid confusion of setting variables, use link_parameter instead")
    def remove_parameter(self, *args, **kwargs):
        raise DeprecationWarning("remove_parameter was renamed to unlink_parameter to avoid confusion of setting variables, use unlink_parameter instead")

    def _connect_parameters(self, ignore_added_names=False):
        # connect parameterlist to this parameterized object
        # This just sets up the right connection for the params objects
        # to be used as parameters
        # it also sets the constraints for each parameter to the constraints
        # of their respective parents
        if not hasattr(self, "parameters") or len(self.parameters) < 1:
            # no parameters for this class
            return
        if self.param_array.size != self.size:
            self._param_array_ = np.empty(self.size, dtype=np.float64)
        if self.gradient.size != self.size:
            self._gradient_array_ = np.empty(self.size, dtype=np.float64)

        old_size = 0
        self._param_slices_ = []
        for i, p in enumerate(self.parameters):
            if not p.param_array.flags['C_CONTIGUOUS']:
                raise ValueError("This should not happen! Please write an email to the developers with the code, which reproduces this error. All parameter arrays must be C_CONTIGUOUS")

            p._parent_ = self
            p._parent_index_ = i

            pslice = slice(old_size, old_size + p.size)

            # first connect all children
            p._propagate_param_grad(self.param_array[pslice], self.gradient_full[pslice])

            # then connect children to self
            self.param_array[pslice] = p.param_array.flat  # , requirements=['C', 'W']).ravel(order='C')
            self.gradient_full[pslice] = p.gradient_full.flat  # , requirements=['C', 'W']).ravel(order='C')

            p.param_array.data = self.param_array[pslice].data
            p.gradient_full.data = self.gradient_full[pslice].data

            self._param_slices_.append(pslice)

            self._add_parameter_name(p, ignore_added_names=ignore_added_names)
            old_size += p.size

    #===========================================================================
    # Get/set parameters:
    #===========================================================================
    def grep_param_names(self, regexp):
        """
        create a list of parameters, matching regular expression regexp
        """
        if not isinstance(regexp, _pattern_type): regexp = compile(regexp)
        found_params = []
        for n, p in zip(self.parameter_names(False, False, True), self.flattened_parameters):
            if regexp.match(n) is not None:
                found_params.append(p)
        return found_params

    def __getitem__(self, name, paramlist=None):
        if isinstance(name, (int, slice, tuple, np.ndarray)):
            return self.param_array[name]
        else:
            if paramlist is None:
                paramlist = self.grep_param_names(name)
            if len(paramlist) < 1: raise AttributeError(name)
            if len(paramlist) == 1:
                if isinstance(paramlist[-1], Parameterized):
                    paramlist = paramlist[-1].flattened_parameters
                    if len(paramlist) != 1:
                        return ParamConcatenation(paramlist)
                return paramlist[-1]
            return ParamConcatenation(paramlist)

    def __setitem__(self, name, value, paramlist=None):
        if value is None:
            return # nothing to do here
        if isinstance(name, (slice, tuple, np.ndarray)):
            try:
                self.param_array[name] = value
            except:
                raise ValueError("Setting by slice or index only allowed with array-like")
            self.trigger_update()
        else:
            try: param = self.__getitem__(name, paramlist)
            except: raise
            param[:] = value

    def __setattr__(self, name, val):
        # override the default behaviour, if setting a param, so broadcasting can by used
        if hasattr(self, "parameters"):
            try:
                pnames = self.parameter_names(False, adjust_for_printing=True, recursive=False)
                if name in pnames:
                    param = self.parameters[pnames.index(name)]
                    param[:] = val; return
            except AttributeError:
                pass
        return object.__setattr__(self, name, val);

    #===========================================================================
    # Pickling
    #===========================================================================
    def __setstate__(self, state):
        super(Parameterized, self).__setstate__(state)
        try:
            self._connect_parameters()
            self._connect_fixes()
            self._notify_parent_change()
            self.parameters_changed()
        except Exception as e:
            print("WARNING: caught exception {!s}, trying to continue".format(e))

    def copy(self, memo=None):
        if memo is None:
            memo = {}
        memo[id(self.optimizer_array)] = None # and param_array
        memo[id(self.param_array)] = None # and param_array
        copy = super(Parameterized, self).copy(memo)
        copy._connect_parameters()
        copy._connect_fixes()
        copy._notify_parent_change()
        return copy

    #===========================================================================
    # Printing:
    #===========================================================================
    def _short(self):
        return self.hierarchy_name()
    @property
    def flattened_parameters(self):
        return [xi for x in self.parameters for xi in x.flattened_parameters]
    @property
    def _parameter_sizes_(self):
        return [x.size for x in self.parameters]
    @property
    def parameter_shapes(self):
        return [xi for x in self.parameters for xi in x.parameter_shapes]
    @property
    def _constraints_str(self):
        return [cs for p in self.parameters for cs in p._constraints_str]
    @property
    def _priors_str(self):
        return [cs for p in self.parameters for cs in p._priors_str]
    @property
    def _description_str(self):
        return [xi for x in self.parameters for xi in x._description_str]
    @property
    def _ties_str(self):
        return [','.join(x._ties_str) for x in self.flattened_parameters]

    def _repr_html_(self, header=True):
        """Representation of the parameters in html for notebook display."""
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
        format_spec = "<tr><td class=tg-left>{{name:<{0}s}}</td><td class=tg-right>{{desc:>{1}s}}</td><td class=tg-left>{{const:^{2}s}}</td><td class=tg-left>{{pri:^{3}s}}</td><td class=tg-left>{{t:^{4}s}}</td></tr>".format(nl, sl, cl, pl, tl)
        to_print = []
        for n, d, c, t, p in zip(names, desc, constrs, ts, prirs):
            to_print.append(format_spec.format(name=n, desc=d, const=c, t=t, pri=p))
        sep = '-' * (nl + sl + cl + + pl + tl + 8 * 2 + 3)
        if header:
            header = """
<tr>
  <th><b>{name}</b></th>
  <th><b>Value</b></th>
  <th><b>Constraint</b></th>
  <th><b>Prior</b></th>
  <th><b>Tied to</b></th>
</tr>""".format(name=name)
            to_print.insert(0, header)
        style = """<style type="text/css">
.tg  {font-family:"Courier New", Courier, monospace !important;padding:2px 3px;word-break:normal;border-collapse:collapse;border-spacing:0;border-color:#DCDCDC;margin:0px auto;width:100%;}
.tg td{font-family:"Courier New", Courier, monospace !important;font-weight:bold;color:#444;background-color:#F7FDFA;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
.tg th{font-family:"Courier New", Courier, monospace !important;font-weight:normal;color:#fff;background-color:#26ADE4;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#DCDCDC;}
.tg .tg-left{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:left;}
.tg .tg-right{font-family:"Courier New", Courier, monospace !important;font-weight:normal;text-align:right;}
</style>"""
        return style + '\n' + '<table class="tg">' + '\n'.format(sep).join(to_print) + '\n</table>'

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
        for n, d, c, t, p in zip(names, desc, constrs, ts, prirs):
            to_print.append(format_spec.format(name=n, desc=d, const=c, t=t, pri=p))
        sep = '-' * (nl + sl + cl + + pl + tl + 8 * 2 + 3)
        if header:
            header = "  {{0:<{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  |  {{3:^{3}s}}  |  {{4:^{4}s}}".format(nl, sl, cl, pl, tl).format(name, "Value", "Constraint", "Prior", "Tied to")
            to_print.insert(0, header)
        return '\n'.format(sep).join(to_print)
    pass


