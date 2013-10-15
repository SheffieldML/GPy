# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy
import copy
import cPickle
import transformations
from parameter import ParamConcatenation
from index_operations import ConstraintIndexOperations,\
    TieIndexOperations, create_raveled_indices, index_empty
import itertools
from re import compile, _pattern_type

#===============================================================================
# Printing:
__fixed__ = "fixed"
#===============================================================================


class Parameterized(object):
    """
    Parameterized class
    
    Say m is a handle to a parameterized class.

    Printing parameters:
    
        - print m:           prints a nice summary over all parameters
        - print m.name:      prints details for all the parameters 
                             which start with name
        - print m['.*name']: prints details for all the parameters 
                             which contain "name"
        - print m['']:       prints details for all parameters
    
        Fields:

            Name:       The name of the parameter, can be renamed!
            Value:      Shape or value, if one-valued
            Constrain:  constraint of the parameter, curly "{c}" brackets indicate 
                        some parameters are constrained by c. See detailed print
                        to get exact constraints.
            Tied_to:    which paramter it is tied to.
    
    Getting and setting parameters:
        
        Two ways to get parameters:
            
            - m.name regular expression matches all parameters beginning with name
            - m['name'] regular expression matches all parameters with name
    
    Handling of constraining, fixing and tieing parameters:
        
        You can constrain parameters by calling the constrain on the parameter itself, e.g:
        
            - m.name[:,1].constrain_positive()
            - m.name[0].tie_to(m.name[1])
            
        Fixing parameters will fix them to the value they are right now. If you change
        the paramters value, the parameter will be fixed to the new value!
        
        If you want to operate on all parameters use m[''] to wildcard select all paramters 
        and concatenate them. Printing m[''] will result in printing of all parameters in detail.
    """
    def __init__(self, parameterlist, prefix=None, *args, **kwargs):
        self._init = True
        self._params = []
        for p in parameterlist: # flatten paramter list
            if isinstance(p, Parameterized):
                self._params.extend(p._params)
            else:
                self._params.append(p)
        self._constraints = ConstraintIndexOperations()
        self._ties = TieIndexOperations(self)
        self._ties_fixes = None
        self._connect_parameters()
        self._init = False
    def _connect_parameters(self):
        # connect parameterlist to this parameterized object
        # This just sets up the right connection for the params objects 
        # to be used as parameters
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
        # don't overright this anymore!
        return numpy.hstack([x._get_params() for x in self._params])#numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params(), self._params)), dtype=numpy.float64, count=sum(self.parameter_sizes))    
    def _set_params(self, params):
        # don't overright this anymore!
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
        # translate an index in parameterized indexing into the index of param
        ind = ind-self._offset(param)
        ind = ind[ind >= 0]
        internal_offset = (numpy.arange(param._realsize).reshape(param._realshape)[param._current_slice]).flat[0]
        ind = ind[ind < param.size + internal_offset]
        return ind
    def _offset(self, param):
        # get the offset in the parameterized index array for param
        return self._param_slices[param._parent_index].start 
    #===========================================================================
    # Handle ties:
    #===========================================================================
    def _add_tie(self, param, tied_to):
        # tie param to tie_to, if the values match (with broadcasting)
        try:
            param[...] = tied_to
        except ValueError:
            raise ValueError("Trying to tie {} with shape {} to {} with shape {}".format(self.name, self.shape, param.name, param.shape))            
        self._ties.add(param, tied_to)
        if self._ties_fixes is None: self._ties_fixes = numpy.ones(self.parameter_size, dtype=bool)
        f = create_raveled_indices(param._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[f] = False
    def _remove_tie(self, param, *params):
        # remove the tie from param to all *params (can be None, so all ties get deleted for param)
        if len(params) == 0:
            params = self._ties.properties()
        for p in self._ties.properties():
            for a in params: 
                if numpy.all(a==p):
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
    def _fix(self, param, warning=True):
        self._add_constrain(param, __fixed__, warning)
        if self._ties_fixes is None: self._ties_fixes = numpy.ones(self.parameter_size, dtype=bool)
        f = create_raveled_indices(param._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[f] = False
    def _unfix(self, param):
        self._remove_constrain(param, __fixed__)
        ind = create_raveled_indices(param._current_slice, param._realshape, self._offset(param))
        self._ties_fixes[ind] = True
        if numpy.all(self._ties_fixes): self._ties_fixes = None
    #===========================================================================
    # Constraint Handling:
    #===========================================================================
    def _add_constrain(self, param, transform, warning=True):
        reconstrained = self._remove_constrain(param, None) # remove constraints before
        self._constraints.add(transform, param._current_slice, param._realshape, self._offset(param))
        if warning and any(reconstrained):
            # if you want to print the whole params object, which was reconstrained use:
            # m = str(param[self._backtranslate_index(param, reconstrained)])
            m = param.name + str("".join(map(str,param._indices()[self._backtranslate_index(param, reconstrained)])))
            print "Warning: re-constraining parameters:\n{}".format(m)
    def _remove_constrain(self, param, *transforms):
        if transforms is ():
            transforms = self._constraints.properties()
        removed_indices = numpy.array([]).astype(int)
        for constr in transforms:
            removed = self._constraints.remove(constr, param._current_slice, param._realshape, self._offset(param))
            removed_indices = numpy.union1d(removed_indices, removed)
        return removed_indices
    # convienience for iterating over items
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
        """
        create a list of parameters, matching regular expression regexp
        """
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
        # override the default behaviour, if setting a parameter, so broadcasting can by used
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


# class Parameterized_old(object):
#     def __init__(self):
#         """
#         This is the base class for model and kernel. Mostly just handles tieing and constraining of parameters
#         """
#         self.tied_indices = []
#         self.fixed_indices = []
#         self.fixed_values = []
#         self.constrained_indices = []
#         self.constraints = []
# 
#     def _get_params(self):
#         raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
#     def _set_params(self, x):
#         raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
# 
#     def _get_param_names(self):
#         raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
#     #def _get_print_names(self):
#     #    """ Override for which names to print out, when using print m """
#     #    return self._get_param_names()
# 
#     def pickle(self, filename, protocol=None):
#         if protocol is None:
#             if self._has_get_set_state():
#                 protocol = 0
#             else:
#                 protocol = -1
#         with open(filename, 'w') as f:
#             cPickle.dump(self, f, protocol)
# 
#     def copy(self):
#         """Returns a (deep) copy of the current model """
#         return copy.deepcopy(self)
# 
#     def __getstate__(self):
#         if self._has_get_set_state():
#             return self.getstate()
#         return self.__dict__
# 
#     def __setstate__(self, state):
#         if self._has_get_set_state():
#             self.setstate(state) # set state
#             self._set_params(self._get_params()) # restore all values
#             return
#         self.__dict__ = state
# 
#     def _has_get_set_state(self):
#         return 'getstate' in vars(self.__class__) and 'setstate' in vars(self.__class__)
# 
#     def getstate(self):
#         """
#         Get the current state of the class,
#         here just all the indices, rest can get recomputed
#         For inheriting from Parameterized:
# 
#         Allways append the state of the inherited object
#         and call down to the inherited object in setstate!!
#         """
#         return [self.tied_indices,
#                 self.fixed_indices,
#                 self.fixed_values,
#                 self.constrained_indices,
#                 self.constraints]
# 
#     def setstate(self, state):
#         self.constraints = state.pop()
#         self.constrained_indices = state.pop()
#         self.fixed_values = state.pop()
#         self.fixed_indices = state.pop()
#         self.tied_indices = state.pop()
# 
#     def __getitem__(self, regexp, return_names=False):
#         """
#         Get a model parameter by name.  The name is applied as a regular
#         expression and all parameters that match that regular expression are
#         returned.
#         """
#         matches = self.grep_param_names(regexp)
#         if len(matches):
#             if return_names:
#                 return self._get_params()[matches], np.asarray(self._get_param_names())[matches].tolist()
#             else:
#                 return self._get_params()[matches]
#         else:
#             raise AttributeError, "no parameter matches %s" % regexp
# 
#     def __setitem__(self, name, val):
#         """
#         Set model parameter(s) by name. The name is provided as a regular
#         expression. All parameters matching that regular expression are set to
#         the given value.
#         """
#         matches = self.grep_param_names(name)
#         if len(matches):
#             val = np.array(val)
#             assert (val.size == 1) or val.size == len(matches), "Shape mismatch: {}:({},)".format(val.size, len(matches))
#             x = self._get_params()
#             x[matches] = val
#             self._set_params(x)
#         else:
#             raise AttributeError, "no parameter matches %s" % name
# 
#     def tie_params(self, regexp):
#         """
#         Tie (all!) parameters matching the regular expression `regexp`. 
#         """
#         matches = self.grep_param_names(regexp)
#         assert matches.size > 0, "need at least something to tie together"
#         if len(self.tied_indices):
#             assert not np.any(matches[:, None] == np.hstack(self.tied_indices)), "Some indices are already tied!"
#         self.tied_indices.append(matches)
#         # TODO only one of the priors will be evaluated. Give a warning message if the priors are not identical
#         if hasattr(self, 'prior'):
#             pass
# 
#         self._set_params_transformed(self._get_params_transformed()) # sets tied parameters to single value
# 
#     def untie_everything(self):
#         """Unties all parameters by setting tied_indices to an empty list."""
#         self.tied_indices = []
# 
#     def grep_param_names(self, regexp, transformed=False, search=False):
#         """
#         :param regexp: regular expression to select parameter names
#         :type regexp: re | str | int
#         :rtype: the indices of self._get_param_names which match the regular expression.
# 
#         Note:-
#           Other objects are passed through - i.e. integers which weren't meant for grepping
#         """
# 
#         if transformed:
#             names = self._get_param_names_transformed()
#         else:
#             names = self._get_param_names()
# 
#         if type(regexp) in [str, np.string_, np.str]:
#             regexp = re.compile(regexp)
#         elif type(regexp) is re._pattern_type:
#             pass
#         else:
#             return regexp
#         if search:
#             return np.nonzero([regexp.search(name) for name in names])[0]
#         else:
#             return np.nonzero([regexp.match(name) for name in names])[0]
# 
#     def num_params_transformed(self):
#         removed = 0
#         for tie in self.tied_indices:
#             removed += tie.size - 1
# 
#         for fix in self.fixed_indices:
#             removed += fix.size
# 
#         return len(self._get_params()) - removed
# 
#     def unconstrain(self, regexp):
#         """Unconstrain matching parameters.  Does not untie parameters"""
#         matches = self.grep_param_names(regexp)
# 
#         # tranformed contraints:
#         for match in matches:
#             self.constrained_indices = [i[i <> match] for i in self.constrained_indices]
# 
#         # remove empty constraints
#         tmp = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
#         if tmp:
#             self.constrained_indices, self.constraints = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
#             self.constrained_indices, self.constraints = list(self.constrained_indices), list(self.constraints)
# 
#         # fixed:
#         self.fixed_values = [np.delete(values, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices, values in zip(self.fixed_indices, self.fixed_values)]
#         self.fixed_indices = [np.delete(indices, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices in self.fixed_indices]
# 
#         # remove empty elements
#         tmp = [(i, v) for i, v in zip(self.fixed_indices, self.fixed_values) if len(i)]
#         if tmp:
#             self.fixed_indices, self.fixed_values = zip(*tmp)
#             self.fixed_indices, self.fixed_values = list(self.fixed_indices), list(self.fixed_values)
#         else:
#             self.fixed_indices, self.fixed_values = [], []
# 
#     def constrain_negative(self, regexp, warning=True):
#         """ Set negative constraints. """
#         self.constrain(regexp, transformations.NegativeLogexp(), warning)
# 
#     def constrain_positive(self, regexp, warning=True):
#         """ Set positive constraints. """
#         self.constrain(regexp, transformations.Logexp(), warning)
# 
#     def constrain_bounded(self, regexp, lower, upper, warning=True):
#         """ Set bounded constraints. """
#         self.constrain(regexp, transformations.Logistic(lower, upper), warning)
# 
#     def all_constrained_indices(self):
#         if len(self.constrained_indices) or len(self.fixed_indices):
#             return np.hstack(self.constrained_indices + self.fixed_indices)
#         else:
#             return np.empty(shape=(0,))
# 
#     def constrain(self, regexp, transform, warning=True):
#         assert isinstance(transform, transformations.Transformation)
# 
#         matches = self.grep_param_names(regexp)
#         overlap = set(matches).intersection(set(self.all_constrained_indices()))
#         if overlap:
#             self.unconstrain(np.asarray(list(overlap)))
#             if warning:
#                 print 'Warning: re-constraining these parameters'
#                 pn = self._get_param_names()
#                 for i in overlap:
#                     print pn[i]
# 
#         self.constrained_indices.append(matches)
#         self.constraints.append(transform)
#         x = self._get_params()
#         x[matches] = transform.initialize(x[matches])
#         self._set_params(x)
# 
#     def constrain_fixed(self, regexp, value=None, warning=True):
#         """
# 
#         :param regexp: which parameters need to be fixed.
#         :type regexp: ndarray(dtype=int) or regular expression object or string
#         :param value: the vlaue to fix the parameters to. If the value is not specified,
#                  the parameter is fixed to the current value
#         :type value: float
# 
#         **Notes**
# 
#         Fixing a parameter which is tied to another, or constrained in some way will result in an error.
# 
#         To fix multiple parameters to the same value, simply pass a regular expression which matches both parameter names, or pass both of the indexes.
# 
#         """
#         matches = self.grep_param_names(regexp)
#         overlap = set(matches).intersection(set(self.all_constrained_indices()))
#         if overlap:
#             self.unconstrain(np.asarray(list(overlap)))
#             if warning:
#                 print 'Warning: re-constraining these parameters'
#                 pn = self._get_param_names()
#                 for i in overlap:
#                     print pn[i]
# 
#         self.fixed_indices.append(matches)
#         if value != None:
#             self.fixed_values.append(value)
#         else:
#             self.fixed_values.append(self._get_params()[self.fixed_indices[-1]])
# 
#         # self.fixed_values.append(value)
#         self._set_params_transformed(self._get_params_transformed())
# 
#     def _get_params_transformed(self):
#         """use self._get_params to get the 'true' parameters of the model, which are then tied, constrained and fixed"""
#         x = self._get_params()
#         [np.put(x, i, t.finv(x[i])) for i, t in zip(self.constrained_indices, self.constraints)]
# 
#         to_remove = self.fixed_indices + [t[1:] for t in self.tied_indices]
#         if len(to_remove):
#             return np.delete(x, np.hstack(to_remove))
#         else:
#             return x
# 
#     def _set_params_transformed(self, x):
#         """ takes the vector x, which is then modified (by untying, reparameterising or inserting fixed values), and then call self._set_params"""
#         self._set_params(self._untransform_params(x))
# 
#     def _untransform_params(self, x):
#         """
#         The Transformation required for _set_params_transformed.
# 
#         This moves the vector x seen by the optimiser (unconstrained) to the
#         valid parameter vector seen by the model
# 
#         Note:
#           - This function is separate from _set_params_transformed for downstream flexibility
#         """
#         # work out how many places are fixed, and where they are. tricky logic!
#         fix_places = self.fixed_indices + [t[1:] for t in self.tied_indices]
#         if len(fix_places):
#             fix_places = np.hstack(fix_places)
#             Nfix_places = fix_places.size
#         else:
#             Nfix_places = 0
# 
#         free_places = np.setdiff1d(np.arange(Nfix_places + x.size, dtype=np.int), fix_places)
# 
#         # put the models values in the vector xx
#         xx = np.zeros(Nfix_places + free_places.size, dtype=np.float64)
# 
#         xx[free_places] = x
#         [np.put(xx, i, v) for i, v in zip(self.fixed_indices, self.fixed_values)]
#         [np.put(xx, i, v) for i, v in [(t[1:], xx[t[0]]) for t in self.tied_indices] ]
# 
#         [np.put(xx, i, t.f(xx[i])) for i, t in zip(self.constrained_indices, self.constraints)]
#         if hasattr(self, 'debug'):
#             stop # @UndefinedVariable
# 
#         return xx
# 
#     def _get_param_names_transformed(self):
#         """
#         Returns the parameter names as propagated after constraining,
#         tying or fixing, i.e. a list of the same length as _get_params_transformed()
#         """
#         n = self._get_param_names()
# 
#         # remove/concatenate the tied parameter names
#         if len(self.tied_indices):
#             for t in self.tied_indices:
#                 n[t[0]] = "<tie>".join([n[tt] for tt in t])
#             remove = np.hstack([t[1:] for t in self.tied_indices])
#         else:
#             remove = np.empty(shape=(0,), dtype=np.int)
# 
#         # also remove the fixed params
#         if len(self.fixed_indices):
#             remove = np.hstack((remove, np.hstack(self.fixed_indices)))
# 
#         # add markers to show that some variables are constrained
#         for i, t in zip(self.constrained_indices, self.constraints):
#             for ii in i:
#                 n[ii] = n[ii] + t.__str__()
# 
#         n = [nn for i, nn in enumerate(n) if not i in remove]
#         return n
# 
#     #@property
#     #def all(self):
#     #    return self.__str__(self._get_param_names())
# 
# 
#     #def __str__(self, names=None, nw=30):
#     def __str__(self, nw=30):
#         """
#         Return a string describing the parameter names and their ties and constraints
#         """
#         names = self._get_param_names()
#         #if names is None:
#         #    names = self._get_print_names()
#         #name_indices = self.grep_param_names("|".join(names))
#         N = len(names)
# 
#         if not N:
#             return "This object has no free parameters."
#         header = ['Name', 'Value', 'Constraints', 'Ties']
#         values = self._get_params() # map(str,self._get_params())
#         #values = self._get_params()[name_indices] # map(str,self._get_params())
#         # sort out the constraints
#         constraints = [''] * len(names)
#         #constraints = [''] * len(self._get_param_names())
#         for i, t in zip(self.constrained_indices, self.constraints):
#             for ii in i:
#                 constraints[ii] = t.__str__()
#         for i in self.fixed_indices:
#             for ii in i:
#                 constraints[ii] = 'Fixed'
#         # sort out the ties
#         ties = [''] * len(names)
#         for i, tie in enumerate(self.tied_indices):
#             for j in tie:
#                 ties[j] = '(' + str(i) + ')'
# 
#         if values.size == 1:
#             values = ['%.4f' %float(values)]
#         else:
#             values = ['%.4f' % float(v) for v in values]
#         max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
#         max_values = max([len(values[i]) for i in range(len(values))] + [len(header[1])])
#         max_constraint = max([len(constraints[i]) for i in range(len(constraints))] + [len(header[2])])
#         max_ties = max([len(ties[i]) for i in range(len(ties))] + [len(header[3])])
#         cols = np.array([max_names, max_values, max_constraint, max_ties]) + 4
#         # columns = cols.sum()
# 
#         header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
#         header_string = map(lambda x: '|'.join(x), [header_string])
#         separator = '-' * len(header_string[0])
#         param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]
# 
# 
#         return ('\n'.join([header_string[0], separator] + param_string)) + '\n'
# 
#     def grep_model(self,regexp):
#         regexp_indices = self.grep_param_names(regexp)
#         all_names = self._get_param_names()
# 
#         names = [all_names[pj] for pj in regexp_indices]
#         N = len(names)
# 
#         if not N:
#             return "Match not found."
# 
#         header = ['Name', 'Value', 'Constraints', 'Ties']
#         all_values = self._get_params()
#         values = np.array([all_values[pj] for pj in regexp_indices])
#         constraints = [''] * len(names)
# 
#         _constrained_indices,aux = self._pick_elements(regexp_indices,self.constrained_indices)
#         _constraints = [self.constraints[pj] for pj in aux]
# 
#         for i, t in zip(_constrained_indices, _constraints):
#             for ii in i:
#                 iii = regexp_indices.tolist().index(ii)
#                 constraints[iii] = t.__str__()
# 
#         _fixed_indices,aux = self._pick_elements(regexp_indices,self.fixed_indices)
#         for i in _fixed_indices:
#             for ii in i:
#                 iii = regexp_indices.tolist().index(ii)
#                 constraints[ii] = 'Fixed'
# 
#         _tied_indices,aux = self._pick_elements(regexp_indices,self.tied_indices)
#         ties = [''] * len(names)
#         for i,ti in zip(_tied_indices,aux):
#             for ii in i:
#                 iii = regexp_indices.tolist().index(ii)
#                 ties[iii] = '(' + str(ti) + ')'
# 
#         if values.size == 1:
#             values = ['%.4f' %float(values)]
#         else:
#             values = ['%.4f' % float(v) for v in values]
# 
#         max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
#         max_values = max([len(values[i]) for i in range(len(values))] + [len(header[1])])
#         max_constraint = max([len(constraints[i]) for i in range(len(constraints))] + [len(header[2])])
#         max_ties = max([len(ties[i]) for i in range(len(ties))] + [len(header[3])])
#         cols = np.array([max_names, max_values, max_constraint, max_ties]) + 4
# 
#         header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
#         header_string = map(lambda x: '|'.join(x), [header_string])
#         separator = '-' * len(header_string[0])
#         param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]
# 
#         print header_string[0]
#         print separator
#         for string in param_string:
#             print string
# 
#     def _pick_elements(self,regexp_ind,array_list):
#         """Removes from array_list the elements different from regexp_ind"""
#         new_array_list = [] #New list with elements matching regexp_ind
#         array_indices = [] #Indices that matches the arrays in new_array_list and array_list
# 
#         array_index = 0
#         for array in array_list:
#             _new = []
#             for ai in array:
#                 if ai in regexp_ind:
#                     _new.append(ai)
#             if len(_new):
#                 new_array_list.append(np.array(_new))
#                 array_indices.append(array_index)
#             array_index += 1
#         return new_array_list, array_indices
