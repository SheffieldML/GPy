# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy; np = numpy
import copy
import cPickle
from parameter import ParamConcatenation, Param
from index_operations import ParameterIndexOperations,\
    index_empty
import transformations
import itertools
from re import compile, _pattern_type
import re

#===============================================================================
# Printing:
__fixed__ = "fixed"
#===============================================================================

#===============================================================================
# constants
FIXED = False
UNFIXED = True
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
    def __init__(self):
        self._constraints_ = ParameterIndexOperations()
        #self._fixes_ = TieIndexOperations(self)
        #self._fixes_ = None
        self._fixes_ = None
        self._in_init_ = True
        if not hasattr(self, "_parameters_"):
            self._parameters_ = []
        self._connect_parameters()
        del self._in_init_
    #===========================================================================
    # Parameter connection for model creation:
    #===========================================================================
    def set_as_parameter(self, name, array, gradient, index=None, highest_parent=None):
        """
        :param name:     name of the parameter (in print and plots), can be callable without parameters
        :type name:      str, callable
        :param array:    array which the parameter consists of
        :type array:     array-like
        :param gradient: gradient method of the parameter
        :type gradient:  callable
        :param index:    (optional) index of the parameter when printing
        
        (:param highest_parent:  connect these parameters to this class, but tell
                        updates to highest_parent, this is needed when parameterized classes
                        contain parameterized classes, but want to access the parameters
                        of their children) 
                         

        Set array (e.g. self.X) as parameter with name and gradient.
        I.e: self.set_as_parameter('curvature', self.lengthscale, self.dK_dlengthscale)
        
        Note: the order in which parameters are added can be adjusted by 
              giving an index, of where to put this parameter in printing  
        """
        if index is None:
            self._parameters_.append(Param(name, array, gradient))
        else:
            self._parameters_.insert(index, Param(name, array, gradient))
        self._connect_parameters(highest_parent=highest_parent)
    def set_as_parameters(self, *parameters, **kwargs):
        """
        :param parameters: the parameters to add
        :param index: index of where to put parameters
        
        
        Add all parameters to this parameter class, you can insert parameters 
        at any given point using the :py:func:`list.insert` syntax 
        """
        if kwargs.get('index',None) is None:
            self._parameters_.extend(parameters)
        else:
            self._parameters_.insert(kwargs['index'], parameters)
        self._connect_parameters(highest_parent=kwargs.get('highest_parent', self))
#     def remove_parameter(self, *names_params_indices):
#         """
#         :param names_params_indices: mix of parameter_names, parameter objects, or indices 
#             to remove from being a parameter of this parameterized object. 
#             
#             note: if it is a string object it will be regexp-matched automatically
#         """
#         self._parameters_ = [p for p in self._parameters_ 
#                         if not (p._parent_index_ in names_params_indices 
#                                 or p.name in names_params_indices
#                                 or p in names_params_indices)]
#         self._connect_parameters()
    def parameters_changed(self):
        # will be called as soon as paramters have changed
        pass
    def _connect_parameters(self, highest_parent=None):
        # connect parameterlist to this parameterized object
        # This just sets up the right connection for the params objects 
        # to be used as parameters
        if not hasattr(self, "_parameters_") or len(self._parameters_) < 1:
            # no parameters for this class
            return
        sizes = numpy.cumsum([0] + self._parameter_sizes_)
        self._parameter_size_ = sizes[-1] 
        self._param_slices_ = [slice(start, stop) for start,stop in zip(sizes, sizes[1:])]
        for i, p in enumerate(self._parameters_):
            p._parent_ = self
            p._parent_index_ = i
            p._updates_parent_ = highest_parent or self
            not_unique = []
            if p.name in self.__dict__:
                if p.base is self.__dict__[p.name] or p is self.__dict__[p.name]:
                    self.__dict__[p.name] = p
                else:
                    not_unique.append(p.name)
                    del self.__dict__[p.name]
            elif not (p.name in not_unique):
                self.__dict__[p.name] = p
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
        return copy.deepcopy(self)
    def __getstate__(self):
        if self._has_get_set_state():
            return self.getstate()
        return self.__dict__
    def __setstate__(self, state):
        if self._has_get_set_state():
            self.setstate(state) # set state
            #self._set_params(self._get_params()) # restore all values
            return
        self.__dict__ = state
    def _has_get_set_state(self):
        return 'getstate' in vars(self.__class__) and 'setstate' in vars(self.__class__)
    def getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        For inheriting from Parameterized:
 
        Allways append the state of the inherited object
        and call down to the inherited object in setstate!!
        """
        return [
                self._fixes_,
                self._constraints_,
                self._priors_,
                self._parameters_,
                ]
        
    def setstate(self, state):
        self._parameters_ = state.pop()
        self._connect_parameters()
        self._priors = state.pop()
        self._constraints_ = state.pop()
        self._fixes_ = state.pop()
        self.parameters_changed()
    #===========================================================================
    # Optimization handles:
    #===========================================================================
    def _get_param_names_transformed(self):
        return numpy.array([p.name+str(i) for p in self._parameters_ for i in p._indices()])[self._fixes_][0]
    
    def _get_params(self):
        # don't overwrite this anymore!
        return numpy.hstack([x._get_params() for x in self._parameters_])#numpy.fromiter(itertools.chain(*itertools.imap(lambda x: x._get_params(), self._parameters_)), dtype=numpy.float64, count=sum(self._parameter_sizes_))    
    def _set_params(self, params):
        # don't overwrite this anymore!
        [p._set_params(params[s]) for p,s in itertools.izip(self._parameters_,self._param_slices_)]
        self.parameters_changed()
    def _get_params_transformed(self):
        p = self._get_params()
        [numpy.put(p, ind, c.finv(p[ind])) for c,ind in self._constraints_.iteritems() if c != __fixed__]
        if self._fixes_ is not None:
            return p[self._fixes_]
        return p
    def _set_params_transformed(self, p):
        if self._fixes_ is not None: tmp = self._get_params(); tmp[self._fixes_] = p; p = tmp; del tmp
        [numpy.put(p, ind, c.f(p[ind])) for c,ind in self._constraints_.iteritems() if c != __fixed__]
        self._set_params(p)
    def _name_changed(self, param, old_name):
        if hasattr(self, old_name):
            delattr(self, old_name)
        self.__dict__[param.name] = param
    #===========================================================================
    # Index Handling
    #===========================================================================
    def _backtranslate_index(self, param, ind):
        # translate an index in parameterized indexing into the index of param
        ind = ind-self._offset_for(param)
        ind = ind[ind >= 0]
        internal_offset = param._internal_offset()
        ind = ind[ind < param.size + internal_offset]
        return ind
    def _offset_for(self, param):
        # get the offset in the parameterized index array for param
        return self._param_slices_[param._parent_index_].start
    def _raveled_index_for(self, param):
        return param._raveled_index() + self._offset_for(param)
    #===========================================================================
    # Handle ties:
    #===========================================================================
    def _set_fixed(self, param_or_index):
        if self._fixes_ is None: self._fixes_ = numpy.ones(self._parameter_size_, dtype=bool)
        try:
            param_or_index = self._raveled_index_for(param_or_index)
        except AttributeError:
            pass
        self._fixes_[param_or_index] = FIXED
        if numpy.all(self._fixes_): self._fixes_ = None # ==UNFIXED
    def _set_unfixed(self, param_or_index):
        if self._fixes_ is None: self._fixes_ = numpy.ones(self._parameter_size_, dtype=bool)
        try:
            param_or_index = self._raveled_index_for(param_or_index)
        except AttributeError:
            pass
        self._fixes_[param_or_index] = UNFIXED
        if numpy.all(self._fixes_): self._fixes_ = None # ==UNFIXED
#     def _add_tie(self, param, tied_to):
#         # tie param to tie_to, if the values match (with broadcasting)
#         self._remove_tie(param) # delete if multiple ties should be allowed
#         f, _ = self._fixes_.add(param, tied_to)
#         if self._fixes_ is None: self._fixes_ = numpy.ones(self._parameter_size_, dtype=bool)
#         self._fixes_[f] = False
#     def _remove_tie(self, param, *params):
#         # remove the tie from param to all *params (can be None, so all ties get deleted for param)
#         if len(params) == 0:
#             params = self._fixes_.properties()
#         for p in params: 
#             _, t = self._fixes_.remove(param, p)
#             self._fixes_[t] = True
#         if numpy.all(self._fixes_): self._fixes_ = None # ==UNFIXED
#     def _ties_iter_items(self, param):
#         for tied_to, ind in self._fixes_.iter_from_items():
#             ind = self._backtranslate_index(param, ind)
#             if not index_empty(ind):
#                 yield tied_to, ind
#     def _ties_iter(self, param):
#         for constr, _ in self._ties_iter_items(param):
#             yield constr
#     def _ties_iter_indices(self, param):
#         for _, ind in self._ties_iter_items(param):
#             yield ind
#     def _ties_for(self, param, rav_index):
#         return self._fixes_.from_to_for(rav_index+self._offset_for(param))
    #===========================================================================
    # Fixing parameters:
    #===========================================================================
    def _fix(self, param, warning=True):
        f = self._add_constrain(param, __fixed__, warning)
        self._set_fixed(f)
    def _unfix(self, param):
        if self._fixes_ is not None:
            f = self._remove_constrain(param, __fixed__)
            self._set_unfixed(f)
    #===========================================================================
    # Convenience for fixed, tied checking of parameter:
    #===========================================================================
    def _is_fixed(self, param):
        # returns if the whole parameter is fixed
        if self._fixes_ is None:
            return False
        return not self._fixes_[self._offset_for(param): self._offset_for(param)+param._realsize_].any()
    def _get_original(self, param):
        # if advanced indexing is activated it happens that the array is a copy
        # you can retrieve the original parameter through this method, by passing
        # the copy here
        return self._parameters_[param._parent_index_]
    #===========================================================================
    # Constraint Handling:
    #===========================================================================
    def _add_constrain(self, param, transform, warning=True):
        rav_i = self._raveled_index_for(param)
        reconstrained = self._remove_constrain(param, index=rav_i) # remove constraints before
        # if removing constraints before adding new is not wanted, just delete the above line!        
        self._constraints_.add(transform, rav_i)
        if warning and any(reconstrained):
            # if you want to print the whole params object, which was reconstrained use:
            # m = str(param[self._backtranslate_index(param, reconstrained)])
            print "Warning: re-constraining parameters:\n{}".format(param._short())
        return rav_i
    def _remove_constrain(self, param, *transforms, **kwargs):
        if transforms is ():
            transforms = self._constraints_.properties()
        removed_indices = numpy.array([]).astype(int)
        if "index" in kwargs: index = kwargs['index'] 
        else: index = self._raveled_index_for(param)
        for constr in transforms:
            removed = self._constraints_.remove(constr, index)
            removed_indices = numpy.union1d(removed_indices, removed)
        return removed_indices
    # convienience for iterating over items
    def _constraints_iter_items(self, param):
        for constr, ind in self._constraints_.iteritems():
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
        return self._backtranslate_index(param, self._constraints_[constraint])
    def _constraints_for(self, param, rav_index):
        return self._constraints_.properties_for(rav_index+self._offset_for(param))
    #===========================================================================
    # Get/set parameters:
    #===========================================================================
    def grep_param_names(self, regexp):
        """
        create a list of parameters, matching regular expression regexp
        """
        if not isinstance(regexp, _pattern_type): regexp = compile(regexp)
        return [param for param in self._parameters_ if regexp.match(param.name) is not None]
    def __getitem__(self, name, paramlist=None):
        if paramlist is None:
            paramlist = self.grep_param_names(name)
        if len(paramlist) < 1: raise AttributeError, name
        if len(paramlist) == 1: return paramlist[-1]
        return ParamConcatenation(paramlist)
    def __setitem__(self, name, value, paramlist=None):
        try: param = self.__getitem__(name, paramlist)
        except AttributeError as a: raise a
        param[:] = value
    def __getattr__(self, name):
        return self.__getitem__(name)
#     def __getattribute__(self, name):
#         #try: 
#             return object.__getattribute__(self, name)
        #except AttributeError:
        #    _, a, tb = sys.exc_info()
        #    try:
        #        return self.__getitem__(name)
        #    except AttributeError:
        #        raise AttributeError, a.message, tb
    def __setattr__(self, name, val):
        # override the default behaviour, if setting a parameter, so broadcasting can by used
        if hasattr(self, "_parameters_"):
            paramlist = self.grep_param_names(name)
            if len(paramlist) == 1: self.__setitem__(name, val, paramlist); return
        object.__setattr__(self, name, val);
    #===========================================================================
    # Printing:        
    #===========================================================================
    @property
    def parameter_names(self):
        return [x.name for x in self._parameters_]
    @property
    def _parameter_sizes_(self):
        return [x.size for x in self._parameters_]
    @property
    def _parameter_size_transformed_(self):
        return sum(self._fixes_)
    @property
    def _parameter_shapes_(self):
        return [x.shape for x in self._parameters_]
    @property
    def _constrs(self):
        return [p._constr for p in self._parameters_]
    @property
    def _descs(self):
        return [x._desc for x in self._parameters_]
    @property
    def _ts(self):
        return [' '.join([t._short() for t in x._tied_to_]) for x in self._parameters_]
    def __str__(self, header=True):
        nl = max([len(str(x)) for x in self.parameter_names + ["Name"]])
        sl = max([len(str(x)) for x in self._descs + ["Value"]])
        constrs = self._constrs; ts = self._ts
        cl = max([len(str(x)) if x else 0 for x in constrs  + ["Constraint"]])
        tl = max([len(str(x)) if x else 0 for x in ts + ["Tied to"]])
        format_spec = "  \033[1m{{p.name:^{0}s}}\033[0;0m  |  {{p._desc:^{1}s}}  |  {{const:^{2}s}}  |  {{t:^{3}s}}".format(nl, sl, cl, tl)
        to_print = [format_spec.format(p=p, const=c, t=t) for p, c, t in itertools.izip(self._parameters_, constrs, ts)]
        sep = '-'*(nl+sl+cl+tl+8*2+3)
        if header:
            header = "  {{0:^{0}s}}  |  {{1:^{1}s}}  |  {{2:^{2}s}}  |  {{3:^{3}s}}".format(nl, sl, cl, tl).format("Name", "Value", "Constraint", "Tied to")
            header += '\n' + sep
            to_print.insert(0, header)
        return '\n'.format(sep).join(to_print)
    pass


class Parameterized_old(object):
    def __init__(self):
        """
        This is the base class for model and kernel. Mostly just handles tieing and constraining of parameters
        """
        self.tied_indices = []
        self.fixed_indices = []
        self.fixed_values = []
        self.constrained_indices = []
        self.constraints = []
 
    def _get_params(self):
        raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
    def _set_params(self, x):
        raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
 
    def _get_param_names(self):
        raise NotImplementedError, "this needs to be implemented to use the Parameterized class"
    #def _get_print_names(self):
    #    """ Override for which parameter_names to print out, when using print m """
    #    return self._get_param_names()
 
    def pickle(self, filename, protocol=None):
        if protocol is None:
            if self._has_get_set_state():
                protocol = 0
            else:
                protocol = -1
        with open(filename, 'w') as f:
            cPickle.dump(self, f, protocol)
 
    def copy(self):
        """Returns a (deep) copy of the current model """
        return copy.deepcopy(self)
 
    def __getstate__(self):
        if self._has_get_set_state():
            return self.getstate()
        return self.__dict__
 
    def __setstate__(self, state):
        if self._has_get_set_state():
            self.setstate(state) # set state
            self._set_params(self._get_params()) # restore all values
            return
        self.__dict__ = state
 
    def _has_get_set_state(self):
        return 'getstate' in vars(self.__class__) and 'setstate' in vars(self.__class__)
 
    def getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        For inheriting from Parameterized:
 
        Allways append the state of the inherited object
        and call down to the inherited object in setstate!!
        """
        return [self.tied_indices,
                self.fixed_indices,
                self.fixed_values,
                self.constrained_indices,
                self.constraints]
 
    def setstate(self, state):
        self.constraints = state.pop()
        self.constrained_indices = state.pop()
        self.fixed_values = state.pop()
        self.fixed_indices = state.pop()
        self.tied_indices = state.pop()
 
    def __getitem__(self, regexp, return_names=False):
        """
        Get a model parameter by name.  The name is applied as a regular
        expression and all parameters that match that regular expression are
        returned.
        """
        matches = self.grep_param_names(regexp)
        if len(matches):
            if return_names:
                return self._get_params()[matches], np.asarray(self._get_param_names())[matches].tolist()
            else:
                return self._get_params()[matches]
        else:
            raise AttributeError, "no parameter matches %s" % regexp
 
    def __setitem__(self, name, val):
        """
        Set model parameter(s) by name. The name is provided as a regular
        expression. All parameters matching that regular expression are set to
        the given value.
        """
        matches = self.grep_param_names(name)
        if len(matches):
            val = np.array(val)
            assert (val.size == 1) or val.size == len(matches), "Shape mismatch: {}:({},)".format(val.size, len(matches))
            x = self._get_params()
            x[matches] = val
            self._set_params(x)
        else:
            raise AttributeError, "no parameter matches %s" % name
 
    def tie_params(self, regexp):
        """
        Tie (all!) parameters matching the regular expression `regexp`. 
        """
        matches = self.grep_param_names(regexp)
        assert matches.size > 0, "need at least something to tie together"
        if len(self.tied_indices):
            assert not np.any(matches[:, None] == np.hstack(self.tied_indices)), "Some indices are already tied!"
        self.tied_indices.append(matches)
        # TODO only one of the priors will be evaluated. Give a warning message if the priors are not identical
        if hasattr(self, 'prior'):
            pass
 
        self._set_params_transformed(self._get_params_transformed()) # sets tied parameters to single value
 
    def untie_everything(self):
        """Unties all parameters by setting tied_indices to an empty list."""
        self.tied_indices = []
 
    def grep_param_names(self, regexp, transformed=False, search=False):
        """
        :param regexp: regular expression to select parameter parameter_names
        :type regexp: re | str | int
        :rtype: the indices of self._get_param_names which match the regular expression.
 
        Note:-
          Other objects are passed through - i.e. integers which weren't meant for grepping
        """
 
        if transformed:
            parameter_names = self._get_param_names_transformed()
        else:
            parameter_names = self._get_param_names()
 
        if type(regexp) in [str, np.string_, np.str]:
            regexp = re.compile(regexp)
        elif type(regexp) is re._pattern_type:
            pass
        else:
            return regexp
        if search:
            return np.nonzero([regexp.search(name) for name in parameter_names])[0]
        else:
            return np.nonzero([regexp.match(name) for name in parameter_names])[0]
 
    def num_params_transformed(self):
        removed = 0
        for tie in self.tied_indices:
            removed += tie.size - 1
 
        for fix in self.fixed_indices:
            removed += fix.size
 
        return len(self._get_params()) - removed
 
    def unconstrain(self, regexp):
        """Unconstrain matching parameters.  Does not untie parameters"""
        matches = self.grep_param_names(regexp)
 
        # tranformed contraints:
        for match in matches:
            self.constrained_indices = [i[i <> match] for i in self.constrained_indices]
 
        # remove empty constraints
        tmp = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
        if tmp:
            self.constrained_indices, self.constraints = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
            self.constrained_indices, self.constraints = list(self.constrained_indices), list(self.constraints)
 
        # fixed:
        self.fixed_values = [np.delete(values, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices, values in zip(self.fixed_indices, self.fixed_values)]
        self.fixed_indices = [np.delete(indices, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices in self.fixed_indices]
 
        # remove empty elements
        tmp = [(i, v) for i, v in zip(self.fixed_indices, self.fixed_values) if len(i)]
        if tmp:
            self.fixed_indices, self.fixed_values = zip(*tmp)
            self.fixed_indices, self.fixed_values = list(self.fixed_indices), list(self.fixed_values)
        else:
            self.fixed_indices, self.fixed_values = [], []
 
    def constrain_negative(self, regexp, warning=True):
        """ Set negative constraints. """
        self.constrain(regexp, transformations.NegativeLogexp(), warning)
 
    def constrain_positive(self, regexp, warning=True):
        """ Set positive constraints. """
        self.constrain(regexp, transformations.Logexp(), warning)
 
    def constrain_bounded(self, regexp, lower, upper, warning=True):
        """ Set bounded constraints. """
        self.constrain(regexp, transformations.Logistic(lower, upper), warning)
 
    def all_constrained_indices(self):
        if len(self.constrained_indices) or len(self.fixed_indices):
            return np.hstack(self.constrained_indices + self.fixed_indices)
        else:
            return np.empty(shape=(0,))
 
    def constrain(self, regexp, transform, warning=True):
        assert isinstance(transform, transformations.Transformation)
 
        matches = self.grep_param_names(regexp)
        overlap = set(matches).intersection(set(self.all_constrained_indices()))
        if overlap:
            self.unconstrain(np.asarray(list(overlap)))
            if warning:
                print 'Warning: re-constraining these parameters'
                pn = self._get_param_names()
                for i in overlap:
                    print pn[i]
 
        self.constrained_indices.append(matches)
        self.constraints.append(transform)
        x = self._get_params()
        x[matches] = transform.initialize(x[matches])
        self._set_params(x)
 
    def constrain_fixed(self, regexp, value=None, warning=True):
        """
 
        :param regexp: which parameters need to be fixed.
        :type regexp: ndarray(dtype=int) or regular expression object or string
        :param value: the vlaue to fix the parameters to. If the value is not specified,
                 the parameter is fixed to the current value
        :type value: float
 
        **Notes**
 
        Fixing a parameter which is tied to another, or constrained in some way will result in an error.
 
        To fix multiple parameters to the same value, simply pass a regular expression which matches both parameter parameter_names, or pass both of the indexes.
 
        """
        matches = self.grep_param_names(regexp)
        overlap = set(matches).intersection(set(self.all_constrained_indices()))
        if overlap:
            self.unconstrain(np.asarray(list(overlap)))
            if warning:
                print 'Warning: re-constraining these parameters'
                pn = self._get_param_names()
                for i in overlap:
                    print pn[i]
 
        self.fixed_indices.append(matches)
        if value != None:
            self.fixed_values.append(value)
        else:
            self.fixed_values.append(self._get_params()[self.fixed_indices[-1]])
 
        # self.fixed_values.append(value)
        self._set_params_transformed(self._get_params_transformed())
 
    def _get_params_transformed(self):
        """use self._get_params to get the 'true' parameters of the model, which are then tied, constrained and fixed"""
        x = self._get_params()
        [np.put(x, i, t.finv(x[i])) for i, t in zip(self.constrained_indices, self.constraints)]
 
        to_remove = self.fixed_indices + [t[1:] for t in self.tied_indices]
        if len(to_remove):
            return np.delete(x, np.hstack(to_remove))
        else:
            return x
 
    def _set_params_transformed(self, x):
        """ takes the vector x, which is then modified (by untying, reparameterising or inserting fixed values), and then call self._set_params"""
        self._set_params(self._untransform_params(x))
 
    def _untransform_params(self, x):
        """
        The Transformation required for _set_params_transformed.
 
        This moves the vector x seen by the optimiser (unconstrained) to the
        valid parameter vector seen by the model
 
        Note:
          - This function is separate from _set_params_transformed for downstream flexibility
        """
        # work out how many places are fixed, and where they are. tricky logic!
        fix_places = self.fixed_indices + [t[1:] for t in self.tied_indices]
        if len(fix_places):
            fix_places = np.hstack(fix_places)
            Nfix_places = fix_places.size
        else:
            Nfix_places = 0
 
        free_places = np.setdiff1d(np.arange(Nfix_places + x.size, dtype=np.int), fix_places)
 
        # put the models values in the vector xx
        xx = np.zeros(Nfix_places + free_places.size, dtype=np.float64)
 
        xx[free_places] = x
        [np.put(xx, i, v) for i, v in zip(self.fixed_indices, self.fixed_values)]
        [np.put(xx, i, v) for i, v in [(t[1:], xx[t[0]]) for t in self.tied_indices] ]
 
        [np.put(xx, i, t.f(xx[i])) for i, t in zip(self.constrained_indices, self.constraints)]
        if hasattr(self, 'debug'):
            stop # @UndefinedVariable
 
        return xx
 
    def _get_param_names_transformed(self):
        """
        Returns the parameter parameter_names as propagated after constraining,
        tying or fixing, i.e. a list of the same length as _get_params_transformed()
        """
        n = self._get_param_names()
 
        # remove/concatenate the tied parameter parameter_names
        if len(self.tied_indices):
            for t in self.tied_indices:
                n[t[0]] = "<tie>".join([n[tt] for tt in t])
            remove = np.hstack([t[1:] for t in self.tied_indices])
        else:
            remove = np.empty(shape=(0,), dtype=np.int)
 
        # also remove the fixed params
        if len(self.fixed_indices):
            remove = np.hstack((remove, np.hstack(self.fixed_indices)))
 
        # add markers to show that some variables are constrained
        for i, t in zip(self.constrained_indices, self.constraints):
            for ii in i:
                n[ii] = n[ii] + t.__str__()
 
        n = [nn for i, nn in enumerate(n) if not i in remove]
        return n
 
    #@property
    #def all(self):
    #    return self.__str__(self._get_param_names())
 
 
    #def __str__(self, parameter_names=None, nw=30):
    def __str__(self, nw=30):
        """
        Return a string describing the parameter parameter_names and their ties and constraints
        """
        parameter_names = self._get_param_names()
        #if parameter_names is None:
        #    parameter_names = self._get_print_names()
        #name_indices = self.grep_param_names("|".join(parameter_names))
        N = len(parameter_names)
 
        if not N:
            return "This object has no free parameters."
        header = ['Name', 'Value', 'Constraints', 'Ties']
        values = self._get_params() # map(str,self._get_params())
        #values = self._get_params()[name_indices] # map(str,self._get_params())
        # sort out the constraints
        constraints = [''] * len(parameter_names)
        #constraints = [''] * len(self._get_param_names())
        for i, t in zip(self.constrained_indices, self.constraints):
            for ii in i:
                constraints[ii] = t.__str__()
        for i in self.fixed_indices:
            for ii in i:
                constraints[ii] = 'Fixed'
        # sort out the ties
        ties = [''] * len(parameter_names)
        for i, tie in enumerate(self.tied_indices):
            for j in tie:
                ties[j] = '(' + str(i) + ')'
 
        if values.size == 1:
            values = ['%.4f' %float(values)]
        else:
            values = ['%.4f' % float(v) for v in values]
        max_names = max([len(parameter_names[i]) for i in range(len(parameter_names))] + [len(header[0])])
        max_values = max([len(values[i]) for i in range(len(values))] + [len(header[1])])
        max_constraint = max([len(constraints[i]) for i in range(len(constraints))] + [len(header[2])])
        max_ties = max([len(ties[i]) for i in range(len(ties))] + [len(header[3])])
        cols = np.array([max_names, max_values, max_constraint, max_ties]) + 4
        # columns = cols.sum()
 
        header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
        header_string = map(lambda x: '|'.join(x), [header_string])
        separator = '-' * len(header_string[0])
        param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=parameter_names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]
 
 
        return ('\n'.join([header_string[0], separator] + param_string)) + '\n'
 
    def grep_model(self,regexp):
        regexp_indices = self.grep_param_names(regexp)
        all_names = self._get_param_names()
 
        parameter_names = [all_names[pj] for pj in regexp_indices]
        N = len(parameter_names)
 
        if not N:
            return "Match not found."
 
        header = ['Name', 'Value', 'Constraints', 'Ties']
        all_values = self._get_params()
        values = np.array([all_values[pj] for pj in regexp_indices])
        constraints = [''] * len(parameter_names)
 
        _constrained_indices,aux = self._pick_elements(regexp_indices,self.constrained_indices)
        _constraints_ = [self.constraints[pj] for pj in aux]
 
        for i, t in zip(_constrained_indices, _constraints_):
            for ii in i:
                iii = regexp_indices.tolist().index(ii)
                constraints[iii] = t.__str__()
 
        _fixed_indices,aux = self._pick_elements(regexp_indices,self.fixed_indices)
        for i in _fixed_indices:
            for ii in i:
                iii = regexp_indices.tolist().index(ii)
                constraints[ii] = 'Fixed'
 
        _tied_indices,aux = self._pick_elements(regexp_indices,self.tied_indices)
        ties = [''] * len(parameter_names)
        for i,ti in zip(_tied_indices,aux):
            for ii in i:
                iii = regexp_indices.tolist().index(ii)
                ties[iii] = '(' + str(ti) + ')'
 
        if values.size == 1:
            values = ['%.4f' %float(values)]
        else:
            values = ['%.4f' % float(v) for v in values]
 
        max_names = max([len(parameter_names[i]) for i in range(len(parameter_names))] + [len(header[0])])
        max_values = max([len(values[i]) for i in range(len(values))] + [len(header[1])])
        max_constraint = max([len(constraints[i]) for i in range(len(constraints))] + [len(header[2])])
        max_ties = max([len(ties[i]) for i in range(len(ties))] + [len(header[3])])
        cols = np.array([max_names, max_values, max_constraint, max_ties]) + 4
 
        header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
        header_string = map(lambda x: '|'.join(x), [header_string])
        separator = '-' * len(header_string[0])
        param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=parameter_names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]
 
        print header_string[0]
        print separator
        for string in param_string:
            print string
 
    def _pick_elements(self,regexp_ind,array_list):
        """Removes from array_list the elements different from regexp_ind"""
        new_array_list = [] #New list with elements matching regexp_ind
        array_indices = [] #Indices that matches the arrays in new_array_list and array_list
 
        array_index = 0
        for array in array_list:
            _new = []
            for ai in array:
                if ai in regexp_ind:
                    _new.append(ai)
            if len(_new):
                new_array_list.append(np.array(_new))
                array_indices.append(array_index)
            array_index += 1
        return new_array_list, array_indices
