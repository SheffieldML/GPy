# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Core module for parameterization.
This module implements all parameterization techniques, split up in modular bits.

HierarchyError:
raised when an error with the hierarchy occurs (circles etc.)

Observable:
Observable Pattern for patameterization


"""

from .transformations import Transformation,Logexp, NegativeLogexp, Logistic, __fixed__, FIXED, UNFIXED
import numpy as np
import re
import logging
from .updateable import Updateable
from functools import reduce

class HierarchyError(Exception):
    """
    Gets thrown when something is wrong with the parameter hierarchy.
    """

def adjust_name_for_printing(name):
    """
    Make sure a name can be printed, alongside used as a variable name.
    """
    if name is not None:
        name2 = name
        name = name.replace(" ", "_").replace(".", "_").replace("-", "_m_")
        name = name.replace("+", "_p_").replace("!", "_I_")
        name = name.replace("**", "_xx_").replace("*", "_x_")
        name = name.replace("/", "_l_").replace("@", '_at_')
        name = name.replace("(", "_of_").replace(")", "")
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9-_]*$', name) is None:
            raise NameError("name {} converted to {} cannot be further converted to valid python variable name!".format(name2, name))
        return name
    return ''



class Parentable(object):
    """
    Enable an Object to have a parent.

    Additionally this adds the parent_index, which is the index for the parent
    to look for in its parameter list.
    """
    _parent_ = None
    _parent_index_ = None
    def __init__(self, *args, **kwargs):
        super(Parentable, self).__init__()

    def has_parent(self):
        """
        Return whether this parentable object currently has a parent.
        """
        return self._parent_ is not None

    def _parent_changed(self):
        """
        Gets called, when the parent changed, so we can adjust our
        inner attributes according to the new parent.
        """
        raise NotImplementedError("shouldnt happen, Parentable objects need to be able to change their parent")

    def _disconnect_parent(self, *args, **kw):
        """
        Disconnect this object from its parent
        """
        raise NotImplementedError("Abstract superclass")

    @property
    def _highest_parent_(self):
        """
        Gets the highest parent by traversing up to the root node of the hierarchy.
        """
        if self._parent_ is None:
            return self
        return self._parent_._highest_parent_

    def _notify_parent_change(self):
        """
        Dont do anything if in leaf node
        """
        pass

class Pickleable(object):
    """
    Make an object pickleable (See python doc 'pickling').

    This class allows for pickling support by Memento pattern.
    _getstate returns a memento of the class, which gets pickled.
    _setstate(<memento>) (re-)sets the state of the class to the memento
    """
    def __init__(self, *a, **kw):
        super(Pickleable, self).__init__()

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
        try: #Py2
            import cPickle as pickle
        except ImportError: #Py3
            import pickle
        if isinstance(f, str):
            with open(f, 'wb') as f:
                pickle.dump(self, f, protocol)
        else:
            pickle.dump(self, f, protocol)

    #===========================================================================
    # copy and pickling
    #===========================================================================
    def copy(self, memo=None, which=None):
        """
        Returns a (deep) copy of the current parameter handle.

        All connections to parents of the copy will be cut.

        :param dict memo: memo for deepcopy
        :param Parameterized which: parameterized object which started the copy process [default: self]
        """
        #raise NotImplementedError, "Copy is not yet implemented, TODO: Observable hierarchy"
        if memo is None:
            memo = {}
        import copy
        # the next part makes sure that we do not include parents in any form:
        parents = []
        if which is None:
            which = self
        which.traverse_parents(parents.append) # collect parents
        for p in parents:
            if not id(p) in memo :memo[id(p)] = None # set all parents to be None, so they will not be copied
        if not id(self.gradient) in memo:memo[id(self.gradient)] = None # reset the gradient
        if not id(self._fixes_) in memo :memo[id(self._fixes_)] = None # fixes have to be reset, as this is now highest parent
        copy = copy.deepcopy(self, memo) # and start the copy
        copy._parent_index_ = None
        copy._trigger_params_changed()
        return copy

    def __deepcopy__(self, memo):
        s = self.__new__(self.__class__) # fresh instance
        memo[id(self)] = s # be sure to break all cycles --> self is already done
        import copy
        s.__setstate__(copy.deepcopy(self.__getstate__(), memo)) # standard copy
        return s

    def __getstate__(self):
        ignore_list = ['_param_array_', # parameters get set from bottom to top
                       '_gradient_array_', # as well as gradients
                       '_optimizer_copy_',
                       'logger',
                       'observers',
                       '_fixes_', # and fixes
                       '_Cacher_wrap__cachers', # never pickle cachers
                       ]
        dc = dict()
        #py3 fix
        #for k,v in self.__dict__.iteritems():
        for k,v in self.__dict__.items():
            if k not in ignore_list:
                dc[k] = v
        return dc

    def __setstate__(self, state):
        self.__dict__.update(state)
        from .lists_and_dicts import ObserverList
        self.observers = ObserverList()
        self._setup_observers()
        self._optimizer_copy_transformed = False


class Gradcheckable(Pickleable, Parentable):
    """
    Adds the functionality for an object to be gradcheckable.
    It is just a thin wrapper of a call to the highest parent for now.
    TODO: Can be done better, by only changing parameters of the current parameter handle,
    such that object hierarchy only has to change for those.
    """
    def __init__(self, *a, **kw):
        super(Gradcheckable, self).__init__(*a, **kw)

    def checkgrad(self, verbose=0, step=1e-6, tolerance=1e-3, df_tolerance=1e-12):
        """
        Check the gradient of this parameter with respect to the highest parent's
        objective function.
        This is a three point estimate of the gradient, wiggling at the parameters
        with a stepsize step.
        The check passes if either the ratio or the difference between numerical and
        analytical gradient is smaller then tolerance.

        :param bool verbose: whether each parameter shall be checked individually.
        :param float step: the stepsize for the numerical three point gradient estimate.
        :param float tolerance: the tolerance for the gradient ratio or difference.
        :param float df_tolerance: the tolerance for df_tolerance

        Note:-
           The *dF_ratio* indicates the limit of accuracy of numerical gradients.
           If it is too small, e.g., smaller than 1e-12, the numerical gradients
           are usually not accurate enough for the tests (shown with blue).
        """
        if self.has_parent():
            return self._highest_parent_._checkgrad(self, verbose=verbose, step=step, tolerance=tolerance, df_tolerance=df_tolerance)
        return self._checkgrad(self, verbose=verbose, step=step, tolerance=tolerance, df_tolerance=df_tolerance)

    def _checkgrad(self, param, verbose=0, step=1e-6, tolerance=1e-3):
        """
        Perform the checkgrad on the model.
        TODO: this can be done more efficiently, when doing it inside here
        """
        raise HierarchyError("This parameter is not in a model with a likelihood, and, therefore, cannot be gradient checked!")

class Nameable(Gradcheckable):
    """
    Make an object nameable inside the hierarchy.
    """
    def __init__(self, name, *a, **kw):
        super(Nameable, self).__init__(*a, **kw)
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        """
        The name of this object
        """
        return self._name
    @name.setter
    def name(self, name):
        """
        Set the name of this object.
        Tell the parent if the name has changed.
        """
        from_name = self.name
        assert isinstance(name, str)
        self._name = name
        if self.has_parent():
            self._parent_._name_changed(self, from_name)
    def hierarchy_name(self, adjust_for_printing=True):
        """
        return the name for this object with the parents names attached by dots.

        :param bool adjust_for_printing: whether to call :func:`~adjust_for_printing()`
        on the names, recursively
        """
        if adjust_for_printing: adjust = lambda x: adjust_name_for_printing(x)
        else: adjust = lambda x: x
        if self.has_parent():
            return self._parent_.hierarchy_name() + "." + adjust(self.name)
        return adjust(self.name)


class Indexable(Nameable, Updateable):
    """
    Make an object constrainable with Priors and Transformations.
    TODO: Mappings!!
    Adding a constraint to a Parameter means to tell the highest parent that
    the constraint was added and making sure that all parameters covered
    by this object are indeed conforming to the constraint.

    :func:`constrain()` and :func:`unconstrain()` are main methods here
    """
    def __init__(self, name, default_constraint=None, *a, **kw):
        super(Indexable, self).__init__(name=name, *a, **kw)
        self._default_constraint_ = default_constraint
        from .index_operations import ParameterIndexOperations
        self.constraints = ParameterIndexOperations()
        self.priors = ParameterIndexOperations()
        if self._default_constraint_ is not None:
            self.constrain(self._default_constraint_)

    def _disconnect_parent(self, constr=None, *args, **kw):
        """
        From Parentable:
        disconnect the parent and set the new constraints to constr
        """
        if constr is None:
            constr = self.constraints.copy()
        self.constraints.clear()
        self.constraints = constr
        self._parent_ = None
        self._parent_index_ = None
        self._connect_fixes()
        self._notify_parent_change()

    #===========================================================================
    # Indexable
    #===========================================================================
    def _offset_for(self, param):
        """
        Return the offset of the param inside this parameterized object.
        This does not need to account for shaped parameters, as it
        basically just sums up the parameter sizes which come before param.
        """
        if param.has_parent():
            p = param._parent_._get_original(param)
            if p in self.parameters:
                return reduce(lambda a,b: a + b.size, self.parameters[:p._parent_index_], 0)
            return self._offset_for(param._parent_) + param._parent_._offset_for(param)
        return 0

    def _raveled_index_for(self, param):
        """
        get the raveled index for a param
        that is an int array, containing the indexes for the flattened
        param inside this parameterized logic.
        """
        from .param import ParamConcatenation
        if isinstance(param, ParamConcatenation):
            return np.hstack((self._raveled_index_for(p) for p in param.params))
        return param._raveled_index() + self._offset_for(param)

    def _raveled_index(self):
        """
        Flattened array of ints, specifying the index of this object.
        This has to account for shaped parameters!
        """
        return np.r_[:self.size]

    #===========================================================================
    # Fixing Parameters:
    #===========================================================================
    def constrain_fixed(self, value=None, warning=True, trigger_parent=True):
        """
        Constrain this parameter to be fixed to the current value it carries.

        :param warning: print a warning for overwriting constraints.
        """
        if value is not None:
            self[:] = value

        index = self.unconstrain()
        index = self._add_to_index_operations(self.constraints, index, __fixed__, warning)
        self._highest_parent_._set_fixed(self, index)
        self.notify_observers(self, None if trigger_parent else -np.inf)
        return index
    fix = constrain_fixed

    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        unconstrained = self.unconstrain(__fixed__)
        self._highest_parent_._set_unfixed(self, unconstrained)
        return unconstrained
    unfix = unconstrain_fixed

    def _ensure_fixes(self):
        # Ensure that the fixes array is set:
        # Parameterized: ones(self.size)
        # Param: ones(self._realsize_
        if not self._has_fixes(): self._fixes_ = np.ones(self.size, dtype=bool)

    def _set_fixed(self, param, index):
        self._ensure_fixes()
        offset = self._offset_for(param)
        self._fixes_[index+offset] = FIXED
        if np.all(self._fixes_): self._fixes_ = None  # ==UNFIXED

    def _set_unfixed(self, param, index):
        self._ensure_fixes()
        offset = self._offset_for(param)
        self._fixes_[index+offset] = UNFIXED
        if np.all(self._fixes_): self._fixes_ = None  # ==UNFIXED

    def _connect_fixes(self):
        fixed_indices = self.constraints[__fixed__]
        if fixed_indices.size > 0:
            self._ensure_fixes()
            self._fixes_[fixed_indices] = FIXED
        else:
            self._fixes_ = None
            del self.constraints[__fixed__]

    #===========================================================================
    # Convenience for fixed
    #===========================================================================
    def _has_fixes(self):
        return hasattr(self, "_fixes_") and self._fixes_ is not None and self._fixes_.size == self.size

    @property
    def is_fixed(self):
        for p in self.parameters:
            if not p.is_fixed: return False
        return True

    def _get_original(self, param):
        # if advanced indexing is activated it happens that the array is a copy
        # you can retrieve the original param through this method, by passing
        # the copy here
        return self.parameters[param._parent_index_]

    #===========================================================================
    # Prior Operations
    #===========================================================================
    def set_prior(self, prior, warning=True):
        """
        Set the prior for this object to prior.
        :param :class:`~GPy.priors.Prior` prior: a prior to set for this parameter
        :param bool warning: whether to warn if another prior was set for this parameter
        """
        repriorized = self.unset_priors()
        self._add_to_index_operations(self.priors, repriorized, prior, warning)

        from .domains import _REAL, _POSITIVE, _NEGATIVE
        if prior.domain is _POSITIVE:
            self.constrain_positive(warning)
        elif prior.domain is _NEGATIVE:
            self.constrain_negative(warning)
        elif prior.domain is _REAL:
            rav_i = self._raveled_index()
            assert all(all(False if c is __fixed__ else c.domain is _REAL for c in con) for con in self.constraints.properties_for(rav_i)), 'Domain of prior and constraint have to match, please unconstrain if you REALLY wish to use this prior'

    def unset_priors(self, *priors):
        """
        Un-set all priors given (in *priors) from this parameter handle.
        """
        return self._remove_from_index_operations(self.priors, priors)

    def log_prior(self):
        """evaluate the prior"""
        if self.priors.size == 0:
            return 0.
        x = self.param_array
        #evaluate the prior log densities
        log_p = reduce(lambda a, b: a + b, (p.lnpdf(x[ind]).sum() for p, ind in self.priors.items()), 0)

        #account for the transformation by evaluating the log Jacobian (where things are transformed)
        log_j = 0.
        priored_indexes = np.hstack([i for p, i in self.priors.items()])
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    log_j += c.log_jacobian(x[jj])
        return log_p + log_j

    def _log_prior_gradients(self):
        """evaluate the gradients of the priors"""
        if self.priors.size == 0:
            return 0.
        x = self.param_array
        ret = np.zeros(x.size)
        #compute derivate of prior density
        [np.put(ret, ind, p.lnpdf_grad(x[ind])) for p, ind in self.priors.items()]
        #add in jacobian derivatives if transformed
        priored_indexes = np.hstack([i for p, i in self.priors.items()])
        for c,j in self.constraints.items():
            if not isinstance(c, Transformation):continue
            for jj in j:
                if jj in priored_indexes:
                    ret[jj] += c.log_jacobian_grad(x[jj])
        return ret

    #===========================================================================
    # Tie parameters together
    #===========================================================================

    def _has_ties(self):
        if self._highest_parent_.tie.tied_param is None:
            return False
        if self.has_parent():
            return self._highest_parent_.tie.label_buf[self._highest_parent_._raveled_index_for(self)].sum()>0
        return True

    def tie_together(self):
        self._highest_parent_.tie.add_tied_parameter(self)
        self._highest_parent_._set_fixed(self,self._raveled_index())
        self._trigger_params_changed()

    #===========================================================================
    # Constrain operations -> done
    #===========================================================================

    def constrain(self, transform, warning=True, trigger_parent=True):
        """
        :param transform: the :py:class:`GPy.core.transformations.Transformation`
                          to constrain the this parameter to.
        :param warning: print a warning if re-constraining parameters.

        Constrain the parameter to the given
        :py:class:`GPy.core.transformations.Transformation`.
        """
        if isinstance(transform, Transformation):
            self.param_array[...] = transform.initialize(self.param_array)
        reconstrained = self.unconstrain()
        added = self._add_to_index_operations(self.constraints, reconstrained, transform, warning)
        self.trigger_update(trigger_parent)
        return added

    def unconstrain(self, *transforms):
        """
        :param transforms: The transformations to unconstrain from.

        remove all :py:class:`GPy.core.transformations.Transformation`
        transformats of this parameter object.
        """
        return self._remove_from_index_operations(self.constraints, transforms)

    def constrain_positive(self, warning=True, trigger_parent=True):
        """
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to the default positive constraint.
        """
        self.constrain(Logexp(), warning=warning, trigger_parent=trigger_parent)

    def constrain_negative(self, warning=True, trigger_parent=True):
        """
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to the default negative constraint.
        """
        self.constrain(NegativeLogexp(), warning=warning, trigger_parent=trigger_parent)

    def constrain_bounded(self, lower, upper, warning=True, trigger_parent=True):
        """
        :param lower, upper: the limits to bound this parameter to
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to lie within the given range.
        """
        self.constrain(Logistic(lower, upper), warning=warning, trigger_parent=trigger_parent)

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

    def _parent_changed(self, parent):
        """
        From Parentable:
        Called when the parent changed

        update the constraints and priors view, so that
        constraining is automized for the parent.
        """
        from .index_operations import ParameterIndexOperationsView
        #if getattr(self, "_in_init_"):
            #import ipdb;ipdb.set_trace()
            #self.constraints.update(param.constraints, start)
            #self.priors.update(param.priors, start)
        offset = parent._offset_for(self)
        self.constraints = ParameterIndexOperationsView(parent.constraints, offset, self.size)
        self.priors = ParameterIndexOperationsView(parent.priors, offset, self.size)
        self._fixes_ = None
        for p in self.parameters:
            p._parent_changed(parent)

    def _add_to_index_operations(self, which, reconstrained, what, warning):
        """
        Helper preventing copy code.
        This adds the given what (transformation, prior etc) to parameter index operations which.
        reconstrained are reconstrained indices.
        warn when reconstraining parameters if warning is True.
        TODO: find out which parameters have changed specifically
        """
        if warning and reconstrained.size > 0:
            # TODO: figure out which parameters have changed and only print those
            print("WARNING: reconstraining parameters {}".format(self.hierarchy_name() or self.name))
        index = self._raveled_index()
        which.add(what, index)
        return index

    def _remove_from_index_operations(self, which, transforms):
        """
        Helper preventing copy code.
        Remove given what (transform prior etc) from which param index ops.
        """
        if len(transforms) == 0:
            transforms = which.properties()
        removed = np.empty((0,), dtype=int)
        for t in list(transforms):
            unconstrained = which.remove(t, self._raveled_index())
            removed = np.union1d(removed, unconstrained)
            if t is __fixed__:
                self._highest_parent_._set_unfixed(self, unconstrained)

        return removed

class OptimizationHandlable(Indexable):
    """
    This enables optimization handles on an Object as done in GPy 0.4.

    `..._optimizer_copy_transformed`: make sure the transformations and constraints etc are handled
    """
    def __init__(self, name, default_constraint=None, *a, **kw):
        super(OptimizationHandlable, self).__init__(name, default_constraint=default_constraint, *a, **kw)
        self._optimizer_copy_ = None
        self._optimizer_copy_transformed = False

    #===========================================================================
    # Optimizer copy
    #===========================================================================
    @property
    def optimizer_array(self):
        """
        Array for the optimizer to work on.
        This array always lives in the space for the optimizer.
        Thus, it is untransformed, going from Transformations.

        Setting this array, will make sure the transformed parameters for this model
        will be set accordingly. It has to be set with an array, retrieved from
        this method, as e.g. fixing will resize the array.

        The optimizer should only interfere with this array, such that transformations
        are secured.
        """
        if self.__dict__.get('_optimizer_copy_', None) is None or self.size != self._optimizer_copy_.size:
            self._optimizer_copy_ = np.empty(self.size)

        if not self._optimizer_copy_transformed:
            self._optimizer_copy_.flat = self.param_array.flat
            #py3 fix
            #[np.put(self._optimizer_copy_, ind, c.finv(self.param_array[ind])) for c, ind in self.constraints.iteritems() if c != __fixed__]
            [np.put(self._optimizer_copy_, ind, c.finv(self.param_array[ind])) for c, ind in self.constraints.items() if c != __fixed__]
            if self.has_parent() and (self.constraints[__fixed__].size != 0 or self._has_ties()):
                fixes = np.ones(self.size).astype(bool)
                fixes[self.constraints[__fixed__]] = FIXED
                return self._optimizer_copy_[np.logical_and(fixes, self._highest_parent_.tie.getTieFlag(self))]
            elif self._has_fixes():
                return self._optimizer_copy_[self._fixes_]

            self._optimizer_copy_transformed = True

        return self._optimizer_copy_

    @optimizer_array.setter
    def optimizer_array(self, p):
        """
        Make sure the optimizer copy does not get touched, thus, we only want to
        set the values *inside* not the array itself.

        Also we want to update param_array in here.
        """
        f = None
        if self.has_parent() and self.constraints[__fixed__].size != 0:
            f = np.ones(self.size).astype(bool)
            f[self.constraints[__fixed__]] = FIXED
        elif self._has_fixes():
            f = self._fixes_
        if f is None:
            self.param_array.flat = p
            [np.put(self.param_array, ind, c.f(self.param_array.flat[ind]))
             #py3 fix
             #for c, ind in self.constraints.iteritems() if c != __fixed__]
             for c, ind in self.constraints.items() if c != __fixed__]
        else:
            self.param_array.flat[f] = p
            [np.put(self.param_array, ind[f[ind]], c.f(self.param_array.flat[ind[f[ind]]]))
             #py3 fix
             #for c, ind in self.constraints.iteritems() if c != __fixed__]
             for c, ind in self.constraints.items() if c != __fixed__]
        #self._highest_parent_.tie.propagate_val()

        self._optimizer_copy_transformed = False
        self.trigger_update()

    def _get_params_transformed(self):
        raise DeprecationWarning("_get|set_params{_optimizer_copy_transformed} is deprecated, use self.optimizer array insetad!")
#
    def _set_params_transformed(self, p):
        raise DeprecationWarning("_get|set_params{_optimizer_copy_transformed} is deprecated, use self.optimizer array insetad!")

    def _trigger_params_changed(self, trigger_parent=True):
        """
        First tell all children to update,
        then update yourself.

        If trigger_parent is True, we will tell the parent, otherwise not.
        """
        [p._trigger_params_changed(trigger_parent=False) for p in self.parameters if not p.is_fixed]
        self.notify_observers(None, None if trigger_parent else -np.inf)

    def _size_transformed(self):
        """
        As fixes are not passed to the optimiser, the size of the model for the optimiser
        is the size of all parameters minus the size of the fixes.
        """
        return self.size - self.constraints[__fixed__].size

    def _transform_gradients(self, g):
        """
        Transform the gradients by multiplying the gradient factor for each
        constraint to it.
        """
        self._highest_parent_.tie.collate_gradient()
        #py3 fix
        #[np.put(g, i, c.gradfactor(self.param_array[i], g[i])) for c, i in self.constraints.iteritems() if c != __fixed__]
        [np.put(g, i, c.gradfactor(self.param_array[i], g[i])) for c, i in self.constraints.items() if c != __fixed__]
        if self._has_fixes(): return g[self._fixes_]
        return g

    def _transform_gradients_non_natural(self, g):
        """
        Transform the gradients by multiplying the gradient factor for each
        constraint to it.
        """
        self._highest_parent_.tie.collate_gradient()
        #py3 fix
        #[np.put(g, i, c.gradfactor_non_natural(self.param_array[i], g[i])) for c, i in self.constraints.iteritems() if c != __fixed__]
        [np.put(g, i, c.gradfactor_non_natural(self.param_array[i], g[i])) for c, i in self.constraints.items() if c != __fixed__]
        if self._has_fixes(): return g[self._fixes_]
        return g


    @property
    def num_params(self):
        """
        Return the number of parameters of this parameter_handle.
        Param objects will always return 0.
        """
        raise NotImplemented("Abstract, please implement in respective classes")

    def parameter_names(self, add_self=False, adjust_for_printing=False, recursive=True):
        """
        Get the names of all parameters of this model.

        :param bool add_self: whether to add the own name in front of names
        :param bool adjust_for_printing: whether to call `adjust_name_for_printing` on names
        :param bool recursive: whether to traverse through hierarchy and append leaf node names
        """
        if adjust_for_printing: adjust = lambda x: adjust_name_for_printing(x)
        else: adjust = lambda x: x
        if recursive: names = [xi for x in self.parameters for xi in x.parameter_names(add_self=True, adjust_for_printing=adjust_for_printing)]
        else: names = [adjust(x.name) for x in self.parameters]
        if add_self: names = map(lambda x: adjust(self.name) + "." + x, names)
        return names

    def _get_param_names(self):
        n = np.array([p.hierarchy_name() + '[' + str(i) + ']' for p in self.flattened_parameters for i in p._indices()])
        return n

    def _get_param_names_transformed(self):
        n = self._get_param_names()
        if self._has_fixes():
            return n[self._fixes_]
        return n

    #===========================================================================
    # Randomizeable
    #===========================================================================
    def randomize(self, rand_gen=None, *args, **kwargs):
        """
        Randomize the model.
        Make this draw from the prior if one exists, else draw from given random generator

        :param rand_gen: np random number generator which takes args and kwargs
        :param flaot loc: loc parameter for random number generator
        :param float scale: scale parameter for random number generator
        :param args, kwargs: will be passed through to random number generator
        """
        if rand_gen is None:
            rand_gen = np.random.normal
        # first take care of all parameters (from N(0,1))
        x = rand_gen(size=self._size_transformed(), *args, **kwargs)
        updates = self.update_model()
        self.update_model(False) # Switch off the updates
        self.optimizer_array = x  # makes sure all of the tied parameters get the same init (since there's only one prior object...)
        # now draw from prior where possible
        x = self.param_array.copy()
        #Py3 fix
        #[np.put(x, ind, p.rvs(ind.size)) for p, ind in self.priors.iteritems() if not p is None]
        [np.put(x, ind, p.rvs(ind.size)) for p, ind in self.priors.items() if not p is None]
        unfixlist = np.ones((self.size,),dtype=np.bool)
        unfixlist[self.constraints[__fixed__]] = False
        self.param_array.flat[unfixlist] = x.view(np.ndarray).ravel()[unfixlist]
        self.update_model(updates)

    #===========================================================================
    # For shared memory arrays. This does nothing in Param, but sets the memory
    # for all parameterized objects
    #===========================================================================
    @property
    def gradient_full(self):
        """
        Note to users:
        This does not return the gradient in the right shape! Use self.gradient
        for the right gradient array.

        To work on the gradient array, use this as the gradient handle.
        This method exists for in memory use of parameters.
        When trying to access the true gradient array, use this.
        """
        self.gradient # <<< ensure _gradient_array_
        return self._gradient_array_

    def _propagate_param_grad(self, parray, garray):
        """
        For propagating the param_array and gradient_array.
        This ensures the in memory view of each subsequent array.

        1.) connect param_array of children to self.param_array
        2.) tell all children to propagate further
        """
        if self.param_array.size != self.size:
            self._param_array_ = np.empty(self.size, dtype=np.float64)
        if self.gradient.size != self.size:
            self._gradient_array_ = np.empty(self.size, dtype=np.float64)

        pi_old_size = 0
        for pi in self.parameters:
            pislice = slice(pi_old_size, pi_old_size + pi.size)

            self.param_array[pislice] = pi.param_array.flat  # , requirements=['C', 'W']).flat
            self.gradient_full[pislice] = pi.gradient_full.flat  # , requirements=['C', 'W']).flat

            pi.param_array.data = parray[pislice].data
            pi.gradient_full.data = garray[pislice].data

            pi._propagate_param_grad(parray[pislice], garray[pislice])
            pi_old_size += pi.size

    def _connect_parameters(self):
        pass

_name_digit = re.compile("(?P<name>.*)_(?P<digit>\d+)$")
class Parameterizable(OptimizationHandlable):
    """
    A parameterisable class.

    This class provides the parameters list (ArrayList) and standard parameter handling,
    such as {link|unlink}_parameter(), traverse hierarchy and param_array, gradient_array
    and the empty parameters_changed().

    This class is abstract and should not be instantiated.
    Use GPy.core.Parameterized() as node (or leaf) in the parameterized hierarchy.
    Use GPy.core.Param() for a leaf in the parameterized hierarchy.
    """
    def __init__(self, *args, **kwargs):
        super(Parameterizable, self).__init__(*args, **kwargs)
        from GPy.core.parameterization.lists_and_dicts import ArrayList
        self.parameters = ArrayList()
        self._param_array_ = None
        self._added_names_ = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__visited = False # for traversing in reverse order we need to know if we were here already

    @property
    def param_array(self):
        """
        Array representing the parameters of this class.
        There is only one copy of all parameters in memory, two during optimization.

        !WARNING!: setting the parameter array MUST always be done in memory:
        m.param_array[:] = m_copy.param_array
        """
        if (self.__dict__.get('_param_array_', None) is None) or (self._param_array_.size != self.size):
            self._param_array_ = np.empty(self.size, dtype=np.float64)
        return self._param_array_

    @property
    def unfixed_param_array(self):
        """
        Array representing the parameters of this class.
        There is only one copy of all parameters in memory, two during optimization.

        !WARNING!: setting the parameter array MUST always be done in memory:
        m.param_array[:] = m_copy.param_array
        """
        if self.__dict__.get('_param_array_', None) is None:
            self._param_array_ = np.empty(self.size, dtype=np.float64)

        if self.constraints[__fixed__].size !=0:
            fixes = np.ones(self.size).astype(bool)
            fixes[self.constraints[__fixed__]] = FIXED
            return self._param_array_[fixes]
        else:
            return self._param_array_

    @param_array.setter
    def param_array(self, arr):
        self._param_array_ = arr

    def traverse(self, visit, *args, **kwargs):
        """
        Traverse the hierarchy performing visit(self, *args, **kwargs)
        at every node passed by downwards. This function includes self!

        See "visitor pattern" in literature. This is implemented in pre-order fashion.

        Example:
        Collect all children:

        children = []
        self.traverse(children.append)
        print children
        """
        if not self.__visited:
            visit(self, *args, **kwargs)
            self.__visited = True
            for c in self.parameters:
                c.traverse(visit, *args, **kwargs)
            self.__visited = False

    def traverse_parents(self, visit, *args, **kwargs):
        """
        Traverse the hierarchy upwards, visiting all parents and their children except self.
        See "visitor pattern" in literature. This is implemented in pre-order fashion.

        Example:

        parents = []
        self.traverse_parents(parents.append)
        print parents
        """
        if self.has_parent():
            self.__visited = True
            self._parent_._traverse_parents(visit, *args, **kwargs)
            self.__visited = False

    def _traverse_parents(self, visit, *args, **kwargs):
        if not self.__visited:
            self.__visited = True
            visit(self, *args, **kwargs)
            if self.has_parent():
                self._parent_._traverse_parents(visit, *args, **kwargs)
                self._parent_.traverse(visit, *args, **kwargs)
            self.__visited = False

    #=========================================================================
    # Gradient handling
    #=========================================================================
    @property
    def gradient(self):
        if (self.__dict__.get('_gradient_array_', None) is None) or self._gradient_array_.size != self.size:
            self._gradient_array_ = np.empty(self.size, dtype=np.float64)
        return self._gradient_array_

    @gradient.setter
    def gradient(self, val):
        self._gradient_array_[:] = val

    @property
    def num_params(self):
        return len(self.parameters)

    def _add_parameter_name(self, param, ignore_added_names=False):
        pname = adjust_name_for_printing(param.name)
        if ignore_added_names:
            self.__dict__[pname] = param
            return

        def warn_and_retry(param, match=None):
            #===================================================================
            # print """
            # WARNING: added a parameter with formatted name {},
            # which is already assigned to {}.
            # Trying to change the parameter name to
            #
            # {}.{}
            # """.format(pname, self.hierarchy_name(), self.hierarchy_name(), param.name + "_")
            #===================================================================
            if match is None:
                param.name += "_1"
            else:
                param.name = match.group('name') + "_" + str(int(match.group('digit'))+1)
            self._add_parameter_name(param, ignore_added_names)
        # and makes sure to not delete programmatically added parameters
        for other in self.parameters[::-1]:
            if other is not param and other.name == param.name:
                warn_and_retry(param, _name_digit.match(other.name))
                return
        if pname not in dir(self):
            self.__dict__[pname] = param
            self._added_names_.add(pname)
        elif pname in self.__dict__:
            if pname in self._added_names_:
                other = self.__dict__[pname]
                if not (param is other):
                    del self.__dict__[pname]
                    self._added_names_.remove(pname)
                    warn_and_retry(other)
                    warn_and_retry(param, _name_digit.match(other.name))
            return

    def _remove_parameter_name(self, param=None, pname=None):
        assert param is None or pname is None, "can only delete either param by name, or the name of a param"
        pname = adjust_name_for_printing(pname) or adjust_name_for_printing(param.name)
        if pname in self._added_names_:
            del self.__dict__[pname]
            self._added_names_.remove(pname)
        self._connect_parameters()

    def _name_changed(self, param, old_name):
        self._remove_parameter_name(None, old_name)
        self._add_parameter_name(param)

    def __setstate__(self, state):
        super(Parameterizable, self).__setstate__(state)
        self.logger = logging.getLogger(self.__class__.__name__)
        return self

    #===========================================================================
    # notification system
    #===========================================================================
    def _parameters_changed_notification(self, me, which=None):
        """
        In parameterizable we just need to make sure, that the next call to optimizer_array
        will update the optimizer_array to the latest parameters
        """
        self._optimizer_copy_transformed = False # tells the optimizer array to update on next request
        self.parameters_changed()
    def _pass_through_notify_observers(self, me, which=None):
        self.notify_observers(which=which)
    def _setup_observers(self):
        """
        Setup the default observers

        1: parameters_changed_notify
        2: pass through to parent, if present
        """
        self.add_observer(self, self._parameters_changed_notification, -100)
        if self.has_parent():
            self.add_observer(self._parent_, self._parent_._pass_through_notify_observers, -np.inf)
    #===========================================================================
    # From being parentable, we have to define the parent_change notification
    #===========================================================================
    def _notify_parent_change(self):
        """
        Notify all parameters that the parent has changed
        """
        for p in self.parameters:
            p._parent_changed(self)

    def parameters_changed(self):
        """
        This method gets called when parameters have changed.
        Another way of listening to param changes is to
        add self as a listener to the param, such that
        updates get passed through. See :py:function:``GPy.core.param.Observable.add_observer``
        """
        pass

    def save(self, filename, ftype='HDF5'):
        """
        Save all the model parameters into a file (HDF5 by default).
        """
        from . import Param
        from ...util.misc import param_to_array
        def gather_params(self, plist):
            if isinstance(self,Param):
                plist.append(self)
        plist = []
        self.traverse(gather_params, plist)
        names = self.parameter_names(adjust_for_printing=True)
        if ftype=='HDF5':
            try:
                import h5py
                f = h5py.File(filename,'w')
                for p,n in zip(plist,names):
                    n = n.replace('.','_')
                    p = param_to_array(p)
                    d = f.create_dataset(n,p.shape,dtype=p.dtype)
                    d[:] = p
                if hasattr(self, 'param_array'):
                    d = f.create_dataset('param_array',self.param_array.shape, dtype=self.param_array.dtype)
                    d[:] = self.param_array
                f.close()
            except:
                raise 'Fails to write the parameters into a HDF5 file!'

