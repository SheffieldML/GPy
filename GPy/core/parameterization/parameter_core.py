# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from transformations import Transformation, Logexp, NegativeLogexp, Logistic, __fixed__, FIXED, UNFIXED

__updated__ = '2013-12-16'

def adjust_name_for_printing(name):
    if name is not None:
        return name.replace(" ", "_").replace(".", "_").replace("-","").replace("+","").replace("!","").replace("*","").replace("/","")
    return ''

class Observable(object):
    _observers_ = {}
    def add_observer(self, observer, callble):
        self._observers_[observer] = callble
        #callble(self)
    def remove_observer(self, observer):
        del self._observers_[observer]
    def _notify_observers(self):
        [callble(self) for callble in self._observers_.itervalues()]
    
class Pickleable(object):
    def _getstate(self):
        """
        Returns the state of this class in a memento pattern.
        The state must be a list-like structure of all the fields
        this class needs to run.

        See python doc "pickling" (`__getstate__` and `__setstate__`) for details.
        """
        raise NotImplementedError, "To be able to use pickling you need to implement this method"
    def _setstate(self, state):
        """
        Set the state (memento pattern) of this class to the given state.
        Usually this is just the counterpart to _getstate, such that
        an object is a copy of another when calling

            copy = <classname>.__new__(*args,**kw)._setstate(<to_be_copied>._getstate())

        See python doc "pickling" (`__getstate__` and `__setstate__`) for details.
        """
        raise NotImplementedError, "To be able to use pickling you need to implement this method"

#===============================================================================
# Foundation framework for parameterized and param objects:
#===============================================================================

class Parentable(object):
    def __init__(self, direct_parent=None, parent_index=None):
        super(Parentable,self).__init__()
        self._direct_parent_ = direct_parent
        self._parent_index_ = parent_index
        
    def has_parent(self):
        return self._direct_parent_ is not None

    def _notify_parent_change(self):
        for p in self._parameters_:
            p._parent_changed(self)

    def _parent_changed(self):
        raise NotImplementedError, "shouldnt happen, Parentable objects need to be able to change their parent"

    @property
    def _highest_parent_(self):
        if self._direct_parent_ is None:
            return self
        return self._direct_parent_._highest_parent_

class Nameable(Parentable):
    _name = None
    def __init__(self, name, direct_parent=None, parent_index=None):
        super(Nameable,self).__init__(direct_parent, parent_index)
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        from_name = self.name
        self._name = name
        if self.has_parent():
            self._direct_parent_._name_changed(self, from_name)    


class Parameterizable(Parentable):
    def __init__(self, *args, **kwargs):
        super(Parameterizable, self).__init__(*args, **kwargs)
        from GPy.core.parameterization.array_core import ParamList
        _parameters_ = ParamList()
    
    def parameter_names(self, add_name=False):
        if add_name:
            return [adjust_name_for_printing(self.name) + "." + xi for x in self._parameters_ for xi in x.parameter_names(add_name=True)]
        return [xi for x in self._parameters_ for xi in x.parameter_names(add_name=True)]

    def _collect_gradient(self, target):
        import itertools
        [p._collect_gradient(target[s]) for p, s in itertools.izip(self._parameters_, self._param_slices_)]

    def _get_params(self):
        import numpy as np
        # don't overwrite this anymore!
        if not self.size:
            return np.empty(shape=(0,), dtype=np.float64)
        return np.hstack([x._get_params() for x in self._parameters_ if x.size > 0])

    def _set_params(self, params, update=True):
        # don't overwrite this anymore!
        import itertools
        [p._set_params(params[s], update=update) for p, s in itertools.izip(self._parameters_, self._param_slices_)]
        self.parameters_changed()

    def parameters_changed(self):
        """
        This method gets called when parameters have changed.
        Another way of listening to param changes is to
        add self as a listener to the param, such that
        updates get passed through. See :py:function:``GPy.core.param.Observable.add_observer``
        """
        pass

    def _notify_parameters_changed(self):
        self.parameters_changed()
        if self.has_parent():
            self._direct_parent_._notify_parameters_changed()


class Gradcheckable(Parentable):
    #===========================================================================
    # Gradchecking
    #===========================================================================
    def checkgrad(self, verbose=0, step=1e-6, tolerance=1e-3):
        if self.has_parent():
            return self._highest_parent_._checkgrad(self, verbose=verbose, step=step, tolerance=tolerance)
        return self._checkgrad(self[''], verbose=verbose, step=step, tolerance=tolerance)
    def _checkgrad(self, param):
        raise NotImplementedError, "Need log likelihood to check gradient against"

class Indexable(object):
    def _raveled_index(self):
        raise NotImplementedError, "Need to be able to get the raveled Index"
        
    def _internal_offset(self):
        return 0
    
    def _offset_for(self, param):
        raise NotImplementedError, "shouldnt happen, offset required from non parameterization object?"
    
    def _raveled_index_for(self, param):
        """
        get the raveled index for a param
        that is an int array, containing the indexes for the flattened
        param inside this parameterized logic.
        """
        raise NotImplementedError, "shouldnt happen, raveld index transformation required from non parameterization object?"        
        
class Constrainable(Nameable, Indexable, Parameterizable):
    def __init__(self, name, default_constraint=None):
        super(Constrainable,self).__init__(name)
        self._default_constraint_ = default_constraint
        from index_operations import ParameterIndexOperations
        self.constraints = ParameterIndexOperations()
        self.priors = ParameterIndexOperations()
        if self._default_constraint_ is not None:
            self.constrain(self._default_constraint_)
    
    #===========================================================================
    # Fixing Parameters:
    #===========================================================================
    def constrain_fixed(self, value=None, warning=True):
        """
        Constrain this paramter to be fixed to the current value it carries.

        :param warning: print a warning for overwriting constraints.
        """
        if value is not None:
            self[:] = value
        self.constrain(__fixed__, warning=warning)
        rav_i = self._highest_parent_._raveled_index_for(self)
        self._highest_parent_._set_fixed(rav_i)
    fix = constrain_fixed
    
    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        unconstrained = self.unconstrain(__fixed__)
        self._highest_parent_._set_unfixed(unconstrained)    
    unfix = unconstrain_fixed
    
    def _set_fixed(self, index):
        import numpy as np
        if not self._has_fixes(): self._fixes_ = np.ones(self.size, dtype=bool)
        self._fixes_[index] = FIXED
        if np.all(self._fixes_): self._fixes_ = None  # ==UNFIXED
    
    def _set_unfixed(self, index):
        import numpy as np
        if not self._has_fixes(): self._fixes_ = np.ones(self.size, dtype=bool)
        #rav_i = self._raveled_index_for(param)[index]
        self._fixes_[index] = UNFIXED
        if np.all(self._fixes_): self._fixes_ = None  # ==UNFIXED

    def _connect_fixes(self):
        import numpy as np
        fixed_indices = self.constraints[__fixed__]
        if fixed_indices.size > 0:
            self._fixes_ = np.ones(self.size, dtype=bool) * UNFIXED
            self._fixes_[fixed_indices] = FIXED
        else:
            self._fixes_ = None
    
    def _has_fixes(self):
        return hasattr(self, "_fixes_") and self._fixes_ is not None

    #===========================================================================
    # Prior Operations
    #===========================================================================
    def set_prior(self, prior, warning=True, update=True):
        repriorized = self.unset_priors()
        self._add_to_index_operations(self.priors, repriorized, prior, warning, update)
    
    def unset_priors(self, *priors):
        return self._remove_from_index_operations(self.priors, priors)
    
    def log_prior(self):
        """evaluate the prior"""
        import numpy as np
        if self.priors.size > 0:
            x = self._get_params()
            return np.sum([p.lnpdf(x[ind]) for p, ind in self.priors.iteritems()])
        return 0.
    
    def _log_prior_gradients(self):
        """evaluate the gradients of the priors"""
        import numpy as np
        if self.priors.size > 0:
            x = self._get_params()
            ret = np.zeros(x.size)
            [np.put(ret, ind, p.lnpdf_grad(x[ind])) for p, ind in self.priors.iteritems()]
            return ret
        return 0.
        
    #===========================================================================
    # Constrain operations -> done
    #===========================================================================

    def constrain(self, transform, warning=True, update=True):
        """
        :param transform: the :py:class:`GPy.core.transformations.Transformation`
                          to constrain the this parameter to.
        :param warning: print a warning if re-constraining parameters.

        Constrain the parameter to the given
        :py:class:`GPy.core.transformations.Transformation`.
        """
        if isinstance(transform, Transformation):
            self._set_params(transform.initialize(self._get_params()), update=False)
        reconstrained = self.unconstrain()
        self._add_to_index_operations(self.constraints, reconstrained, transform, warning, update)

    def unconstrain(self, *transforms):
        """
        :param transforms: The transformations to unconstrain from.

        remove all :py:class:`GPy.core.transformations.Transformation`
        transformats of this parameter object.
        """
        return self._remove_from_index_operations(self.constraints, transforms)
    
    def constrain_positive(self, warning=True, update=True):
        """
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to the default positive constraint.
        """
        self.constrain(Logexp(), warning=warning, update=update)

    def constrain_negative(self, warning=True, update=True):
        """
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to the default negative constraint.
        """
        self.constrain(NegativeLogexp(), warning=warning, update=update)

    def constrain_bounded(self, lower, upper, warning=True, update=True):
        """
        :param lower, upper: the limits to bound this parameter to
        :param warning: print a warning if re-constraining parameters.

        Constrain this parameter to lie within the given range.
        """
        self.constrain(Logistic(lower, upper), warning=warning, update=update)

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
        from index_operations import ParameterIndexOperationsView
        self.constraints = ParameterIndexOperationsView(parent.constraints, parent._offset_for(self), self.size)
        self.priors = ParameterIndexOperationsView(parent.priors, parent._offset_for(self), self.size)
        self._fixes_ = None
        for p in self._parameters_:
            p._parent_changed(parent)

    def _add_to_index_operations(self, which, reconstrained, transform, warning, update):
        if warning and reconstrained.size > 0:
            print "WARNING: reconstraining parameters {}".format(self.parameter_names() or self.name)
        which.add(transform, self._raveled_index())
        if update:
            self._notify_parameters_changed()

    def _remove_from_index_operations(self, which, transforms):
        if len(transforms) == 0:
            transforms = which.properties()
        import numpy as np
        removed = np.empty((0, ), dtype=int)
        for t in transforms:
            unconstrained = which.remove(t, self._raveled_index())
            removed = np.union1d(removed, unconstrained)
            if t is __fixed__:
                self._highest_parent_._set_unfixed(unconstrained)
        
        return removed




