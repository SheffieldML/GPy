# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from transformations import Logexp, NegativeLogexp, Logistic

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
        self._highest_parent_ = highest_parent

    def has_parent(self):
        return self._direct_parent_ is not None
    
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

class Constrainable(Nameable):
    def __init__(self, name):
        super(Constrainable,self).__init__(name)
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
        self._highest_parent_._fix(self,warning)
    fix = constrain_fixed
    def unconstrain_fixed(self):
        """
        This parameter will no longer be fixed.
        """
        self._highest_parent_._unfix(self)
    unfix = unconstrain_fixed


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
        if self.has_parent():
            self._highest_parent_._add_constrain(self, transform, warning)
            if update:
                self._highest_parent_.parameters_changed()
        else:
            for p in self._parameters_:
                self._add_constrain(p, transform, warning)
            if update:
                self.parameters_changed()

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

    def unconstrain(self, *transforms):
        """
        :param transforms: The transformations to unconstrain from.

        remove all :py:class:`GPy.core.transformations.Transformation`
        transformats of this parameter object.
        """
        if self.has_parent():
            self._highest_parent_._remove_constrain(self, *transforms)
        else:
            for p in self._parameters_:
                self._remove_constrain(p, *transforms)

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
