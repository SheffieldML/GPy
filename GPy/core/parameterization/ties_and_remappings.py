# Copyright (c) 2014, James Hensman, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from parameterized import Parameterized
from param import Param

class Remapping(Parameterized):
    def mapping(self):
        """
        The return value of this function gives the values which the re-mapped
        parameters should take. Implement in sub-classes.
        """
        raise NotImplementedError

    def callback(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def parameters_changed(self):
        #ensure all out parameters have the correct value, as specified by our mapping
        index = self._highest_parent_.constraints[self]
        self._highest_parent_.param_array[index] = self.mapping()
        [p.notify_observers(which=self) for p in self.tied_parameters]

class Fix(Remapping):
    pass




class Tie(Remapping):
    def __init__(self, value, name):
        super(Tie, self).__init__(name)
        self.tied_parameters = []
        self.value = Param('val', value)
        self.add_parameter(self.value)

    def add_tied_parameter(self, p):
        self.tied_parameters.append(p)
        p.add_observer(self, self.callback)
        self.parameters_changed()

    def callback(self, param=None, which=None):
        """
        This gets called whenever any of the tied parameters changes. we spend
        considerable effort working out what has changed and to what value.
        Then we store that value in self.value, and broadcast it everywhere
        with parameters_changed.
        """
        if which is self:return
        index = self._highest_parent_.constraints[self]
        if len(index)==0:
            return # nothing to tie together, this tie exists without any tied parameters
        self.collate_gradient()
        vals = self._highest_parent_.param_array[index]
        uvals = np.unique(vals)
        if len(uvals)==1:
            #all of the tied things are at the same value
            if (self.value==uvals[0]).all():
                return # DO NOT DO ANY CHANGES IF THE TIED PART IS NOT CHANGED!
            self.value[...] = uvals[0]
        elif len(uvals)==2:
            #only *one* of the tied things has changed. it must be different to self.value
            newval = uvals[uvals != self.value*1]
            self.value[...] = newval
        else:
            #more than one of the tied things changed. panic.
            raise ValueError, "something is wrong with the tieing"
    def parameters_changed(self):
        #ensure all out parameters have the correct value, as specified by our mapping
        index = self._highest_parent_.constraints[self]
        if (self._highest_parent_.param_array[index]==self.value).all():
            return # STOP TRIGGER THE UPDATE LOOP MULTIPLE TIMES!!!
        self._highest_parent_.param_array[index] = self.mapping()
        [p.notify_observers(which=self) for p in self.tied_parameters]
        self.collate_gradient()

    def mapping(self):
        return self.value

    def collate_gradient(self):
        index = self._highest_parent_.constraints[self]
        self.value.gradient = np.sum(self._highest_parent_.gradient[index])





