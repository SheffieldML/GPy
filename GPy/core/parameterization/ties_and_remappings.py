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


class Tie(Parameterized):
    """
    The new parameter tie framework. (under development)
    
    All the parameters tied together get a new parameter inside the *Tie* object. 
    Its value should always be equal to all the tied parameters, and its gradient
    is the sum of all the tied parameters.
    
    =====Implementation Details=====
    The *Tie* object should only exist on the top of param tree (the highest parent).
    
    Each tied param object has the attribute _tie_ which stores the labels for tied parameters.
    
    self.label_buf:
    It uses a label buffer that has the same length as all the parameters (self._highest_parent_.param_array).
    The buffer keeps track of all the tied parameters. All the tied parameters have a label (an interger) higher 
    than 0, and the parameters that have the same label are tied together.
    
    self.buf_index:
    An auxiliary index list for the global index of the tie parameter inside the *Tie* object.
    
    ================================
    
    TODO:
    1. Add the support for multiple parameter tie_together and tie_vector
    2. Properly handling parameters with constraints
    3. Properly handling the merging of two models
    4. Properly handling initialization
    
    """
    def __init__(self, name='Ties'):
        # whether it has just propagated tied parameter values during optimization
        # If ture, it does not need to check consistency
        self._PROPAGATE_VAL_ = False
        super(Tie, self).__init__(name)
        self.size = sum(p.size for p in self.parameters)
        self.tied_param = None
        # The buffer keeps track of tie status
        self.label_buf = None
        self.buf_idx = None
        
    def mergeTies(self, p):
        """Merge the tie tree with another tie tree"""
        assert hasattr(p,'ties') and isinstance(p.ties,Tie)
        self.updates = False
        if p.ties.tied_param is not None:
            tie_labels = self._expand_tie_param(p.ties.tied_param.size)
            self.tied_param[-p.ties.tied_param.size:] = p.ties.tied_param
            pairs = zip(self.tied_param.tie,tie_labels)
            self._replace_labels(p, pairs)
        p.remove_observer(p.ties)
        p.remove_parameter(p.ties)
        del p.ties
        self._update_label_buf()
        self.updates = True
        
    def _sync_val_group(self, plist):
        val = np.asarray([p.param_array for p in plist]).mean()
        def _set_val(p):
            p[:] = val
        for p in plist:
            self._traverse_param(_set_val, p, [])
        return val
        
    def _traverse_param(self, func, p, res):
        """
        Traverse a param tree starting with *p*
        Apply *func* to every leaves (param objects),
        and collect return values into *res*
        """
        if isinstance(p, Param):
            res.append(func(p))
        else:
            for pc in p.parameters:
                self._traverse_param(func,pc,res)

    def _get_labels(self, plist):
        labels = []
        for p in plist:
            self._traverse_param(lambda x: x.tie, p, labels)
        return np.unique(np.asarray(labels))
    
    def _set_labels(self, plist, labels):
        """
        If there is only one label, set all the param objects to that label,
        otherwise each parameter take a label.
        """
        def _set_l1(p):
            p.tie[:] = labels[0]
        if len(labels)==1:
            for p in plist:
                self._traverse_param(_set_l1, p, [])
    
    def _replace_labels(self, p, label_pairs):
        def _replace_l(p):
            for l1,l2 in label_pairs:
                p.tie[p.tie==l1] = l2
        self._traverse_param(_replace_l, p, [])

    def _expand_tie_param(self, num):
        """Expand the tie param with the number of *num* parameters"""
        if self.tied_param is None:
            start_label = 1
            new_buf = np.empty((num,))
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:] = range(start_label,start_label+num)
        else:
            start_label = self.tied_param.tie.max()+1
            new_buf = np.empty((self.tied_param.size+num,))
            new_buf[:self.tied_param.size] = self.tied_param.param_array.copy()
            old_tie_ = self.tied_param.tie.copy()
            old_size = self.tied_param.size
            self.remove_parameter(self.tied_param)
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:old_size] = old_tie_
            self.tied_param.tie[old_size:] = range(start_label,start_label+num)
        self.add_parameter(self.tied_param)
        self.size = sum(p.size for p in self.parameters)
        return range(start_label,start_label+num)

    def _remove_tie_param(self, labels):
        """Remove the tie param corresponding to *labels*"""
        if len(labels) == self.tied_param.size:
            self.remove_parameter(self.tied_param)
            self.tied_param = None
        else:
            new_buf = np.empty((self.tied_param.size-len(labels),))
            idx = np.logical_not(np.in1d(self.tied_param.tie,labels))
            new_buf[:] = self.tied_param[idx]
            old_tie_ = self.tied_param.tie.copy()
            self.remove_parameter(self.tied_param)            
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:] = old_tie_[idx]
            self.add_parameter(self.tied_param)
        self.size = sum(p.size for p in self.parameters)
    
    def _merge_tie_labels(self, labels):
        """Merge all the labels in the list to the first one"""
        if len(labels)<2:
            return
        self._remove_tie_param(labels[1:])
        self._replace_labels(self._highest_parent_, [(l,labels[0]) for l in labels[1:]])

    def _update_label_buf(self):
        if self.tied_param is None:
            self.label_buf = None
            self.buf_idx = None
        else:
            self.label_buf = np.zeros((self._highest_parent_.size,),dtype=np.uint32)
            self._traverse_param(lambda x:np.put(self.label_buf,self._highest_parent_._raveled_index_for(x),x.tie), self._highest_parent_, [])
            self.buf_idx = self._highest_parent_._raveled_index_for(self.tied_param)
        
    def tie_together(self,plist):
        """tie a list of parameters"""
        self.updates = False
        labels = self._get_labels(plist)
        val = self._sync_val_group(plist)
        if labels[0]==0 and labels.size==1:
            # None of parameters in plist has been tied before.
            tie_labels = self._expand_tie_param(1)
            self._set_labels(plist, tie_labels)
        else:
            # Some of parameters has been tied already.
            # Merge the tie param
            tie_labels = labels[labels>0]
            if tie_labels.size>1:
                self._merge_tie_labels(tie_labels)
            self._set_labels(plist, [tie_labels[0]])
        self._update_label_buf()
        self.tied_param[self.tied_param.tie==tie_labels[0]] = val
        self.updates = True

    def _check_change(self):
        changed = False
        if self.tied_param is not None:
            for i in xrange(self.tied_param.size):
                b0 = self.label_buf==self.label_buf[self.buf_idx[i]]
                b = self._highest_parent_.param_array[b0]!=self.tied_param[i]
                if b.sum()==0:
                    # All the tied parameters are the same
                    continue
                elif b.sum()==1:
                    # One of the tied parameter is different.
                    # It must be recently changed one.
                    # The rest will be set to its value.
                    val = self._highest_parent_.param_array[b0][b][0]
                    self._highest_parent_.param_array[b0] = val
                else:
                    # It is most likely that the tie parameter is changed.
                    # Set all the tied parameter to the value of tie parameter.
                    self._highest_parent_.param_array[b0] = self.tied_param[i]
                changed = True
        return changed
    
    def _parameters_changed_notification(self, me, which=None):
        if which is not self:
            self._optimizer_copy_transformed = False # tells the optimizer array to update on next request
            self.parameters_changed()

    def parameters_changed(self):
        #ensure all out parameters have the correct value, as specified by our mapping
        if self._PROPAGATE_VAL_:
            self._PROPAGATE_VAL_ = False
        else:
            if self._check_change():
                self._highest_parent_._trigger_params_changed()
        self.collate_gradient()

    def collate_gradient(self):
        if self.tied_param is not None:
            self.tied_param.gradient = 0.
            [np.put(self.tied_param.gradient, i, self._highest_parent_.gradient[self.label_buf==self.label_buf[self.buf_idx[i]]].sum()) 
                for i in xrange(self.tied_param.size)]
    
    def propagate_val(self):
        if self.tied_param is not None:
            for i in xrange(self.tied_param.size):
                self._highest_parent_.param_array[self.label_buf==self.label_buf[self.buf_idx[i]]] = self.tied_param[i]
        self._PROPAGATE_VAL_ = True



