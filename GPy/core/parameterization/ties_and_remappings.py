# Copyright (c) 2014, James Hensman, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .parameterized import Parameterized
from .param import Param

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
    
    self.label_buf:
    It uses a label buffer that has the same length as all the parameters (self._highest_parent_.param_array).
    The buffer keeps track of all the tied parameters. All the tied parameters have a label (an interger) higher 
    than 0, and the parameters that have the same label are tied together.
    
    self.buf_index:
    An auxiliary index list for the global index of the tie parameter inside the *Tie* object.
    
    ================================
    
    TODO:
    * EVERYTHING
    
    """
    def __init__(self, name='tie'):
        super(Tie, self).__init__(name)
        self.tied_param = None
        # The buffer keeps track of tie status
        self.label_buf = None
        # The global indices of the 'tied' param
        self.buf_idx = None
        # A boolean array indicating non-tied parameters
        self._tie_ = None
        
    def getTieFlag(self, p=None):
        if self.tied_param is None:
            if self._tie_ is None or self._tie_.size != self._highest_parent_.param_array.size:
                self._tie_ = np.ones((self._highest_parent_.param_array.size,),dtype=np.bool)
        if p is not None:
            return self._tie_[p._highest_parent_._raveled_index_for(p)]
        return self._tie_
    
    def _init_labelBuf(self):
        if self.label_buf is None:
            self.label_buf = np.zeros(self._highest_parent_.param_array.shape, dtype=np.int)
        if self._tie_ is None or self._tie_.size != self._highest_parent_.param_array.size:
            self._tie_ = np.ones((self._highest_parent_.param_array.size,),dtype=np.bool)
            
    def _updateTieFlag(self):
        if self._tie_.size != self.label_buf.size:
            self._tie_ = np.ones((self._highest_parent_.param_array.size,),dtype=np.bool)
        self._tie_[self.label_buf>0] = False
        self._tie_[self.buf_idx] = True

    def add_tied_parameter(self, p, p2=None):
        """
        Tie the list of parameters p together (p2==None) or 
        Tie the list of parameters p with the list of parameters p2 (p2!=None) 
        """
        self._init_labelBuf()
        if p2 is None:
            idx = self._highest_parent_._raveled_index_for(p)
            val = self._sync_val_group(idx)            
            if np.all(self.label_buf[idx]==0):
                # None of p has been tied before.
                tie_idx = self._expandTieParam(1)
                print(tie_idx)
                tie_id = self.label_buf.max()+1
                self.label_buf[tie_idx] = tie_id
            else:
                b = self.label_buf[idx]
                ids = np.unique(b[b>0])
                tie_id, tie_idx = self._merge_tie_param(ids)
            self._highest_parent_.param_array[tie_idx] = val
            idx = self._highest_parent_._raveled_index_for(p)
            self.label_buf[idx] = tie_id
        else:
            pass
        self._updateTieFlag()
        
    def _merge_tie_param(self, ids):
        """Merge the tie parameters with ids in the list."""
        if len(ids)==1:
            id_final_idx = self.buf_idx[self.label_buf[self.buf_idx]==ids[0]][0]
            return ids[0],id_final_idx
        id_final = ids[0]
        ids_rm = ids[1:]
        label_buf_param = self.label_buf[self.buf_idx]
        idx_param = [np.where(label_buf_param==i)[0][0] for i in ids_rm]
        self._removeTieParam(idx_param)
        [np.put(self.label_buf, np.where(self.label_buf==i), id_final) for i in ids_rm]
        id_final_idx = self.buf_idx[self.label_buf[self.buf_idx]==id_final][0]
        return id_final, id_final_idx
        
    def _sync_val_group(self, idx):
        self._highest_parent_.param_array[idx] = self._highest_parent_.param_array[idx].mean()
        return self._highest_parent_.param_array[idx][0]
        
    def _expandTieParam(self, num):
        """Expand the tie param with the number of *num* parameters"""
        if self.tied_param is None:
            new_buf = np.empty((num,))
        else:
            new_buf = np.empty((self.tied_param.size+num,))
            new_buf[:self.tied_param.size] = self.tied_param.param_array.copy()
            self.remove_parameter(self.tied_param)
        self.tied_param = Param('tied',new_buf)
        self.add_parameter(self.tied_param)
        buf_idx_new = self._highest_parent_._raveled_index_for(self.tied_param)
        self._expand_label_buf(self.buf_idx, buf_idx_new)
        self.buf_idx = buf_idx_new
        return self.buf_idx[-num:]

    def _removeTieParam(self, idx):
        """idx within tied_param"""
        new_buf = np.empty((self.tied_param.size-len(idx),))
        bool_list = np.ones((self.tied_param.size,),dtype=np.bool)
        bool_list[idx] = False
        new_buf[:] = self.tied_param.param_array[bool_list]
        self.remove_parameter(self.tied_param)
        self.tied_param = Param('tied',new_buf)
        self.add_parameter(self.tied_param)
        buf_idx_new = self._highest_parent_._raveled_index_for(self.tied_param)
        self._shrink_label_buf(self.buf_idx, buf_idx_new, bool_list)
        self.buf_idx = buf_idx_new
        
    def _expand_label_buf(self, idx_old, idx_new):
        """Expand label buffer accordingly"""
        if idx_old is None:
            self.label_buf = np.zeros(self._highest_parent_.param_array.shape, dtype=np.int)
        else:
            bool_old = np.zeros((self.label_buf.size,),dtype=np.bool)
            bool_old[idx_old] = True
            bool_new = np.zeros((self._highest_parent_.param_array.size,),dtype=np.bool)
            bool_new[idx_new] = True
            label_buf_new = np.zeros(self._highest_parent_.param_array.shape, dtype=np.int)
            label_buf_new[np.logical_not(bool_new)] = self.label_buf[np.logical_not(bool_old)]
            label_buf_new[idx_new[:len(idx_old)]] = self.label_buf[idx_old]
            self.label_buf = label_buf_new

    def _shrink_label_buf(self, idx_old, idx_new, bool_list):
        bool_old = np.zeros((self.label_buf.size,),dtype=np.bool)
        bool_old[idx_old] = True
        bool_new = np.zeros((self._highest_parent_.param_array.size,),dtype=np.bool)
        bool_new[idx_new] = True
        label_buf_new = np.empty(self._highest_parent_.param_array.shape, dtype=np.int)
        label_buf_new[np.logical_not(bool_new)] = self.label_buf[np.logical_not(bool_old)]
        label_buf_new[idx_new] = self.label_buf[idx_old[bool_list]]
        self.label_buf = label_buf_new

    def _check_change(self):
        changed = False
        if self.tied_param is not None:
            for i in range(self.tied_param.size):
                b0 = self.label_buf==self.label_buf[self.buf_idx[i]]
                b = self._highest_parent_.param_array[b0]!=self.tied_param[i]
                if b.sum()==0:
                    print('XXX')
                    continue
                elif b.sum()==1:
                    print('!!!')
                    val = self._highest_parent_.param_array[b0][b][0]
                    self._highest_parent_.param_array[b0] = val
                else:
                    print('@@@')
                    self._highest_parent_.param_array[b0] = self.tied_param[i]
                changed = True
        return changed

    def parameters_changed(self):
        #ensure all out parameters have the correct value, as specified by our mapping
        changed = self._check_change()
        if changed:
            self._highest_parent_._trigger_params_changed()
        self.collate_gradient()

    def collate_gradient(self):
        if self.tied_param is not None:
            self.tied_param.gradient = 0.
            [np.put(self.tied_param.gradient, i, self._highest_parent_.gradient[self.label_buf==self.label_buf[self.buf_idx[i]]].sum()) 
                for i in range(self.tied_param.size)]
    
    def propagate_val(self):
        if self.tied_param is not None:
            for i in range(self.tied_param.size):
                self._highest_parent_.param_array[self.label_buf==self.label_buf[self.buf_idx[i]]] = self.tied_param[i]





