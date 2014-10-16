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
    1. Add the support for multiple parameter tie_together and tie_vector [Preliminary]
    2. Properly handling parameters with constraints [DONE]
    3. Properly handling the merging of two models [DONE]
    4. Properly handling initialization [DONE]
    
    """
    def __init__(self, name='Ties'):
        # whether it has just propagated tied parameter values during optimization
        # If ture, it does not need to check consistency
        self._PROPAGATE_VAL_ = False
        super(Tie, self).__init__(name)
        self.tied_param = None
        # The buffer keeps track of tie status
        self.label_buf = None
        self.buf_idx = None
        self._untie_ = None
        
    def getTiedParamList(self):
        if self.tied_param is None:
            return []
        labels = np.unique(self.label_buf)
        labels = labels[labels>0]
        return [np.where(self.label_buf==l)[0] for l in labels]
            
    def _sync_val(self, plist, toTiedParam=True):
        """
        Ensure the consistency of the values of tied parameters.
        if toTieParam is true, the values of tied parameters will be synchronized among themselves and to the *TiedParam*,
        otherwise all the tied parameters will be synchronized according to the values in *TiedParam*.
        """
        assert self.tied_param is not None
        read = np.zeros((self.tied_param.size,),dtype=np.uint8)
        def _sync_val_p(p, tieparam, read):
            if p.tie is not None:
                labels = np.unique(p.tie)
                labels = labels[labels>0]
                for l in labels:
                    if toTiedParam and read[tieparam.tie==l] == 0:
                        val = p[p.tie==l][0]
                        tieparam[tieparam.tie==l] = val
                        read[tieparam.tie==l] = 1
                    else:
                        val = tieparam[tieparam.tie==l]                        
                    p[p.tie==l] = val
        for p in plist:
            self._traverse_param(_sync_val_p, (p,self.tied_param,read), [])
            
    def _sync_constraints(self, plist, toTiedParam=True):
        """
        Ensure the consistency of the constraints of tied parameters.
        if toTieParam is true, the constraints of tied parameters will be synchronized among themselves and to the *TiedParam*,
        otherwise all the tied parameters will be synchronized according to the constraints in *TiedParam*.
        """
        assert self.tied_param is not None
        read = np.zeros((self.tied_param.size,),dtype=np.uint8)
        cons = [c[0] if len(c)>0 else None for c in self.tied_param.constraints.properties_for(range(self.tied_param.size))]
        def _sync_constraints_p(p, tieparam, read, cons):
            if p.tie is not None:
                p = p._original_
                if p is tieparam:
                    return
                labels = np.unique(p.tie)
                labels = labels[labels>0]
                for l in labels:
                    idx = np.where(tieparam.tie==l)[0][0]
                    conslist = p.constraints.properties_to_index_dict(np.where(p.tie.flat==l)[0])
                    if toTiedParam and read[tieparam.tie==l] == 0:
                        if len(conslist)==0:
                            if cons[idx] is not None:
                                tieparam[idx:idx+1].unconstrain()
                        else:
                            c = conslist.keys()[0]
                            if len(conslist)>1 or len(conslist[c])!= (p.tie==l).sum():
                                p[p.tie==l].constrain(c)
                            if c != cons[idx]:
                                tieparam[idx:idx+1].constrain(c)
                                cons[idx] = c
                        read[tieparam.tie==l] = 1
                    else:
                        if cons[idx] is None:
                            if len(conslist)>0:
                                p[p.tie==l].unconstrain()
                        else:
                            if len(conslist)!=1 or conslist.keys()[0]!=cons[idx] or len(conslist[cons[idx]])!= (p.tie==l).sum():
                                p[p.tie==l].constrain(cons[idx])
        for p in plist:
            self._traverse_param(_sync_constraints_p, (p,self.tied_param,read, cons), [])        
        
        
    @staticmethod
    def recoverTies(p):
        """Recover the Tie object from the param objects"""
        if not p.has_parent():
            p.ties = Tie()
            p.link_parameter(p.ties, -1)
            
            p.update_model(False)
            labels = p.ties._get_labels([p])
            labels = labels[labels>0]
            if len(labels)>0:
                p.ties._expand_tie_param(len(labels))
                p.ties.tied_param.tie[:] = labels
                p.ties._sync_val([p],toTiedParam=True)
                p.ties._sync_constraints([p], toTiedParam=True)
            p.ties._update_label_buf()
            p.add_observer(p.ties, p.ties._parameters_changed_notification, priority=-500)
            p.update_model(True)

    def mergeTies(self, p):
        """Merge the tie tree with another tie tree"""
        assert hasattr(p,'ties') and isinstance(p.ties,Tie), str(type(p))
        #self.update_model(False)
        if p.ties.tied_param is not None:
            tie_labels,_ = self._expand_tie_param(p.ties.tied_param.size)
            self.tied_param[-p.ties.tied_param.size:] = p.ties.tied_param
            pairs = zip(self.tied_param.tie,tie_labels)
            self._replace_labels(p, pairs)
            self._sync_constraints([self._parent_], toTiedParam=True)
        p.remove_observer(p.ties)
        p.unlink_parameter(p.ties)
        del p.ties
        self._update_label_buf()
        #self.update_model(True)

    def splitTies(self, p):
        """Split the tie subtree from the original tie tree"""
        p.ties = Tie()
        p.link_parameter(p.ties, -1)
        p.add_observer(p.ties, p.ties._parameters_changed_notification, priority=-500)
        if self.tied_param is not None:
            self.update_model(False)
            labels = self._get_labels([p])
            labels = labels[labels>0]
            if len(labels)>0:
                Tie.recoverTies(p)
#                 p._expand_tie_param(len(labels))
#                 idx = np.in1d(self.tied_param.tie,labels)
#                 p.tied_param[:] = self.tied_param[idx]
#                 p.tied_param.tie[:] = self.tied_param.tie[idx]
            self._remove_unnecessary_ties()
            self._sync_constraints([self._parent_], toTiedParam=True)
            self._update_label_buf()
            p.ties._update_label_buf()
            self.update_model(True)
        
    def _traverse_param(self, func, p, res):
        """
        Traverse a param tree starting with *p*
        Apply *func* to every leaves (param objects),
        and collect return values into *res*
        """
        if isinstance(p[0], Param):
            res.append(func(*p))
        else:
            for pc in p[0].parameters:
                self._traverse_param(func, (pc,)+p[1:] ,res)

    def _get_labels(self, plist):
        labels = []
        for p in plist:
            self._traverse_param(lambda x: x.tie.flat, (p,), labels)
        return np.unique(np.hstack(labels))
    
    def _get_labels_vector(self, p1,p2):
        label1 = []
        self._traverse_param(lambda x: x.tie.flat, (p1,), label1)
        label1 = np.hstack(label1)
        label2 = []
        self._traverse_param(lambda x: x.tie.flat, (p2,), label2)
        label2 = np.hstack(label2)
        expandlist = np.where(label1+label2==0)[0]
        labellist =label1.copy()
        idx = np.logical_and(label1==0,label2>0)
        labellist[idx] = label2[idx]
        idx = np.logical_and(label1*label2>0,label1!=label2)
        removelist = (label1[idx],label2[idx])
        return expandlist,removelist,labellist
    
    def _set_labels(self, plist, labels):
        """
        If there is only one label, set all the param objects to that label,
        otherwise each parameter take a label.
        """
        def _set_l1(p):
            p.tie[:] = labels[0]
        def _set_list(p, offset):
            p.tie.flat[:] = labels[offset[0]:offset[0]+p.size]
            offset[0] = offset[0]+ p.size
        if len(labels)==1:
            for p in plist:
                self._traverse_param(_set_l1, (p,), [])
        else:
            idx = [0]
            for p in plist:
                self._traverse_param(_set_list, (p,idx), [])

    def _replace_labels(self, p, label_pairs):
        def _replace_l(p):
            for l1,l2 in label_pairs:
                p.tie[p.tie==l1] = l2
        self._traverse_param(_replace_l, (p,), [])

    def _expand_tie_param(self, num):
        """Expand the tie param with the number of *num* parameters"""
        if self.tied_param is None:
            start_label = 1
            labellist = np.array(range(start_label,start_label+num),dtype=np.int)
            idxlist = np.array(range(0,num),dtype=np.int)
            new_buf = np.empty((num,))
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:] = labellist
            self.link_parameter(self.tied_param)
        else:
            start_label = self.tied_param.tie.max()+1
            new_buf = np.empty((self.tied_param.size+num,))
            new_buf[:self.tied_param.size] = self.tied_param.param_array.copy()
            old_tie_ = self.tied_param.tie.copy()
            old_size = self.tied_param.size
            labellist = np.array(range(start_label,start_label+num),dtype=np.int)
            idxlist = np.array(range(old_size,old_size+num),dtype=np.int)
            cons = self.tied_param.constraints.copy()
            self.unlink_parameter(self.tied_param)
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:old_size] = old_tie_
            self.tied_param.tie[old_size:] = labellist
            self.link_parameter(self.tied_param)
            self.tied_param.constraints.update(cons)
        return labellist, idxlist

    def _remove_tie_param(self, labels):
        """Remove the tie param corresponding to *labels*"""
        if len(labels) == self.tied_param.size:
            self.unlink_parameter(self.tied_param)
            self.tied_param = None
        else:
            new_buf = np.empty((self.tied_param.size-len(labels),))
            idx = np.logical_not(np.in1d(self.tied_param.tie,labels))
            new_buf[:] = self.tied_param[idx]
            old_tie_ = self.tied_param.tie.copy()
            cons = {}
            for c,ind in self.tied_param.constraints.iteritems():
                buf = np.zeros((old_tie_.size,),dtype=np.uint8)
                buf[ind] = 1
                if (buf[idx]==1).sum()>0:
                    cons[c] = np.where(buf[idx]==1)[0]
            self.unlink_parameter(self.tied_param)
            self.tied_param = Param('tied',new_buf)
            self.tied_param.tie[:] = old_tie_[idx]
            self.link_parameter(self.tied_param)
            [self.tied_param.constraints.add(c,ind) for c,ind in cons.iteritems()]
    
    def _merge_tie_labels(self, labels):
        """Merge all the labels in the list to the first one"""
        if len(labels)<2:
            return
        self._remove_tie_param(labels[1:])
        self._replace_labels(self._highest_parent_, [(l,labels[0]) for l in labels[1:]])

    def _merge_tie_labelpair(self, labelpair):
        """Merge the second list in labelpair to the first list"""
        self._remove_tie_param(labelpair[1])
        self._replace_labels(self._highest_parent_, zip(labelpair[1],labelpair[0]))
        
    def _remove_unnecessary_ties(self):
        """Remove the unnecessary ties"""
        if self.tied_param is not None:
            labels = [l for l in self.tied_param.tie if (self.label_buf==l).sum()<=2]
            if len(labels)>0:
                self._remove_tie_param(labels)
                self._replace_labels(self._highest_parent_, zip(labels,[0]*len(labels)))

    def _update_label_buf(self):
        if self.tied_param is None:
            self.label_buf = None
            self.buf_idx = None
            self._untie_ = None
        else:
            self.label_buf = np.zeros((self._highest_parent_.param_array.size,),dtype=np.uint32)
            self._traverse_param(lambda x:np.put(self.label_buf,self._highest_parent_._raveled_index_for(x),x.tie.flat), (self._highest_parent_,), [])
            self.buf_idx = self._highest_parent_._raveled_index_for(self.tied_param)
            self._untie_ = self.label_buf==0
            self._untie_[self.buf_idx] = True
            assert(np.all(self.tied_param.tie>0))
            
    def _keepParamList(self,plist):
        paramlist = []
        for p in plist:
            self._traverse_param(lambda p: (p._original_, p._current_slice_), (p,), paramlist)
        return paramlist
    
    def _updateParamList(self, p_split):
        return [p_org[p_slice] for p_org,p_slice in p_split]

    def tie_together(self,plist):
        """tie a list of parameters"""        
        self.update_model(False)
        labels = self._get_labels(plist)
        if labels[0]==0 and labels.size==1:
            # None of parameters in plist has been tied before.
            p_split = self._keepParamList(plist)
            tie_labels,_ = self._expand_tie_param(1)
            plist = self._updateParamList(p_split)
            self._set_labels(plist, tie_labels)
            toTiedParam = True
        else:
            # Some of parameters has been tied already.
            # Merge the tie param
            tie_labels = labels[labels>0]
            if tie_labels.size>1:
                self._merge_tie_labels(tie_labels)
            self._set_labels(plist, [tie_labels[0]])
            toTiedParam = False
        self._sync_val(plist,toTiedParam)
        self._sync_constraints(plist, toTiedParam)
        self._update_label_buf()
        self.update_model(True)
    
    def tie_vector(self, plist):
        p_splits = [self._keepParamList([p]) for p in plist]
                
    def _tie_vector(self, p1, p2):
        """tie a pair of vectors"""
        self.update_model(False)        
        expandlist,removelist,labellist = self._get_labels_vector(p1, p2)
        p_split1 = self._keepParamList([p1])
        p_split2 = self._keepParamList([p2])
        if len(expandlist)>0:
            tie_labels,idxlist = self._expand_tie_param(len(expandlist))
            labellist[expandlist] = tie_labels
        if len(removelist[0])>0:
            self._merge_tie_labelpair(removelist)
        p1 = self._updateParamList(p_split1)
        p2 = self._updateParamList(p_split2)
        ps = p1+p2
        self._set_labels(p1, labellist)
        self._set_labels(p2, labellist)
        self._sync_val(ps,toTiedParam=True)
        self._sync_constraints(ps, toTiedParam=True)
        self._update_label_buf()
        self.update_model(True)
        
    def untie(self,plist):
        """Untie a list of parameters"""
        self.update_model(False)
        self._set_labels(plist,[0])
        self._update_label_buf()
        self._remove_unnecessary_ties()
        self._update_label_buf()
        self.update_model(True)
        
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
        
    #=========================================
    # Functions for checking consistency
    #=========================================
         
    def checkValueConsistency(self):
        return not self._check_change()
    
    def checkConstraintConsistency(self):
        if self.tied_param is not None:
            tlist = self.getTiedParamList()
            for l in tlist:
                for _,ind in self._highest_parent_.constraints.iteritems():
                    f = np.in1d(l,ind)
                    if not np.all(f) and np.any(f):
                        return False
        return True
    
    def checkTieTogether(self, plist):
        idx = []
        for p in plist:
            idx.extend(self._highest_parent_._raveled_index_for(p))
        labels = np.unique(self.label_buf[idx])
        if len(labels)==1 and labels[0]>0:
            return True
        else:
            return False
        
    def checkTieVector(self, plist):
        p1 = plist[0]
        idx1 = self._highest_parent_._raveled_index_for(p1)
        if np.any(self.label_buf[idx1]==0):
            return False
        for p2 in plist[1:]:
            idx2 = self._highest_parent_._raveled_index_for(p2)
            if np.any(self.label_buf[idx2]==0) or np.any(self.label_buf[idx1]!=self.label_buf[idx2]):
                return False
        return True

