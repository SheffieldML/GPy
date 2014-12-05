'''
Created on 11 Nov 2014

@author: maxz
'''
from observable import Observable


class Updateable(Observable):
    """
    A model can be updated or not.
    Make sure updates can be switched on and off.
    """
    _updates = True
    def __init__(self, *args, **kwargs):
        super(Updateable, self).__init__(*args, **kwargs)

    def update_model(self, updates=None):
        """
        Get or set, whether automatic updates are performed. When updates are
        off, the model might be in a non-working state. To make the model work
        turn updates on again.

        :param bool|None updates:

            bool: whether to do updates
            None: get the current update state
        """
        if updates is None:
            p = getattr(self, '_highest_parent_', None)
            if p is not None:
                self._updates = p._updates
            return self._updates
        assert isinstance(updates, bool), "updates are either on (True) or off (False)"
        p = getattr(self, '_highest_parent_', None)
        if p is not None:
            p._updates = updates
        self._updates = updates
        self.trigger_update()

    def toggle_update(self):
        self.update_model(not self.update_model())

    def trigger_update(self, trigger_parent=True):
        """
        Update the model from the current state.
        Make sure that updates are on, otherwise this
        method will do nothing

        :param bool trigger_parent: Whether to trigger the parent, after self has updated
        """
        if not self.update_model() or (hasattr(self, "_in_init_") and self._in_init_):
            #print "Warning: updates are off, updating the model will do nothing"
            return
        self._trigger_params_changed(trigger_parent)
