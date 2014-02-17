from ..core.parameterization.array_core import ObservableArray, ParamList
class Cacher(object):
    def __init__(self, operation, limit=5):
        self.limit = int(limit)
        self.operation=operation
        self.cached_inputs = ParamList([])
        self.cached_outputs = []
        self.inputs_changed = []

    def __call__(self, X):
        assert isinstance(X, ObservableArray)
        if X in self.cached_inputs:
            i = self.cached_inputs.index(X)
            if self.inputs_changed[i]:
                self.cached_outputs[i] = self.operation(X)
                self.inputs_changed[i] = False
            return self.cached_outputs[i]
        else:
            if len(self.cached_inputs) == self.limit:
                X_ = self.cached_inputs.pop(0)
                X_.remove_observer(self)
                self.inputs_changed.pop(0)
                self.cached_outputs.pop(0)

            self.cached_inputs.append(X)
            self.cached_outputs.append(self.operation(X))
            self.inputs_changed.append(False)
            X.add_observer(self, self.on_cache_changed)
            return self.cached_outputs[-1]

    def on_cache_changed(self, X):
        #print id(X)
        i = self.cached_inputs.index(X)
        self.inputs_changed[i] = True

                




