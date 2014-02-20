from ..core.parameterization.parameter_core import Observable
from ..core.parameterization.array_core import ParamList

class Cacher(object):
    def __init__(self, operation, limit=5, reset_on_first=False):
        self.limit = int(limit)
        self._reset_on_first = reset_on_first
        self.operation=operation
        self.cached_inputs = []
        self.cached_outputs = []
        self.inputs_changed = []

    def __call__(self, *args):
        if self._reset_on_first:
            assert isinstance(args[0], Observable)
            args[0].add_observer(args[0], self.reset)
            cached_args = args
        else:
            cached_args = args[1:]


        if not all([isinstance(arg, Observable) for arg in cached_args]):
            return self.operation(*args)
        if cached_args in self.cached_inputs:
            i = self.cached_inputs.index(cached_args)
            if self.inputs_changed[i]:
                self.cached_outputs[i] = self.operation(*args)
                self.inputs_changed[i] = False
            return self.cached_outputs[i]
        else:
            if len(self.cached_inputs) == self.limit:
                args_ = self.cached_inputs.pop(0)
                [a.remove_observer(self) for a in args_]
                self.inputs_changed.pop(0)
                self.cached_outputs.pop(0)

            self.cached_inputs.append(cached_args)
            self.cached_outputs.append(self.operation(*args))
            self.inputs_changed.append(False)
            [a.add_observer(self, self.on_cache_changed) for a in args]
            return self.cached_outputs[-1]

    def on_cache_changed(self, arg):
        self.inputs_changed = [any([a is arg for a in args]) or old_ic for args, old_ic in zip(self.cached_inputs, self.inputs_changed)]

    def reset(self, obj):
        [[a.remove_observer(self) for a in args] for args in self.cached_inputs]
        self.cached_inputs = []
        self.cached_outputs = []
        self.inputs_changed = []




def cache_this(limit=5, reset_on_self=False):
    def limited_cache(f):
        c = Cacher(f, limit, reset_on_first=reset_on_self)
        def f_wrap(*args):
            return c(*args)
        f_wrap._cacher = c
        return f_wrap
    return limited_cache












        #Xbase = X
        #while Xbase is not None:
            #try:
                #i = self.cached_inputs.index(X)
                #break
            #except ValueError:
                #Xbase = X.base
                #continue
        #self.inputs_changed[i] = True







