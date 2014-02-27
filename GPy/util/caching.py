from ..core.parameterization.parameter_core import Observable

class Cacher(object):
    def __init__(self, operation, limit=5, ignore_args=()):
        self.limit = int(limit)
        self.ignore_args = ignore_args
        self.operation=operation
        self.cached_inputs = []
        self.cached_outputs = []
        self.inputs_changed = []

    def __call__(self, *args):
        if len(self.ignore_args) != 0:
            ca = [a for i,a in enumerate(args) if i not in self.ignore_args]
        else:
            ca = args
        # this makes sure we only add an observer once, and that None can be in args
        cached_args = []
        for a in ca:
            if (not any(a is ai for ai in cached_args)) and a is not None:
                cached_args.append(a)
        if not all([isinstance(arg, Observable) for arg in cached_args]):
            print cached_args
            import ipdb;ipdb.set_trace()
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
                [a.remove_observer(self, self.on_cache_changed) for a in args_]
                self.inputs_changed.pop(0)
                self.cached_outputs.pop(0)

            self.cached_inputs.append(cached_args)
            self.cached_outputs.append(self.operation(*args))
            self.inputs_changed.append(False)
            [a.add_observer(self, self.on_cache_changed) for a in cached_args]
            return self.cached_outputs[-1]

    def on_cache_changed(self, arg):
        self.inputs_changed = [any([a is arg for a in args]) or old_ic for args, old_ic in zip(self.cached_inputs, self.inputs_changed)]

    def reset(self, obj):
        [[a.remove_observer(self, self.on_cache_changed) for a in args] for args in self.cached_inputs]
        [[a.remove_observer(self, self.reset) for a in args] for args in self.cached_inputs]
        self.cached_inputs = []
        self.cached_outputs = []
        self.inputs_changed = []

class Cache_this(object):
    def __init__(self, limit=5, ignore_args=()):
        self.limit = limit
        self.ignore_args = ignore_args
        self.c = None
    def __call__(self, f):
        def f_wrap(*args):
            if self.c is None:
                self.c = Cacher(f, self.limit, ignore_args=self.ignore_args)
            return self.c(*args)
        f_wrap._cacher = self
        return f_wrap