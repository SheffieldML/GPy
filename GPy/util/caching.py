from ..core.parameterization.parameter_core import Observable
import itertools, collections, weakref

class Cacher(object):


    def __init__(self, operation, limit=5, ignore_args=(), force_kwargs=()):
        """
        Parameters:
        ***********
        :param callable operation: function to cache
        :param int limit: depth of cacher
        :param [int] ignore_args: list of indices, pointing at arguments to ignore in *args of operation(*args). This includes self!
        :param [str] force_kwargs: list of kwarg names (strings). If a kwarg with that name is given, the cacher will force recompute and wont cache anything.
        """
        self.limit = int(limit)
        self.ignore_args = ignore_args
        self.force_kwargs = force_kwargs
        self.operation=operation
        self.order = collections.deque()
        self.cached_inputs = {} # point from cache_ids to a list of [ind_ids], which where used in cache cache_id

        #=======================================================================
        # point from each ind_id to [ref(obj), cache_ids]
        # 0: a weak reference to the object itself
        # 1: the cache_ids in which this ind_id is used (len will be how many times we have seen this ind_id)
        self.cached_input_ids = {} 
        #=======================================================================

        self.cached_outputs = {} # point from cache_ids to outputs
        self.inputs_changed = {} # point from cache_ids to bools

    def combine_args_kw(self, args, kw):
        "Combines the args and kw in a unique way, such that ordering of kwargs does not lead to recompute"
        return args + tuple(c[1] for c in sorted(kw.items(), key=lambda x: x[0]))

    def preprocess(self, combined_args_kw, ignore_args):
        "get the cacheid (conc. string of argument ids in order) ignoring ignore_args"
        return "".join(str(id(a)) for i,a in enumerate(combined_args_kw) if i not in ignore_args)

    def ensure_cache_length(self, cache_id):
        "Ensures the cache is within its limits and has one place free"
        if len(self.order) == self.limit:
            # we have reached the limit, so lets release one element
            cache_id = self.order.popleft()
            combined_args_kw = self.cached_inputs[cache_id]
            for ind in combined_args_kw:
                ind_id = id(ind)
                ref, cache_ids = self.cached_input_ids[ind_id]
                if len(cache_ids) == 1 and ref() is not None:
                    ref().remove_observer(self, self.on_cache_changed)
                    del self.cached_input_ids[ind_id]
                else:
                    cache_ids.remove(cache_id)
                    self.cached_input_ids[ind_id] = [ref, cache_ids]
            del self.cached_outputs[cache_id]
            del self.inputs_changed[cache_id]
            del self.cached_inputs[cache_id]

    def add_to_cache(self, cache_id, combined_args_kw, output):
        self.inputs_changed[cache_id] = False
        self.cached_outputs[cache_id] = output
        self.order.append(cache_id)
        self.cached_inputs[cache_id] = combined_args_kw
        for a in combined_args_kw:
            ind_id = id(a)
            v = self.cached_input_ids.get(ind_id, [weakref.ref(a), []])
            v[1].append(cache_id)
            if len(v[1]) == 1:
                a.add_observer(self, self.on_cache_changed)
            self.cached_input_ids[ind_id] = v

    def __call__(self, *args, **kw):
        """
        A wrapper function for self.operation,
        """

        # 1: Check whether we have forced recompute arguments:
        if len(self.force_kwargs) != 0:
            for k in self.force_kwargs:
                if k in kw and kw[k] is not None:
                    return self.operation(*args, **kw)

        # 2: preprocess and get the unique id string for this call
        combined_args_kw = self.combine_args_kw(args, kw)
        cache_id = self.preprocess(combined_args_kw, self.ignore_args)

        # 2: if anything is not cachable, we will just return the operation, without caching
        if reduce(lambda a,b: a or (not isinstance(b, Observable)), combined_args_kw, False):
            return self.operation(*args, **kw)
        # 3&4: check whether this cache_id has been cached, then has it changed?
        try:
            if(self.inputs_changed[cache_id]):
                # 4: This happens, when one element has changed for this cache id
                self.inputs_changed[cache_id] = False
                self.cached_outputs[cache_id] = self.operation(*args, **kw)
        except KeyError:
            # 3: This is when we never saw this chache_id:
            self.ensure_cache_length(cache_id)
            self.add_to_cache(cache_id, combined_args_kw, self.operation(*args, **kw))
        except:
            self.reset()
            raise
        # 5: We have seen this cache_id and it is cached:
        return self.cached_outputs[cache_id]

    def on_cache_changed(self, direct, which=None):
        """
        A callback funtion, which sets local flags when the elements of some cached inputs change

        this function gets 'hooked up' to the inputs when we cache them, and upon their elements being changed we update here.
        """
        for ind_id in [id(direct), id(which)]:
            _, cache_ids = self.cached_input_ids.get(ind_id, [None, []])
            for cache_id in cache_ids:
                self.inputs_changed[cache_id] = True

    def reset(self):
        """
        Totally reset the cache
        """
        [a().remove_observer(self, self.on_cache_changed) if (a() is not None) else None for [a, _] in self.cached_input_ids.values()]
        self.cached_input_ids = {}
        self.cached_outputs = {}
        self.inputs_changed = {}

    def __deepcopy__(self, memo=None):
        return Cacher(self.operation, self.limit, self.ignore_args, self.force_kwargs)

    def __getstate__(self, memo=None):
        raise NotImplementedError, "Trying to pickle Cacher object with function {}, pickling functions not possible.".format(str(self.operation))

    def __setstate__(self, memo=None):
        raise NotImplementedError, "Trying to pickle Cacher object with function {}, pickling functions not possible.".format(str(self.operation))

    @property
    def __name__(self):
        return self.operation.__name__

from functools import partial, update_wrapper

class Cacher_wrap(object):
    def __init__(self, f, limit, ignore_args, force_kwargs):
        self.limit = limit
        self.ignore_args = ignore_args
        self.force_kwargs = force_kwargs
        self.f = f
        update_wrapper(self, self.f)
    def __get__(self, obj, objtype=None):
        return partial(self, obj)
    def __call__(self, *args, **kwargs):
        obj = args[0]
        #import ipdb;ipdb.set_trace()
        try:
            caches = obj.__cachers
        except AttributeError:
            caches = obj.__cachers = {}
        try:
            cacher = caches[self.f]
        except KeyError:
            cacher = caches[self.f] = Cacher(self.f, self.limit, self.ignore_args, self.force_kwargs)
        return cacher(*args, **kwargs)

class Cache_this(object):
    """
    A decorator which can be applied to bound methods in order to cache them
    """
    def __init__(self, limit=5, ignore_args=(), force_kwargs=()):
        self.limit = limit
        self.ignore_args = ignore_args
        self.force_args = force_kwargs
    def __call__(self, f):
        return Cacher_wrap(f, self.limit, self.ignore_args, self.force_args)
