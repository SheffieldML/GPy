# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Copyright (c) 2014, James Hensman, Max Zwiessele, Zhenwen Dai
# Copyright (c) 2015, Max Zwiessele
#
# Redistribution and use under the BSD 3-clause license (see LICENSE.txt).
#
# CIP-000F Stage 3: GPy-owned Cache_this / Cacher.
# - ndarray arguments are cacheable (ObsAr no longer required)
# - owner ``_cache_version`` is part of the cache id (invalidation without observers)
# - Observable inputs still receive observer hooks when present

import collections
import contextlib
import weakref
from functools import reduce
from numbers import Number
from pickle import PickleError

from decorator import decorate

from paramz.core.observable import Observable


class Cacher:
    def __init__(
        self,
        operation,
        limit=3,
        ignore_args=(),
        force_kwargs=(),
        cacher_enabled=True,
    ):
        self.limit = int(limit)
        self.ignore_args = ignore_args
        self.force_kwargs = force_kwargs
        self.operation = operation
        self.cacher_enabled = cacher_enabled
        if getattr(self.operation, "__self__", None) is not None:
            obj = self.operation.__self__
            if not hasattr(obj, "cache"):
                obj.cache = FunctionCache()
            cache = obj.cache
            self.cacher_enabled = self.cacher_enabled and cache.caching_enabled
            cache[self.operation] = self

        self.cached_input_ids = {}
        self.reset()

    def disable_cacher(self):
        self.cacher_enabled = False
        self.reset()

    def enable_cacher(self):
        self.cacher_enabled = True

    def id(self, obj):
        return hex(id(obj))

    def combine_inputs(self, args, kw, ignore_args):
        inputs = args + tuple(c[1] for c in sorted(kw.items(), key=lambda x: x[0]))
        return [a for i, a in enumerate(inputs) if i not in ignore_args]

    def prepare_cache_id(self, combined_args_kw, owner=None):
        parts = [self.id(a) for a in combined_args_kw]
        # Stage 3: version counter on the method owner (even when self is in
        # ignore_args). ``operation`` is usually an unbound function, so the
        # caller passes ``args[0]`` as owner.
        if owner is None:
            owner = getattr(self.operation, "__self__", None)
        if owner is not None:
            ver = getattr(owner, "_cache_version", None)
            if ver is not None:
                parts.append(f"v{ver}")
        for a in combined_args_kw:
            av = getattr(a, "_cache_version", None)
            if av is not None:
                parts.append(f"a{av}")
        return "".join(parts)

    @staticmethod
    def _is_cacheable(b):
        if b is None or isinstance(b, (Number, str)):
            return True
        if isinstance(b, Observable):
            return True
        # Stage 3: plain ndarrays (replacing ObsAr as the data container)
        try:
            import numpy as np

            if isinstance(b, np.ndarray):
                return True
        except ImportError:
            pass
        return False

    def ensure_cache_length(self):
        if len(self.order) == self.limit:
            cache_id = self.order.popleft()
            combined_args_kw = self.cached_inputs[cache_id]
            for ind in combined_args_kw:
                ind_id = self.id(ind)
                tmp = self.cached_input_ids.get(ind_id, None)
                if tmp is not None:
                    ref, cache_ids = tmp
                    if len(cache_ids) == 1 and ref() is not None:
                        obj = ref()
                        if hasattr(obj, "remove_observer"):
                            obj.remove_observer(self, self.on_cache_changed)
                        del self.cached_input_ids[ind_id]
                    else:
                        cache_ids.remove(cache_id)
                        self.cached_input_ids[ind_id] = [ref, cache_ids]
            for store in (
                self.cached_outputs,
                self.inputs_changed,
                self.cached_inputs,
            ):
                with contextlib.suppress(KeyError):
                    del store[cache_id]

    def add_to_cache(self, cache_id, inputs, output):
        self.inputs_changed[cache_id] = False
        self.cached_outputs[cache_id] = output
        self.order.append(cache_id)
        self.cached_inputs[cache_id] = inputs
        for a in inputs:
            if a is not None and not isinstance(a, Number) and not isinstance(a, str):
                if not hasattr(a, "add_observer"):
                    # ndarray / non-Observable: keyed by id + owner version only
                    continue
                ind_id = self.id(a)
                v = self.cached_input_ids.get(ind_id, [weakref.ref(a), []])
                v[1].append(cache_id)
                if len(v[1]) == 1:
                    a.add_observer(self, self.on_cache_changed)
                self.cached_input_ids[ind_id] = v

    def __call__(self, *args, **kw):
        if not self.cacher_enabled:
            return self.operation(*args, **kw)

        if len(self.force_kwargs) != 0:
            for k in self.force_kwargs:
                if k in kw and kw[k] is not None:
                    return self.operation(*args, **kw)

        inputs = self.combine_inputs(args, kw, self.ignore_args)
        owner = args[0] if args else None
        cache_id = self.prepare_cache_id(inputs, owner=owner)
        if reduce(lambda a, b: a or (not self._is_cacheable(b)), inputs, False):
            return self.operation(*args, **kw)

        not_seen = cache_id not in self.inputs_changed
        changed = (not not_seen) and self.inputs_changed[cache_id]
        if changed or not_seen:
            try:
                new_output = self.operation(*args, **kw)
            except Exception:
                self.reset()
                raise
            if changed:
                self.inputs_changed[cache_id] = False
                self.cached_outputs[cache_id] = new_output
            else:
                self.ensure_cache_length()
                self.add_to_cache(cache_id, inputs, new_output)
        return self.cached_outputs[cache_id]

    def on_cache_changed(self, direct, which=None):
        for what in [direct, which]:
            ind_id = self.id(what)
            _, cache_ids = self.cached_input_ids.get(ind_id, [None, []])
            for cache_id in cache_ids:
                self.inputs_changed[cache_id] = True

    def reset(self):
        for a, _ in list(self.cached_input_ids.values()):
            obj = a()
            if obj is not None and hasattr(obj, "remove_observer"):
                obj.remove_observer(self, self.on_cache_changed)

        self.order = collections.deque()
        self.cached_inputs = {}
        self.cached_input_ids = {}
        self.cached_outputs = {}
        self.inputs_changed = {}

    def __deepcopy__(self, memo=None):
        return Cacher(self.operation, self.limit, self.ignore_args, self.force_kwargs)

    def __getstate__(self, memo=None):
        raise PickleError(
            f"Trying to pickle Cacher object with function {str(self.operation)}, pickling functions not possible."
        )

    def __setstate__(self, memo=None):
        raise PickleError(
            f"Trying to pickle Cacher object with function {str(self.operation)}, pickling functions not possible."
        )

    @property
    def __name__(self):
        return self.operation.__name__

    def __str__(self, *args, **kwargs):
        return f"Cacher({self.__name__})\n  limit={self.limit}\n  \\#cached={len(self.cached_input_ids)}"


class FunctionCache(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.caching_enabled = True

    def disable_caching(self):
        self.caching_enabled = False
        for c in self.values():
            c.disable_cacher()

    def enable_caching(self):
        self.caching_enabled = True
        for c in self.values():
            c.enable_cacher()

    def reset(self):
        for c in self.values():
            c.reset()


class Cache_this:
    """Decorator for caching bound methods with version-counter invalidation."""

    def __init__(self, limit=5, ignore_args=(), force_kwargs=()):
        self.limit = limit
        self.ignore_args = ignore_args
        self.force_kwargs = force_kwargs
        self.f = None

    def __call__(self, f):
        self.f = f

        def g(obj, *args, **kw):
            obj = args[0]
            if not hasattr(obj, "cache"):
                obj.cache = FunctionCache()
            cache = obj.cache
            if self.f in cache:
                cacher = cache[self.f]
            else:
                cacher = cache[self.f] = Cacher(
                    self.f,
                    self.limit,
                    self.ignore_args,
                    self.force_kwargs,
                    cacher_enabled=cache.caching_enabled,
                )
            return cacher(*args, **kw)

        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g = decorate(self.f, g)
        return g
