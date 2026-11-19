"""Guarded line search on an explicitly identified W32 runtime."""
from contextlib import contextmanager
import copy
import hashlib
import json
import weakref

from ..artifacts import file_pin
from .backend_pen_runtime import BackendSupportPenRuntime
from .explicit_collision_backend import active_backend
from .pen_stable_friction_binding_v2 import FactoryBinding


_active = None


@contextmanager
def guarded_w32(manifest):
    import warp as wp
    from . import pen_stable_friction_profile_v2

    global _active
    collision = active_backend()
    if _active is not None or collision.runtime_set or pen_stable_friction_profile_v2._ACTIVE is not None:
        raise ValueError('W32 requires an exclusive scope with no existing runtime')
    identity = dict(schema='pen_guarded_friction_W32:1', worlds=32, nv=26, nv_pad=28,
                    stopping='INSTALLED_ORIGINAL_OR', algorithm='STABLE_SAME_REGIME_FRICTION_OR',
                    runtime=file_pin(__file__), collision_backend_id=collision.contract['backend_id'], factory=manifest)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    contract = dict(backend_id=digest, identity=copy.deepcopy(identity))
    live = weakref.WeakSet()
    with FactoryBinding(manifest, synchronize=wp.synchronize) as binding:
        _active = binding, contract, live
        try:
            yield binding
            if live or collision.runtime_set:
                raise ValueError('Close all W32 runtimes before releasing the guarded factory')
        finally:
            _active = None


class GuardedW32PenRuntime(BackendSupportPenRuntime):
    def __init__(self, objective_config, num_envs):
        if num_envs != 32 or _active is None:
            raise ValueError('Guarded W32 scope and exactly 32 worlds required')
        binding, _, live = _active
        binding.verify()
        live.add(self)
        try:
            super().__init__(objective_config, num_envs)
        except BaseException:
            live.discard(self)
            raise

    def _guarded_identity(self):
        if _active is None or self not in _active[2]:
            raise ValueError('Runtime does not belong to the active guarded W32 scope')
        _active[0].verify()
        return _active[1]

    def metadata(self):
        return dict(super().metadata(), termination_backend=copy.deepcopy(self._guarded_identity()))

    def validate_resources(self):
        self._guarded_identity()
        return super().validate_resources()

    def step(self, *args, **kwargs):
        self._guarded_identity()
        return super().step(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._guarded_identity()
        return super().reset(*args, **kwargs)

    def close(self):
        self._guarded_identity()
        try:
            super().close()
        finally:
            _active[2].discard(self)
