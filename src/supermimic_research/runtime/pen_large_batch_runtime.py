"""Run the existing guarded physics and collector at an explicitly named batch size."""
import copy
from contextlib import contextmanager
import inspect
import textwrap
from unittest.mock import patch

from .backend_pen_runtime import BackendSupportPenRuntime
from .pen_native_policy import NativePenPolicyAdapter
from .pen_stable_friction_binding_v2 import FactoryBinding
from ..artifacts import file_pin

_active = None


def _scaled_constructor():
    # The published W1/32 constructor is also an input to archived experiments.
    # Specialize its single world-count gate in memory, retaining its other checks.
    old = NativePenPolicyAdapter.__init__
    source = textwrap.dedent(inspect.getsource(old))
    guard = "if num_envs not in [1,32]:raise ValueError('Predeclared integration worlds1/32')"
    if source.count(guard) != 1:
        raise RuntimeError('Unexpected upstream adapter constructor')
    namespace = {}
    exec(compile(source.replace(guard, "if num_envs not in (256,512,1024,2048,4096): raise ValueError('Unsupported batch size')"),
                 __file__, 'exec'), old.__globals__, namespace)
    return namespace['__init__']


class BatchedPen(BackendSupportPenRuntime):
    def metadata(self):
        if _active is None or _active[0] != self.num_envs:
            raise ValueError('Inactive guarded batch')
        _active[1].verify()
        return dict(super().metadata(), termination_backend=copy.deepcopy(_active[2]))


@contextmanager
def batch_runtime(objective, worlds, manifest):
    global _active
    import warp as wp
    from .explicit_collision_backend import active_backend
    collision = active_backend()
    if _active is not None or collision.runtime_set or worlds not in (256, 512, 1024, 2048, 4096):
        raise ValueError('Exclusive batch scope required')
    with FactoryBinding(manifest, synchronize=wp.synchronize) as binding:
        identity = dict(schema='large_batch_guarded_pen:1', worlds=worlds, factory=manifest['generated'],
                        runtime=file_pin(__file__), collision_backend=collision.contract['backend_id'])
        _active = worlds, binding, identity
        try:
            with patch.object(NativePenPolicyAdapter, '__init__', _scaled_constructor()):
                runtime = BatchedPen(objective, worlds)
            try:
                sim = runtime.env.sim
                if not binding.audit['calls'] or sim.step_graph is None or sim.wp_model.nv_pad != 28 or sim.wp_model.nv != 26:
                    raise RuntimeError('Missing guarded graph or unexpected joint layout')
                yield runtime
            finally:
                runtime.close()
        finally:
            _active = None

