"""Fresh full task on an explicitly attested collision backend."""
import copy
from .pen_support_runtime import SupportPenRuntime
from .explicit_collision_backend import active_backend


class BackendSupportPenRuntime(SupportPenRuntime):
    def __init__(self,objective_config,num_envs):
        backend=active_backend();self._collision_backend_contract=copy.deepcopy(backend.contract);backend.runtime_set.add(self)
        try:
            super().__init__(objective_config,num_envs)
            if self.env.sim.wp_model.opt.ccd_iterations!=50 or any(float(v)!=float(__import__('numpy').float32(1e-6)) for v in self.env.sim.wp_model.opt.ccd_tolerance.numpy()):
                raise ValueError('Qualified backend numerical settings differ')
            self.validate_resources()
        except BaseException:
            try:
                if hasattr(self,'env'):super().close()
            finally:backend.runtime_set.discard(self)
            raise

    def validate_resources(self):
        backend=active_backend(self._collision_backend_contract)
        if self not in backend.runtime_set:raise ValueError('Runtime not registered with its backend')
        return super().validate_resources()

    def step(self,*args,**kwargs):
        active_backend(self._collision_backend_contract)
        return super().step(*args,**kwargs)

    def reset(self,*args,**kwargs):
        active_backend(self._collision_backend_contract)
        return super().reset(*args,**kwargs)

    def metadata(self):
        active_backend(self._collision_backend_contract)
        return dict(super().metadata(),collision_backend=copy.deepcopy(self._collision_backend_contract),
            collision_state_contract='explicit_collision_binding_and_owned_state:1')

    def close(self):
        backend=active_backend(self._collision_backend_contract)
        try:return super().close()
        finally:backend.runtime_set.discard(self)
