"""Resident full dynamic-array restore for same-source physical diagnostics.

Array copy fidelity is distinct from deterministic physics or rollout accuracy.
The caller must use the owning Warp device/stream boundary and must not replace
captured-graph array allocations. Unused solver scratch may contain NaN bytes;
copy them exactly and validate ACTIVE physical fields separately.
"""
import hashlib

from .gpu_model_snapshot import model_leaves


class ResidentDataSnapshot:
    def __init__(self,data):
        import warp as wp
        arrays,self.scalars=model_leaves(data)
        self.layout={k:(v.shape,v.dtype,v.device,v.strides,v.ptr) for k,v in arrays.items()}
        self.arrays={k:wp.clone(v) for k,v in arrays.items()}

    def restore(self,data):
        import warp as wp
        targets,scalars=model_leaves(data)
        if scalars!=self.scalars or set(targets)!=set(self.arrays):
            raise ValueError('Exact dynamic scalar/array-key compatibility required')
        # Validate EVERY destination before any copy; preserve original pointers.
        for key,target in targets.items():
            if (target.shape,target.dtype,target.device,target.strides,target.ptr)!=self.layout[key]:
                raise ValueError(f'Dynamic array layout mismatch: {key}')
        for key,target in targets.items():
            if target.size:wp.copy(target,self.arrays[key])

    def host_values(self):
        return {k:v.numpy().copy() for k,v in self.arrays.items()}

    def audit(self,data):
        targets,scalars=model_leaves(data);values=self.host_values()
        if set(targets)!=set(values):raise ValueError('Dynamic snapshot keys changed')
        differences=[];fingerprints={}
        for key,value in values.items():
            before=value.tobytes();after=targets[key].numpy().tobytes()
            fingerprints[key]=hashlib.sha256(before).hexdigest()
            if before!=after:differences.append(key)
        return dict(arrays=len(values),scalar_leaves=len(self.scalars),scalar_equal=scalars==self.scalars,
            all_array_bytes_equal=not differences,differing_arrays=differences,fingerprints=fingerprints)
