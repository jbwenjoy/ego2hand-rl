"""Parameter-free current-state hand FK and phone-relative geometry poses.

No native calls, future labels, contact-force inputs, distance queries or NN.
The asset constructor reads only the separately exported compiled constants.
All pose arithmetic, including q derivatives, is Torch and supports leading
batch dimensions. R is BODY -> world; the pair origin is the phone COM.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

SCHEMA = 'phone_current_pair_geometry_features:2'
STATE_SCHEMA = 'phone_COM_world_state58:1'
MODEL_SHA256 = 'ff88ccc33fd1190328639925d82c6c11550b44dea67eec0f51c0a3cfd4339826'
from . import rotation as _exp_module
from ._artifacts import checked_data_pin
from .geometry_context import CurrentGeometry, ImmutableTensorCache


def _check_file(pin, override=None):
    return checked_data_pin(pin, override=override)


def _proper_rotation(R, label):
    tol = 1e-5 if R.dtype == torch.float32 else 1e-10
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    if not torch.isfinite(R).all() or torch.any(abs(R.transpose(-2,-1) @ R - eye) > tol) or torch.any(abs(torch.linalg.det(R)-1) > tol):
        raise ValueError(label + ': raw proper SO3 required; no repair/projection')


class PhoneGeometryProvider:
    """Compiled fixed-wrist model constants; zero trainable parameters.

    `poses(state58)` returns every body/geom transform in native ID order.
    `features(state58, absolute_u20, native_h_s, n_substeps)` returns the exact
    v2 current-input contract. q20 is state58[..., :20]; replacing q requires
    replacing that slice, so a separate q override cannot silently disagree.
    """
    def __init__(self, metadata_path, *, geometry_arrays_path=None):
        self.metadata_path = Path(metadata_path).resolve(strict=True)
        self.meta = json.loads(self.metadata_path.read_text(encoding='utf8'))
        if self.meta['schema'] != 'phone_compiled_geometry_constants:1' or self.meta['model']['sha256'] != MODEL_SHA256:
            raise ValueError('Only this compiled phone model is supported')
        if self.meta['phone_geom_id'] != 71 or self.meta['designated_pad_geom_ids'] != [15,28] or len(self.meta['pairs']) != 22:
            raise ValueError('Original two-pad 22-boundary-pair identity')
        with np.load(_check_file(self.meta['geometry_arrays'], geometry_arrays_path), allow_pickle=False) as z:
            self.geometry_arrays = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        self.parents = self.meta['body_parentid']
        self.body_joint_qindex = self.meta['body_joint_qindex']
        self.phone_body_id = self.meta['phone_body_id']
        self.pair_geom_ids = [r['other_geom_id'] for r in self.meta['pairs']]
        self.pair_names = [r['name'] for r in self.meta['pairs']]
        self._constants = {k: torch.tensor(self.meta[k], dtype=torch.float64) for k in (
            'body_pos', 'body_R', 'body_joint_axis', 'body_joint_anchor', 'joint_qpos0',
            'geom_pos', 'geom_R', 'phone_body_ipos')}
        self.geom_bodyid = self.meta['geom_bodyid']
        self.geometry_by_id = {r['geom_id']: r for r in self.meta['geometry_bindings']}
        if any(t.requires_grad for t in self._constants.values()) or any(t.requires_grad for t in self.geometry_arrays.values()):
            raise ValueError('Geometry constants cannot be trainable')
        self._constant_cache = ImmutableTensorCache(self._constants)
        vertex_keys = {b['vertices_key'] for b in self.geometry_by_id.values() if b['kind'] == 'mesh'}
        self._array_cache = ImmutableTensorCache({k: self.geometry_arrays[k] for k in vertex_keys})
        self.counts = {'pure_Torch_FK_calls': 0}

    def bundle(self, state58, *, retain_all_directions=False):
        return CurrentGeometry(self, state58, retain_all_directions=retain_all_directions)

    def _c(self, key, x):
        return self._constant_cache.get(x)[key]

    def poses(self, state58):
        x = state58
        if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32,torch.float64) or x.shape[-1:] != (58,) or not torch.isfinite(x).all():
            raise ValueError('Finite floating Torch state58, any leading batch shape')
        Rp = x[...,43:52].reshape(x.shape[:-1]+(3,3))
        _proper_rotation(Rp, 'BODY orientation')
        self.counts['pure_Torch_FK_calls'] += 1
        constants = self._constant_cache.get(x)
        shape = x.shape[:-1]
        I = torch.eye(3, dtype=x.dtype, device=x.device).expand(shape+(3,3))
        zero = torch.zeros(shape+(3,), dtype=x.dtype, device=x.device)
        off, localR = constants['body_pos'], constants['body_R']
        axes, anchors = constants['body_joint_axis'], constants['body_joint_anchor']
        q0 = constants['joint_qpos0']
        p, R = [zero], [I]
        for b in range(1,len(self.parents)):
            if b == self.phone_body_id:
                Rb = Rp
                pb = x[...,40:43] - (Rb @ constants['phone_body_ipos'][...,None])[...,0]
            else:
                parent = self.parents[b]
                if not 0 <= parent < b:
                    raise ValueError('Native parent-before-child ordering')
                qi = self.body_joint_qindex[b]
                if qi >= 0:
                    rr = _exp_module.so3_exp_torch((x[...,qi]-q0[qi])[...,None]*axes[b])
                    Rb = R[parent] @ localR[b] @ rr
                    local_p = off[b] + (localR[b] @ (anchors[b] - (rr @ anchors[b][...,None])[...,0])[...,None])[...,0]
                else:
                    Rb = R[parent] @ localR[b]
                    local_p = off[b]
                pb = p[parent] + (R[parent] @ local_p[...,None])[...,0]
            p.append(pb); R.append(Rb)
        bp, bR = torch.stack(p,-2), torch.stack(R,-3)
        gp0, gR0 = constants['geom_pos'], constants['geom_R']
        gR = bR[...,self.geom_bodyid,:,:] @ gR0
        gp = bp[...,self.geom_bodyid,:] + (bR[...,self.geom_bodyid,:,:] @ gp0[...,None])[...,0]
        Rt = Rp.transpose(-2,-1)
        relp = (Rt[...,None,:,:] @ (gp[...,self.pair_geom_ids,:]-x[...,None,40:43])[...,None])[...,0]
        relR = Rt[...,None,:,:] @ gR[...,self.pair_geom_ids,:,:]
        return dict(body_position_world_m=bp, body_rotation_world=bR,
                    geom_position_world_m=gp, geom_rotation_world=gR,
                    phone_root_world_m=bp[...,self.phone_body_id,:],
                    pair_relative_translation_BODY_COM_m=relp,
                    pair_relative_rotation_BODY9=relR.flatten(-2))

    def features(self, state58, absolute_u20, native_h_s, n_substeps, *, context=None):
        """Current-observable features only; unknown distance is NaN, never0."""
        x = state58; lead = x.shape[:-1]
        if absolute_u20.shape != lead+(20,) or absolute_u20.device != x.device or absolute_u20.dtype != x.dtype or not torch.isfinite(absolute_u20).all():
            raise ValueError('Finite same-dtype/device actual absolute u20')
        if native_h_s.shape != lead or n_substeps.shape != lead or not native_h_s.is_floating_point() or n_substeps.dtype not in (torch.int32,torch.int64):
            raise ValueError('Explicit batch-native h and integer substeps')
        if not torch.isfinite(native_h_s).all() or torch.any(native_h_s <= 0) or torch.any(n_substeps <= 0):
            raise ValueError('Positive finite native h and substeps')
        if torch.any((n_substeps != 64)&(n_substeps != 128)) or torch.any(abs(native_h_s*n_substeps-.02) > 4*torch.finfo(native_h_s.dtype).eps*.02):
            raise ValueError('This phone20ms contract requires64/128 integer ticks')
        if context is None:
            v = self.poses(x)
        else:
            context.require_provider(self, x); v = context.poses()
        shape = lead+(22,)
        yes = torch.ones(shape,dtype=torch.bool,device=x.device)
        no = torch.zeros_like(yes)
        nan = torch.full(shape,float('nan'),dtype=x.dtype,device=x.device)
        return dict(schema=SCHEMA,source_state_schema=STATE_SCHEMA,model_sha256=MODEL_SHA256,
            state58=x,absolute_u20=absolute_u20,native_h_s=native_h_s,n_substeps=n_substeps,
            pair_names=self.pair_names,geometry_bindings=self.meta['geometry_bindings'],
            pair_relative_translation_BODY_COM_m=v['pair_relative_translation_BODY_COM_m'],
            pair_relative_rotation_BODY9=v['pair_relative_rotation_BODY9'],
            pair_pose_known=yes,pair_geometry_binding_known=yes,
            pair_signed_gap_m=nan,pair_gap_known=no,pair_gap_lower_bound_m=nan.clone(),pair_lower_bound_known=no.clone(),
            normal_on_phone_BODY3=torch.full(lead+(22,3),float('nan'),dtype=x.dtype,device=x.device),
            normal_known=no.clone(),normal_unique_known=no.clone())

    def mesh_vertices_world(self, poses, geom_id):
        """Raw compiled geom-frame vertices transformed once; not an SDF."""
        binding = self.geometry_by_id[geom_id]
        if binding['kind'] != 'mesh':
            raise ValueError('This geometry is not a mesh')
        R = poses['geom_rotation_world'][...,geom_id,:,:]
        p = poses['geom_position_world_m'][...,geom_id,:]
        verts = self._array_cache.get(R)[binding['vertices_key']]
        return (R[...,None,:,:] @ verts[...,None])[...,0]+p[...,None,:]

    def capsule_centerline_world(self, poses, geom_id):
        binding = self.geometry_by_id[geom_id]
        if binding['kind'] != 'capsule':
            raise ValueError('This geometry is not a capsule')
        R = poses['geom_rotation_world'][...,geom_id,:,:]
        p = poses['geom_position_world_m'][...,geom_id,:]
        half = binding['size_m'][1]
        delta = R[..., :,2]*half
        return p-delta,p+delta,binding['size_m'][0]


def validate_features(v):
    """v2 separates pose/binding knowledge from gap and normal knowledge."""
    expected = {'schema','source_state_schema','model_sha256','state58','absolute_u20','native_h_s','n_substeps',
        'pair_names','geometry_bindings','pair_relative_translation_BODY_COM_m','pair_relative_rotation_BODY9',
        'pair_pose_known','pair_geometry_binding_known','pair_signed_gap_m','pair_gap_known','pair_gap_lower_bound_m',
        'pair_lower_bound_known','normal_on_phone_BODY3','normal_known','normal_unique_known'}
    if set(v) != expected or v['schema'] != SCHEMA or v['source_state_schema'] != STATE_SCHEMA or v['model_sha256'] != MODEL_SHA256:
        raise ValueError('Exact current-input schema; future labels/forces are not inputs')
    x = v['state58']; shape = x.shape[:-1]+(22,)
    if len(v['pair_names']) != 22 or len(set(v['pair_names'])) != 22:
        raise ValueError('All22 pair identities')
    _proper_rotation(v['pair_relative_rotation_BODY9'].reshape(shape+(3,3)), 'Relative orientation')
    for key in ('pair_pose_known','pair_geometry_binding_known','pair_gap_known','pair_lower_bound_known','normal_known','normal_unique_known'):
        if v[key].shape != shape or v[key].dtype != torch.bool:
            raise ValueError('Explicit separate known masks')
    if not v['pair_pose_known'].all() or not v['pair_geometry_binding_known'].all() or not torch.isfinite(v['pair_relative_translation_BODY_COM_m']).all():
        raise ValueError('This provider supplies all compiled poses/bindings')
    # This implementation has no query/certificate. Reject fake known gaps.
    for known,value in [('pair_gap_known','pair_signed_gap_m'),('pair_lower_bound_known','pair_gap_lower_bound_m'),('normal_known','normal_on_phone_BODY3')]:
        if v[known].any() or not torch.isnan(v[value]).all():
            raise ValueError('No distance/normal query is implemented in this provider')
    if v['normal_unique_known'].any():
        raise ValueError('No normal uniqueness proof')
    return {'schema':SCHEMA,'pairs':22,'gap_queries':0,'trainable_parameters':0,'future_inputs':0}
