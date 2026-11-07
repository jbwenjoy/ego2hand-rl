"""New16-frame declared-geometry intent binding; never human-contact labels.

The unchanged forward/score/P math is vendored. This adapter changes only frozen
data/lineage validation and the geometry model instance to actual context
geom_size/pos/friction. It does not upload this host model to physics.
"""
from pathlib import Path
import copy,hashlib,json,random,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor'))
from frozen_contact_math.contact_objective import GeometricContactObjective
from frozen_contact_math.contact_geometry import ContactGeometryProxy
from frozen_contact_math.penalty_only_objective import PenaltyOnlyPrior
from .io import read,sha,checked

VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
KNOWN_SEMANTICS='declared_H123_robot_kinematic_geometry_hypothesis_only_not_human_observation'
HYPER={'bandwidth_m':.001,'tolerance':.1,'var':.1}


def code_hashes():
    v=ROOT/'vendor/frozen_contact_math'
    return {'binding_module_sha256':sha(__file__),'binding_io_module_sha256':sha(ROOT/'src/rough_prefix/io.py'),'objective_module_sha256':sha(v/'contact_objective.py'),'geometry_module_sha256':sha(v/'contact_geometry.py'),'contacts_module_sha256':sha(v/'contacts.py'),'penalty_wrapper_module_sha256':sha(v/'penalty_only_objective.py')}


class RoughPrefixContactObjective(GeometricContactObjective):
    """Old formula and buffers; strict new lineage and actual geometry copy."""
    def __init__(self,spec):
        import mujoco
        if spec.get('schema')!='rough_prefix_contact_objective_spec:1' or spec.get('hyperparameters')!=HYPER:raise ValueError('Frozen roughprefix schema/hyperparameters required')
        expected=spec['expected_hashes']
        for key,value in code_hashes().items():
            if expected.get(key)!=value:raise ValueError('Contact binding/source changed: '+key)
        binding_path=checked(spec['binding_manifest']);binding=read(binding_path)
        if binding.get('schema')!='rough_prefix_contact_intent_binding:1' or binding['retarget_variant_id']!=VARIANT or binding['source_window_half_open']!=[200,216]:raise ValueError('Wrong new16/window/variant target binding')
        if binding['known_semantics']!=KNOWN_SEMANTICS or binding['human_contact_observation_known'] is not False or binding['target_defined_from_policy_contacts'] is not False or binding['local_proxy_qualification']!='FAILED_RETAINED_EXPLORATORY_ONLY':raise ValueError('Unqualified contact semantics or policy-defined target')
        for key,pin in binding['source_pins'].items():checked(pin)
        for role,path,hashkey in [('reference',spec['reference_file'],'reference_npz_sha256'),('reference_json',str(Path(spec['reference_file']).with_suffix('.json')),'reference_json_sha256'),('model',spec['model_file'],'model_sha256'),('actual_context',spec['actual_context_file'],'actual_context_sha256')]:
            if sha(path)!=expected[hashkey] or Path(path).resolve()!=Path(binding[role]['path']).resolve() or expected[hashkey]!=binding[role]['sha256']:raise ValueError('Wrong frozen '+role)
        if Path(spec['bundle_dir']).resolve()!=binding_path.parent:raise ValueError('Bundle directory identity differs')
        meta=read(Path(spec['reference_file']).with_suffix('.json'))
        if meta['retarget_variant_id']!=VARIANT or meta['source_window_half_open']!=[200,216] or meta['human_contact_observation_known'] is not False:raise ValueError('Bridge metadata cannot become measured labels')
        super().__init__(spec['bundle_dir'],reference_file=spec['reference_file'],model_file=spec['model_file'],expected_reference_sha256=expected['reference_npz_sha256'],dtype=torch.float32,**HYPER)
        if self.frame_count!=16 or self.metadata['reference_npz_sha256']!=binding['reference']['sha256']:raise ValueError('Only exactnew16 is accepted')
        with np.load(Path(spec['bundle_dir'])/'contact_targets.npz',allow_pickle=False) as z:
            if z['human_contact_observation_known'].shape!=(16,) or np.any(z['human_contact_observation_known']):raise ValueError('Human contact observations remain all unknown')
            known=z['contact_known'].copy();absent=z['known_no_contact'].copy();phase=z['phase'].copy()
        if np.any(absent):raise ValueError('No explicit release annotation exists; absence cannot become known_no_contact')
        model=mujoco.MjModel.from_binary_path(spec['model_file']);differences={}
        with np.load(spec['actual_context_file'],allow_pickle=False) as z:
            for field in ('geom_size','geom_pos','geom_friction'):
                value=z[field]
                if value.shape[1:]!=getattr(model,field).shape or not np.isfinite(value).all() or not np.array_equal(value,np.broadcast_to(value[:1],value.shape)):raise ValueError('Nonfixed actual geometry '+field)
                differences[field]=float(np.max(np.abs(getattr(model,field)-value[0])))
                getattr(model,field)[:]=value[0]
        self.geometry_proxy=ContactGeometryProxy(model,meta['joint_names'],self.basis512[self.basis_subset_indices].cpu().numpy(),float(meta['object_radius_m']),points_per_geom=64,dtype=torch.float32)
        self.metadata.update(schema='rough_prefix_declared_geometry_shaping:1',retarget_variant_id=VARIANT,source_window_half_open=[200,216],known_semantics=KNOWN_SEMANTICS,human_contact_observation_known=False,
                             target_defined_from_policy_contacts=False,local_proxy_qualification='FAILED_RETAINED_EXPLORATORY_ONLY',physical_reference_qualified=False,qualified_contact_predictor=False,dynamics_gradient_qualified=False,
                             compiled_precision_qualification='FAILED_RETAINED_NOT_REVALIDATED',
                             binding_manifest_sha256=sha(binding_path),actual_context_sha256=expected['actual_context_sha256'],**code_hashes(),geometry=self.geometry_proxy.metadata,
                             actual_geometry_host_copy_fields=['geom_size','geom_pos','geom_friction'],nominal_to_actual_geometry_max_abs=differences,host_model_used_for_physics=False,
                             support_radius_definition='Frozen nominalMJB enclosing radius normalization, not re-fit under actualcontext; COM from sameMJBbody_ipos. Actualgeometry copied to proxy only.',
                             known_frame_count=int(known.sum()),unknown_frame_count=int((~known).sum()),support_experiment_degenerate_to_penalty_only=not bool(self.phase_weight.any()),
                             scoring={**self.metadata['scoring'],'not_native_friction_or_force_amplitude':True},sampled_hand_surface_points=self.geometry_proxy.metadata['sample_count'])
        for key,value in expected.items():
            if key!='binding_manifest_sha256' and self.metadata.get(key)!=value:raise ValueError('Constructed frozen metadata differs: '+key)
        if expected.get('binding_manifest_sha256')!=sha(binding_path):raise ValueError('Binding manifest hash differs')


def build_contact_objective(spec,device='cpu'):
    """File-only factory, preserving caller RNG; no env/current policy argument."""
    python_rng=random.getstate();numpy_rng=np.random.get_state();torch_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:module=RoughPrefixContactObjective(spec).to(device=device,dtype=torch.float32)
    finally:
        random.setstate(python_rng);np.random.set_state(numpy_rng);torch.set_rng_state(torch_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    return module.eval().requires_grad_(False)


def build_penalty_only_objective(spec,device='cpu'):
    module=PenaltyOnlyPrior(build_contact_objective(spec,device)).eval().requires_grad_(False)
    module.metadata.update(schema='rough_prefix_penalty_only_shaping:1',retarget_variant_id=VARIANT,known_semantics=KNOWN_SEMANTICS,penalty_wrapper_sha256=sha(ROOT/'vendor/frozen_contact_math/penalty_only_objective.py'))
    return module
