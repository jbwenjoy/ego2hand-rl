"""Exploratory matched-training factory for the unchanged declared-H123 prior.

A separately authorized spec:2 may move the frozen representation to CUDA.
This is enrollment/placement only: no physical contact, dynamics-gradient,
forceclosure or training-integration qualification follows from construction.
"""
from pathlib import Path
import hashlib,importlib.util,random,sys
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from full151_native.io import checked,read,pin,canonical,loadnp

ROOT=Path(__file__).resolve().parents[2]
FROZEN=Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/frozen_contact_math')
FROZEN_HASHES={
 '__init__.py':'a43a9d504ac69c2afce750e17ae28e2aa0d2af4b8c2dc7ca3b9fb19169b346da',
 'contact_objective.py':'54d9507d824d08baab3639e29d028b325109f51ca3e3fdd3118be341729502ea',
 'contact_geometry.py':'a9e0e19cc826cefe4e6a97a5c7b33f02f04c23585065cb0a273436eb07846d7a',
 'contacts.py':'79b53c63dd5573c9908d052f29c678f1724ff5be3ffa5b7c077421cbf4788212',
 'penalty_only_objective.py':'cfa53515fb7dc12ae49a2aec2d9ca37603061c1752fc8875f879cf5ea235c914'}
for _n,_h in FROZEN_HASHES.items():
    if hashlib.sha256((FROZEN/_n).read_bytes()).hexdigest()!=_h:raise ValueError('Frozen math source changed: '+_n)
_package='_full151_intent_frozen_contact_math'
if _package not in sys.modules:
    _sp=importlib.util.spec_from_file_location(_package,FROZEN/'__init__.py',submodule_search_locations=[str(FROZEN)])
    _module=importlib.util.module_from_spec(_sp);sys.modules[_package]=_module;_sp.loader.exec_module(_module)
_objective=importlib.import_module(_package+'.contact_objective')
_geometry=importlib.import_module(_package+'.contact_geometry')
_penalty=importlib.import_module(_package+'.penalty_only_objective')
for _module,_name in ((_objective,'contact_objective.py'),(_geometry,'contact_geometry.py'),(_penalty,'penalty_only_objective.py')):
    if Path(_module.__file__).resolve()!=(FROZEN/_name).resolve():raise ValueError('Foreign cached math module')
GeometricContactObjective=_objective.GeometricContactObjective
ContactGeometryProxy=_geometry.ContactGeometryProxy
PenaltyOnlyPrior=_penalty.PenaltyOnlyPrior
VARIANT='legacy_H123_full151_SCP8_knownrough_v1'
INTENT='source_rigid_H123_surface_projection_all151_v1'
SEMANTICS='declared_H123_observation_surface_projection_design_hypothesis_only_not_human_contact'
HYPER={'bandwidth_m':.001,'tolerance':.1,'var':.1}
JOINT_NAMES=[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
ARRAY_SHAPES={
 'time_s':(151,),'finger_ids':(3,),'landmark_indices':(3,),'mapped_object_pose_w':(151,7),
 'geometry_valid_mask':(151,3),'declared_intent_mask':(151,3),'declared_intent_active':(151,3),'declared_intent_frame_known':(151,),
 'known_no_contact':(151,),'human_contact_known':(151,3),'human_contact_present':(151,3),'human_contact_unknown_mask':(151,3),
 'landmark_confidence_known':(151,3),'object_confidence_known':(151,),
 'point_com_body_m':(151,3,3),'inward_normal_body':(151,3,3),'object_com_offset_body_m':(3,),'object_nominal_enclosing_radius_m':(),
 'raw_source_signed_distance_m':(151,3),'raw_source_unsigned_distance_m':(151,3),'surface_feature_index':(151,3),'surface_tie_count':(151,3),
 'phase':(151,),'phase_weight':(151,),'support_basis':(512,6),'support512':(151,1,1,512)}


def _require(ok,message):
    if not bool(ok):raise ValueError(message)


def _same_pin(a,b):
    return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def code_pins():
    return {'binding':pin(__file__),'predecessor_binding':pin(Path(__file__).with_name('binding_v2.py')),'binding_package':pin(Path(__file__).with_name('__init__.py')),
            'native_io':pin(ROOT/'src/full151_native/io.py'),'native_bundle_loader':pin(ROOT/'src/full151_native/bundle.py'),
            'native_metadata':pin(ROOT/'src/full151_native/metadata.py'),
            **{'frozen__'+n:pin(FROZEN/n) for n in FROZEN_HASHES}}


def validate_arrays(a):
    """Pure numeric/shape/mask checks; no distance cutoff and no contact query."""
    _require(set(a)==set(ARRAY_SHAPES),'Unexpected target key or missing new151 field; old16/q/link injection refused')
    for key,shape in ARRAY_SHAPES.items():
        value=np.asarray(a[key]);_require(value.shape==shape,'Wrong151/3/512 target shape: '+key)
        _require(value.dtype.kind in 'biuf' and np.isfinite(value).all(),'Nonfinite/nonnumeric target: '+key)
    boolean=[k for k in a if k.endswith('_mask') or k in ('declared_intent_active','declared_intent_frame_known','known_no_contact','human_contact_known','human_contact_present','landmark_confidence_known','object_confidence_known')]
    _require(all(a[k].dtype==np.bool_ for k in boolean),'Masks must be actual boolean arrays')
    _require(np.array_equal(a['time_s'],np.arange(151)*.02),'Wrong151 source clock')
    _require(np.array_equal(a['finger_ids'],[1,2,3]) and np.array_equal(a['landmark_indices'],[3,7,11]),'Declared H123 slot mapping changed')
    active=a['geometry_valid_mask']&a['declared_intent_mask'];known=active.any(1)
    _require(a['declared_intent_mask'].all() and np.array_equal(active,a['declared_intent_active']) and np.array_equal(known,a['declared_intent_frame_known']),'Declared mask is only geometric-valid AND persistent H123')
    _require(not a['known_no_contact'].any() and not a['human_contact_known'].any() and not a['human_contact_present'].any() and a['human_contact_unknown_mask'].all() and not a['landmark_confidence_known'].any() and not a['object_confidence_known'].any(),'Unknown human contact/confidence cannot become observed or release labels')
    _require(np.array_equal(a['phase'],np.where(known,3,0)) and np.array_equal(a['phase_weight'],known.astype(float)),'Phase3 means declared active only, not measured motion; weights1/0')
    basis=a['support_basis'];support=a['support512']
    _require(np.allclose(np.linalg.norm(basis,axis=1),1,rtol=0,atol=1e-6) and (support>=0).all(),'Basis must be unit/nonnegative support')
    _require(np.allclose(np.linalg.norm(a['inward_normal_body'][active],axis=1),1,rtol=0,atol=1e-6),'Active inward normals must be unit')
    _require(np.isfinite(a['object_nominal_enclosing_radius_m']).all() and a['object_nominal_enclosing_radius_m']>0,'Positive nominal enclosing radius required')
    _require(np.array_equal(support[~known],np.zeros_like(support[~known])),'Unknown frames have no attraction target')
    _require(known.any() and (support[known]>.001).any(-1).all() and (support[known,...,::8]>.001).any(-1).all(),'A nondegenerate support experiment needs known frames and nonzero512/64 fingerprints')
    return active,known


SPEC_FIELDS={'schema','binding_manifest','target_receipt','independent_target_review','input_bundle',
 'expected_code_pins','hyperparameters','execution_scope','CPU_authorized','GPU_authorized',
 'training_authorized','prior_CPU_binding_review','actual_material_contract','matched_material_screen',
 'matched_material_saved_review','legacy_proxy_dynamic_cone_review','matched_material_lifecycle_review'}


def _passed(value,schema,status):
    _require(value.get('schema')==schema and value.get('status')==status,'Wrong prerequisite evidence schema/status')
    _require(type(value.get('passed')) is int and type(value.get('total')) is int
             and value['passed']==value['total']==len(value['checks']) and value['total']>0
             and all(v is True for v in value['checks'].values()),'Incomplete prerequisite checks')


def validate_spec(spec):
    """Content-bound enrollment; no Env, query, solver or device allocation."""
    _require(type(spec) is dict and set(spec)==SPEC_FIELDS
             and spec['schema']=='full151_declared_H123_objective_spec:2','Wrong exploratory objective specification')
    _require(spec['hyperparameters']==HYPER and spec['execution_scope']=='EXPLORATORY_MATCHED_TRAINING'
             and spec['CPU_authorized'] is True and spec['training_authorized'] is True
             and type(spec['GPU_authorized']) is bool,'Explicit exploratory CPU/GPU/training scope required')
    _require(spec['expected_code_pins']==code_pins(),'Adapter/frozen source bytes changed')
    prior=read(checked(spec['prior_CPU_binding_review']))
    _passed(prior,'full151_declared_H123_binding_CPU_contract:1','PASS_CPU_REPRESENTATION_AND_GEOMETRY_AUTOGRAD_ONLY')
    _require(prior['source_pins']['binding']==spec['expected_code_pins']['predecessor_binding'],'CPU predecessor was not this frozen v2 source')
    checked(prior['checker']);oldspec=read(checked(prior['spec']))
    _require(oldspec['schema']=='full151_declared_H123_objective_spec:1'
             and all(oldspec[k]==spec[k] for k in ('binding_manifest','target_receipt','independent_target_review','input_bundle','hyperparameters')),'v2 CPU evidence belongs to a different target/bundle/math')
    material=read(checked(spec['actual_material_contract']))
    # The passive-material proof uses its own verified status, not a physics certificate.
    _require(material['schema']=='full151_actual_material_native_pyramid_contract:1'
             and material['status']=='VERIFIED_PASSIVE_CONSTANTS_AND_NUMPY_ALGEBRA'
             and type(material['passed']) is int and material['passed']==material['total']==len(material['checks'])
             and all(v is True for v in material['checks'].values())
             and material['condim']==4 and material['model_cone']==0
             and material['contact_override_enabled'] is False,'Actual native-material proof missing')
    for value in material['inputs'].values():checked(value)
    for value in material['sources'].values():checked(value)
    matched=read(checked(spec['matched_material_screen']))
    _require(matched['schema']=='full151_matched_actual_material_cone_CPU_report:1'
             and matched['status']=='COMPLETED_MATCHED_CONE_CPU_SCREEN' and matched['error'] is None
             and matched['LP_calls']==matched['accounted_slots']==302
             and matched['condition_counts']['H123']=={'FEASIBLE_CONDITIONAL_WITNESS':151}
             and matched['physical_reference_qualified'] is False,'H123 fixed native-parameter conditional cone evidence missing')
    for value in matched['source_pins'].values():checked(value)
    _require(_same_pin(matched['source_pins']['material_contract'],spec['actual_material_contract']),'Matched cone used another material')
    numeric=read(checked(spec['matched_material_saved_review']))
    _passed(numeric,'full151_matched_material_cones_saved_numeric_independent:1','PASS_SAVED_NUMERIC_AUDIT')
    _require(_same_pin(numeric['saved_run_report'],spec['matched_material_screen'])
             and _same_pin(numeric['relevant_input_pins']['material_contract'],spec['actual_material_contract'])
             and numeric['physical_reference_qualified'] is False,'Independent matched-cone review aliases differ')
    for key in ('source','authorized_config','raw_arrays','raw_frame_results','numeric_evidence'):checked(numeric[key])
    life=read(checked(spec['matched_material_lifecycle_review']))
    _passed(life,'full151_matched_material_cones_lifecycle_saved_independent:1','PASS_SAVED_LIFECYCLE_SOURCE_AND_COVERAGE')
    _require(_same_pin(life['evidence']['report'],spec['matched_material_screen']),'Lifecycle evidence bound elsewhere')
    checked(life['auditor'])
    legacy=read(checked(spec['legacy_proxy_dynamic_cone_review']))
    _require(legacy['schema']=='full151_declared_H123_dynamic_cone_LP_report:1'
             and legacy['status']=='COMPLETED_DECLARED_CONE_DIAGNOSTIC'
             and legacy['frame_status_counts']=={'FEASIBLE':23,'INFEASIBLE':128,'SOLVER_UNKNOWN':0}
             and legacy['physical_reference_qualified'] is False,'Preserve the earlier mu.1 proxy-cone negative result')
    _require(_same_pin(legacy['source_pins']['bundle'],spec['input_bundle']),'Legacy proxy evidence used another bundle')
    return {'prior':prior,'material':material,'matched':matched,'numeric':numeric,'legacy':legacy}


@contextmanager
def _preserve_rng():
    """Do not initialize CUDA to snapshot it; restore existing device states."""
    py=random.getstate();nps=np.random.get_state();cpu=torch.get_rng_state().clone()
    cuda_initialized=bool(torch.cuda.is_initialized())
    cuda=torch.cuda.get_rng_state_all() if cuda_initialized else None
    try:yield cuda_initialized
    finally:
        random.setstate(py);np.random.set_state(nps);torch.set_rng_state(cpu)
        if cuda_initialized:torch.cuda.set_rng_state_all(cuda)


def _device(spec,device):
    # Deny unapproved CUDA before constructor, CUDA discovery, RNG/device access.
    value=torch.device(device)
    _require(value.type in ('cpu','cuda'),'Only CPU/CUDA objective placement supported')
    _require(type(spec) is dict and spec.get('schema')=='full151_declared_H123_objective_spec:2'
             and spec.get('CPU_authorized') is True and spec.get('training_authorized') is True
             and type(spec.get('GPU_authorized')) is bool,'Wrong explicit exploratory factory authorization')
    if value.type=='cuda':_require(spec['GPU_authorized'] is True,'GPU objective factory is not authorized by this spec')
    return value


class Full151DeclaredH123ContactObjective(GeometricContactObjective):
    """New strict loader with exactly the frozen inherited forward/score/P."""
    def __init__(self,spec):
        # Do not call the old kinematic-reference manifest constructor.
        nn.Module.__init__(self)
        evidence=validate_spec(spec)
        manifest=read(checked(spec['binding_manifest']));receipt=read(checked(spec['target_receipt']))
        independent=read(checked(spec['independent_target_review']))
        _require(manifest['schema']=='full151_observation_declared_intent_binding:1' and manifest['target_schema']=='full151_observation_declared_surface_intent_targets:1','Do not masquerade as a legacy kinematic manifest')
        _require(receipt['status']=='INDEPENDENT_TARGET_DATA_BUILT_NOT_OBJECTIVE_READY' and receipt['passed']==receipt['total']==len(receipt['checks']) and all(v is True for v in receipt['checks'].values()) and receipt['binding_manifest']==spec['binding_manifest'] and receipt['target']==manifest['target'],'Actual numeric target receipt/aliases do not match')
        _require(independent['schema']=='full151_declared_targets_independent_numpy:1' and independent['status']=='PASS_DECLARED_NUMERIC_DATA_ONLY' and independent['passed']==independent['total']==len(independent['checks']) and all(v is True for v in independent['checks'].values()) and independent['binding']==spec['binding_manifest'] and independent['target']==manifest['target'],'Independent target representation review missing or bound elsewhere')
        checked(independent['auditor'])
        _require(manifest['reference_states']==151 and manifest['control_intervals']==150 and manifest['source_window_half_open']==[200,351] and manifest['retarget_variant_id']==VARIANT and manifest['contact_intent_variant_id']==INTENT,'Wrong window/variant/horizon')
        _require(manifest['known_semantics']==SEMANTICS and manifest['target_array_numeric_ready'] is True and manifest['objective_loading_permitted'] is False and manifest['adapter_status']=='PENDING_NEW_FULL151_BINDING_ADAPTER_REVIEW','Preserve the producer stage; this CPU adapter is separately assessed')
        for field in ('target_defined_from_policy_contacts','target_defined_from_generated_robot_contacts','robotq_used_to_define_targets','human_contact_observation_known','source_confidence_known','physical_reference_qualified','qualified_contact_predictor','forceclosure_qualified','dynamics_gradient_qualified','training_authorized'):
            _require(manifest[field] is False,'Unqualified/source-independent boundary changed: '+field)
        _require(manifest['local_proxy_qualification']=='FAILED_RETAINED_EXPLORATORY_ONLY' and manifest['compiled_precision_qualification']=='FAILED_RETAINED_NOT_REVALIDATED','Cannot inherit predictive or compiled qualification')
        for value in manifest['source_pins'].values():checked(value)
        _require(_same_pin(manifest['full151_input_bundle'],spec['input_bundle']),'Wrong full151 input-bundle pin')
        from full151_native.bundle import load_bundle
        bundle=load_bundle(checked(spec['input_bundle']),expected_sha256=spec['input_bundle']['sha256'])
        _require(_same_pin(manifest['reference_arrays_hash_binding_only'],bundle['spec']['source_pins']['reference_arrays']),'Reference array identity differs')
        _require(_same_pin(manifest['source_pins']['model'],bundle['model']) and _same_pin(manifest['source_pins']['actual_context'],bundle['spec']['context']),'Nominal model/actual-context aliases differ')
        _require(_same_pin(evidence['material']['inputs']['model'],bundle['model'])
                 and _same_pin(evidence['material']['inputs']['actual_context'],bundle['spec']['context']),'Actual material/bundle aliases differ')
        _require(_same_pin(evidence['matched']['source_pins']['old_H123_targets'],manifest['target'])
                 and _same_pin(evidence['numeric']['relevant_input_pins']['old_H123_targets'],manifest['target'])
                 and _same_pin(evidence['legacy']['source_pins']['targets'],manifest['target']),'Cone evidence belongs to another H123 target')
        a=loadnp(checked(manifest['target']));active,known=validate_arrays(a)
        # Full-bundle validation is an identity/ABI check. No q is used to
        # construct the already-frozen observation-only target support.
        reference=bundle['arrays']['reference']
        _require(np.array_equal(a['time_s'],reference['time_s']) and np.array_equal(a['mapped_object_pose_w'],reference['mapped_object_pose_w']),'Target time/source object pose not aligned with current reference')
        s=manifest['support'];_require(all(s[k]==v for k,v in {'basis_count':512,'training_stride':8,'mu':.1,'edges':8,'normal_ray':True,'unit_force_rays':True,'active_threshold':.001,'moment_radius_divisions':1,'moment_origin':'objectCOM_BODY','normal':'force-on-object inward=-outward'}.items()),'Frozen cone/origin/radius convention changed')
        import mujoco
        model=mujoco.MjModel.from_binary_path(str(checked(bundle['model'])))
        body=model.body('object/cube').id;com=model.body_ipos[body].copy()
        radius=max(float(np.linalg.norm(model.geom_pos[g]-com)+model.geom_rbound[g]) for g in np.flatnonzero(model.geom_bodyid==body))
        _require(np.array_equal(a['object_com_offset_body_m'],com) and np.isclose(float(a['object_nominal_enclosing_radius_m']),radius,rtol=0,atol=1e-9),'Target COM/nominal enclosing radius differs; do not use crosssection radius or subtract COM twice')
        differences={}
        for name in ('geom_size','geom_pos','geom_friction'):
            value=np.asarray(bundle['context'][name]);_require(value.shape[1:]==getattr(model,name).shape and np.array_equal(value,np.broadcast_to(value[:1],value.shape)),'Actual geometry must be fixed/shared: '+name)
            differences[name]=float(np.max(np.abs(getattr(model,name)-value[0])))
            getattr(model,name)[:]=value[0]
        basis=a['support_basis'];support=a['support512'];subset=np.arange(0,512,8,dtype=np.int64)
        self.geometry_proxy=ContactGeometryProxy(model,JOINT_NAMES,basis[subset],radius,points_per_geom=64,dtype=torch.float32)
        for name,value,dtype in [('basis512',basis,torch.float32),('basis_subset_indices',subset,torch.long),('reference_support512',support,torch.float32),('reference_support64',support[...,subset],torch.float32),('contact_known',known,torch.bool),('known_no_contact',a['known_no_contact'],torch.bool),('phase',a['phase'],torch.long),('phase_weight',a['phase_weight'],torch.float32)]:
            self.register_buffer(name,torch.tensor(value,dtype=dtype))
        self.frame_count=151;self.bandwidth_m=.001;self.tolerance=.1;self.var=.1
        self.metadata={'schema':'full151_observation_declared_geometry_shaping:1','scientific_role':'exploratory soft geometric prior, not true force/contact dynamics','retarget_variant_id':VARIANT,'contact_intent_variant_id':INTENT,'source_window_half_open':[200,351],
          'input_bundle':spec['input_bundle'],'binding_manifest':spec['binding_manifest'],'target':manifest['target'],'target_receipt':spec['target_receipt'],'independent_target_review':spec['independent_target_review'],'source_pins':code_pins(),'actual_context':manifest['source_pins']['actual_context'],
          'known_semantics':SEMANTICS,'human_contact_observation_known':False,'source_confidence_known':False,'phase3_semantics':'declared persistent H123 intent with any geometric-valid slot; not measured movement','known_frame_count':int(known.sum()),'active_slot_count':int(active.sum()),
          'target_defined_from_policy_contacts':False,'robotq_used_to_define_targets':False,'producer_objective_loading_permitted':False,'producer_status_retained':manifest['adapter_status'],
          'frame_count':151,'active_real_imagination_intervals':[0,150],'terminal_target150':'finite diagnostic; caller full151runtime masks time>=150 reward/bootstrap/critic','frame_index':'frozen (start+clamp_min(time,0)).minimum(end-1); requires start0/end151 and integer phases from runtime',
          'scoring':{'kind':'official tolerance-band object-centric union-support score','tolerance':.1,'variance':.1,'active_threshold':.001,'support_coefficient':1.,'bandwidth_m':.001,'no_finger_identity_or_extra_contact_penalty':True,'not_native_friction_or_force_amplitude':True},
          'formula':'phase_weight*official_support_match-.1*penetration_prior','unknown_masks_attraction_only':True,'penetration_allowance_m':.003,'penetration_normalization_m':.001,'dt_applied_inside_module':False,'caller_dt_s':.02,
          'basis':{'training_directions':64,'evaluation_directions':512,'training_indices':subset.tolist(),'basis512_sha256':hashlib.sha256(np.ascontiguousarray(basis).tobytes()).hexdigest()},
          'geometry':self.geometry_proxy.metadata,'joint_names':JOINT_NAMES,'actual_geometry_host_copy_fields':list(differences),'nominal_to_actual_geometry_max_abs':differences,'host_model_uploaded_to_physics':False,
          'COM_origin':'target already COM-relative; proxy subtracts its BODY COM once before cross product; torque divided by nominal radius once','object_nominal_enclosing_radius_m':radius,'object_com_offset_body_m':com.tolist(),
          'local_proxy_qualification':'FAILED_RETAINED_EXPLORATORY_ONLY','compiled_precision_qualification':'FAILED_RETAINED_NOT_REVALIDATED','qualified_contact_predictor':False,'forceclosure_qualified':False,'dynamics_gradient_qualified':False,'physical_reference_qualified':False,'training_authorized':False,'GPU_authorized':False,'training_runtime_integration':'PENDING_NOT_INSERTED_IN_A1L'}


        self.metadata.update(schema='full151_observation_declared_geometry_shaping:2',
          execution_scope='EXPLORATORY_MATCHED_TRAINING',CPU_authorized=True,
          GPU_authorized=spec['GPU_authorized'],training_authorized=True,
          training_runtime_integration='SEPARATE_LEARNING_HOOK_REQUIRED_NOT_QUALIFIED_BY_FACTORY',
          objective_spec_schema=spec['schema'],objective_spec_sha256=hashlib.sha256(canonical(spec).encode()).hexdigest(),
          prerequisites={k:spec[k] for k in ('prior_CPU_binding_review','actual_material_contract','matched_material_screen','matched_material_saved_review','matched_material_lifecycle_review','legacy_proxy_dynamic_cone_review')},
          matched_native_parameter_cone={'condition':'H123','feasible_frames':151,'frame_count':151,
            'material_mu_sliding':evidence['material']['friction5'][0],
            'material_torsion_m':evidence['material']['friction5'][2],'condim':4,
            'columns_per_contact':6,'normalforce_uppercap':None,
            'qualification':'conditional hypothetical source-surface cone only; not physical/actuator/forceclosure/dynamics gradient',
            'selection':'H123 was declared before this matched comparison; no H12345 winner selection'},
          representation_vs_native={'target_support_mu':.1,'target_unitforce_edges':8,'target_normal_ray':True,
            'target_direct_torsion':False,'native_material_mu_sliding':evidence['material']['friction5'][0],
            'native_condim':4,'same_representation':False,'forward_and_targets_changed':False},
          legacy_proxy_cone_status={'feasible':23,'solver_infeasible':128,'frame0_feasible':False,
            'retained':True,'explanation':'Different mu.1 unit-force nine-ray representation; not relabeled by native-parameter evidence'},
          GPU_runtime_qualification='PENDING_REAL_GPU_FACTORY_AND_LEARNING_SMOKE',
          physical_reference_qualified=False,forceclosure_qualified=False,dynamics_gradient_qualified=False)


def _build(spec,device,penalty_only):
    value=_device(spec,device)
    with _preserve_rng() as cuda_preinitialized:
        module=Full151DeclaredH123ContactObjective(spec)
        if penalty_only:
            module=PenaltyOnlyPrior(module)
            module.metadata.update(schema='full151_declared_H123_penalty_only:2')
        module=module.to(device=value).eval().requires_grad_(False)
        module.metadata.update(factory_device=str(value),CUDA_initialized_before_factory=cuda_preinitialized,
          RNG_contract='Python/NumPy/TorchCPU restored; CUDA states restored iff already initialized; no CUDA initialization for RNG snapshot')
    return module


def build_contact_objective(spec,device='cpu'):
    """Frozen A2 math, new explicit enrollment and device/RNG contract."""
    return _build(spec,device,False)


def build_penalty_only_objective(spec,device='cpu'):
    """Frozen eager P_ONLY math; source support forward remains diagnostic."""
    return _build(spec,device,True)
