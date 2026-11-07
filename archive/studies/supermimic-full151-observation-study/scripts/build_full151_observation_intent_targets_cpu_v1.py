"""Declared H123 source-surface intent, independent of robot/policy contacts.

One CPU math pass. No distance threshold, release inference, query or physics.
Output is a new target-data schema; no objective adapter is implemented here.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(p):
    if set(p)!={'path','sha256'} or not Path(p['path']).is_absolute() or sha(p['path'])!=p['sha256']:raise ValueError('Pinned source changed')
    return Path(p['path'])
def selected(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main(protocol_path,output):
    tick=time.perf_counter();p=read(protocol_path);sources=p['sources'];inputs=p['inputs']
    for v in [*inputs.values(),*sources.values()]:checked(v)
    if p['schema']!='full151_observation_H123_support_CPU_protocol:1' or checked(sources['producer']).resolve()!=Path(__file__).resolve():raise ValueError('Wrong producer protocol')
    if p['numeric_support']!={'basis_count':512,'basis_seed':0,'training_stride':8,'mu':.1,'edges':8,'normal_ray':True,'unit_force_rays':True,'active_threshold':.001,'moment_radius_divisions':1}:raise ValueError('Frozen support math changed')
    if p['mask_rule']!='declared_H123_all151_AND_numeric_unambiguous_geometry; frame_known=any_active; no_distance_threshold':raise ValueError('Wrong declaration/mask rule')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks={};calls={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    import torch
    torch.set_num_threads(1)
    check('CPU_only_CUDA_uninitialized',not torch.cuda.is_initialized() and not torch.cuda.is_available())
    import mujoco
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos'):
        calls[name]=0
        def deny(*args,_name=name,**kwargs):calls[_name]+=1;raise AssertionError('Target producer forbids '+_name)
        setattr(mujoco,name,deny)
    keys=('time_s','finger_ids','landmark_indices','mapped_object_pose_w','object_surface_closest_body_m','object_surface_outward_normal_body',
          'surface_point_numeric_valid','surface_normal_valid_unambiguous','geometric_branch_unknown','surface_feature_index','surface_tie_count',
          'signed_distance_m','unsigned_distance_m','human_contact_known','human_contact_present','human_contact_unknown_mask',
          'landmark_confidence_known','object_confidence_known','declared_robot_intent_known')
    a=selected(checked(inputs['observation_cues']),keys);meta=read(checked(inputs['cue_metadata']))
    check('original_cues_stay_not_objective_targets',meta['target_ready'] is False and meta['objective_loading_permitted'] is False and meta['saved_matrix_only'] is True and meta['no_material_anchor_transport_or_repeat16'] is True)
    check('H123_declared_membership_only',read(checked(inputs['H123_declaration']))['hypothesis_id']=='H123' and np.array_equal(a['finger_ids'],[1,2,3]) and np.array_equal(a['landmark_indices'],[3,7,11]))
    check('all151_canonical_source_frames',a['time_s'].shape==(151,) and np.array_equal(a['time_s'],np.arange(151)*.02) and meta['source_window_half_open']==[200,351])
    check('human_contact_and_confidence_remain_UNKNOWN',not a['human_contact_known'].any() and not a['human_contact_present'].any() and a['human_contact_unknown_mask'].all() and not a['landmark_confidence_known'].any() and not a['object_confidence_known'].any() and not a['declared_robot_intent_known'].any())
    model=mujoco.MjModel.from_binary_path(str(checked(inputs['model'])));obj=model.body('object/cube').id;og=np.flatnonzero(model.geom_bodyid==obj)
    com=model.body_ipos[obj].copy();radius=max(float(np.linalg.norm(model.geom_pos[g]-com)+model.geom_rbound[g]) for g in og)
    check('bound_actual_surface_geometry_source',meta['surface_metadata']['model_sha256']==inputs['model']['sha256'] and meta['surface_metadata']['context_sha256']==inputs['actual_context']['sha256'] and meta['surface_metadata']['seams_abut_and_box_crosssection_encloses_cylinder'] is True)
    check('COM_enclosing_radius_positive_not_cylinder_radius',np.isfinite(radius) and radius>meta['surface_metadata']['radius_m'] and np.isfinite(com).all())
    # The actual new reference is crossbound using time/pose ONLY. No robot q or
    # generated native contact field is read, even from the saved reference zip.
    bundle=read(checked(inputs['full151_input_bundle']));rpin=bundle['source_pins']['reference_arrays']
    ref=selected(checked(rpin),('time_s','mapped_object_pose_w'))
    check('full151_binding_same_source_pose_time_only',np.array_equal(ref['time_s'],a['time_s']) and np.array_equal(ref['mapped_object_pose_w'],a['mapped_object_pose_w']) and bundle['retarget_variant_id']=='legacy_H123_full151_SCP8_knownrough_v1' and bundle['physical_reference_qualified'] is False and bundle['geometric_candidate_qualified'] is False)
    geometry_valid=a['surface_point_numeric_valid']&a['surface_normal_valid_unambiguous']&~a['geometric_branch_unknown']
    declared=np.ones((151,3),bool);active=declared&geometry_valid;known=active.any(1)
    check('separate_geometric_and_declared_masks',geometry_valid.shape==declared.shape==(151,3) and geometry_valid.dtype==np.bool_ and declared.dtype==np.bool_)
    point=a['object_surface_closest_body_m']-com;inward=-a['object_surface_outward_normal_body']
    point=np.where(active[...,None],point,0.);inward=np.where(active[...,None],inward,0.)
    check('finite_COM_points_and_inward_unit_normals',np.isfinite(point).all() and np.isfinite(inward).all() and np.all(np.abs(np.linalg.norm(inward,axis=-1)[active]-1)<=1e-6))
    legacy=selected(checked(inputs['legacy_basis_archive_only']),('support_basis',))['support_basis']
    sp=importlib.util.spec_from_file_location('_full151_intent_frozen_contacts',checked(sources['contacts_math']));math=importlib.util.module_from_spec(sp);sp.loader.exec_module(math)
    basis=math.sample_basis(512,0,dtype=torch.float64).numpy();check('same_frozen512_basis_no_old_targets_read',np.array_equal(basis,legacy))
    support=math.wrench_support(torch.from_numpy(point),torch.from_numpy(inward),torch.from_numpy(active),torch.from_numpy(basis),radius,mu=.1,edges=8).numpy()
    empty=math.wrench_support(torch.from_numpy(point),torch.from_numpy(inward),torch.zeros_like(torch.from_numpy(active)),torch.from_numpy(basis),radius,mu=.1,edges=8).numpy()
    count512=np.sum(support>.001,axis=-1);count64=np.sum(support[:,::8]>.001,axis=-1)
    check('support_finite_nonnegative_151x512',support.shape==(151,512) and np.isfinite(support).all() and (support>=0).all())
    check('known_frames_nonzero512_and_training64',bool(known.any()) and np.all(count512[known]>0) and np.all(count64[known]>0))
    check('all_unknown_negative_control_zero_support',np.array_equal(empty,np.zeros_like(empty)))
    phase=np.where(known,3,0).astype(np.int32);weights=known.astype(np.float64)
    target={'time_s':a['time_s'],'finger_ids':a['finger_ids'],'landmark_indices':a['landmark_indices'],'mapped_object_pose_w':a['mapped_object_pose_w'],
            'geometry_valid_mask':geometry_valid,'declared_intent_mask':declared,'declared_intent_active':active,'declared_intent_frame_known':known,
            'known_no_contact':np.zeros(151,bool),'human_contact_known':a['human_contact_known'],'human_contact_present':a['human_contact_present'],'human_contact_unknown_mask':a['human_contact_unknown_mask'],
            'landmark_confidence_known':a['landmark_confidence_known'],'object_confidence_known':a['object_confidence_known'],
            'point_com_body_m':point,'inward_normal_body':inward,'object_com_offset_body_m':com,'object_nominal_enclosing_radius_m':np.array(radius),
            'raw_source_signed_distance_m':a['signed_distance_m'],'raw_source_unsigned_distance_m':a['unsigned_distance_m'],'surface_feature_index':a['surface_feature_index'],'surface_tie_count':a['surface_tie_count'],
            'phase':phase,'phase_weight':weights,'support_basis':basis,'support512':support[:,None,None,:]}
    np.savez_compressed(out/'intent_targets.npz',**target)
    shifts=np.linalg.norm(a['object_surface_closest_body_m']-a['object_surface_closest_body_m'][:1],axis=-1)
    n=a['object_surface_outward_normal_body'];angles=np.rad2deg(np.arccos(np.clip(np.sum(n*n[:1],axis=-1),-1,1)))
    adjacent=np.rad2deg(np.arccos(np.clip(np.sum(n[1:]*n[:-1],axis=-1),-1,1)))
    facts={'known_hypothesis_frames':int(known.sum()),'active_declared_slots':int(active.sum()),'geometry_invalid_slots':int((~geometry_valid).sum()),'human_known_slots':0,
           'basis512_active_direction_count_minmax':[int(count512[known].min()),int(count512[known].max())],'basis64_active_direction_count_minmax':[int(count64[known].min()),int(count64[known].max())],
           'support512_max':float(support.max()),'support512_mean':float(support.mean()),'object_COM_body_m':com.tolist(),'support_nominal_enclosing_radius_m':radius,'surface_cylinder_crosssection_radius_m':meta['surface_metadata']['radius_m'],
           'per_finger':[{ 'finger_id':int(a['finger_ids'][j]),'landmark_index':int(a['landmark_indices'][j]),'source_signed_distance_minmax_m':[float(a['signed_distance_m'][:,j].min()),float(a['signed_distance_m'][:,j].max())],
               'closest_surface_BODY_shift_from_frame0_max_m':float(shifts[:,j].max()),'normal_from_frame0_max_deg':float(angles[:,j].max()),'normal_adjacent_frame_max_deg':float(adjacent[:,j].max())} for j in range(3)]}
    binding={'schema':'full151_observation_declared_intent_binding:1','target_schema':'full151_observation_declared_surface_intent_targets:1',
       'target':pin(out/'intent_targets.npz'),'target_array_numeric_ready':True,'objective_loading_permitted':False,'adapter_status':'PENDING_NEW_FULL151_BINDING_ADAPTER_REVIEW',
       'retarget_variant_id':'legacy_H123_full151_SCP8_knownrough_v1','contact_intent_variant_id':'source_rigid_H123_surface_projection_all151_v1','source_window_half_open':[200,351],
       'reference_states':151,'control_intervals':150,'full151_input_bundle':inputs['full151_input_bundle'],'reference_arrays_hash_binding_only':rpin,
       'known_semantics':'declared_H123_observation_surface_projection_design_hypothesis_only_not_human_contact',
       'target_defined_from_policy_contacts':False,'target_defined_from_generated_robot_contacts':False,'robotq_used_to_define_targets':False,
       'human_contact_observation_known':False,'source_confidence_known':False,'known_no_contact_frames':0,'support_experiment_degenerate_to_penalty_only':not bool(known.any()),
       'phase_rule':'3=continuous declared H123 intent when any valid slot;0=all geometric slots unknown. No speed/distance threshold, no release/switch inference. Weights1/0; 2/4 unused.',
       'support':p['numeric_support']|{'moment_origin':'objectCOM_BODY','normal':'force-on-object inward=-outward','radius_definition':'Nominal hashedMJB max(norm(geom_pos-COM)+geom_rbound); not pen crosssection radius; divide moment once','not_native_friction_or_force_amplitude':True},
       'frame_contract':'targets0..150 from151 distinct sourceframes; native reward uses current preactionphase0..149; terminal150 buffer is finite but real/active-imagination reward is disabled by separately reviewed full151runtime.',
       'missing_contact':'Geometrically invalid slots unknown, not release/prohibition. Global union support; no finger identity or extra-contact penalty.',
       'physical_reference_qualified':False,'qualified_contact_predictor':False,'local_proxy_qualification':'FAILED_RETAINED_EXPLORATORY_ONLY','compiled_precision_qualification':'FAILED_RETAINED_NOT_REVALIDATED',
       'forceclosure_qualified':False,'dynamics_gradient_qualified':False,'training_authorized':False,
       'source_pins':inputs|sources|{'protocol':pin(protocol_path)},'read_fields':{'cue_npz':list(keys),'reference_npz':['time_s','mapped_object_pose_w'],'H123_declaration':['hypothesis_id'],'legacy_basis_archive':['support_basis']}}
    write(out/'binding_manifest.json',binding)
    check('all_input_source_pins_unchanged',all(checked(v) for v in [*inputs.values(),*sources.values()]))
    check('all_native_queries_zero',not any(calls.values()))
    check('no_CUDA_initialization',not torch.cuda.is_initialized())
    report={'schema':'full151_declared_H123_surface_support_CPU_receipt:1','status':'INDEPENDENT_TARGET_DATA_BUILT_NOT_OBJECTIVE_READY','protocol':pin(protocol_path),'target':pin(out/'intent_targets.npz'),'binding_manifest':pin(out/'binding_manifest.json'),
       'checks':checks,'passed':sum(checks.values()),'total':len(checks),'facts':facts,'wall_s':time.perf_counter()-tick,'forbidden_native_calls':calls,
       'support_nondegenerate_is_not_forceclosure':True,'not_observed_human_contact_or_force':True,'objective_loading_permitted':False,'training_authorized':False,'GPU_used':False,
       'scope':'151 source-only projected surface points/normals under a separate explicit persistentH123 design hypothesis; no robotq/contact/policy target mining, no16padding orframe0materialtransport. Nonzero unit-wrench fingerprint is a representation check, not actualfriction/forceclosure/trajectory-demand feasibility or useful dynamics gradients.'}
    write(out/'receipt.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'facts':facts,'receipt':pin(out/'receipt.json')},indent=2));return 0


if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--protocol',required=True,type=Path);x.add_argument('--output',required=True,type=Path);a=x.parse_args();raise SystemExit(main(a.protocol.resolve(strict=True),a.output))
