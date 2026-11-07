"""Saved-only independent native151 source/frame/initial/ABI review; no FK."""
from __future__ import annotations
import argparse,ast,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def rot(q):
    q=np.asarray(q,np.float64);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def main(bundle_path,output):
    tick=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks={};numeric={}
    def check(k,v):
        checks[k]=bool(v)
        if not checks[k]:raise AssertionError(k)
    try:
        spec=read(bundle_path);pins=spec['source_pins']
        for name,p in pins.items():check('source_pin__'+name,pin(p['path'])==p)
        parent=read(pins['parent_scp_receipt']['path']);native_meta=read(pins['native_manifest']['path']);proof=read(pins['native_reader_proof']['path'])
        sidecar=read(pins['native_sidecar']['path']);r=load(pins['reference_arrays']['path']);i=load(pins['init_arrays']['path']);c=load(pins['control_arrays']['path']);n=load(pins['native_data']['path']);reader=load(pins['native_reader_arrays']['path'])
        descriptor=read(Path(bundle_path).with_name('descriptor.json'))
        canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
        check('descriptor_input_signature',hashlib.sha256(canonical({k:v for k,v in descriptor.items() if k!='input_signature_sha256'}).encode()).hexdigest()==descriptor['input_signature_sha256'])
        check('descriptor_bundle_pin',descriptor['input_bundle']==pin(bundle_path))
        check('descriptor_pins_match_spec',descriptor['source_pins']==pins)
        check('new_schema_mode_variant',spec['schema']=='full151_knownrough_input_bundle:1' and descriptor['schema']=='full151_knownrough_controller_input_descriptor:1' and spec['mode']==descriptor['mode']=='rough_observation_reference_full151_v1' and spec['retarget_variant_id']==descriptor['retarget_variant_id']=='legacy_H123_full151_SCP8_knownrough_v1')
        check('same_legacy_source_window_not_shared_clip',spec['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,351] and spec['raw_source_identity']['local_reference_range_half_open']==[0,151] and sidecar['provenance']['source_window_half_open']==[200,351])
        check('N151_M150_no_extension_or_truncation',r['q_goal_rad'].shape==(151,20) and r['mapped_object_pose_w'].shape==(151,7) and c['u_postC2_150_FP32_rad'].shape==(150,20) and reader['trajectory_lengths'].tolist()==[151])
        check('parent_source_FAIL_preserved',parent['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' and parent['last_accepted_major']==8 and {k for k,v in parent['checks'].items() if not v}=={'last_accepted_eta_final_gate','final_fresh_sampled_geometry'})
        check('no_geometric_physical_training_shared_promotion',all(spec[k] is False for k in ('physical_reference_qualified','geometric_candidate_qualified','training_authorized','runtime_implementation_ready','shared10_enrollment')) and descriptor['rough_input_status']=='NEW_ENGINEERING_INPUT_ONLY_NOT_CPU_GEOMETRY_PASS')
        check('no_human_contact_labels_or_A2_targets',spec['contact_intent_bundle'] is None and spec['human_contact_observation_known'] is False and descriptor['contact_intent_target_ready'] is False)
        check('sourceq_is_derived_not_proprioception',sidecar['provenance']['source_robot_labels_used'] is False and sidecar['provenance']['is_real_ego'] is False and sidecar['provenance']['geometric_derivative']==pins['reference_arrays'] and sidecar['provenance']['observation_source']==pins['observation_fixture'])
        check('native_aliases',native_meta['native']==pins['native_data'] and native_meta['reader_arrays']==pins['native_reader_arrays'] and native_meta['reader_proof']==pins['native_reader_proof'] and native_meta['recipe']==pins['native_recipe'])
        check('sidecar_aliases_exact_current_data',native_meta['native_provenance']==pins['native_sidecar'] and Path(spec['native_reference']['provenance_path']).resolve()==Path(pins['native_sidecar']['path']).resolve() and sidecar['native_npz_sha256']==pins['native_data']['sha256'])
        check('actual_Reader_source_pin',proof['native_reader_source']==pins['build_source__native_reader'] and proof['native_reader_arrays']==pins['native_reader_arrays'])
        check('native_only_four_fields',set(n)=={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} and n['link_pos_w'].shape==(151,21,3))
        names=['right_palm_link']+[f'right_finger{f}_link{l}' for f in range(1,6) for l in range(1,5)]
        check('BODY21_names_are_not_IK_macro_points',sidecar['native_body_link_names']==names and 'Fresh BODY FK' in sidecar['link_source'])
        check('q_native_source64_exact',exact(n['reference_qpos'],r['q_goal_rad']))
        check('object_native_source64_exact',exact(n['object_pose_w'],r['mapped_object_pose_w']))
        check('reader_q_cast_exact',exact(reader['joint_pos'],r['q_goal_rad'].astype(np.float32)))
        check('fixed_default_native_palm_wrist',exact(n['wrist_pose_w'],np.broadcast_to(n['wrist_pose_w'][:1],(151,7))))
        Rw=rot(n['wrist_pose_w'][:,3:]);Ro=rot(n['object_pose_w'][:,3:])
        expected_obj=np.einsum('tji,tj->ti',Rw,n['object_pose_w'][:,:3]-n['wrist_pose_w'][:,:3])
        expected_R=np.swapaxes(Rw,-1,-2)@Ro;expected_links=np.einsum('tji,tkj->tki',Rw,n['link_pos_w']-n['wrist_pose_w'][:,None,:3])
        numeric['reader_object_p_transform_max_m']=float(np.max(np.abs(reader['object_pose'][:,:3]-expected_obj)))
        numeric['reader_object_R_transform_max']=float(np.max(np.abs(rot(reader['object_pose'][:,3:])-expected_R)))
        numeric['reader_BODY21_transform_max_m']=float(np.max(np.abs(reader['link_pos']-expected_links)))
        check('independent_world_to_palm_transform_parity',all(v<=1e-6 for v in numeric.values()))
        q32=r['q_goal_rad'].astype(np.float32);fd32=np.empty_like(q32);fd32[:-1]=(q32[1:]-q32[:-1])/np.float32(.02);fd32[-1]=fd32[-2]
        check('native_FD32_forward_lastcopy_preserved',exact(reader['joint_vel'],fd32) and exact(fd32,r['native_forward_FD32_rad_s']))
        numeric['nativeFD_vs_analytic_initial_max_rad_s']=float(np.max(np.abs(reader['joint_vel'][0]-r['analytic_qd151_rad_s'][0].astype(np.float32))))
        check('analytic_initial_qd_authority_distinct_from_nativeFD',numeric['nativeFD_vs_analytic_initial_max_rad_s']>0 and exact(i['qd_init_requested_FP32_rad_s'],r['analytic_qd151_rad_s'][0].astype(np.float32)))
        check('init_unprojected_q0_and_object0_exactcast',exact(i['q_init_requested_FP32_rad'],r['q_goal_rad'][0].astype(np.float32)) and exact(i['object_pose_init_requested_FP32_w'],r['mapped_object_pose_w'][0].astype(np.float32)))
        check('init_nonzero_world_v_and_world_omega',exact(i['object_bodyorigin_v_requested_FP32_world_m_s'],r['object_v_world_m_s'][0].astype(np.float32)) and exact(i['object_angular_requested_FP32_world_rad_s'],r['object_omega_world_rad_s'][0].astype(np.float32)))
        wanted_body=(rot(i['object_pose_init_requested_FP32_w'][3:]).T@r['object_omega_world_rad_s'][0]).astype(np.float32)
        check('body_omega_from_FP32_pose_world_omega_exact',exact(wanted_body,i['object_angular_requested_FP32_body_rad_s']))
        check('same_u0_and_150_controls',exact(i['u_planned0_FP32_rad'],c['u_postC2_150_FP32_rad'][0]) and exact(c['u_pre150_FP32_rad'],c['u_postC2_150_FP32_rad']))
        check('canonical_151_601_times',exact(r['time_s'],np.arange(151)*.02) and exact(r['planned_time601_s'],np.arange(601)*.005))
        trial=load(Path(pins['parent_scp_receipt']['path']).parent/'major08/trial_control_points.npz')
        check('final_reference_is_lastaccepted8_controlpoints',any(exact(v,r['control_points_rad']) for v in trial.values()))
        from scipy.interpolate import BSpline
        bs=BSpline(r['Bspline_knots_s'],r['control_points_rad'],3,extrapolate=False)
        numeric['saved_qcurve_reconstruction_max_rad']=float(np.max(np.abs(bs(r['time_s'])-r['q_goal_rad'])))
        numeric['saved_analytic_qd_reconstruction_max_rad_s']=float(np.max(np.abs(bs(r['time_s'],nu=1)-r['analytic_qd151_rad_s'])))
        check('saved_Bspline_q_and_qd_relation',numeric['saved_qcurve_reconstruction_max_rad']<=1e-12 and numeric['saved_analytic_qd_reconstruction_max_rad_s']<=1e-10)
        import mujoco
        m=mujoco.MjModel.from_binary_path(pins['model']['path'])
        palm=m.body('robot/right_palm_link').id;mount=m.body('robot/mocap_base').id;obj=m.body('object/cube').id
        check('bound_asset_palm_identity_child_of_root_mocap',m.body_parentid[palm]==mount and m.body_mocapid[mount]==0 and m.body_mocapid[palm]==-1 and m.body_jntnum[palm]==0 and np.array_equal(m.body_pos[palm],np.zeros(3)) and np.array_equal(m.body_quat[palm],np.array([1.,0,0,0])) and m.body_parentid[mount]==0)
        check('saved_wrist_matches_default_model_root_pose',exact(n['wrist_pose_w'][0],np.r_[m.body_pos[mount],m.body_quat[mount]]))
        context=load(pins['actual_context']['path'])
        check('object_COM_offset_zero_for_world_v_convention',np.array_equal(m.body_ipos[obj],np.zeros(3)))
        check('actual8_context_pinned_separate_from_nominal_geometry',all(k in context for k in ('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')))
        proto=read(pins['build_protocol']['path']);builder=Path(proto['sources']['builder']['path']).read_text(encoding='utf-8-sig');tree=ast.parse(builder)
        check('static_builder_consumes_only_derived_qpose_as_native_inputs',"ref['q_goal_rad'],ref['mapped_object_pose_w']" in builder and "source_robot_labels_used':False" in builder)
        check('static_collision_physics_solver_guards',all("'"+name+"'" in builder for name in ('mj_step','mj_forward','mj_collision','mj_geomDistance','linprog','minimize','least_squares')))
        check('source_bound_151FK_counts_and_forbidden0_not_new_queries',proof['kinematics_calls']==151 and not any(proof['forbidden_calls'].values()))
        check('all_sources_remain_unchanged',all(pin(p['path'])==p for p in pins.values()))
        status='PASS_SAVED_NATIVE151_INPUT_BINDING_ONLY'
    except Exception as e:status='INDEPENDENT_REVIEW_FAILURE_RETAINED';numeric['exception']=repr(e);numeric['traceback']=traceback.format_exc()
    report={'schema':'full151_native_input_saved_independent:1','status':status,'reviewer':pin(__file__),'bundle':pin(bundle_path),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'numeric':numeric,'wall_s':time.perf_counter()-tick,
            'source_bound_only':['The exporter151 FK count and zero forbidden-call claims are the pinned original proof, not an offline replay of the native library.','BODY21 link origins are saved FK results. This reviewer only transforms their coordinates; it does not recompute FK.'],
            'scope':'151 states/150 targets of same legacy observed-source lineage[200,351). Geometry and parentSCP FAIL retained; unknown human contact. No checkpoint, training, dynamics or physical qualification.',
            'independent_new_FK':0,'geometry_queries':0,'Env_created':False,'physics_steps':0,'GPU_used':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print({'status':status,'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')},flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.bundle,a.output))
