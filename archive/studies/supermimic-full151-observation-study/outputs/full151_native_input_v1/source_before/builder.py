"""Fresh BODY21/native Reader151 binding of a retained known-rough candidate."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import read,write,pin,checked,loadnp
from full151_native.metadata import MODE,VARIANT,BUNDLE_SCHEMA


def main(protocol_path):
    tick=time.perf_counter();p=read(protocol_path)
    if p['schema']!='full151_knownrough_native_input_CPU:1':raise ValueError('Wrong build protocol')
    for value in [*p['inputs'].values(),*p['sources'].values()]:checked(value)
    if checked(p['sources']['builder'])!=Path(__file__).resolve():raise ValueError('Wrong builder source')
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'full151_knownrough_input_CPU_receipt:1','status':'STARTED','protocol':pin(protocol_path),'checks':{},'physics_steps':0,'geometry_queries':0,'training_executed':False,'GPU_started':False}
    original={};blocked={};kin_calls=0
    try:
        import mujoco
        import scipy.optimize as optimize
        ref=loadnp(checked(p['inputs']['reference_arrays']));init=loadnp(checked(p['inputs']['init_arrays']));control=loadnp(checked(p['inputs']['control_arrays']))
        source=read(checked(p['inputs']['parent_scp_receipt']))
        if source['status']!='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' or source['last_accepted_major']!=8 or {k for k,v in source['checks'].items() if not v}!={'last_accepted_eta_final_gate','final_fresh_sampled_geometry'}:raise ValueError('Retained candidate failure identity changed')
        def deny(name):
            def call(*a,**kw):blocked[name]+=1;raise AssertionError('Input ABI builder forbids '+name)
            return call
        for ns,names in [(mujoco,['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision']),
                         (optimize,['linprog','minimize','least_squares'])]:
            for name in names:original[(ns,name)]=getattr(ns,name);blocked[name]=0;setattr(ns,name,deny(name))
        oldkin=mujoco.mj_kinematics;original[(mujoco,'mj_kinematics')]=oldkin
        def kinematics(*a,**kw):
            nonlocal kin_calls
            kin_calls+=1;return oldkin(*a,**kw)
        mujoco.mj_kinematics=kinematics
        (out/'source_before').mkdir()
        for name,value in p['sources'].items():(out/'source_before'/(name+Path(value['path']).suffix)).write_bytes(checked(value).read_bytes())
        name='_full151_knownrough_frozen_native_exporter';s=importlib.util.spec_from_file_location(name,checked(p['sources']['native_exporter']));abi=importlib.util.module_from_spec(s);s.loader.exec_module(abi)
        group='legacy_full151_knownrough';clip='legacy007_f0200_n0151_H123_SCP8_v1';groupdir=out/'clips'/group
        provenance={'is_real_ego':False,'source_robot_labels_used':False,'observation_source':p['inputs']['observation_fixture'],'geometric_derivative':p['inputs']['reference_arrays'],
          'retarget_variant_id':VARIANT,'source_window_half_open':[200,351],'parent_scp_receipt':p['inputs']['parent_scp_receipt'],
          'parent_candidate_status':source['status'],'known_rough_reference':True,'physical_reference_qualified':False,'geometric_candidate_qualified':False,
          'training_authorized':False,'shared10_enrollment':False,'contact_intent_target_ready':False,'human_contact_observation_known':False}
        native,sidecar,info=abi.export_clip(groupdir,clip,ref['time_s'],ref['q_goal_rad'],ref['mapped_object_pose_w'],model_path=checked(p['inputs']['model']),constants_path=checked(p['sources']['native_constants']),provenance=provenance)
        write(groupdir/'index.json',{'schema':'wuji_pen_spin_clips:1','group_id':group,'group_name':'Known-rough legacy[200,351), independentofshared[0,151)','frame_dt':.02,'clip_count':1,'clips':[{'clip_id':clip,'file':native.name,'sha256':pin(native)['sha256']}]})
        recipe=out/'recipe.json';write(recipe,{'schema':'wuji_pen_spin_recipe:1','clips':[group+'/'+clip]})
        import torch
        from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
        from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
        from full151_native.bundle import rotation,load_bundle
        reader_source=Path(sys.modules[SingleHandTrackingReader.__module__].__file__).resolve()
        if pin(reader_source)['sha256']!=p['sources']['native_reader']['sha256'] or reader_source!=checked(p['sources']['native_reader']).resolve():raise ValueError('Wrong actual native reader import')
        selection=resolve_recipe(recipe,clips_root=out/'clips',verify_clip_hashes=True)
        loaded=SingleHandTrackingReader(selection.clip_sets[0],device='cpu').read();n=loadnp(native)
        tensors={key:value.cpu().numpy() for key,value in loaded.items() if torch.is_tensor(value)}
        np.savez_compressed(out/'native_reader_arrays.npz',**tensors)
        Rw=rotation(n['wrist_pose_w'][:,3:]);Ro=rotation(n['object_pose_w'][:,3:]);native_obj=tensors['object_pose']
        expected_obj=np.einsum('tji,tj->ti',Rw,n['object_pose_w'][:,:3]-n['wrist_pose_w'][:,:3]);expected_R=np.swapaxes(Rw,-1,-2)@Ro
        expected_links=np.einsum('tji,tkj->tki',Rw,n['link_pos_w']-n['wrist_pose_w'][:,None,:3])
        errors={'object_wristlocal_m':float(np.max(np.abs(native_obj[:,:3]-expected_obj))),'object_wristlocal_R':float(np.max(np.abs(rotation(native_obj[:,3:])-expected_R))),
          'BODY21_wristlocal_m':float(np.max(np.abs(tensors['link_pos']-expected_links)))}
        checks={'one_distinct_151clip':len(selection.clip_sets)==1 and len(selection.clip_sets[0].clips)==1 and loaded['trajectory_lengths'].tolist()==[151] and loaded['frame_dt']==.02,
          'native_four_fields_BODY21':set(n)=={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} and n['link_pos_w'].shape==(151,21,3),
          'q151_source64_exact':np.array_equal(n['reference_qpos'],ref['q_goal_rad']),'object151_source64_exact':np.array_equal(n['object_pose_w'],ref['mapped_object_pose_w']),
          'native_q_FP32_exact':np.array_equal(tensors['joint_pos'],ref['q_goal_rad'].astype(np.float32)),
          'native_FD151_preserved':np.array_equal(tensors['joint_vel'],ref['native_forward_FD32_rad_s']),
          'world_to_default_wrist_parity':all(value<=1e-6 for value in errors.values()),
          'fixed_default_wrist':np.max(np.abs(n['wrist_pose_w']-n['wrist_pose_w'][:1]))<=1e-12,
          'all_native_tensors_CPU':all(not value.is_cuda for value in loaded.values() if torch.is_tensor(value)),
          'Torch_CUDA_uninitialized':not torch.cuda.is_initialized(),'exact151_FK_no_forbidden':kin_calls==151 and not any(blocked.values()),
          'analytic_init_not_replaced_by_nativeFD':np.array_equal(init['qd_init_requested_FP32_rad_s'],ref['analytic_qd151_rad_s'][0].astype(np.float32)),
          'source_failure_not_promoted':source['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED'}
        proof={'status':'PASS_ACTUAL_NATIVE151_CPU_READER_ONLY' if all(checks.values()) else 'FAILED_NATIVE151_ABI_RETAINED','checks':checks,'errors':errors,
          'frames':151,'native_reader_source':pin(reader_source),'native_reader_arrays':pin(out/'native_reader_arrays.npz'),'kinematics_calls':kin_calls,
          'forbidden_calls':blocked,'physical_reference_qualified':False,'geometric_candidate_qualified':False,'training_authorized':False,'known_rough_reference':True,
          'nativeFD_vs_analytic_qd0_max_abs_rad_s':float(np.max(np.abs(tensors['joint_vel'][0]-init['qd_init_requested_FP32_rad_s'])))}
        write(out/'native_reader_proof.json',proof)
        if not all(checks.values()):raise ValueError('Actual151 Reader CPU proof failed')
        native_meta={'schema':'full151_knownrough_native_ABI:1','retarget_variant_id':VARIANT,'source_window_half_open':[200,351],
          'frames':151,'control_intervals':150,'native':pin(native),'recipe':pin(recipe),'native_provenance':pin(sidecar),'clips_root':str((out/'clips').resolve()),
          'reader_arrays':pin(out/'native_reader_arrays.npz'),'reader_proof':pin(out/'native_reader_proof.json'),
          'physical_reference_qualified':False,'training_authorized':False,'shared_registry_enrollment':None,'known_rough_reference':True,'geometric_candidate_qualified':False}
        write(out/'native_manifest.json',native_meta)
        pins=dict(p['inputs']);pins.update(native_data=pin(native),native_recipe=pin(recipe),native_sidecar=pin(sidecar),native_reader_arrays=pin(out/'native_reader_arrays.npz'),native_reader_proof=pin(out/'native_reader_proof.json'),native_manifest=pin(out/'native_manifest.json'),build_protocol=pin(protocol_path))
        pins.update({'build_source__'+k:value for k,value in p['sources'].items()});pins['pure_control760']=p['sources']['pure_control760']
        identity={'primary_pose_source_frame_range_half_open':[200,351],'local_reference_range_half_open':[0,151],'goal_states':151,'control_intervals':150,'source_lineage_group':'Existingcontrolledcachedfixture_unknown_original_video_parent','source_lineage_N':1,'independent_ego_N':0}
        spec={'schema':BUNDLE_SCHEMA,'mode':MODE,'retarget_variant_id':VARIANT,'raw_source_identity':identity,
          'native_reference':{'recipe_path':str(recipe.resolve()),'clips_root':native_meta['clips_root'],'duration_s':3.,'provenance_path':str(sidecar.resolve())},
          'source_pins':pins,'model':p['inputs']['model'],'context':p['inputs']['actual_context'],
          'physical_reference_qualified':False,'shared10_enrollment':False,'training_authorized':False,'runtime_implementation_ready':False,
          'geometric_candidate_qualified':False,'known_rough_reference':True,'contact_intent_bundle':None,'human_contact_observation_known':False}
        write(out/'bundle.json',spec);bundle=load_bundle(out/'bundle.json')
        write(out/'descriptor.json',bundle['descriptor'])
        checks['actual_new_bundle_loader_executed']=bundle['baseline_u150'].shape==(150,20)
        checks['no_extra_native_queries_after_loader']=kin_calls==151 and not any(blocked.values())
        checks['all_source_inputs_unchanged']=all(checked(value) for value in [*p['inputs'].values(),*p['sources'].values()])
        report.update(status='PASS_FULL151_KNOWNROUGH_INPUT_DATA_ABI_ONLY' if all(checks.values()) else 'FAILED_NEW_INPUT_RETAINED',checks=checks,errors=errors,
          bundle=pin(out/'bundle.json'),descriptor=pin(out/'descriptor.json'),native_export_info=info,kinematics_calls=kin_calls,
          forbidden_calls=blocked,Torch_CUDA_initialized=torch.cuda.is_initialized(),physical_reference_qualified=False,geometric_candidate_qualified=False,
          runtime_ready=False,training_authorized=False)
    except Exception as e:
        report.update(status='INPUT_BUILD_EXCEPTION_RETAINED',exception=repr(e),traceback=traceback.format_exc(),kinematics_calls=kin_calls,forbidden_calls=blocked)
    finally:
        for (ns,name),value in original.items():setattr(ns,name,value)
    report['wall_s']=time.perf_counter()-tick
    report['artifacts']=[pin(f) for f in sorted(out.rglob('*')) if f.is_file()]
    write(out/'receipt.json',report)
    print(json.dumps({'status':report['status'],'checks_passed':sum(bool(v) for v in report['checks'].values()),'checks_total':len(report['checks']),'kinematics_calls':kin_calls,'receipt':pin(out/'receipt.json')}),flush=True)
    return 0 if report['status']=='PASS_FULL151_KNOWNROUGH_INPUT_DATA_ABI_ONLY' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);args=parser.parse_args();raise SystemExit(main(Path(args.protocol)))
