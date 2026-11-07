"""Independent CPU data/signature/negative gates; no native reader/FK/model/physics."""
from __future__ import annotations
import argparse,copy,hashlib,importlib,importlib.util,json,os,sys,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,default=ROOT/'outputs/input_bundle_v1/bundle.json');ap.add_argument('--module',default='rough_prefix.bundle_v2');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=[];negatives=[]
    module=importlib.import_module(args.module);loader_path=Path(module.__file__);spec=read(args.bundle);frozen={'loader':sha(loader_path),'checker':sha(__file__),'bundle':sha(args.bundle)}
    (out/'loader_source.py').write_bytes(loader_path.read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes())
    b=module.load_bundle(args.bundle,expected_sha256=frozen['bundle']);ref,init,control=(b['arrays'][k] for k in ('reference','init','control'));native=read(spec['source_pins']['native_manifest']['path'])
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
    def rejected(call):
        try:call()
        except (ValueError,KeyError,TypeError,FileNotFoundError,AssertionError):return True
        return False
    def spec_negative(name,change,expected=True):
        d=copy.deepcopy(spec);change(d);p=out/(name+'.json');p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        result=rejected(lambda:module.load_bundle(p));negatives.append({'name':name,'rejected':result,'expect_rejected':expected,'passed':result==expected,'seam':'changed local bundle file; optional outer SHA deliberately omitted to test intrinsic semantic checks'})
    check('all_original_source_pins_exact',all(sha(v['path'])==v['sha256'] for v in spec['source_pins'].values()))
    check('physical_context_aliases_exact',spec['model']==spec['source_pins']['export_source__model'] and spec['context']==spec['source_pins']['export_source__actual_context'])
    check('three_bundle_artifacts_read_separately',all(np.array_equal(b['arrays'][k][key],a) for k in b['arrays'] for key,a in module.loadnp(spec['source_pins'][k+'_arrays']['path']).items()))
    check('native_legacy16_metadata_not_shared',native['source_window_half_open']==[200,216] and native['native_local_range_half_open']==[0,16] and native['physical_reference_qualified'] is False and native['training_authorized'] is False and native['shared_registry_enrollment'] is None)
    check('contact_reference_semantic_join_positive',all(np.array_equal(module.loadnp(spec['source_pins']['contact_reference']['path'])[a],v) for a,v in [('time_s',ref['time_s']),('joint_pos_rad',ref['q_goal_rad']),('object_pose_w',ref['mapped_object_pose_w']),('wrist_pose_w',module.loadnp(spec['source_pins']['native_data']['path'])['wrist_pose_w'])]))
    check('initial_q_qd_u0_exact',np.array_equal(init['q_init_requested_FP32_rad'],ref['q_goal_rad'][0].astype(np.float32)) and np.array_equal(init['qd_init_requested_FP32_rad_s'],ref['analytic_qd16_rad_s'][0].astype(np.float32)) and np.array_equal(init['u_planned0_FP32_rad'],b['baseline_u15'][0]))
    check('baseline15_finite_nonconstant_and_rate',b['baseline_u15'].shape==(15,20) and b['baseline_u15'].dtype==np.float32 and np.isfinite(b['baseline_u15']).all() and np.any(np.diff(b['baseline_u15'],axis=0)) and np.max(np.abs(np.diff(b['baseline_u15'].astype(float),axis=0)))/.02<=15)
    desc=copy.deepcopy(b['descriptor']);sig=desc.pop('input_signature_sha256')
    check('descriptor_signature_independently_recomputed',sig==hashlib.sha256(json.dumps(desc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
    check('descriptor_binds_bundles_actualphysics_velocity_phase',b['descriptor']['source_pins']==spec['source_pins'] and b['descriptor']['input_bundle']['sha256']==frozen['bundle'] and b['descriptor']['control_intervals']==15 and b['descriptor']['goal_velocity_authority']=='frozen_Bspline_analytic_joint_qd16_and_quintic_object_world_v_omega16' and '104' in b['descriptor']['checkpoint'])
    check('outer_expected_bundle_SHA_rejects_wrongfile',rejected(lambda:module.load_bundle(args.bundle,expected_sha256='0'*64)))
    spec_negative('wrong_source_SHA',lambda d:d['source_pins']['reference_arrays'].__setitem__('sha256','0'*64))
    spec_negative('old151_source_window',lambda d:d['raw_source_identity'].__setitem__('primary_pose_source_frame_range_half_open',[200,351]))
    spec_negative('old151_goalstates',lambda d:d['raw_source_identity'].__setitem__('goal_states',151))
    spec_negative('shared10_enrollment_injected',lambda d:d.__setitem__('shared10_enrollment',True))
    spec_negative('physical_qualification_injected',lambda d:d.__setitem__('physical_reference_qualified',True))
    spec_negative('wrong_model_alias',lambda d:d.__setitem__('model',d['context']))
    spec_negative('wrong_native_duration',lambda d:d['native_reference'].__setitem__('duration_s',3.))
    # Altering bytes must fail independently of mathematical consistency; never
    # edit an original array or re-sign a frozen producer lineage.
    changed=out/'modified_reference_arrays.npz';z={k:v.copy() for k,v in ref.items()};z['analytic_qd16_rad_s'][1,0]+=.01;np.savez_compressed(changed,**z)
    spec_negative('velocity_file_changed_retaining_expected_SHA',lambda d:d['source_pins']['reference_arrays'].__setitem__('path',str(changed)))
    changed_u=out/'modified_control_arrays.npz';z={k:v.copy() for k,v in control.items()};z[module.CONTROL_KEY][3,0]+=.001;np.savez_compressed(changed_u,**z)
    spec_negative('baseline_file_changed_retaining_expected_SHA',lambda d:d['source_pins']['control_arrays'].__setitem__('path',str(changed_u)))
    # In-memory load seam specifically tests semantic init checks after valid
    # file hashes. It is not a claim about re-signed end-to-end artifacts.
    original_loadnp=module.loadnp
    for field in ('qd_init_requested_FP32_rad_s','object_angular_requested_FP32_body_rad_s','u_planned0_FP32_rad'):
        def corrupt(path,field=field):
            z=original_loadnp(path)
            if Path(path).resolve()==Path(spec['source_pins']['init_arrays']['path']).resolve():z[field][0]+=.01
            return z
        module.loadnp=corrupt
        try:check('semantic_init_corruption_'+field,rejected(lambda:module.load_bundle(args.bundle)))
        finally:module.loadnp=original_loadnp
    for field in ('time_s','joint_pos_rad','object_pose_w','wrist_pose_w'):
        def corrupt_bridge(path,field=field):
            z=original_loadnp(path)
            if Path(path).resolve()==Path(spec['source_pins']['contact_reference']['path']).resolve():z[field].flat[0]+=.001
            return z
        module.loadnp=corrupt_bridge
        try:check('semantic_contact_bridge_corruption_'+field,rejected(lambda:module.load_bundle(args.bundle)))
        finally:module.loadnp=original_loadnp
    original_read=module.read
    for field,value in [('physical_reference_qualified',True),('training_authorized',True),('shared_registry_enrollment',{'claimed':True})]:
        def corrupt_native(path,field=field,value=value):
            d=original_read(path)
            if Path(path).resolve()==Path(spec['source_pins']['native_manifest']['path']).resolve():d[field]=value
            return d
        module.read=corrupt_native
        try:check('semantic_native_claim_'+field,rejected(lambda:module.load_bundle(args.bundle)))
        finally:module.read=original_read
    for field in ('physical_reference_qualified','human_contact_observation_known','source_robot_labels_used'):
        def corrupt_bridge_meta(path,field=field):
            d=original_read(path)
            if Path(path).resolve()==Path(spec['source_pins']['contact_reference_metadata']['path']).resolve():d[field]=True
            return d
        module.read=corrupt_bridge_meta
        try:check('semantic_contact_meta_claim_'+field,rejected(lambda:module.load_bundle(args.bundle)))
        finally:module.read=original_read
    # These are source-review gaps if currently accepted; preserve the result
    # rather than silently moving to an outer-SHA-only test.
    spec_negative('training_authority_flag_injected',lambda d:d.__setitem__('training_authorized',True))
    spec_negative('runtime_ready_flag_injected',lambda d:d.__setitem__('runtime_implementation_ready',True))
    check('sources_and_original_bundle_unchanged',sha(loader_path)==frozen['loader'] and sha(__file__)==frozen['checker'] and sha(args.bundle)==frozen['bundle'])
    check('no_native_reader_FK_or_GPU_libraries_imported',not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')))
    passed=all(c['passed'] for c in checks+negatives)
    report={'status':'PASS' if passed else 'FAIL_INTRINSIC_CONTRACT_GAPS_RETAINED','checks':checks,'negative_checks':negatives,'check_count':len(checks)+len(negatives),'passed':sum(c['passed'] for c in checks+negatives),'source_sha256':frozen,'loader_path':str(loader_path),'input_signature_sha256':sig,'positive_actual_bundle_valid':all(c['passed'] for c in checks),'boundaries':'Negative local bundle files omit optional expected_sha to exercise intrinsic checks; the real entry must also pin and require the outer bundle SHA. In-memory seams test init mathematics only. No original source/data modified or producer/Reader/FK/solver/physics/GPU called.','execution':{'producer_calls':0,'reader_calls':0,'FK_calls':0,'geometry_queries':0,'solver_calls':0,'physics_steps':0,'GPU':0}}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'count':report['check_count'],'failed':[c['name'] for c in checks+negatives if not c['passed']],'report_sha256':sha(out/'report.json')},indent=2));raise SystemExit(not passed)
if __name__=='__main__':main()
