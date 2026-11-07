"""Small synthetic CPU contracts; does not read future candidates or start physics."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,ast,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/track_prefix_warp.py'
spec=importlib.util.spec_from_file_location('new_dynamic_prefix_worker_contract',SOURCE);w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture():
    t=np.arange(16)*.02;ts=np.arange(61)*.005;angle=.3*ts;quat=np.c_[np.cos(angle/2),np.zeros((61,2)),np.sin(angle/2)];poses=np.c_[.2*ts,np.zeros(61),np.full(61,.2),quat]
    reference={'time_s':t,'planned_time61_s':ts,'q_goal_rad':np.repeat((.1*t)[:,None],20,axis=1),'analytic_qd16_rad_s':np.full((16,20),.1),'mapped_object_pose_w':poses[::4].copy(),'object_COM_world_m':poses[:,:3].copy(),'object_rotation_w':w.rotation(quat),'object_v_world_m_s':np.tile([.2,0,0],(61,1)),'object_omega_world_rad_s':np.tile([0,0,.3],(61,1))}
    proposed=np.repeat((.1*t).astype(np.float32)[:,None],20,axis=1);track=proposed[:15].copy();hold=np.broadcast_to(track[0],(15,20)).copy()
    control={w.CONTROL_KEYS['track']:track,w.CONTROL_KEYS['hold']:hold,w.CONTROL_KEYS['proposed']:proposed}
    initial={'q_init_requested_FP32_rad':reference['q_goal_rad'][0].astype(np.float32),'qd_init_requested_FP32_rad_s':np.full(20,.1,np.float32),'object_pose_init_requested_FP32_w':poses[0].astype(np.float32),'object_bodyorigin_v_requested_FP32_world_m_s':np.array([.2,0,0],np.float32),'object_angular_requested_FP32_body_rad_s':np.array([0,0,.3],np.float32),'object_angular_requested_FP32_world_rad_s':np.array([0,0,.3],np.float32),'u_planned0_FP32_rad':track[0].copy()}
    return reference,initial,control,np.full(20,-1.),np.full(20,1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();started=time.perf_counter();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=[]
    def check(name,call):
        try:
            result=call()
            if result is not None and not bool(result):raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    def rejects(call):
        try:call()
        except (ValueError,KeyError,TypeError):return True
        raise AssertionError('Invalid input was accepted')
    r,i,c,lo,hi=fixture()
    validate=lambda rr=r,ii=i,cc=c,condition=w.CONDITIONS[1]:w.validate_arrays(rr,ii,cc,condition,lo,hi)
    check('no_Warp_or_Torch_imported',lambda:not any(k in sys.modules for k in ('warp','mujoco_warp','torch')))
    check('track_synthetic_ABI',lambda:np.array_equal(validate()[0],c[w.CONTROL_KEYS['track']]));check('hold_synthetic_ABI',lambda:np.array_equal(validate(condition=w.CONDITIONS[0])[0],c[w.CONTROL_KEYS['hold']]))
    check('exact14_update_boundaries',lambda:[k for k in range(1,61) if w.is_target_update_before(k)]==list(range(5,58,4)))
    check('state4u0_before5u1_solverpre4',lambda:w.control_index_for_raw_state(4)==0 and w.control_index_for_solver_step(5)==1 and w.is_target_update_before(5))
    check('state8u1_before9u2',lambda:w.control_index_for_raw_state(8)==1 and w.control_index_for_solver_step(9)==2)
    check('state60_and_solver60_interval14',lambda:w.control_index_for_raw_state(60)==14 and w.control_index_for_solver_step(60)==14)
    check('scheduler_reject_outside_scope',lambda:all(rejects(lambda k=k:w.control_index_for_solver_step(k)) for k in (0,61,1.0,True)))
    def mutation(name,which,key,modify):
        rr,ii,cc=copy.deepcopy((r,i,c));target={'reference':rr,'initial':ii,'control':cc}[which];modify(target,key)
        check(name,lambda:rejects(lambda:validate(rr,ii,cc)))
    mutation('missing_field_rejected','reference','q_goal_rad',lambda d,k:d.pop(k))
    mutation('wrong_window_rejected','reference','time_s',lambda d,k:d.__setitem__(k,d[k]+.02))
    mutation('nonfinite_reference_rejected','reference','object_COM_world_m',lambda d,k:d[k].__setitem__((2,0),np.nan))
    mutation('bad_SO3_rejected','reference','object_rotation_w',lambda d,k:d[k].__setitem__((3,0,0),2.))
    mutation('float64_target_rejected','control',w.CONTROL_KEYS['track'],lambda d,k:d.__setitem__(k,d[k].astype(np.float64)))
    mutation('u0_mismatch_rejected','initial','u_planned0_FP32_rad',lambda d,k:d[k].__setitem__(0,.1))
    mutation('hold_not_constant_rejected','control',w.CONTROL_KEYS['hold'],lambda d,k:d[k].__setitem__((5,0),.001))
    mutation('q_init_mismatch_rejected','initial','q_init_requested_FP32_rad',lambda d,k:d[k].__setitem__(0,.001))
    mutation('bodyangular_frame_mismatch_rejected','initial','object_angular_requested_FP32_body_rad_s',lambda d,k:d[k].__setitem__(0,.001))
    mutation('linearvelocity_mismatch_rejected','initial','object_bodyorigin_v_requested_FP32_world_m_s',lambda d,k:d[k].__setitem__(0,0.))
    mutation('C2_bounds_violation_rejected','control',w.CONTROL_KEYS['track'],lambda d,k:d[k].__setitem__((3,0),.99))
    mutation('rate_exceed_rejected','control',w.CONTROL_KEYS['track'],lambda d,k:(d[k].__setitem__((3,0),.8),d[w.CONTROL_KEYS['proposed']].__setitem__((3,0),.8)))
    mutation('proposal_postC2_mismatch_rejected','control',w.CONTROL_KEYS['proposed'],lambda d,k:d[k].__setitem__((4,0),.01))
    mutation('constant_track_is_not_intervention','control',w.CONTROL_KEYS['track'],lambda d,k:(d.__setitem__(k,np.zeros_like(d[k])),d.__setitem__(w.CONTROL_KEYS['proposed'],np.zeros_like(d[w.CONTROL_KEYS['proposed']]))))
    rr,ii,cc=copy.deepcopy((r,i,c));rr['analytic_qd16_rad_s'][:]=0;rr['object_v_world_m_s'][:]=0;rr['object_omega_world_rad_s'][:]=0
    for key in ('qd_init_requested_FP32_rad_s','object_bodyorigin_v_requested_FP32_world_m_s','object_angular_requested_FP32_body_rad_s','object_angular_requested_FP32_world_rad_s'):ii[key][:]=0
    check('old_zero_rest_rejected',lambda:rejects(lambda:validate(rr,ii,cc)));check('old_condition_rejected',lambda:rejects(lambda:validate(condition='QREF_C2')))
    states=[{'nacon':0,'nefc':0} for _ in range(61)]
    check('capacity_zero_OK',lambda:w.classify_capacity(np.array([0],np.int32),states,192,4096)['status']=='OK')
    check('capacity_nonzero_OVERFLOW',lambda:w.classify_capacity(np.array([4],np.int32),states,192,4096)['status']=='OVERFLOW')
    check('capacity_missing_UNKNOWN',lambda:w.classify_capacity(None,states,192,4096)['status']=='UNKNOWN')
    check('capacity_wrong_worlds_UNKNOWN',lambda:w.classify_capacity(np.array([0,0],np.int32),states,192,4096)['status']=='UNKNOWN')
    check('capacity_floatmask_UNKNOWN',lambda:w.classify_capacity(np.array([0.]),states,192,4096)['status']=='UNKNOWN')
    check('capacity_at_limit_OVERFLOW',lambda:w.classify_capacity(np.array([0],np.int32),[{'nacon':192,'nefc':0}],192,4096)['status']=='OVERFLOW')
    ss=[];sol=[]
    for k in range(61):
        ss.append({'joint_rad':np.zeros(20),'full_qpos':np.zeros(27),'full_qvel':np.zeros(26),'ctrl_rad':np.zeros(20),'object_pose_w':r['mapped_object_pose_w'][0].copy() if k==0 else np.r_[r['object_COM_world_m'][k],np.cos(.3*r['planned_time61_s'][k]/2),0,0,np.sin(.3*r['planned_time61_s'][k]/2)],'actuator_force_native':np.zeros(20),'external_force_max_abs':0.,'ctrl_exact':True,'raw_time_s':k*.005,'expected_time_s':k*.005})
        if k:sol.append({'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0})
    pp={'gates':w.FIXED_GATES,'joint_lower_rad':lo.tolist(),'joint_upper_rad':hi.tolist()};gg=lambda sts=ss,so=sol,cap={'status':'OK'},complete=True:w.gates(sts,so,cap,pp,r,complete,np.full(20,-1),np.full(20,1),.3)
    check('moving_reference16_grade_pass',lambda:gg()['short_dynamic_pass'] is True)
    check('incomplete_remains_UNKNOWN',lambda:gg(complete=False)['short_dynamic_pass'] is None)
    check('capacity_unqualified_remains_UNKNOWN',lambda:gg(cap={'status':'UNKNOWN'})['short_dynamic_pass'] is None)
    check('capacity_overflow_remains_UNKNOWN',lambda:gg(cap={'status':'OVERFLOW'})['short_dynamic_pass'] is None)
    bad=copy.deepcopy(ss);bad[4]['object_pose_w'][0]+=.03
    check('quality_failure_retains_full_horizon_grade',lambda:gg(sts=bad)['short_dynamic_pass'] is False and gg(sts=bad)['primary_reference_samples']==16)
    between=copy.deepcopy(ss);between[3]['object_pose_w'][0]+=.03
    check('61diagnostic_not_silently_primary',lambda:gg(sts=between)['short_dynamic_pass'] is True and gg(sts=between)['all61_position_diagnostic_pass'] is False)
    tree=ast.parse(SOURCE.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run');outside=ast.Module(body=[n for n in tree.body if n is not run],type_ignores=[])
    check('GPU_import_only_in_authorized_run',lambda:not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(k in ast.unparse(n) for k in ('import warp','import mujoco_warp')) for n in ast.walk(outside)))
    check('one_startup_forward_callsite',lambda:sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='mjwarp.forward' for n in ast.walk(run))==1)
    check('one_physics_step_callsite',lambda:sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='mjwarp.step' for n in ast.walk(run))==1)
    check('end_only_sticky_read_callsite',lambda:sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='data.overflow.numpy' for n in ast.walk(run))==1)
    check('no_runtime_state_or_force_write_methods',lambda:not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset_data','reset','set_state','set_qpos','set_qvel') for n in ast.walk(run)))
    check('source_and_vendor_original_hashes',lambda:sha(ROOT/'vendor/runtime_io.py')=='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1' and sha(ROOT/'vendor/measured_contacts.py')=='478c3117489bd4a02238f6299a91638e791a751352b772321ce2676b0ec4f96f')
    check('end_no_Torch_Warp_import',lambda:not any(k in sys.modules for k in ('warp','mujoco_warp','torch')))
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_sha256':sha(SOURCE),'checker_sha256':sha(__file__),'supervisor_sha256':sha(ROOT/'scripts/run_track_prefix.py'),'synthetic_only':True,'future_candidate_read':False,'physics_steps':0,'GPU_started':False,'Torch_imported':'torch' in sys.modules,'Warp_imported':any(k in sys.modules for k in ('warp','mujoco_warp')),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'wall_seconds':time.perf_counter()-started}
    for source in (SOURCE,Path(__file__),ROOT/'scripts/run_track_prefix.py'):(out/source.name).write_bytes(source.read_bytes())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');(out/'schema.json').write_text(json.dumps(w.describe_contract(),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
