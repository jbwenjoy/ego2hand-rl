"""Synthetic CPU proof of the saved-policy auditor, not an actual rollout."""
from pathlib import Path
import copy,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];src=ROOT/'scripts/audit_native_policy_saved_cpu.py'
spec=importlib.util.spec_from_file_location('saved_audit',src);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def fixture():
    s={'index':np.arange(61),'raw_time_s':np.arange(61)*.005,'joint_rad':np.zeros((61,20)),'joint_lower_rad':-np.ones(20),'joint_upper_rad':np.ones(20),'ctrl_rad':np.zeros((61,20)),'ctrl_exact':np.ones(61,bool),'object_pose_w':np.tile([0.,0.,.5,1.,0.,0.,0.],(61,1)),'external_force_max_abs':np.zeros(61),'force_lower_Nm':-np.ones(20),'force_upper_Nm':np.ones(20),'nacon':np.zeros(61,dtype=np.int64),'nefc':np.zeros(61,dtype=np.int64)}
    sol=[{'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0} for _ in range(60)];events=[{'actualu':np.zeros((1,20))} for _ in range(15)];cap={'flags_by_world':[0],'raw_flags_dtype':'int32','raw_flags_shape':[1],'contact_capacity':192,'constraint_capacity':4096};ref={'object_rotation_w':np.tile(np.eye(3),(61,1,1)),'object_COM_world_m':np.tile([0.,0.,.5],(61,1))}
    return [s,sol,events,cap,ref,.5]
def main():
    out=ROOT/'outputs/native_policy_saved_auditor_cpu_v1';out.mkdir(parents=True,exist_ok=False);checks={}
    f=fixture();r=a.grade(*f);checks['complete_positive_original_gates']=r['qualified'] and r['native_policy_pass'] is True and all(r['gates'].values())
    f=fixture()
    for e in f[2]:e['actualu'][0,0]=.31
    r=a.grade(*f);checks['startup_jump_detected_despite_zero_later_changes']=r['native_policy_pass'] is False and r['startup_and_update_rate']['startup_rate_rad_s']==15.5 and r['startup_and_update_rate']['subsequent_updates_max_rate_rad_s']==0 and r['startup_and_update_rate']['rate_difference_count']==15
    f=fixture();f[2].pop();checks['partial_events_UNKNOWN']=a.grade(*f)['native_policy_pass'] is None
    f=fixture();f[0]['index']=np.arange(60);checks['partial_states_UNKNOWN']=a.grade(*f)['native_policy_pass'] is None
    f=fixture();f[3]['flags_by_world']=[1];r=a.grade(*f);checks['overflow_complete_coverage_UNKNOWN']=r['qualified'] is False and r['native_policy_pass'] is None
    f=fixture();f[3]['raw_flags_dtype']='float32';checks['noninteger_capacity_UNKNOWN']=a.capacity_status(f[3],f[0])=='UNKNOWN'
    f=fixture();f[3]['flags_by_world']=[.1];checks['fractional_flag_not_silently_cast_to_zero']=a.capacity_status(f[3],f[0])=='UNKNOWN'
    f=fixture();f[0]['nacon'][1]=192;checks['count_at_capacity_UNKNOWN']=a.capacity_status(f[3],f[0])=='UNKNOWN'
    f=fixture();f[0]['joint_rad'][0,0]=np.nan;checks['nonfinite_raw_UNKNOWN']=a.grade(*f)['native_policy_pass'] is None
    f=fixture();f[0]['object_pose_w'][60,0]=.020001;checks['position_gate_above20mm']=a.grade(*f)['gates']['object_position_16'] is False
    f=fixture();f[0]['object_pose_w'][1,0]=.03;r=a.grade(*f);checks['position16_primary_61_diagnostic_distinct']=r['gates']['object_position_16'] and r['max_object_position_error_61_m']==.03
    f=fixture();f[0]['joint_rad'][1,0]=1.00101;checks['actual_q_all61_gate']=a.grade(*f)['gates']['actual_q_all61'] is False
    f=fixture();f[1][0]['penetration_m']=.003001;checks['penetration_original3mm_gate']=a.grade(*f)['gates']['penetration_solver60'] is False
    f=fixture();f[1][0]['actuator_force_solver_native'][0]=-1.01001;checks['negative_actuator_load_gate']=a.grade(*f)['gates']['load_ratio'] is False
    f=fixture();f[0]['object_pose_w'][:,3:]*=-1;checks['quaternion_sign_equivalence']=a.grade(*f)['native_policy_pass'] is True
    p=a.predict(np.full((1,20),8.),np.full((1,20),.3),np.full((1,20),.9),-np.ones(20),np.ones(20));checks['original_EMA_keeps_residual_despite_target_C2']=a.same32(p['residual'],np.full((1,20),.4,np.float32)) and a.same32(p['hard_bounded'],np.ones((1,20),np.float32)) and a.same32(p['actual'],np.full((1,20),.98,np.float32))
    raw={'geom':[5,1],'distance_m':-.002,'pos_w':[0.,1.,0.],'frame_rows_w':np.eye(3).tolist(),'force6_contact_frame':[1.,.2,.3,.01,.02,.03]};depth,n,rows,w=a.contact_rows([raw],{1},{5},[0.,0.,0.,1.,0.,0.,0.],np.zeros(3));checks['force_sign_and_directtorsion_COMmoment']=np.allclose(w,[1.,.2,.3,.31,.02,-.97],rtol=0,atol=1e-15) and depth['hand_object']==.002 and n==0
    rev=copy.deepcopy(raw);rev['geom']=[1,5];_,_,_,wrev=a.contact_rows([rev],{1},{5},[0.,0.,0.,1.,0.,0.,0.],np.zeros(3));checks['object_geom_order_force_and_torque_sign']=np.array_equal(wrev,-w)
    no=copy.deepcopy(raw);no['geom']=[5,6];d,n,rr,ww=a.contact_rows([no],{1},{5,6},[0.,0.,0.,1.,0.,0.,0.],np.zeros(3));checks['self_contact_not_object_force_or_false_label']=d['hand_self']==.002 and not rr and n==0 and np.array_equal(ww,np.zeros(6))
    absent=a.audit(out/'no_such_config.json',out/'not_run',out/'no_a0')[0];checks['missing_run_NOT_RUN_not_failure']=absent['status']=='NOT_RUN' and absent['native_policy_pass'] is None and absent['total']==0 and absent['engineering_qualified'] is False
    fp=out/'pin_fixture.json';fp.write_text('{}\n');p=a.pin(fp);checks['sourcepin_positive']=a.checked(p)==fp;rejected=False
    try:a.checked({**p,'sha256':'0'*64})
    except ValueError:rejected=True
    checks['changed_sourcepin_rejected']=rejected
    checks['CPU_no_environment_or_device']=not a.torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    report={'schema':'native_policy_saved_auditor_CPU_fixture_proof:1','status':'PASS_SYNTHETIC_NOT_ACTUAL_PHYSICS' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':a.pin(src),'checker':a.pin(__file__),'GPU_used':False,'created_environment':False,'physics_steps':0,'scope':'Synthetic state/control/contact/capacity/source fixtures only; doesnot qualify an actual evaluation run.'};p=out/'report.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':a.pin(p),'source':a.pin(src)},indent=2));return int(not all(checks.values()))
if __name__=='__main__':raise SystemExit(main())
