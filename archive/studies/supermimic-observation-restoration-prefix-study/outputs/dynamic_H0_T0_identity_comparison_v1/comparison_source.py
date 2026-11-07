"""Saved-only H0/T0 identity comparison; no native, optimizer, force or GPU call."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(hold_plan,track_plan,hold_audit,track_audit,output):
    start=time.perf_counter();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'comparison_source.py').write_bytes(Path(__file__).read_bytes());hp,tp=read(hold_plan),read(track_plan);hr,tr=read(hold_audit),read(track_audit);hdir,tdir=Path(hp['output']),Path(tp['output']);h=arrays(hdir/'raw_states.npz');t=arrays(tdir/'raw_states.npz');checks=[];inputs={};diagnostics={}
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    check('both_independent_engineering_audits_pass',hr['engineering_passed'] is True and tr['engineering_passed'] is True and hr['passed_checks']==hr['check_count']==152 and tr['passed_checks']==tr['check_count']==152)
    check('correct_H0_T0_ordered_conditions',hp['control_condition']=='H0_MATCHED_DYNAMIC_INIT_HOLD' and tp['control_condition']=='T0_MATCHED_DYNAMIC_INIT_TRACK')
    diffs=[k for k in sorted(set(hp)|set(tp)) if hp.get(k)!=tp.get(k)]
    check('plans_differ_only_condition_output',diffs==['control_condition','output'],diffs);check('all_plan_sourcepins_identical',hp['source_pins']==tp['source_pins'])
    for field in ('full_qpos','full_qvel','joint_rad','joint_velocity_rad_s','object_pose_w','object_twist_native','ctrl_rad','full_ctrl_native','mocap_pos_w','mocap_quat_wxyz'):
        check('initial_raw_exact__'+field,np.array_equal(h[field][0],t[field][0]))
    for name in ('expected_context_fields','actual_context_applied','actual_context_readbacks','requested_initialization_and_control'):
        ha,ta=arrays(hdir/(name+'.npz')),arrays(tdir/(name+'.npz'));check(name+'__keys_exact',set(ha)==set(ta))
        for field in sorted(set(ha)&set(ta)):
            # Schedules intentionally differ after the shared first command.
            if name=='requested_initialization_and_control' and field=='selected_planned15_FP32_rad':continue
            check(name+'__'+field,np.array_equal(ha[field],ta[field]))
    check('same_initial_u0_but_real_control_intervention',np.array_equal(h['selected_planned15_FP32_rad'][0],t['selected_planned15_FP32_rad'][0]) and np.array_equal(h['selected_planned15_FP32_rad'],np.broadcast_to(h['selected_planned15_FP32_rad'][0],(15,20))) and np.any(h['selected_planned15_FP32_rad'][1:]!=t['selected_planned15_FP32_rad'][1:]))
    check('actual8_context_same_model_context_pins',hp['model']==tp['model'] and hp['context']==tp['context'])
    for name,hpath,tpath in [('plan',hold_plan,track_plan),('independent_audit',hold_audit,track_audit),('raw_states',hdir/'raw_states.npz',tdir/'raw_states.npz'),('runtime',hdir/'runtime.json',tdir/'runtime.json'),('context_readbacks',hdir/'actual_context_readbacks.npz',tdir/'actual_context_readbacks.npz')]:inputs['hold_'+name]=pin(hpath);inputs['track_'+name]=pin(tpath)
    for field in ('joint_rad','joint_velocity_rad_s','actuator_force_native','object_pose_w','object_twist_native'):
        delta=np.abs(h[field]-t[field]);perstep=np.max(delta.reshape(61,-1),axis=1);nonzero=np.flatnonzero(perstep>0);diagnostics[field]={'max_abs_difference_all61':float(delta.max()),'first_nonzero_state_index':int(nonzero[0]) if len(nonzero) else None,'initial_max_abs_difference':float(perstep[0]),'scope':'Exploratory trajectory difference, no attribution of early floating-point variation to the scheduled intervention'}
    hm,tm=hr['metrics'],tr['metrics'];summary={'hold':hm,'track':tm,'sequential_N_per_condition':1,'same_initial_state':True,'quality_outcomes':[hm['recomputed_short_dynamic_pass'],tm['recomputed_short_dynamic_pass']],'failed_quality_gates_hold':[k for k,v in hm['recomputed_quality_gates'].items() if not v],'failed_quality_gates_track':[k for k,v in tm['recomputed_quality_gates'].items() if not v],'cost_seconds':{'hold_worker':hm['cost']['worker_wall_seconds'],'track_worker':tm['cost']['worker_wall_seconds'],'sum_worker':hm['cost']['worker_wall_seconds']+tm['cost']['worker_wall_seconds'],'hold_watchdog':hm['cost']['watchdog_elapsed_seconds'],'track_watchdog':tm['cost']['watchdog_elapsed_seconds'],'sum_watchdog_slots':hm['cost']['watchdog_elapsed_seconds']+tm['cost']['watchdog_elapsed_seconds']},'method_benefit_or_population_inference':False,'full3s_task_assessed':False}
    report={'schema':'restored_dynamic_H0_T0_saved_identity_comparison:1','status':'PASS_IDENTITY_COMPARISON' if all(c['passed'] for c in checks) else 'FAIL_OR_UNKNOWN_COMPARISON','check_count':len(checks),'passed_checks':sum(c['passed'] for c in checks),'checks':checks,'source':pin(__file__),'input_pins':inputs,'summary':summary,'trajectory_difference_diagnostics':diagnostics,'Torch_imported':'torch' in sys.modules,'Warp_imported':any(k in sys.modules for k in ('warp','mujoco_warp')),'physics_steps_executed':0,'wall_seconds':time.perf_counter()-start,'scope':'Two predeclared sequential known-historyN1 conditions with exact matched new nonzero initialization and actual8 fields. Both quality outcomes retained, no selection/retry/causal method-benefit or force-prediction claim.'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--hold-plan',type=Path,required=True);ap.add_argument('--track-plan',type=Path,required=True);ap.add_argument('--hold-audit',type=Path,required=True);ap.add_argument('--track-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=compare(a.hold_plan,a.track_plan,a.hold_audit,a.track_audit,a.output);print(json.dumps({'status':r['status'],'checks':r['check_count'],'passed':r['passed_checks'],'quality_outcomes':r['summary']['quality_outcomes'],'plan_differences':next(c['detail'] for c in r['checks'] if c['name']=='plans_differ_only_condition_output'),'cost':r['summary']['cost_seconds']},indent=2));raise SystemExit(r['status']!='PASS_IDENTITY_COMPARISON')
if __name__=='__main__':main()
