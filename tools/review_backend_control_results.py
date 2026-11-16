"""Independent terminal budget/trace review and archive of the full fixed trial."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_control_study import require_final_training
from supermimic_research.learning.backend_pen_ppo import require_saved_backend
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.runtime.pen_executed_control_audit import audit_executed_control
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def independent_errors(z,seconds=None):
    n=len(z['post_qpos']) if seconds is None else round(seconds/.0025);t=np.arange(1,n+1)*.0025
    target=np.column_stack([np.interp(t,z['time_s'],z['COM_world_m'][:,j]) for j in range(3)])
    distance=np.linalg.norm(z['post_COM'][:n]-target,axis=1)
    target_q=Slerp(z['time_s'],Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(t).as_quat()
    actual=np.asarray(z['post_quaternion_wxyz'][:n,[1,2,3,0]],dtype=np.float64);actual=actual/np.linalg.norm(actual,axis=1,keepdims=True)
    angle=2*np.arccos(np.clip(abs(np.sum(actual*target_q,axis=1)),0,1))
    return dict(actual_duration_s=n*.0025,COM_RMS_m=float(np.sqrt(np.mean(distance**2))),COM_max_m=float(distance.max()),angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),angle_max_rad=float(angle.max()))


def main():
    out=ROOT/'provenance/pen_backend_control_results_v1';analysis=ROOT/'runs/pen_backend_control_results_review_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh terminal review/archive required')
    tp=ROOT/'runs/pen_backend_long_training_v1/report.json';ep=ROOT/'runs/pen_backend_long_evaluation_v1/report.json';cp=ROOT/'configs/pen_backend_long_control_v1.toml';rp=ROOT/'runs/pen_backend_long_registration_v1/registration.json'
    training,evaluation=read(tp),read(ep);contract=load_contract(ROOT/'configs/collision_backend_face_normal_v1.json');prior=ROOT/'provenance/pen_backend_coverage_readiness_v1/ledger.json'
    require_final_training(training,file_pin(cp),contract['backend_id'],file_pin(rp))
    checks=dict(training5=training['passed']==training['total']==5 and all(training['checks'].values()),training_attempts_exact=training['actual_native_integrations']==training['attempted_native_integrations']==65536000 and training['optimizer_steps']==training['attempted_optimizer_steps']==128000,
        complete_final_matrix=[(c['repetition'],c['policy']) for c in evaluation['cases']]==[(r,k) for r in range(3) for k in ['INITIAL','RESIDUAL','SUPPORT']],
        evaluation_same_trial=evaluation['training']==file_pin(tp) and evaluation['backend_id']==contract['backend_id'] and evaluation['registration']==file_pin(rp),
        evaluation_actual_count=evaluation['actual_native_integrations']==evaluation['attempted_native_integrations']==sum(c['native_integrations'] for c in evaluation['cases']) and evaluation['optimizer_steps']==0,
        shared_preparation_once=training['preparation_native_integrations']==2048 and training['cohort_native_before_evaluation']==65538048)
    saved_states=[];health_paths=[]
    for arm in training['arms']:
        label=arm['arm'];state=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False);require_saved_backend(state,contract)
        checks[label+'_final_state_budget']=(state['iteration'],state['optimizer_steps'],state['training_transitions'],state['calibration_transitions'])==(2000,64000,4096000,256)
        checks[label+'_Adam_steps']=bool(state['optimizer']['state']) and all(int(v['step'])==64000 for v in state['optimizer']['state'].values())
        checks[label+'_finite_modules']=all(bool(torch.isfinite(v).all()) for m in state['modules'].values() for v in m.values())
        checks[label+'_finite_Adam']=all(bool(torch.isfinite(v).all()) for node in state['optimizer']['state'].values() for v in node.values() if isinstance(v,torch.Tensor))
        verify_source_snapshot(state['native']['sources'],live=True)
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        paths=[tp.parent/label/f'health_{i:04d}.json' for i in range(1,2001)];rows=[read(p) for p in paths];collectors=[read(verify_file_pin(h['collector'])) for h in rows];health_paths+=paths
        checks[label+'_all2000_work']=[h['iteration'] for h in rows]==list(range(1,2001)) and all((h['actual_native_integrations'],h['optimizer_steps'],h['training_transitions'])==(h['iteration']*16384,h['iteration']*32,h['iteration']*2048) for h in rows)
        checks[label+'_all50000_collector_flags']=all(c['passed']==c['total']==25 and all(c['checks'].values()) for c in collectors)
        checks[label+'_all21_due_saves']=[h['iteration'] for h in rows if h['checkpoint'] is not None]==[1]+list(range(100,2001,100))
        saved_states.append(dict(arm=label,checkpoint=arm['checkpoint'],iterations=state['iteration'],Adam=state['optimizer_steps'],transitions=state['training_transitions']))
    cases=[];prefixes={};expected_pins={'INITIAL':read(verify_file_pin(training['preparation']))['common'],**{a['arm']:a['checkpoint'] for a in training['arms']}}
    for case in evaluation['cases']:
        name=case['name'];z=dict(np.load(verify_file_pin(case['trace'])));metric=independent_errors(z);recorded=case['summary']['metrics']
        checks[name+'_metrics_independent_quaternion_dot']=all(abs(v-recorded[k])<=1e-9 for k,v in metric.items())
        audit=audit_executed_control(z,case);checks[name+'_drive_audit']=audit['passed']==audit['total']==6 and audit==case['executed_control_audit']
        checks[name+'_no_aid_no_overflow_finite']=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']) and not np.any(z['overflow']) and all(np.isfinite(v).all() for k,v in z.items() if k.startswith(('pre_','post_')))
        checks[name+'_actual_control_count']=case['native_integrations']==len(z['post_qpos'])==case['controls']*8
        checks[name+'_fixed_mount_physics']=case['fixed_mount'] and case['post_start_resets']==0 and all(case['physical_matches'].values())
        checks[name+'_checkpoint_backend']=case['checkpoint']==expected_pins[case['policy']] and case['actor_initialization']['collision_backend_id']==contract['backend_id'] and case['actor_initialization']['backend_binding_verified']
        checks[name+'_task_flag_honest']=case['summary']['qualified']==all(case['summary']['gates'].values())
        if case['policy']!='INITIAL':
            prefix=independent_errors(z,evaluation['trained_common_horizon_s']);prefixes[name]=prefix
            checks[name+'_common_prefix']=all(abs(v-evaluation['common_horizon_metrics'][name][k])<=1e-9 for k,v in prefix.items())
        cases.append(dict(name=name,policy=case['policy'],duration_s=metric['actual_duration_s'],COM_RMS_mm=1000*metric['COM_RMS_m'],angle_RMS_deg=float(np.rad2deg(metric['angle_RMS_rad'])),qualified=case['summary']['qualified'],termination_reasons=case['termination_reasons']))
    for label,row in evaluation['aggregate'].items():
        selected=[c for c in cases if c['policy']==label];checks[label+'_aggregate_every_repeat']=row['durations_s']==[c['duration_s'] for c in selected] and row['qualified_count']==sum(c['qualified'] for c in selected)
    rendering=evaluation['rendering'];selection=np.asarray(rendering['selected_post_indices']);lengths=[c['native_integrations'] for c in evaluation['cases'][:3]]
    expected_samples=int(np.ceil(max(rendering['durations_s'])/rendering['speed']*rendering['fps']))+1
    checks['render_frame_selection']=selection.shape==(expected_samples,3) and rendering['frames']==expected_samples+18+30 and np.issubdtype(selection.dtype,np.integer) and bool(np.all(selection>=-1)) and all(np.all(selection[:,j]<n) and np.all(np.diff(selection[:,j])>=0) and selection[-1,j]==n-1 for j,n in enumerate(lengths))
    checks['render_no_physics']=rendering['render_native_integrations']==0 and rendering['render_optimizer_steps']==0 and rendering['durations_s']==evaluation['aggregate']['INITIAL']['durations_s'][:1]+evaluation['aggregate']['RESIDUAL']['durations_s'][:1]+evaluation['aggregate']['SUPPORT']['durations_s'][:1]
    receipts=[training['source_receipt'],evaluation['source_receipt'],read(rp)['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Independent terminal review failed: '+repr([k for k,v in checks.items() if not v]))
    samples=np.asarray([s['values'] for s in training['telemetry']['samples']]);telemetry=dict(samples=len(samples),GPU_utilization_median_pct=float(np.median(samples[:,0])),GPU_utilization_p95_pct=float(np.quantile(samples[:,0],.95)),memory_used_median_MiB=float(np.median(samples[:,1])),memory_used_max_MiB=float(samples[:,1].max()),scope='Device-wide full-run observations include brief parallel unstepped W1 QA;not controlled algorithm throughput/acceleration.')
    docs=[ROOT/'docs'/n for n in ['pen-backend-control-results-2026-11-16.md','research-plan-v1.32.md']];analysis.mkdir();receipt=freeze_sources([Path(__file__)]+docs,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    write(analysis/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),final_states=saved_states,cases=cases,common_horizon_s=evaluation['trained_common_horizon_s'],common_horizon=prefixes,telemetry=telemetry,
        training_native=65536000,training_Adam=128000,preparation_native=2048,evaluation_native=evaluation['actual_native_integrations'],control_cohort_native=65538048+evaluation['actual_native_integrations'],
        training_wall_s=training['wall_s'],evaluation_and_render_wall_s=evaluation['wall_s'],added_actual_native_integrations=0,added_optimizer_steps=0,
        video_QA='Inspected decoded frames1200/1900;correct three-panel title,non-ego/no-aid scope,timestamps and frozen terminal panels.18 lead/30 tail frames are visual holds,outside per-sample selection list.3 numerical repeats are not independent training seeds.'))
    QA=ROOT/'runs/pen_backend_long_evaluation_video_QA_v1';diagnosis=ROOT/'runs/pen_backend_control_review_diagnosis_v1';todo=[file_pin(p) for p in [tp,ep,rp,prior,analysis/'source_receipt.json',analysis/'independent_review.json']+docs+health_paths]+receipts+[file_pin(p) for p in QA.glob('*.png')]+[file_pin(p) for p in diagnosis.iterdir() if p.is_file()]
    allowed={tp.parent,ep.parent,rp.parent,analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    out.mkdir();write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(training=file_pin(tp),evaluation=file_pin(ep),independent_review=file_pin(analysis/'independent_review.json'),checks=[sum(checks.values()),len(checks)],
        aggregate=evaluation['aggregate'],common_horizon_s=evaluation['trained_common_horizon_s'],video=rendering['video'],training_native=65536000,training_Adam=128000,preparation_native=2048,evaluation_native=evaluation['actual_native_integrations'],control_cohort_native=65538048+evaluation['actual_native_integrations'],
        added_actual_native_integrations=0,added_optimizer_steps=0,scope='Complete fixed single-seed/reference model-free trial. Both tasks fail strict7.36s. Support trades lower short-prefix angular error for larger position error/earlier failure;no task benefit,WM Actor,ego or acceleration admission.'))
    write(out/'ledger.json',dict(schema='complete_backend_control_results_ledger:1',prior_evidence=file_pin(prior),summary=file_pin(out/'summary.json'),roots=dict(training=file_pin(tp),evaluation=file_pin(ep)),source_receipts=receipts+[file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        independent_review=file_pin(analysis/'independent_review.json'),verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),added_actual_native_integrations=0,added_optimizer_steps=0))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified),control_native=65538048+evaluation['actual_native_integrations'],telemetry=telemetry)),flush=True)


if __name__=='__main__':main()
