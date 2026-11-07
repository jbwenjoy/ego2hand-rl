"""CPU-only audit of five predeclared fresh-process evaluator repetitions.

No simulation, training, confidence intervals, best-of-five or seed-N inflation.
The instrumented donor is a separate diagnostic, never one of the five repeats.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent/'supermimic-chord-study'
sys.path.insert(0,str(STUDY/'src'))
from chord_study.acceptance import protocol_digest
from chord_study.evaluation import evaluate
from chord_study.geometry import quat_matrix
from chord_study.reference import Reference

FILES=('rollout.npz','rollout.json','evaluation.json','model.mjb','reference.npz','reference.json')
INITIAL=('joint_pos_rad','joint_vel_rad_s','joint_target_rad','object_pose_w','object_lin_vel_w_m_s','object_ang_vel_w_rad_s','wrist_pose_w')
GATES=('complete_clip','unassisted','task_tracking_ok','progress_ok','drop_ok','load_ok','target_rate_ok','task_penetration_ok','task_joint_limits_ok','joint_state_limits_ok','joint_target_limits_ok','task_logging_contract_ok')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
def require(ok,why):
    if not ok:raise ValueError(why)
def same_path(a,b):return Path(a).resolve()==Path(b).resolve()
def check_bound_directory(spec,actual):
    require(same_path(spec['path'],actual),'Directory differs from predeclared plan')
    return {name:sha(Path(actual)/name)==expected for name,expected in spec['sha256'].items()}
def compare_metrics(actual,recorded,prefix=''):
    for k,v in actual.items():
        require(k in recorded,'Missing recorded metric '+prefix+k)
        if isinstance(v,dict):compare_metrics(v,recorded[k],prefix+k+'.')
        elif isinstance(v,(float,np.floating)):require(np.isclose(v,recorded[k],atol=1e-10,rtol=1e-10),'Metric differs '+prefix+k)
        else:require(v==recorded[k],'Metric differs '+prefix+k)

def load_run(path,label,archive):
    a=npz(path/'rollout.npz');m=read(path/'rollout.json');e=read(path/'evaluation.json');r=Reference.load(path/'reference.npz')
    hashes={name:sha(path/name) for name in FILES}
    if (path/'condition.json').exists():hashes['condition.json']=sha(path/'condition.json')
    checks={'reference_file_and_metadata_pin':hashes['reference.npz']==m['reference_bundle_sha256'],
            'model_file_and_metadata_pin':hashes['model.mjb']==m['model_sha256'],
            'protocol_digest':protocol_digest(r.meta['task_acceptance'])==e['task_protocol_sha256'],
            'training_seed1':m['checkpoint_training_config']['seed']==1,
            'evaluation_seed0':m['seed']==0,
            'checkpoint_final2000':m['checkpoint_iteration']==2000,
            'full151_canonical_clock':np.array_equal(a['time_s'],np.arange(151)*.02),
            'no_reset_assist_noise_DR':m['resets_during_rollout']==m['object_state_writes_after_initialization']==0 and all(m[k] is False for k in ('domain_randomization','observation_corruption','external_disturbances'))}
    if archive is not None:
        am=archive['metadata']; ar=archive['reference']
        checks.update({f'archive_pin:{k}':m[k]==am[k] for k in ('checkpoint_sha256','model_sha256','reference_bundle_sha256','script_sha256','physics_dt_s','decimation','control_hz')})
        checks['same_protocol_as_archive']=e['task_protocol_sha256']==archive['evaluation']['task_protocol_sha256']
        checks['same_joint_order_as_archive']=m['joint_names']==am['joint_names']
        checks['same_frozen_criteria_as_archive']=r.meta['task_acceptance']==ar.meta['task_acceptance']
        checks['initial_state_exact_to_archive']={k:np.array_equal(a[k][0],archive['arrays'][k][0]) for k in INITIAL}
        checks['same_bounds_and_force_limits']={k:np.array_equal(a[k],archive['arrays'][k]) for k in ('joint_lower_rad','joint_upper_rad','actuator_force_limit_Nm')}
    recomputed=evaluate(r,a,m);compare_metrics(recomputed,e)
    checks['all_saved_metrics_independently_recomputed']=True
    bools=[v for v in checks.values() if not isinstance(v,dict)]+[v for d in checks.values() if isinstance(d,dict) for v in d.values()]
    valid=all(bools)
    violation=np.maximum(np.maximum(a['joint_lower_rad']-a['joint_pos_rad'],a['joint_pos_rad']-a['joint_upper_rad']),0)
    target_violation=np.maximum(np.maximum(a['joint_lower_rad']-a['joint_target_rad'],a['joint_target_rad']-a['joint_upper_rad']),0)
    axis=quat_matrix(a['object_pose_w'][:,3:])[:,:,0];reference_axis=quat_matrix(r.arrays['object_pose_w'][:,3:])[:,:,0]
    shaft=np.rad2deg(np.arccos(np.clip((axis*reference_axis).sum(-1),-1,1)))
    k,j=np.unravel_index(np.argmax(violation),violation.shape)
    row={'label':label,'path':str(path),'status':'VALID' if valid else 'INVALID_BINDING','training_seed':1,'evaluation_seed':0,'checks':checks,'hashes':hashes,'task_success':e['task_success'] if valid else None,
         'saved_task_success':e['task_success'],'gates':{k:e[k] for k in GATES},'failed_task_gates':[k for k in GATES if e[k] is False],
         'metrics':{'joint_state_max_excess_mrad':float(1000*violation.max()),'joint_target_max_excess_mrad':float(1000*target_violation.max()),'state_peak_time_s':float(a['time_s'][k]),'state_peak_joint':m['joint_names'][j],
         'shaft_mean_deg':e['orientation_error_deg']['mean'],'shaft_max_deg':e['orientation_error_deg']['max'],'position_mean_mm':1000*e['position_error_m']['mean'],'position_max_mm':1000*e['position_error_m']['max'],
         'drop_detected':e['drop_detected'],'progress_endpoint_error_deg':e['rotation_progress_error_deg']['endpoint']},
         'cost':{'evaluator_report_wall_seconds':e.get('wall_seconds'),'rollout_report_wall_seconds':m.get('wall_seconds'),'GPU_slot_seconds':None,'note':'Evaluator wall values are retained separately; launcher/GPU-slot cost is unavailable unless separately recorded.'}}
    return {'row':row,'arrays':a,'metadata':m,'evaluation':e,'reference':r,'curves':{'time_s':a['time_s'].tolist(),'joint_state_max_excess_mrad':(1000*violation.max(-1)).tolist(),'shaft_error_deg':shaft.tolist()}}

def differences(actual,baseline):
    a,b=actual['arrays'],baseline['arrays'];result={}
    fields={'q_rad':(a['joint_pos_rad'],b['joint_pos_rad']),'qd_rad_s':(a['joint_vel_rad_s'],b['joint_vel_rad_s']),'executed_target_rad':(a['joint_target_rad'],b['joint_target_rad']),
            'object_position_m':(a['object_pose_w'][:,:3],b['object_pose_w'][:,:3]),'object_rotation_matrix':(quat_matrix(a['object_pose_w'][:,3:]),quat_matrix(b['object_pose_w'][:,3:])),
            'object_linear_velocity_m_s':(a['object_lin_vel_w_m_s'],b['object_lin_vel_w_m_s']),'object_angular_velocity_rad_s':(a['object_ang_vel_w_rad_s'],b['object_ang_vel_w_rad_s']),
            'actuator_torque_Nm':(a['actuator_torque_Nm'],b['actuator_torque_Nm']),'support512':(a['live_support512'],b['live_support512'])}
    for name,(x,y) in fields.items():
        if x.shape!=y.shape:result[name]={'status':'UNALIGNED_SHAPE','actual':list(x.shape),'baseline':list(y.shape)};continue
        d=np.abs(x-y);frames=d.reshape(len(d),-1).max(-1);nz=np.flatnonzero(frames>0)
        result[name]={'max_abs':float(d.max()),'mean_abs':float(d.mean()),'exactly_equal':bool(np.array_equal(x,y)),'first_exact_difference_time_s':float(a['time_s'][nz[0]]) if len(nz) else None,'peak_difference_time_s':float(a['time_s'][np.argmax(frames)])}
    result['initial_exact']={k:np.array_equal(a[k][0],b[k][0]) for k in INITIAL}
    result['canonical_time_exact']=np.array_equal(a['time_s'],b['time_s'])
    return result

def figures(repeats,archive,donor,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    traces=[archive]+repeats+([donor] if donor else [])
    for run in traces:
        label=run['row']['label'];style=':' if run is donor else '--' if run is archive else '-'
        for ax,key in zip(axes,('joint_state_max_excess_mrad','shaft_error_deg')):ax.plot(run['curves']['time_s'],run['curves'][key],label=label,linestyle=style,alpha=.85)
    axes[0].axhline(1,color='black',linestyle='--',linewidth=.8,label='Frozen1mrad criterion')
    axes[1].axhline(20,color='black',linestyle='--',linewidth=.8,label='Frozen20deg criterion')
    axes[0].set_ylabel('Maximum actual-state excess over20 joints (mrad)');axes[1].set_ylabel('Directed shaft error (deg)');axes[1].set_xlabel('Canonical time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=4)
    fig.suptitle('One frozen A1L checkpoint: five fresh-process repeats (training seed1, evaluation seed0)\nArchive and instrumented donor are separately labeled; repeated runs do not increase training-seed N.')
    for ext in ('png','svg'):fig.savefig(out/f'traces.{ext}',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--manager',type=Path,required=True)
    p.add_argument('--repeat',action='append',default=[],help='repeat1=absolute output path; exactlyrepeat1..repeat5')
    p.add_argument('--archive',type=Path);p.add_argument('--instrumented-donor',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check-archive-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);plan=read(args.plan)
    args.archive=args.archive or Path(plan['archive']['path'])
    args.instrumented_donor=args.instrumented_donor or Path(plan['instrumented_donor']['path'])
    require(plan['training_seed']==1 and plan['evaluation_seed']==0 and plan['reference_repeat_id']=='repeat1' and plan['repeat_count']==5,'Frozen repetition design changed')
    archive_pins=check_bound_directory(plan['archive'],args.archive)
    require(all(archive_pins.values()),'Original archive no longer matches predeclared hashes')
    source_checks={name:sha(pin['path'])==pin['sha256'] for name,pin in plan['source_pins'].items()}
    input_checks={name:sha(plan[name]['path'])==plan[name]['sha256'] for name in ('checkpoint','reference','reference_metadata')}
    require(all(source_checks.values()) and all(input_checks.values()),'Frozen input/source changed')
    header={'schema':'plain_fresh_process_repetition_audit:1','script_sha256':sha(__file__),'plan_sha256':sha(args.plan),'archive_plan_hash_checks':archive_pins,'source_plan_hash_checks':source_checks,'input_plan_hash_checks':input_checks,'gpu_used':False,'torch_imported':'torch' in sys.modules,'training_seed_N':1,'repeats_are_independent_training_seeds':False,'predeclared_anchor':'repeat1','instrumented_donor_in_five_repeat_denominator':False,'confidence_intervals':None,'best_repeat_selection':False}
    archive=load_run(args.archive,'original_archive',None)
    require(archive['row']['status']=='VALID','Original archive metric/source validation failed')
    if args.check_archive_only:
        write(args.output/'archive_preparation_check.json',{**header,'archive':archive['row'],'qualification':'Read-only loader/metric audit on existing archive; five fresh repeats have not been evaluated by this preparation check.'});print('ARCHIVE_PREPARATION_CHECK '+archive['row']['status']);return
    planned={job['repeat_id']:job['output_dir'] for job in plan['repeats']}
    specs=[value.split('=',1) for value in args.repeat] if args.repeat else list(planned.items())
    require(len(specs)==5 and {label for label,path in specs}=={f'repeat{i}' for i in range(1,6)},'Must supply exactly predeclared repeat1..repeat5, no selection')
    require(set(planned)=={label for label,path in specs} and all(same_path(planned[label],path) for label,path in specs),'Repeat directories differ from predeclared plan')
    manager=read(args.manager) if args.manager.exists() else {'status':'PENDING'}
    # Exact terminal status is bound by the final runner schema before execution.
    complete=manager.get('status')=='completed_5_repeats'
    if not complete:
        write(args.output/'comparison.json',{**header,'status':'PENDING_MANAGER_TERMINAL_COMPLETION','manager_status':manager.get('status'),'repeat_results':None,'pass_count':None});print('PENDING_MANAGER_TERMINAL_COMPLETION');return
    require(manager['plan_sha256']==sha(args.plan),'Manager plan binding differs')
    require(len(manager['jobs'])==5 and {job['repeat_id'] for job in manager['jobs']}==set(planned),'Manager repeat coverage differs')
    runs=[];invalid=[]
    for label,folder in sorted(specs):
        try:
            run=load_run(Path(folder),label,archive)
            job=next(job for job in manager['jobs'] if job['repeat_id']==label)
            expected=next(job for job in plan['repeats'] if job['repeat_id']==label)
            condition=read(Path(folder)/'condition.json')
            more={'manager_job_matches_plan':same_path(job['output_dir'],folder) and job['command']==expected['command'],
                  'manager_job_engineering_exit0':job['status']=='completed' and job['process']['status']=='completed' and job['process']['exit_code']==0,
                  'manager_saved_artifact_hashes':all(run['row']['hashes'].get(name)==value for name,value in job['artifacts']['artifact_sha256'].items()),
                  'condition_plan_seed_checkpoint':condition['repeat_id']==label and condition['plan_sha256']==sha(args.plan) and condition['training_seed']==1 and condition['evaluation_seed']==0 and condition['checkpoint_sha256']==plan['checkpoint']['sha256'] and condition['evaluator_sha256']==plan['source_pins']['evaluator']['sha256'],
                  'quality_not_used_for_control':condition['new_independent_training_seed'] is False and condition['task_quality_selects_or_stops'] is False}
            run['row']['checks'].update(more)
            if not all(more.values()):run['row'].update(status='INVALID_BINDING',task_success=None)
            run['row']['cost'].update(child_supervisor_wall_seconds=job['process'].get('elapsed_s'),GPU_slot_seconds=None,note='Evaluator and whole-child-supervisor times differ. Driver holds one shared GPU lock across five jobs; full driver wall is recorded separately and is not GPU kernel busy time.')
            runs.append(run)
        except (OSError,KeyError,ValueError,IndexError) as error:invalid.append({'label':label,'path':folder,'status':'INVALID_OR_MISSING','task_success':None,'reason':str(error)})
    anchor=next((run for run in runs if run['row']['label']=='repeat1'),None)
    for run in runs:
        run['row']['trace_differences_to_archive']=differences(run,archive)
        run['row']['trace_differences_to_predeclared_repeat1']=differences(run,anchor) if anchor else None
    donor_error=None
    try:
        donor=load_run(args.instrumented_donor,'instrumented_donor_separate',archive)
        donor['row']['plan_hash_checks']=check_bound_directory(plan['instrumented_donor'],args.instrumented_donor)
        if not all(donor['row']['plan_hash_checks'].values()):donor['row'].update(status='INVALID_BINDING',task_success=None)
        donor['row']['trace_differences_to_archive']=differences(donor,archive)
        donor['row']['trace_differences_to_predeclared_repeat1']=differences(donor,anchor) if anchor else None
    except (OSError,KeyError,ValueError,IndexError) as error:donor=None;donor_error=str(error)
    valid=len(runs)==5 and not invalid and all(run['row']['status']=='VALID' for run in runs)
    data={**header,'status':'COMPLETE' if valid else 'INVALID_OR_MISSING_REPEATS','manager_status':manager['status'],'manager_sha256':sha(args.manager),
          'repeat_results':[run['row'] for run in runs]+invalid,'archive':archive['row'],'instrumented_donor':donor['row'] if donor else {'status':'UNAVAILABLE','reason':donor_error},
          'pass_count':sum(run['row']['task_success'] is True for run in runs) if valid else None,'repeat_count':5,'curves':{run['row']['label']:run['curves'] for run in runs},'cost':{'whole_driver_wall_seconds':manager.get('whole_driver_wall_seconds'),'within240s_budget':manager.get('within_whole_budget'),'five_evaluator_report_seconds':sum(run['row']['cost']['evaluator_report_wall_seconds'] for run in runs),'five_child_supervisor_seconds':sum(run['row']['cost']['child_supervisor_wall_seconds'] for run in runs) if all(run['row']['cost']['child_supervisor_wall_seconds'] is not None for run in runs) else None,'scope':'Whole driver includes preflight/artifact work and locked gaps; not GPU kernel busy time or training cost.'},
          'scope':'Descriptive numerical/process repeatability of one fixed checkpoint/reference/initial state. No training-seed replication, intervention effect, weak-C1 calibration or method-generalization inference.'}
    write(args.output/'comparison.json',data)
    fields=['label','status','task_success','joint_state_max_excess_mrad','joint_target_max_excess_mrad','shaft_mean_deg','shaft_max_deg','position_mean_mm','position_max_mm','drop_detected','progress_endpoint_error_deg','evaluator_report_wall_seconds','rollout_report_wall_seconds','child_supervisor_wall_seconds','failed_task_gates']
    with (args.output/'repeats.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for run in runs:
            row=run['row'];flat={k:v for group in (row,row['metrics'],row['cost']) for k,v in group.items() if k in fields};flat['failed_task_gates']=';'.join(row['failed_task_gates']);writer.writerow(flat)
    figures(runs,archive,donor,args.output)
    text=[f"Status: {data['status']}",f"Same-checkpoint task passes: {data['pass_count']}/5 (unavailable if any binding/raw invalid).",'Training-seed N remains1; five fresh processes are numerical repeatability, not five learned policies.','Predeclared anchor is repeat1; all trace differences to repeat1 and original archive are retained. Instrumented donor is separate and excluded from the five.']
    for run in runs:
        row=run['row'];v=row['metrics'];text.append(f"{row['label']}: task={row['task_success']}, actual-q excess={v['joint_state_max_excess_mrad']:.6f}mrad, target excess={v['joint_target_max_excess_mrad']:.6f}mrad, shaft max={v['shaft_max_deg']:.6f}deg, position max={v['position_max_mm']:.6f}mm, drop={v['drop_detected']}; failed={row['failed_task_gates']}")
    text+=['No CI, no success-based selection, no best-of-five result.','Evaluator report wall time is distinct from launcher/GPU-slot wall time; no missing cost is treated as zero.','Earlier archive and instrumented donor do not share independent-process qualification merely because hashes/initial conditions match.']
    (args.output/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8');print(json.dumps({'status':data['status'],'pass_count':data['pass_count'],'training_seed_N':1},indent=2))

if __name__=='__main__':main()
