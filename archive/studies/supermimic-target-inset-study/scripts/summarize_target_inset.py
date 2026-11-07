"""Independent CPU analysis of the nine predeclared C2 target-inset jobs.

No simulation/training, confidence interval, selected best run, new training seed
or C1 calibration. Original physical state and task limits are never altered.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT.parent/'supermimic-control-feasibility-draft/scripts/summarize_plain_repeats.py'
HELPER_SHA='5bbe7c85a8ac75d9435f1ed6c64a28ee52a28b8b2ca2171c51787fd5bfbe5b78'
spec=importlib.util.spec_from_file_location('frozen_plain_repeat_audit',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.require(h.sha(HELPER)==HELPER_SHA,'Frozen CPU metric helper changed')
read,write,require,sha,npz=h.read,h.write,h.require,h.sha,h.npz
INITIAL_PHYSICAL=tuple(k for k in h.INITIAL if k!='joint_target_rad')
TARGET_FIELDS=('u_proposed_original_bounded_rad','u_executed_filtered_rad','u_cached_after_step_rad','u_actual_ctrl_readback_rad','ema_residual_before_filter_rad','ema_residual_after_filter_rad')

def native_equal(a,b):
    x,y=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
    return bool(x.shape==y.shape and np.array_equal(x.view(np.uint32),y.view(np.uint32)))
def all_checks(checks):
    return all(all_checks(v) if isinstance(v,dict) else bool(v) for v in checks.values())
def stat(values):
    values=np.asarray(values,float)
    return {'mean':float(values.mean()),'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max())}

def target_diagnostic(folder,run,job,baseline):
    path=folder/'control_targets_50hz.npz';a=npz(path);raw=run['arrays'];margin=np.float32(job['margin_rad'])
    require(all(key in a and a[key].shape==(150,20) and np.isfinite(a[key]).all() for key in TARGET_FIELDS),'Invalid150x20 target/EMA evidence')
    p,e,c,u=(a[key].astype(np.float32) for key in TARGET_FIELDS[:4])
    lo,hi=(a[key].astype(np.float32) for key in ('lower_bound_rad','upper_bound_rad'))
    il,iu=(a[key].astype(np.float32) for key in ('inset_lower_rad','inset_upper_rad'))
    ids=np.asarray(a['position_ctrl_ids']);require(ids.shape==(20,) and np.issubdtype(ids.dtype,np.integer) and len(np.unique(ids))==20 and np.all((ids>=0)&(ids<len(a['initial_ctrl']))),'Missing/invalid explicit actuator mapping')
    checks={'canonical_interval_starts':bool(np.allclose(a['time_start_s'],np.arange(150)*.02,atol=1e-12,rtol=0)),
            'canonical_interval_ends':bool(np.allclose(a['time_end_s'],np.arange(1,151)*.02,atol=1e-12,rtol=0)),
            'original_bounds_match_model_FP32':native_equal(lo,raw['joint_lower_rad']) and native_equal(hi,raw['joint_upper_rad']),
            'nonempty_inset_range':bool(np.all(il<=iu)),
            'inset_bounds_native_FP32_formula':native_equal(il,lo+margin) and native_equal(iu,hi-margin),
            'executed_native_FP32_clip_formula':native_equal(e,np.clip(p,il,iu)),
            'cached_actual_equals_executed':native_equal(c,e) and native_equal(u,e),
            'rollout_actual_target_matches_interval_log':native_equal(raw['joint_target_rad'][1:],u),
            'u_prev_matches_raw_t0_target':native_equal(a['u_prev_rad'],raw['joint_target_rad'][0]),
            'u_prev_matches_initial_ctrl':native_equal(a['u_prev_rad'],a['initial_ctrl'][ids]),
            'u_prev_matches_initial_cached_target':native_equal(a['u_prev_rad'],a['initial_cached_target_rad']),
            'EMA_residual_unchanged_bitwise':native_equal(a[TARGET_FIELDS[4]],a[TARGET_FIELDS[5]])}
    if float(job['margin_mrad'])==0:
        checks['zero_margin_identity_bitwise']=native_equal(p,e) and native_equal(e,c) and native_equal(e,u)
        checks['zero_margin_EMA_identity_bitwise']=native_equal(a[TARGET_FIELDS[4]],a[TARGET_FIELDS[5]])
    if baseline is not None:
        b=baseline['target_arrays']
        checks['full_initial_qpos_qvel_ctrl_exact']={k:bool(np.array_equal(a[k],b[k])) for k in ('initial_qpos','initial_qvel','initial_ctrl','u_prev_rad')}
    delta=e.astype(float)-p.astype(float);changed=delta!=0
    metrics={'clamp_component_count':int(changed.sum()),'clamp_component_fraction':float(changed.mean()),'clamp_tick_count':int(changed.any(-1).sum()),'clamp_tick_fraction':float(changed.any(-1).mean()),
             'clamp_delta_abs_mean_mrad':float(1000*np.abs(delta).mean()),'clamp_delta_abs_max_mrad':float(1000*np.abs(delta).max()),
             'clamp_components_by_joint':{joint:int(changed[:,j].sum()) for j,joint in enumerate(run['metadata']['joint_names'])},
             'proposal_definition':'Original post-hard-limit/post-EMA bounded absolute target; not pre-limit raw proposal',
             'timebase':'150control intervals at50Hz; no4x physics-sample denominator'}
    return a,checks,metrics,sha(path)

def contact_diagnostic(run):
    a=run['arrays'];active=a['live_contact_active'].astype(bool);counts=active.sum(-1);norm=np.linalg.norm(a['live_contact_normal_o'],axis=-1)
    force=a['live_contact_force_w_N'];values=np.linalg.norm(force,axis=-1)[active]
    checks={'active_mask_binary':bool(np.isin(a['live_contact_active'],[0,1]).all()),'logged_count_equals_active_mask':bool(np.array_equal(counts,a['hand_object_active_contact_count'])),
            'active_normals_unit':bool(np.allclose(norm[active],1,atol=1e-5,rtol=0)),
            'measured_force_finite':bool(np.isfinite(force).all()),'support512_finite_nonnegative':bool(np.isfinite(a['live_support512']).all() and np.all(a['live_support512']>=0))}
    metrics={'real_contact_frames':int((counts>0).sum()),'real_contact_frame_fraction':float((counts>0).mean()),'real_contact_count_mean':float(counts.mean()),'real_contact_count_max':int(counts.max()),
             'active_contact_force_norm_mean_N':float(values.mean()) if len(values) else None,'active_contact_force_norm_max_N':float(values.max()) if len(values) else None,
             'hand_object_penetration_max_mm':float(1000*a['hand_object_penetration_m'].max()),'hand_self_penetration_max_mm':float(1000*a['hand_self_penetration_m'].max()),'other_penetration_max_mm':float(1000*a['other_penetration_m'].max()),
             'contact_scope':run['metadata']['hand_contact_scope'],'qualification':'Actual solver-active hand-object measurements; geometric support is not force matching, actuator-only braking or task success.'}
    return checks,metrics

def load_job(job,manager_job,plan,baseline,plan_sha):
    path=Path(job['output_dir']);run=h.load_run(path,job['job_id'],None);row=run['row'];meta=run['metadata'];condition=read(path/'condition.json')
    checks={'checkpoint_plan_pin':meta['checkpoint_sha256']==plan['checkpoint']['sha256'],
            'reference_plan_pin':meta['reference_bundle_sha256']==plan['reference']['sha256'],
            'model_plan_pin':meta['model_sha256']==plan['expected_model_sha256'],
            'original_evaluator_pin':meta['script_sha256']==plan['source_pins']['evaluator']['sha256'],
            'predeclared_initial_anchor_available':baseline is not None or job['job_id']==plan['jobs'][0]['job_id'],
            'canonical_frozen_task_criterion':run['evaluation']['task_protocol_sha256']==read(ROOT/'analysis_protocol_v1.json')['task_protocol_sha256'],
            'manager_output_matches_plan':h.same_path(manager_job['output_dir'],path),
            'manager_complete_exit0':manager_job['status']=='completed' and manager_job['process']['status']=='completed' and manager_job['process']['exit_code']==0,
            'condition_plan_seed_margin':condition['plan_sha256']==plan_sha and condition['training_seed']==1 and condition['evaluation_seed']==0 and condition['margin_mrad']==job['margin_mrad'] and condition['job_id']==job['job_id'],
            'condition_sources_match_plan':condition['source_hashes']=={k:v['sha256'] for k,v in plan['source_pins'].items()},
            'C1_disabled':condition['C1_enabled'] is False and condition['m0'] is None and condition['a_brake'] is None}
    if baseline is not None:
        checks['initial_physical_fields_exact']={k:bool(np.array_equal(run['arrays'][k][0],baseline['arrays'][k][0])) for k in INITIAL_PHYSICAL}
        checks['initial_public_target_exact']=native_equal(run['arrays']['joint_target_rad'][0],baseline['arrays']['joint_target_rad'][0])
        checks['reference_criteria_same']=run['reference'].meta['task_acceptance']==baseline['reference'].meta['task_acceptance']
    target_arrays,target_checks,target_metrics,target_sha=target_diagnostic(path,run,job,baseline)
    contact_checks,contact_metrics=contact_diagnostic(run)
    row['checks'].update({'C2_binding':checks,'C2_target_contract':target_checks,'C2_real_contacts':contact_checks})
    row['hashes']['control_targets_50hz.npz']=target_sha
    artifact_pins=manager_job.get('health',{}).get('artifact_sha256',{})
    row['checks']['manager_artifact_hashes']=bool(artifact_pins) and all(row['hashes'].get(name)==value for name,value in artifact_pins.items())
    row.update({key:job[key] for key in ('job_id','block_id','position','condition_id','margin_mrad','margin_rad','repeat_id')})
    row['control_intervention']=target_metrics;row['real_contacts']=contact_metrics
    row['metrics'].update(load_ratio_max=run['evaluation']['load_ratio_max'],load_charge_max=run['evaluation']['load_charge_max'],target_speed_abs_max_rad_s=run['evaluation']['target_speed_abs_max_rad_s'])
    row['cost'].update(child_supervisor_wall_seconds=manager_job['process'].get('elapsed_s'))
    if not all_checks(row['checks']):row.update(status='INVALID_BINDING_OR_TARGET_CONTRACT',task_success=None)
    run['target_arrays']=target_arrays
    return run

def figures(runs,out):
    plt=h.plt;plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
    colors={0:'#1764ab',5:'#d55e00',20:'#15845c'};styles={1:'-',2:'--',3:':'}
    for run in runs:
        row=run['row'];a=run['arrays'];color=colors[int(row['margin_mrad'])];style=styles[int(row['block_id'])];label=f"{row['margin_mrad']}mrad block{row['block_id']}"
        for ax,value in zip(axes,[run['curves']['joint_state_max_excess_mrad'],run['curves']['shaft_error_deg'],a['hand_object_active_contact_count']]):ax.plot(a['time_s'],value,label=label,color=color,linestyle=style,alpha=.85)
    axes[0].axhline(1,color='black',linestyle='--',linewidth=.8,label='Frozen1mrad criterion');axes[1].axhline(20,color='black',linestyle='--',linewidth=.8,label='Frozen20deg criterion')
    for ax,label in zip(axes,['Maximum actual-state excess over20joints (mrad)','Directed shaft error (deg)','Real solver-active hand-object contacts']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel('Canonical time (s)')
    fig.suptitle('C2: all nine target-inset jobs, same fixed A1L checkpoint (training seed1/evaluation0)\nEach margin has three fresh processes in prespecified Latin-square order. No new training-seed N or dynamics guarantee.')
    for ext in ('png','svg'):fig.savefig(out/f'tracking.{ext}',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,default=ROOT/'configs/target-inset-v1.json');p.add_argument('--manager',type=Path);p.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_v1');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);protocol=read(ROOT/'analysis_protocol_v1.json')
    header={'schema':'C2_TARGET_INSET_analysis:1','script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'analysis_protocol_v1.json'),'frozen_metric_helper_sha256':sha(HELPER),'gpu_used':False,'torch_imported':'torch' in sys.modules,'training_seed_N':1,'m0':None,'a_brake':None,'parameter_enablement':False,'prior_plain_repeats_used_as_C2_baseline':False}
    if not args.plan.exists():write(args.output/'comparison.json',{**header,'status':'PENDING_FROZEN_PLAN','results':None});print('PENDING_FROZEN_PLAN');return
    plan=read(args.plan);header['plan_sha256']=sha(args.plan)
    checks={name:sha(pin['path'])==pin['sha256'] for name,pin in plan['source_pins'].items()}
    checks.update({name:sha(plan[name]['path'])==plan[name]['sha256'] for name in ('checkpoint','reference','reference_metadata')})
    require(all(checks.values()),'Frozen source/input hash changed')
    jobs=plan['jobs'];expected=[(b['block_id'],position,margin) for b in protocol['blocks'] for position,margin in enumerate(b['order_mrad'],1)]
    require(len(jobs)==len({j['job_id'] for j in jobs})==9 and [(int(j['block_id']),int(j['position']),int(j['margin_mrad'])) for j in jobs]==expected,'Prespecified Latin-square job order changed')
    header['plan_source_input_checks']=checks
    if args.prepare_only:write(args.output/'preparation.json',{**header,'status':'CPU_PLAN_PREPARATION_PASS','job_count':9,'physical_results':None});print('CPU_PLAN_PREPARATION_PASS');return
    manager_path=args.manager or Path(plan['output_root'])/'state.json';manager=read(manager_path) if manager_path.exists() else {'status':'PENDING'}
    if manager['status']!='completed_9_jobs':write(args.output/'comparison.json',{**header,'status':'PENDING_TERMINAL_COMPLETION','manager_status':manager['status'],'results':None,'pass_counts':None});print('PENDING_TERMINAL_COMPLETION');return
    require(manager['plan_sha256']==sha(args.plan),'Manager plan mismatch')
    require(len(manager['jobs'])==9 and {j['job_id'] for j in manager['jobs']}=={j['job_id'] for j in jobs},'Manager job coverage differs')
    runs=[];invalid=[];baseline=None
    for job in jobs:
        try:
            mj=next(j for j in manager['jobs'] if j['job_id']==job['job_id']);run=load_job(job,mj,plan,baseline,sha(args.plan));runs.append(run)
            if job['job_id']==jobs[0]['job_id']:baseline=run
        except (OSError,KeyError,ValueError,IndexError,StopIteration) as error:invalid.append({**job,'status':'INVALID_OR_MISSING','task_success':None,'reason':str(error)})
    rows=[r['row'] for r in runs]+invalid;valid=len(runs)==9 and not invalid and all(r['row']['status']=='VALID' for r in runs)
    groups={}
    for margin in protocol['conditions_mrad']:
        members=[r for r in rows if r['margin_mrad']==margin];ok=len(members)==3 and all(r['status']=='VALID' for r in members)
        groups[str(margin)]={'jobs':len(members),'valid':ok,'task_passes':sum(r['task_success'] is True for r in members) if ok else None,'metrics':{key:stat([r['metrics'][key] for r in members]) for key in runs[0]['row']['metrics'] if isinstance(runs[0]['row']['metrics'][key],(int,float)) and not isinstance(runs[0]['row']['metrics'][key],bool)} if ok and runs else None}
    contrasts={}
    for margin in (5,20):
        pairs=[]
        for block in (1,2,3):
            a=next((r for r in rows if int(r['block_id'])==block and r['margin_mrad']==0 and r['status']=='VALID'),None);b=next((r for r in rows if int(r['block_id'])==block and r['margin_mrad']==margin and r['status']=='VALID'),None)
            pairs.append({'block_id':block,'available':a is not None and b is not None,'difference':{key:b['metrics'][key]-a['metrics'][key] for key in ('joint_state_max_excess_mrad','shaft_mean_deg','shaft_max_deg','position_mean_mm','position_max_mm','progress_endpoint_error_deg','load_ratio_max','load_charge_max','target_speed_abs_max_rad_s')} if a and b else None})
        contrasts[f'{margin}_minus_0']={'pairs':pairs,'summary':{key:stat([v['difference'][key] for v in pairs]) for key in pairs[0]['difference']} if all(v['available'] for v in pairs) else None}
    data={**header,'status':'COMPLETE' if valid else 'INVALID_OR_MISSING_JOBS','manager_status':manager['status'],'manager_sha256':sha(manager_path),'results':rows,'conditions':groups,'descriptive_block_contrasts':contrasts,'cost':{'whole_queue_wall_seconds':manager.get('whole_queue_wall_seconds'),'sum_evaluator_report_seconds':sum(r['row']['cost']['evaluator_report_wall_seconds'] for r in runs),'sum_child_supervisor_seconds':sum(r['row']['cost']['child_supervisor_wall_seconds'] for r in runs) if all(r['row']['cost']['child_supervisor_wall_seconds'] is not None for r in runs) else None,'scope':'Distinct evaluator, child-supervisor and whole-queue wall scopes, not GPU kernel busy time.'},'limits':protocol['limits'],'curves':{r['row']['job_id']:r['curves'] for r in runs}}
    write(args.output/'comparison.json',data)
    fields=['job_id','block_id','position','margin_mrad','repeat_id','status','task_success','joint_state_max_excess_mrad','state_peak_time_s','state_peak_joint','joint_target_max_excess_mrad','shaft_mean_deg','shaft_max_deg','position_mean_mm','position_max_mm','drop_detected','load_ratio_max','load_charge_max','target_speed_abs_max_rad_s','clamp_component_fraction','clamp_tick_fraction','clamp_delta_abs_max_mrad','real_contact_frames','real_contact_count_mean','real_contact_count_max','child_supervisor_wall_seconds','evaluator_report_wall_seconds']
    with (args.output/'jobs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:v for group in (r,r.get('metrics',{}),r.get('control_intervention',{}),r.get('real_contacts',{}),r.get('cost',{})) for k,v in group.items() if k in fields})
    if runs:figures(runs,args.output)
    report=[f"Status: {data['status']}",'C2 same fixed A1L checkpoint, training seed1/evaluation0. Three fresh processes per margin do not add training seeds. Original five plain repeats are not this experiment baseline.']
    for margin,group in groups.items():report.append(f"Margin{margin}mrad: task passes={group['task_passes']}/3; all three valid={group['valid']}.")
    for r in rows:report.append(f"{r['job_id']}: margin={r['margin_mrad']}mrad, status={r['status']}, task={r['task_success']}, failed frozen gates={r.get('failed_task_gates')}, metrics={r.get('metrics')}")
    report+=['All9 outcomes retained; no CI, best-repeat/checkpoint selection, threshold relaxation, actual-state clamping or trained adaptation.','Contrasts are descriptive within predeclared Latin-square blocks; feedback trajectories and realized numerical noise are not matched counterfactual action sequences.','Targets inset, physical joint ranges and original task gates unchanged. No actuator-only braking, hard joint-state safety, dynamics guarantee or C1 qualification. m0/a_brake=null; parameter enablement=false.',protocol['limits']]
    (args.output/'REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf-8');print(json.dumps({'status':data['status'],'condition_passes':{k:v['task_passes'] for k,v in groups.items()},'training_seed_N':1},indent=2))

if __name__=='__main__':main()
