"""CPU-only hierarchical audit of all27 fixed common-controller portfolio jobs.

Three processes per checkpoint describe numerical repetition. Method-level N
is three training seeds, never nine processes or151frames. No training/physics.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for helper,expected in [('supermimic-target-inset-study/scripts/summarize_target_inset.py','9c683d8c0a51b3267c1cfa265355f86d7d34a30b4095c01ba8056f31e2d26ba2'),('supermimic-target-inset-study/scripts/join_capacity_v2.py','a29523fefe78dd127d8db53f960ec3cfbc8699b92f6d73317e2873582573e2ac')]:
    if hashlib.sha256((ROOT.parent/helper).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen CPU helper changed: '+helper)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c2=module(ROOT.parent/'supermimic-target-inset-study/scripts/summarize_target_inset.py','frozen_C2_contract')
h=c2.h
cap=module(ROOT.parent/'supermimic-target-inset-study/scripts/join_capacity_v2.py','frozen_capacity_contract')
read,write,sha,require,npz=h.read,h.write,h.sha,h.require,h.npz
from chord_study.acceptance import projected_shaft_progress
METHODS=['A1L','P_ONLY','A2']
KEYS=['position_mean_mm','position_max_mm','shaft_mean_deg','shaft_max_deg','progress_endpoint_error_deg','joint_state_max_excess_mrad','joint_target_max_excess_mrad','load_ratio_max','load_charge_max','target_speed_abs_max_rad_s','support512_score']

def stats(x):
    x=np.asarray(x,float);return {'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())}
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def support_numpy(points,normals,active,basis,radius):
    """FlashCHORD/sidecar Frisvad frame,8cone edges plus normal, COM origin."""
    if points.shape[-2]==0:return np.zeros(points.shape[:-2]+(len(basis),))
    n=norm(normals);sign=np.where(n[...,2]>=0,1.,-1.);a=-1/(sign+n[...,2]);b=n[...,0]*n[...,1]*a
    t1=norm(np.stack([1+sign*n[...,0]**2*a,sign*b,-sign*n[...,0]],-1));t2=norm(np.stack([b,sign+n[...,1]**2*a,-n[...,1]],-1))
    theta=np.arange(8)*(2*np.pi/8)
    rays=norm(n[...,None,:]+.1*(np.cos(theta)[:,None]*t1[...,None,:]+np.sin(theta)[:,None]*t2[...,None,:]))
    rays=np.concatenate([rays,n[...,None,:]],axis=-2)
    wrench=np.concatenate([rays,np.cross(points[...,None,:],rays)/radius],axis=-1)
    values=np.einsum('...keq,nq->...ken',wrench,basis,optimize=True)
    values=np.where(active[...,None,None],values,-np.inf)
    return np.maximum(values.max(axis=(-3,-2)),0)

def score_numpy(actual,target):
    """Official tolerance-band score: basis,body,hand active averages in order."""
    commanded=target>1e-3;current=actual>1e-3
    loss=np.maximum(.9*target-actual,0)**2+np.maximum(actual-1.1*target,0)**2
    score=np.where(commanded&current,np.exp(-loss/.1),0)
    body_active=commanded.any(-1);body_score=score.sum(-1)/np.maximum(commanded.sum(-1),1)
    hand_active=body_active.any(-1);hand_score=body_score.sum(-1)/np.maximum(body_active.sum(-1),1)
    return hand_score.sum(-1)/np.maximum(hand_active.sum(-1),1)

def contact_audit(run,model,targets,manifest):
    a=run['arrays'];active=a['live_contact_active'].astype(bool);obj=model.body('object/cube').id
    objects=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
    hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name)}
    pairs=a['live_contact_geom_pair'][active];simple,metrics=c2.contact_diagnostic(run)
    checks={**simple,'hand_object_pairs_no_mount':all((int(x) in objects and int(y) in hand) or (int(y) in objects and int(x) in hand) for x,y in pairs),
            'COM_shift_matches_model':bool(np.allclose(a['live_contact_pos_com_o_m'][active],a['live_contact_pos_o_m'][active]-model.body_ipos[obj],atol=1e-10,rtol=0)),
            'basis_matches_frozen512':a['support_basis'].shape==(512,6) and bool(np.allclose(a['support_basis'],targets['support_basis'],atol=1e-7,rtol=0)),
            'target_model_and_ref_pins':run['metadata']['model_sha256']==manifest['model_sha256'] and run['metadata']['reference_bundle_sha256']==manifest['reference_npz_sha256'],
            'target_clock_same':bool(np.array_equal(a['time_s'],targets['time_s'])),
            'target_object_intent_same':bool(np.allclose(run['reference'].arrays['object_pose_w'],targets['input_object_pose_w'],atol=1e-7,rtol=0)),
            'target_wrist_intent_same':bool(np.allclose(run['reference'].arrays['wrist_pose_w'],targets['input_wrist_pose_w'],atol=1e-7,rtol=0))}
    raw=support_numpy(a['live_contact_pos_com_o_m'],a['live_contact_normal_o'],active,a['support_basis'],run['reference'].meta['object_radius_m'])
    error=float(np.abs(raw-a['live_support512']).max());checks['raw_contacts_recompute_support512']=error<=1e-10
    scores=score_numpy(raw[:,None,None,:],targets['support512']);phase=targets['phase']
    weights=np.where(np.isin(phase,[2,3]),1.,np.where(phase==4,.5,0.))*targets['contact_known'].astype(bool)
    checks['known_no_contact_not_attracted']=not bool(np.any((weights>0)&targets['known_no_contact'].astype(bool)))
    require(weights.sum()>0,'No known target-contact frames')
    metrics.update(support512_recompute_max_abs_error=error,support512_score=float(np.sum(scores*weights)/weights.sum()),known_contact_weight_sum=float(weights.sum()))
    metrics['qualification']='Auxiliary unit-cone geometry vs frozen kinematic intent, not force amplitude matching, task success or predictor qualification.'
    return checks,metrics

def metric_run(folder,job,model_spec,plan,protocol):
    a=npz(folder/'rollout.npz');m=read(folder/'rollout.json');e=read(folder/'evaluation.json');r=h.Reference.load(folder/'reference.npz')
    hashes={name:sha(folder/name) for name in [*h.FILES,'condition.json']}
    if (folder/'control_targets_50hz.npz').exists():hashes['control_targets_50hz.npz']=sha(folder/'control_targets_50hz.npz')
    checks={'training_seed_matches_real_checkpoint_config':m['checkpoint_training_config']['seed']==job['training_seed']==model_spec['training_seed'],
            'checkpoint_final2000_SHA':m['checkpoint_iteration']==2000 and m['checkpoint_sha256']==model_spec['checkpoint_sha256'],
            'evaluation_seed0':m['seed']==0,'reference_hash':hashes['reference.npz']==m['reference_bundle_sha256']==plan['reference']['sha256'],
            'nominal_model_hash':hashes['model.mjb']==m['model_sha256']==plan['expected_model_sha256'],
            'evaluator_hash':m['script_sha256']==plan['source_pins']['evaluator']['sha256'],
            'canonical_criterion':h.protocol_digest(r.meta['task_acceptance'])==e['task_protocol_sha256']==protocol['task_protocol_sha256'],
            'full151_clock':bool(np.array_equal(a['time_s'],np.arange(151)*.02)),
            'unassisted_no_reset_no_corruption':m['resets_during_rollout']==m['object_state_writes_after_initialization']==0 and all(m[k] is False for k in ['domain_randomization','observation_corruption','external_disturbances'])}
    recomputed=h.evaluate(r,a,m);h.compare_metrics(recomputed,e);checks['original_frozen_metrics_recomputed']=True
    violation=np.maximum(np.maximum(a['joint_lower_rad']-a['joint_pos_rad'],a['joint_pos_rad']-a['joint_upper_rad']),0);target=np.maximum(np.maximum(a['joint_lower_rad']-a['joint_target_rad'],a['joint_target_rad']-a['joint_upper_rad']),0)
    maximum=float(violation.max());i,j=np.unravel_index(np.argmax(violation),violation.shape)
    axes=h.quat_matrix(a['object_pose_w'][:,3:])[:,:,0];refaxes=h.quat_matrix(r.arrays['object_pose_w'][:,3:])[:,:,0]
    actual,_=projected_shaft_progress(axes,r.meta['task_acceptance']['progress_frame']['normal_w']);expected,_=projected_shaft_progress(refaxes,r.meta['task_acceptance']['progress_frame']['normal_w'])
    row={**{k:job[k] for k in ('job_id','model_id','trained_method','training_seed','evaluation_repeat','block_id','position')},'evaluation_seed':0,'directory':str(folder),'checks':checks,'hashes':hashes,'observed_task_success':e['task_success'],'task_success':None,'gates':{k:e[k] for k in h.GATES},'failed_task_gates':[k for k in h.GATES if e[k] is False],
         'metrics':{'position_mean_mm':1000*e['position_error_m']['mean'],'position_max_mm':1000*e['position_error_m']['max'],'shaft_mean_deg':e['orientation_error_deg']['mean'],'shaft_max_deg':e['orientation_error_deg']['max'],'progress_endpoint_error_deg':e['rotation_progress_error_deg']['endpoint'],'joint_state_max_excess_mrad':1000*maximum,'joint_target_max_excess_mrad':1000*float(target.max()),'load_ratio_max':e['load_ratio_max'],'load_charge_max':e['load_charge_max'],'target_speed_abs_max_rad_s':e['target_speed_abs_max_rad_s']},
         'actual_violation_peak':{'time_s':float(a['time_s'][i]),'joint':m['joint_names'][j]} if maximum>0 else None,'drop_detected':e['drop_detected'],'cost':{'evaluator_report_wall_seconds':e.get('wall_seconds'),'rollout_report_wall_seconds':m.get('wall_seconds')}}
    curves={'time_s':a['time_s'].tolist(),'position_error_mm':(1000*np.linalg.norm(a['object_pose_w'][:,:3]-r.arrays['object_pose_w'][:,:3],axis=-1)).tolist(),'shaft_error_deg':np.rad2deg(np.arccos(np.clip((axes*refaxes).sum(-1),-1,1))).tolist(),'progress_error_deg':np.abs(actual-expected).tolist(),'joint_state_excess_mrad':(1000*violation.max(-1)).tolist()}
    return {'row':row,'arrays':a,'metadata':m,'evaluation':e,'reference':r,'curves':curves}

def load_job(job,mj,plan,models,protocol,baseline,physical_model,targets,target_manifest,plan_sha):
    model_spec=models[job['model_id']];folder=Path(job['output_dir']);run=metric_run(folder,job,model_spec,plan,protocol);row=run['row'];condition=read(folder/'condition.json');runtime=read(Path(job['wrapper_metadata_dir'])/'runtime.json');capacity=read(folder/'capacity_audit.json')
    checks={'method_seed_job_model_bind':job['trained_method']==model_spec['method'] and job['training_seed']==model_spec['training_seed'],
            'common_fixed20mrad':job['margin_mrad']==20 and job['margin_rad']==.020,
            'plan_job_command_and_output':mj['command']==job['command'] and h.same_path(mj['output_dir'],folder),
            'engineering_terminal0':mj['status']=='completed' and mj['process']['status']=='completed' and mj['process']['exit_code']==0,
            'condition_common_controller':condition['method']=='C2_TARGET_INSET' and condition['margin_rad']==.020 and condition['evaluation_seed']==0,
            'C1_disabled':condition['C1_enabled'] is False and condition['m0'] is None and condition['a_brake'] is None,
            'global_initial_anchor_available':baseline is not None or job['job_id']==plan['jobs'][0]['job_id']}
    for name,value in [('trained_method',model_spec['method']),('model_id',job['model_id']),('training_seed',job['training_seed']),('checkpoint_sha256',model_spec['checkpoint_sha256'])]:checks['real_identity:'+name]=all(x.get(name)==value for x in [condition,runtime,capacity])
    if baseline is not None:
        checks['initial_physical_exact']={k:bool(np.array_equal(run['arrays'][k][0],baseline['arrays'][k][0])) for k in c2.INITIAL_PHYSICAL}
        checks['initial_public_target_exact']=c2.native_equal(run['arrays']['joint_target_rad'][0],baseline['arrays']['joint_target_rad'][0])
    target_arrays,target_checks,target_metrics,target_sha=c2.target_diagnostic(folder,run,job,baseline);run['target_arrays']=target_arrays
    contact_checks,contact_metrics=contact_audit(run,physical_model,targets,target_manifest)
    local_plan={**plan,'checkpoint':{'path':model_spec['checkpoint'],'sha256':model_spec['checkpoint_sha256']}}
    capacity_checks=cap.binding_checks(capacity,job,local_plan,plan_sha,condition,runtime,mj['health'],run['metadata'],row['hashes']['rollout.npz'],target_sha,sha(folder/'capacity_audit.json'))
    health=mj['health'];checks['health_original8_hashes']=len(health['artifact_sha256'])==8 and all(row['hashes'].get(name)==expected for name,expected in health['artifact_sha256'].items())
    row['checks'].update(portfolio=checks,control=target_checks,recorded_contact=contact_checks,capacity=capacity_checks)
    row['capacity']={'record_status':capacity['status'],'overflow_bitmask':capacity['overflow_bitmask'],'flags_by_world':capacity['flags_by_world'],'read_count':capacity['read_count'],'read_at_control_intervals':capacity['read_at_control_intervals'],'sidecar_sha256':sha(folder/'capacity_audit.json'),'runtime_sha256':sha(Path(job['wrapper_metadata_dir'])/'runtime.json')}
    good=c2.all_checks(row['checks']);qualified=good and cap.capacity_class(capacity)=='OK' and capacity['overflow_bitmask']==0
    row.update(status='VALID_CAPACITY_ZERO' if qualified else 'INVALID_OR_UNQUALIFIED',task_success=row['observed_task_success'] if qualified else None,control_intervention=target_metrics,real_contacts=contact_metrics)
    row['metrics']['support512_score']=contact_metrics['support512_score'];row['cost']['child_supervisor_wall_seconds']=mj['process'].get('elapsed_s')
    return run

def hierarchy(rows,models):
    checkpoints={}
    for model_id,spec in models.items():
        group=[r for r in rows if r['model_id']==model_id];ok=len(group)==3 and {r['evaluation_repeat'] for r in group}=={1,2,3} and all(r['status']=='VALID_CAPACITY_ZERO' for r in group)
        passes=sum(r['task_success'] is True for r in group) if ok else None
        checkpoints[model_id]={'model_id':model_id,'method':spec['method'],'training_seed':spec['training_seed'],'checkpoint_sha256':spec['checkpoint_sha256'],'valid_three_processes':ok,'process_outcomes':[{'evaluation_repeat':r['evaluation_repeat'],'task_success':r['task_success'],'job_id':r['job_id']} for r in group],'process_passes':passes,'all3pass':passes==3 if ok else None,'outcome_class':'ALL3_PASS' if passes==3 else 'ALL3_FAIL' if passes==0 else 'MIXED' if ok else 'INCOMPLETE_OR_INVALID','metrics':{k:{**stats([r['metrics'][k] for r in group]),'worst':min(r['metrics'][k] for r in group) if k=='support512_score' else max(r['metrics'][k] for r in group)} for k in KEYS} if ok else None}
    methods={}
    for method in METHODS:
        group=[r for r in checkpoints.values() if r['method']==method];ok=len(group)==3 and {r['training_seed'] for r in group}=={1,2,3} and all(r['valid_three_processes'] for r in group)
        methods[method]={'N_training_seeds':3,'processes_per_checkpoint':3,'valid_three_training_seeds':ok,'seed_process_pass_profile':{str(r['training_seed']):r['process_passes'] for r in group},'training_seeds_all3pass':sum(r['all3pass'] is True for r in group) if ok else None,'training_seeds_mixed':sum(r['outcome_class']=='MIXED' for r in group) if ok else None,'training_seeds_all3fail':sum(r['outcome_class']=='ALL3_FAIL' for r in group) if ok else None,'across_seed_process_means':{k:stats([r['metrics'][k]['mean'] for r in group]) for k in KEYS} if ok else None,'across_seed_worst':{k:stats([r['metrics'][k]['worst'] for r in group]) for k in KEYS} if ok else None}
    paired={}
    for left,right in [('A1L','P_ONLY'),('P_ONLY','A2'),('A1L','A2')]:
        pairs=[]
        for seed in (1,2,3):
            a=next(r for r in checkpoints.values() if r['method']==left and r['training_seed']==seed);b=next(r for r in checkpoints.values() if r['method']==right and r['training_seed']==seed);ok=a['valid_three_processes'] and b['valid_three_processes']
            pairs.append({'training_seed':seed,'available':ok,'difference_of_process_means':{k:b['metrics'][k]['mean']-a['metrics'][k]['mean'] for k in KEYS} if ok else None,'difference_of_process_worst':{k:b['metrics'][k]['worst']-a['metrics'][k]['worst'] for k in KEYS} if ok else None,'process_passes_left':a['process_passes'],'process_passes_right':b['process_passes']})
        paired[right+'_minus_'+left]={'N_training_seed_pairs':3,'pairs':pairs,'descriptive_mean_differences':{k:stats([p['difference_of_process_means'][k] for p in pairs]) for k in KEYS} if all(p['available'] for p in pairs) else None}
    return checkpoints,methods,paired

def figures(runs,checkpoints,out,criterion):
    plt=h.plt;plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'});colors={'A1L':'#1764ab','P_ONLY':'#15845c','A2':'#d55e00'}
    metrics=[('position_error_mm','Position error (mm)',1000*criterion['position_max_m']),('shaft_error_deg','Shaft error (deg)',criterion['shaft_angle_max_deg']),('progress_error_deg','Progress error (deg)',criterion['progress_abs_error_max_deg']),('joint_state_excess_mrad','Actual joint-state excess (mrad)',1000*criterion['joint_limit_margin_rad'])]
    fig,axes=plt.subplots(4,3,figsize=(13,11),sharex=True,sharey='row',constrained_layout=True)
    for col,seed in enumerate((1,2,3)):
        axes[0,col].set_title(f'Training seed{seed}: three processes per method')
        for method in METHODS:
            group=[r for r in runs if r['row']['training_seed']==seed and r['row']['trained_method']==method]
            for row,(key,label,limit) in enumerate(metrics):
                ax=axes[row,col]
                for run in group:ax.plot(run['curves']['time_s'],run['curves'][key],color=colors[method],alpha=.22,linewidth=.8)
                if len(group)==3:ax.plot(group[0]['curves']['time_s'],np.mean([r['curves'][key] for r in group],axis=0),color=colors[method],label=method,linewidth=1.5)
                ax.axhline(limit,color='.2',linestyle=':',linewidth=.8);ax.grid(alpha=.15)
                if col==0:ax.set_ylabel(label)
        axes[-1,col].set_xlabel('Canonical time (s)');axes[0,col].legend(fontsize=8)
    fig.suptitle('Common20mrad C2 controller: all27 fresh-process traces, nine final2000 checkpoints\nThin lines=each process; thick lines=three-process mean. Method N=three training seeds, no frame-N or CI.')
    for ext in ('png','svg'):fig.savefig(out/f'tracking.{ext}',dpi=175)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,4),constrained_layout=True)
    for ax,key,label in zip(axes,['position_mean_mm','shaft_max_deg','joint_state_max_excess_mrad','support512_score'],['Position mean (mm)','Shaft maximum (deg)','Actual-state excess (mrad)','Auxiliary support score512']):
        for seed,offset in zip((1,2,3),(-.12,0,.12)):
            values=[]
            for x,method in enumerate(METHODS):
                cp=next(v for v in checkpoints.values() if v['method']==method and v['training_seed']==seed)
                if not cp['valid_three_processes']:continue
                v=cp['metrics'][key];ax.plot([x+offset,x+offset],[v['min'],v['max']],color=colors[method],alpha=.7);ax.scatter(x+offset,v['mean'],color=colors[method],marker=['o','s','^'][seed-1],s=30);values.append((x+offset,v['mean']))
            if values:ax.plot(*np.array(values).T,color='.5',alpha=.35,linewidth=.7)
        ax.set_xticks(range(3),METHODS);ax.set_title(label);ax.grid(alpha=.18)
    fig.suptitle('One marker per training seed=mean of three process results; bars=min–max of those processes\nConnected seed pairs are descriptive. No CI, best-of-repeat or unconditional mechanism attribution.')
    for ext in ('png','svg'):fig.savefig(out/f'seed_summary.{ext}',dpi=175)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,default=ROOT/'configs/c2-portfolio-v1.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_v1');p.add_argument('--prepare-only',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    protocol=read(ROOT/'analysis_protocol_v1.json');plan=read(args.plan);plan_sha=sha(args.plan);proposal=read(plan['source_pins']['portfolio_proposal']['path']);models={m['model_id']:m for m in plan['models']}
    checks={k:sha(v['path'])==v['sha256'] for k,v in plan['source_pins'].items()}
    checks.update({k:sha(plan[k]['path'])==plan[k]['sha256'] for k in ['reference','reference_metadata','contact_targets','contact_target_manifest']})
    checks.update({m['model_id']+':checkpoint':sha(m['checkpoint'])==m['checkpoint_sha256'] for m in models.values()})
    require(all(checks.values()),'Frozen input/source changed');require(len(models)==9 and len(plan['jobs'])==len({j['job_id'] for j in plan['jobs']})==27,'Need nine models27unique jobs')
    require(plan['models']==proposal['models'],'Model identity/selection differs from proposal')
    require(all(all(job[k]==spec[k] for k in spec) for job,spec in zip(plan['jobs'],proposal['jobs'])) and len(proposal['jobs'])==27,'Proposal27job order changed')
    header={'schema':'C2_020_followup_hierarchical_analysis:1','source_sha256':sha(__file__),'analysis_protocol_sha256':sha(ROOT/'analysis_protocol_v1.json'),'plan_sha256':plan_sha,'proposal_sha256':sha(plan['source_pins']['portfolio_proposal']['path']),'input_source_checks':checks,'gpu_used':False,'training_seed_N_per_method':3,'process_repeats_per_checkpoint':3,'old_cohorts_pooled':False,'C1_enabled':False,'m0':None,'a_brake':None}
    if args.prepare_only:write(args.output/'preparation.json',{**header,'status':'CPU_PREPARATION_PASS','physical_results':None});print('CPU_PREPARATION_PASS');return
    manager_path=Path(plan['output_root'])/'state.json';manager=read(manager_path) if manager_path.exists() else {'status':'PENDING'}
    if manager['status']!='completed_27_jobs':write(args.output/'comparison.json',{**header,'status':'PENDING_TERMINAL_COMPLETION','manager_status':manager['status'],'results':None,'method_summaries':None});print('PENDING_TERMINAL_COMPLETION');return
    require(manager['plan_sha256']==plan_sha and len(manager['jobs'])==27,'Manager plan/coverage mismatch')
    targets=npz(plan['contact_targets']['path']);manifest=read(plan['contact_target_manifest']['path']);require(sha(plan['contact_targets']['path'])==manifest['target_npz_sha256']==protocol['contact_target_npz_sha256'],'Fixed contact target changed')
    import mujoco  # Readonly MJB topology/COM; no MjData, step, forward or Warp.
    physical_model=mujoco.MjModel.from_binary_path(str(Path(models[next(iter(models))]['training_run'])/'compiled/nominal_model.mjb'))
    require(sha(Path(models[next(iter(models))]['training_run'])/'compiled/nominal_model.mjb')==plan['expected_model_sha256'],'Physical model reader pin changed')
    runs=[];invalid=[];baseline=None
    for job in plan['jobs']:
        try:
            mj=next(x for x in manager['jobs'] if x['job_id']==job['job_id']);run=load_job(job,mj,plan,models,protocol,baseline,physical_model,targets,manifest,plan_sha);runs.append(run)
            if job['job_id']==plan['jobs'][0]['job_id']:baseline=run
        except (OSError,KeyError,ValueError,IndexError,StopIteration,TypeError) as error:invalid.append({**{k:job[k] for k in ['job_id','model_id','trained_method','training_seed','evaluation_repeat','block_id','position']},'status':'INVALID_OR_MISSING','task_success':None,'reason':str(error)})
    rows=[r['row'] for r in runs]+invalid;order={j['job_id']:i for i,j in enumerate(plan['jobs'])};rows.sort(key=lambda r:order[r['job_id']]);checkpoints,methods,paired=hierarchy(rows,models);valid=len(runs)==27 and not invalid and all(r['status']=='VALID_CAPACITY_ZERO' for r in rows)
    training_costs={}
    for model_id,spec in models.items():
        path=Path(spec['training_run'])/'summary.json';summary=read(path);training_costs[model_id]={'summary_sha256':sha(path),'recorded_training_process_seconds_including_failed_attempts':summary['total_wall_seconds_including_failed_attempts'],'nominal_budget_from_frozen_proposal':proposal['training_budget'],'scope':'Historical sequential training process costs; different caching/order/engineering context, not equal compute or total project cost.'}
    data={**header,'status':'COMPLETE' if valid else 'INVALID_OR_INCOMPLETE','manager_sha256':sha(manager_path),'results':rows,'checkpoint_summaries':checkpoints,'method_summaries':methods,'paired_seed_descriptions':paired,'historical_training_costs':training_costs,'evaluation_cost':{'whole_queue_wall_seconds':manager.get('whole_queue_wall_seconds'),'sum_evaluator_report_seconds':sum(r['row']['cost']['evaluator_report_wall_seconds'] for r in runs),'sum_child_supervisor_seconds':sum(r['row']['cost']['child_supervisor_wall_seconds'] for r in runs) if all(r['row']['cost']['child_supervisor_wall_seconds'] is not None for r in runs) else None,'scope':'Queue/child/evaluator walls are separate, not GPU kernel busy time.'},'retained_limits':protocol['retained_limits'],'background_only':[{'path':str(ROOT.parent/'supermimic-chord-penalty-only-study/outputs/three_arm_seed123_comparison_v1/REPORT.md'),'scope':'Historical original-controller evaluation, not pooled'}, {'path':str(ROOT.parent/'supermimic-target-inset-study/outputs/analysis_capacity_v2/REPORT.txt'),'scope':'Single-checkpoint controller engineering trial, not pooled'}],'curves':{r['row']['job_id']:r['curves'] for r in runs}}
    write(args.output/'comparison.json',data)
    fields=['job_id','model_id','trained_method','training_seed','evaluation_seed','evaluation_repeat','block_id','position','status','task_success','observed_task_success',*KEYS,'drop_detected']
    with (args.output/'jobs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v for group in [r,r.get('metrics',{})] for k,v in group.items() if k in fields} for r in rows)
    if runs:figures(runs,checkpoints,args.output,runs[0]['reference'].meta['task_acceptance'])
    lines=[f"Status: {data['status']}",'Common20mrad C2 controller, all27 predetermined jobs. Each checkpoint has3fresh processes; each method has3training seeds. No frame-N, CI, best-repeat or pooled historical cohorts.','Checkpoint process outcomes (not new training seeds):']
    for cp in checkpoints.values():lines.append(f"{cp['model_id']}: passes={cp['process_passes']}/3; class={cp['outcome_class']}; mean/worst/ranges={cp['metrics']}")
    lines+=['Method summaries use the three checkpoint/seed summaries:']
    for method,value in methods.items():lines.append(f"{method}: seed process-pass profile={value['seed_process_pass_profile']}; all3pass seeds={value['training_seeds_all3pass']}/3; mixed={value['training_seeds_mixed']}; all3fail={value['training_seeds_all3fail']}.")
    lines+=['Per-job original gates, capacity/control/identity checks and fixed-target512 diagnostics are retained in comparison.json; rows are in jobs.csv.','Raw contact support is recomputed using COM/no-mount/unit-normal checks. It is auxiliary geometry, not force matching or task success.','Historical original-controller and single-checkpoint C2 engineering reports are background links only.','Paired differences compare three-process means/worst by training seed, conditional on common C2 and historical training. No unconditional support-only mechanism attribution.','Training budgets share nominal updates/transitions, not equal compute. All wall-time scopes remain separate.',*protocol['retained_limits']]
    (args.output/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({'status':data['status'],'method_profiles':{k:v['seed_process_pass_profile'] for k,v in methods.items()},'N_training_seeds_per_method':3},indent=2))

if __name__=='__main__':main()
