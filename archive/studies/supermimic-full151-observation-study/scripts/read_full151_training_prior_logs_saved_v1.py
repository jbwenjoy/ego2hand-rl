"""Frozen-source saved JSON prior log reader; no model/objective/backend import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COUNTERS=('policy_update_calls','objective_forward_calls','active_rows','inactive_rows','dt_applications')
PREFIX='prior/geometry_intent_'
COMMON={'prior/bonus_unscaled',PREFIX+'support_match',PREFIX+'penetration_prior',PREFIX+'phase_weight',PREFIX+'bonus_unscaled',PREFIX+'min_sdf_m',PREFIX+'reference_frame',PREFIX+'phase'}

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def checked(value):
    p=Path(value['path']).resolve(strict=True);require(sha(p)==value['sha256'],'Pin drift: '+str(p));return p
def same_pin(left,right):
    return left['sha256']==right['sha256'] and Path(left['path']).samefile(Path(right['path']))
def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def positive_health(h):
    checks=h.get('checks')
    require(h.get('schema')=='full151105_checkpoint_CPU_health:1' and h.get('status')=='PASS_SAVED_ENGINEERING_HEALTH'
            and type(checks) is dict and checks and all(v is True for v in checks.values())
            and type(h.get('passed')) is int and h['passed']==h.get('total')==len(checks),'Actual final health is not positive')
def metric_rows(path):
    rows=[];h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for i,line in enumerate(f):
            h.update(line);row=json.loads(line)
            require(type(row.get('iteration')) is int and row['iteration']==i,'Metrics iteration not contiguous')
            rows.append(row)
    require(len(rows)==2000,'Exactly2000 final metric rows required')
    return rows,h.hexdigest()

def analyze_method(method,run,health_path,contract):
    """Read JSONL once as a stream; keep only narrow active-mean rows."""
    cfg=read(run/'config.json');effective=read(run/'effective_training_config.json');result=read(run/'result.json')
    marker=read(run/'training_completion.json');health=read(health_path);positive_health(health)
    initial=read(run/'prior_metadata.json');meta=read(run/'train/model_final.meta.json')
    require(cfg['schema']=='full151_knownrough_native_training_run:2' and cfg['method']==method and cfg['role']=='primary_matched','Exact primary schema2 method required')
    require(cfg['training']['max_iterations']==effective['max_iterations']==2000 and effective['wm_only_iters']==200
            and effective['horizon']==8 and effective['pi_batch']==128 and effective['num_envs']==128,'Fixed final2000/H8/pi128 budget differs')
    require(health['method']==method and health['role']=='primary_matched' and health['iteration']==2000 and health['final_iteration_scope'] is True
            and Path(health['run_dir']).resolve()==run.resolve(),'Health method/run/final scope differs')
    require(same_pin(health['auditor'],contract['source_pins']['health_auditor']),'Wrong trusted final health source')
    checked(health['config']);require(sha(run/'config.json')==health['config']['sha256'],'Actual root config vs saved copy bytes differ')
    require(result['schema']=='full151_native_training_completion:2' and result['status']=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY'
            and result['method']==method and result['final_iteration']==2000 and result['from_scratch'] is True and result['cleanup_errors']==[],'Complete actual final training result required')
    require(marker['schema']=='full151_native_worker_training_completion:2' and marker['config_sha256']==health['config']['sha256']
            and canonical(marker['result'])==canonical(result),'Worker completion/result mismatch')
    # These are metadata joins to a separately audited file; no checkpoint bytes
    # or tensors are opened by this reader.
    require(same_pin(result['checkpoint'],health['checkpoint']) and meta['sha256']==health['checkpoint']['sha256']
            and Path(meta['checkpoint']).samefile(health['checkpoint']['path']) and meta['iteration']==2000,'Final checkpoint sidecar/health/result mismatch')
    final=result['full151_prior_evidence']
    require(final['schema']=='full151_imagination_prior_checkpoint_evidence:1' and final['method']==method and final['status']=='IMAGINATION_ONLY'
            and final['dt_s']==.02 and final['imagined_soft_required'] is True and final['real_collect_shaped'] is False
            and final['WMfit_shaped'] is False and final['logging_errors']==[],'Imagination-only final prior scope differs')
    require(same_pin(final['spec_pin'],cfg['contact_binding']) and same_pin(final['source_pins']['prior_hook'],contract['source_pins']['prior_hook']),'Prior spec/source differs')
    spec=read(checked(cfg['contact_binding']));require(spec['schema']=='full151_declared_H123_objective_spec:2','Actual target spec:2 required')
    static={k:v for k,v in final.items() if k not in COUNTERS}
    require(canonical({k:v for k,v in initial.items() if k not in COUNTERS})==canonical(static)
            and all(type(initial.get(k)) is int and initial[k]==0 for k in COUNTERS),'Initial prior metadata/static fields differ')
    require(health['checks'].get('prior_target_actual151_unknownhuman_declared_only') is True,'Frozen all151 target phase_weight1 source proof required')
    metrics,metric_sha=metric_rows(run/'metrics.jsonl')
    require(not any(any(k.startswith('prior/') for k in r) for r in metrics[:200]),'WM-only metric prior leakage')
    expected_keys=COMMON | ({PREFIX+'support_coefficient',PREFIX+'penetration_coefficient'} if method=='P_ONLY' else set())
    totals={k:0 for k in COUNTERS};rows=[];log_hash=hashlib.sha256();byte_count=0
    log=run/'full151_prior_evidence.jsonl'
    with log.open('rb') as stream:
        for i,line in enumerate(stream):
            log_hash.update(line);byte_count+=len(line);r=json.loads(line)
            require(r.get('schema')=='full151_imagination_prior_policy_log:1' and r.get('method')==method and r.get('complete') is True,'Incomplete/foreign prior row')
            require(all(type(r.get(k)) is int for k in ('policy_update_call','iteration','objective_forward_calls','active_rows','inactive_rows','dt_applications')),'Native integer row counters required')
            require(r['policy_update_call']==i+1 and r['iteration']==200+i and r['dt_s']==.02,'Raw0based iteration/call/dt differs')
            active,inactive=r['active_rows'],r['inactive_rows']
            require(active>=0 and inactive>=0 and active+inactive==1024 and r['objective_forward_calls']==r['dt_applications']==8,'H8/actual row denominator mismatch')
            totals['policy_update_calls']+=1
            for key in COUNTERS[1:]:totals[key]+=r[key]
            cumulative=r['cumulative']
            require(all(type(cumulative.get(k)) is int and cumulative[k]==totals[k] for k in COUNTERS),'Cumulative prior progression differs')
            require(canonical({k:v for k,v in cumulative.items() if k not in COUNTERS})==canonical(static),'Static cumulative metadata/source/buildrng changed')
            means=r['active_means']
            require(set(means)==expected_keys and all(type(v) in (int,float) and math.isfinite(v) for v in means.values()),'Unexpected/nonfinite active_mean fields')
            require(active>0 or all(v==0 for v in means.values()),'Zeroactive row must have zero logged means')
            require(all(k in metrics[r['iteration']] and canonical(metrics[r['iteration']][k])==canonical(v) for k,v in means.items()),'Prior means vs actual metric row differ')
            P=means[PREFIX+'penetration_prior'];S=means[PREFIX+'support_match'];w=means[PREFIX+'phase_weight'];bonus=means['prior/bonus_unscaled']
            require(P>=0 and means[PREFIX+'bonus_unscaled']==bonus,'P sign/duplicate bonus differs')
            require(w==(1 if active else 0),'Active meanweight not1 despite frozen all151 target')
            penalty=-.1*P;attraction=S if method=='A2' else 0.;predicted=penalty+attraction
            rows.append(dict(method=method,policy_update_call=i+1,raw_iteration=r['iteration'],completed_training_iteration=r['iteration']+1,
                             objective_forward_calls=r['objective_forward_calls'],dt_applications=r['dt_applications'],active_rows=active,inactive_rows=inactive,
                             P_raw_active_mean=P,S_raw_active_mean=S,phase_weight_active_mean=w,
                             weighted_penalty_active_mean=penalty,weighted_support_active_mean=attraction,
                             bonus_logged_active_mean=bonus,dt_scaled_bonus_active_mean=.02*bonus,
                             dt_scaled_penalty_active_mean=.02*penalty,dt_scaled_support_active_mean=.02*attraction,
                             bonus_identity_arithmetic_residual=bonus-predicted,**means))
    require(len(rows)==1800 and totals['objective_forward_calls']==totals['dt_applications']==14400,'Exactly1800 full policy logs required')
    require(all(totals[k]==final[k] for k in COUNTERS),'Final result cumulative counters differ')
    prefix=health['details']['prior_checkpoint_prefix']
    require(prefix['prefix_policy_calls']==1800 and prefix['later_complete_rows_ignored']==0 and prefix['checkpoint_counters']==totals
            and prefix['prefix_SHA256']==log_hash.hexdigest() and prefix['prefix_byte_count']==byte_count,'Final health/log byteprefix provenance differs')
    require(health['details']['metrics_prefix_sha256']==metric_sha,'Final health/metrics byteprefix differs')
    denominator=totals['active_rows']
    def weighted(key):return None if denominator==0 else math.fsum(r['active_rows']*r[key] for r in rows)/denominator
    keys=('P_raw_active_mean','S_raw_active_mean','weighted_penalty_active_mean','weighted_support_active_mean','bonus_logged_active_mean',
          'dt_scaled_bonus_active_mean','dt_scaled_penalty_active_mean','dt_scaled_support_active_mean')
    summary=dict(method=method,policy_updates=len(rows),raw_iteration_range=[rows[0]['raw_iteration'],rows[-1]['raw_iteration']],
                 completed_training_iteration_range=[rows[0]['completed_training_iteration'],rows[-1]['completed_training_iteration']],
                 cumulative=totals,candidate_imagined_rows=1800*1024,active_weighted_denominator=denominator,
                 active_row_weighted_logged_means={k:weighted(k) for k in keys},
                 positive_P_mean_updates=sum(r['P_raw_active_mean']>0 for r in rows),
                 nonzero_S_mean_updates=sum(r['S_raw_active_mean']!=0 for r in rows),
                 nonzero_bonus_mean_updates=sum(r['bonus_logged_active_mean']!=0 for r in rows),
                 negative_penalty_mean_updates=sum(r['weighted_penalty_active_mean']<0 for r in rows),
                 zeroactive_updates=sum(r['active_rows']==0 for r in rows),
                 P_mean_maximum=max(r['P_raw_active_mean'] for r in rows),
                 P_mean_first_positive_completed_iteration=next((r['completed_training_iteration'] for r in rows if r['P_raw_active_mean']>0),None),
                 maximum_abs_bonus_identity_arithmetic_residual=max(abs(r['bonus_identity_arithmetic_residual']) for r in rows),
                 health={'path':str(health_path),'sha256':sha(health_path),'passed':health['passed'],'total':health['total']},
                 health_prefix_log_sha256=log_hash.hexdigest(),metrics_sha256=metric_sha,
                 qualification='Saved active-phase imagined proxy means only; no per-row positive fraction or actual contact/force/Actor-gradient claim')
    files=['config.json','effective_training_config.json','result.json','training_completion.json','prior_metadata.json','metrics.jsonl','full151_prior_evidence.jsonl','train/model_final.meta.json']
    inputs={name:pin(run/name) for name in files};inputs['health']=pin(health_path);inputs['actual_root_training_config']=health['config'];inputs['objective_spec']=cfg['contact_binding']
    return rows,summary,inputs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();source=pin(__file__);contract_pin=pin(args.contract);contract=read(args.contract)
    require(contract['schema']=='full151_trained_prior_saved_log_reader_contract:1' and contract['status']=='FROZEN_BEFORE_NUMERIC_LOG_READ','Reader equations must be frozen before actual log analysis')
    for value in contract['source_pins'].values():checked(value)
    out=args.output.resolve();require(not out.exists(),'Fresh saved-only output required; no rerun/overwrite')
    out.mkdir(parents=True);(out/'executed_reader_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'frozen_reader_contract.json').write_bytes(args.contract.read_bytes())
    write(out/'source_freeze_before_data.json',{'reader':source,'contract':contract_pin,'source_pins':contract['source_pins'],
            'numeric_log_read_started':False,'equations_and_checklist_frozen':True})
    # A fixed independent arithmetic sanity check before either numeric log read.
    require(math.fsum(a*m for a,m in zip((1,3),(2,4)))/4==3.5,'Active-row weighted arithmetic contract')
    allrows=[];summaries={};inputs={}
    try:
        for method in contract['methods']:
            rows,summary,pins=analyze_method(method,Path(contract['run_directories'][method]),Path(contract['health_reports'][method]),contract)
            allrows.extend(rows);summaries[method]=summary;inputs[method]=pins
        fields=sorted(set().union(*(r.keys() for r in allrows)))
        with (out/'all3600_prior_updates.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,2,figsize=(15.4,13.2),sharex=True)
        for col,method in enumerate(contract['methods']):
            rows=[r for r in allrows if r['method']==method];x=[r['completed_training_iteration'] for r in rows]
            arrays=lambda k:[r[k] for r in rows]
            axes[0,col].plot(x,arrays('P_raw_active_mean'),color='#aa3451',lw=.65)
            axes[0,col].set_title(method+' | all1800 policy updates',fontsize=11)
            axes[0,col].set_ylabel('Raw P mean\n(dimensionless squared proxy)')
            axes[1,col].plot(x,arrays('S_raw_active_mean'),color='#286aa0',lw=.65,label='S (diagnostic only)' if method=='P_ONLY' else 'S (support coefficient1)')
            axes[1,col].set_ylabel('Raw support S mean\n(dimensionless)');axes[1,col].legend(fontsize=8,loc='upper right')
            axes[2,col].plot(x,arrays('weighted_penalty_active_mean'),lw=.65,color='#aa3451',label='-.1 P')
            axes[2,col].plot(x,arrays('weighted_support_active_mean'),lw=.65,color='#286aa0',label='weighted attraction')
            axes[2,col].plot(x,arrays('bonus_logged_active_mean'),lw=.8,color='#202020',alpha=.75,label='logged total bonus')
            axes[2,col].set_ylabel('Unscaled weighted terms\n(dimensionless prior rate)');axes[2,col].legend(fontsize=8,loc='best')
            axes[3,col].plot(x,arrays('active_rows'),lw=.65,color='#2b8061',label='phase<150 active')
            axes[3,col].plot(x,arrays('inactive_rows'),lw=.65,color='#777777',label='inactive end rows')
            axes[3,col].set_ylim(-20,1044);axes[3,col].set_ylabel('Imagined rows / update\n(active+inactive=1024)');axes[3,col].legend(fontsize=8,loc='best')
            axes[3,col].set_xlabel('Completed training iteration (raw iteration +1)')
            for row in range(4):axes[row,col].grid(alpha=.18);axes[row,col].set_xlim(201,2000);axes[row,col].ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
        fig.suptitle('Full recorded H8 training priors: separate P_ONLY and A2 histories',fontsize=15,y=.98)
        summary_lines=[]
        for method in contract['methods']:
            s=summaries[method];value=s['active_row_weighted_logged_means']['P_raw_active_mean'];shown='NA' if value is None else '%.6g'%value
            summary_lines.append('%s: positive P-mean updates %d/1800; active rows %d; active-weighted P=%s'%(method,s['positive_P_mean_updates'],s['active_weighted_denominator'],shown))
        caption='\n'.join(summary_lines)+'\n'+('Every raw update retained, no smoothing/filtering. Means use active phase<150 rows, not survival/gamma weights; dt=.02 applied once (CSV includes scaled terms).\n'
             'Positive-mean update count is not a per-row P-positive fraction. P is imagined sampled hand-object geometry, not measured penetration/contact/force.\n'
             'Schema2 full151 only; no old-cohort pooling. Single source/seed0; WMs/normalizers differed before prior. No causal arm comparison, CI or physical qualification.')
        fig.text(.5,.02,caption,ha='center',va='bottom',fontsize=8.2,linespacing=1.45)
        fig.subplots_adjust(left=.08,right=.98,top=.94,bottom=.165,hspace=.21,wspace=.22)
        fig.savefig(out/'full1800_prior_histories.png',dpi=185);fig.savefig(out/'full1800_prior_histories.svg');plt.close(fig)
        require(pin(__file__)==source and pin(args.contract)==contract_pin,'Reader/contract changed during analysis')
        for pins in inputs.values():
            for value in pins.values():checked(value)
        for value in contract['source_pins'].values():checked(value)
        require('torch' not in sys.modules and 'warp' not in sys.modules,'Saved JSON reader crossed model/backend boundary')
        report=dict(schema='full151_trained_prior_saved_log_summary:1',status='COMPLETED_SAVED_ONLY_FULL_LOG_ANALYSIS',
            reader=source,contract=contract_pin,input_pins=inputs,methods=summaries,total_policy_rows=3600,
            source_and_input_pins_unchanged=True,raw_iteration_zero_based=[200,1999],completed_iteration_one_based=[201,2000],
            model_loads=0,objective_builds=0,WM_forwards=0,gradients=0,Env_created=False,physics_steps=0,GPU_used=False,
            weighted_mean_scope='F64 weighted aggregation of rounded logged FP32 active means, not raw per-row tensors or survival/discounted Actor loss',
            nonzero_rule='Exact logged P_mean>0; no activation epsilon or per-row positive-fraction claim',
            causal_method_comparison=False,physical_contact_or_force_qualification=False,
            actual_visual_QA='PENDING',output_pins={v.name:pin(v) for v in out.iterdir() if v.is_file()})
        write(out/'report.json',report)
        print(json.dumps({'status':report['status'],'methods':summaries,'report':pin(out/'report.json'),'PNG':pin(out/'full1800_prior_histories.png')},indent=2,allow_nan=False))
    except BaseException as exc:
        write(out/'failure.json',{'error':repr(exc),'reader':source,'contract':contract_pin,'no_retry':True,'prior_files_unchanged':True})
        raise

if __name__=='__main__':main()
