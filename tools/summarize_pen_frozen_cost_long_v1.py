"""All18 outcomes,matched starts and common valid horizons;no missing-tail fill."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import settings,read
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.strict_json import write_json


def main(a):
    cp=a.config.resolve(); cfg,paths,_,_,_,_=settings(cp); pp=file_pin(paths['output']/'report.json'); rp=file_pin(paths['reader_output']/'report.json'); producer,review=read(verify_file_pin(pp)),read(verify_file_pin(rp))
    if review['producer']!=pp or any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [producer,review]): raise ValueError('Complete reviewed execution required')
    out=ROOT/'runs/pen_frozen_cost_long_summary_v1'
    if out.exists(): raise ValueError('Fresh complete summary required')
    inputs=[pp,rp]; cases=[]; rows=[]; groups=[]; indexed={}; initial_pairs=[]
    gates=tomllib.loads(paths['gates'].read_text())
    for slot in producer['rows']:
        c=read(verify_file_pin(slot['report'])); inputs.append(slot['report']); cases.append(c); indexed[(c['repetition'],c['algorithm'],c['policy'])]=c
        controls=[read(verify_file_pin(p)) for p in c['controls']]
        row={k:c[k] for k in ['name','repetition','algorithm','policy','selected_controls','selected_duration_s','valid_prefix_s','stop_reason','full_task_success']}
        row.update(missing_controls=len(c['missing_controls']),physical_invalid_controls=sum(not p['physical_audit']['valid'] for p in controls),
            peak_linear_impulse_Ns=max(p['physical_audit']['decoder_metrics']['max_linear_impulse_residual_N_s'] for p in controls),
            final_termination_reasons=controls[-1]['termination_reasons'],executed_prefix_strict_metrics=c['summary']['metrics'])
        rows.append(row)
    for algorithm in cfg['algorithms']:
        for label in cfg['policies']:
            g=[r for r in rows if r['algorithm']==algorithm and r['policy']==label]
            groups.append(dict(algorithm=algorithm,policy=label,cases=len(g),durations_s=[r['selected_duration_s'] for r in g],valid_prefix_s=[r['valid_prefix_s'] for r in g],
                stop_reasons=dict(Counter(r['stop_reason'] for r in g)),full_success=sum(r['full_task_success'] for r in g),peak_linear_impulse_Ns=max(r['peak_linear_impulse_Ns'] for r in g)))
    for repeat in range(3):
        for label in cfg['policies']:
            b,c=[indexed[(repeat,algorithm,label)] for algorithm in cfg['algorithms']]; x,y=load_shared(b['initial']),load_shared(c['initial']); xa,ya=x['native']['data']['arrays'],y['native']['data']['arrays']
            core={k:xa[k].shape==ya[k].shape and xa[k].dtype==ya[k].dtype and xa[k].tobytes()==ya[k].tobytes() for k in ['qpos','qvel','time','ctrl','act','mocap_pos','mocap_quat','qacc_warmstart']}
            changed=[k for k in xa if xa[k].shape!=ya[k].shape or xa[k].dtype!=ya[k].dtype or xa[k].tobytes()!=ya[k].tobytes()]
            initial_pairs.append(dict(repetition=repeat,policy=label,core_equal=core,all_Data_arrays_equal=not changed,differing_Data_fields=changed,
                scope='Complete fresh snapshots retained. Derived caches/unused tails can differ;not a claim of whole-Data matched initialization.'))
    common={}
    for label in cfg['policies']:
        g=[c for c in cases if c['policy']==label]; horizon=min(c['valid_prefix_s'] for c in g); metrics={}
        if horizon>0:
            n=round(horizon/.0025)
            for c in g:
                z=dict(np.load(verify_file_pin(c['trace']))); trace={k:v[:n] for k,v in z.items() if k.startswith(('pre_','post_')) or k=='overflow'}; ref={k:z[k] for k in ['time_s','COM_world_m','quaternion_wxyz']}
                metrics[c['name']]=summarize_trace(trace,ref,.0025,False,gates)['metrics']
        common[label]=dict(valid_horizon_s=horizon,metrics=metrics,scope='Shared valid prefix only;zero horizon yields no trajectory comparison,never imputed.')
    checks=dict(all18=len(rows)==18,all6_groups=len(groups)==6 and all(g['cases']==3 for g in groups),all9_start_pairs=len(initial_pairs)==9,
        native_exact=producer['actual_native_integrations']==8*sum(r['selected_controls'] for r in rows)<=52992,all_outcomes=all(r['stop_reason'] in ['PHYSICAL_AUDIT_STOP','TASK_FAILURE','REFERENCE_COMPLETION'] for r in rows),
        all_missing_tails=sum(r['missing_controls']+r['selected_controls'] for r in rows)==18*368)
    out.mkdir(); receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    with (out/'all18.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig,ax=plt.subplots(figsize=(12,6)); colors={cfg['algorithms'][0]:'#b65c4f',cfg['algorithms'][1]:'#4181ac'}
    labels=[]
    for i,g in enumerate(groups):
        labels.append(('Original OR' if g['algorithm']==cfg['algorithms'][0] else 'Guarded stable cost OR')+' / '+g['policy'])
        for repeat,(duration,prefix) in enumerate(zip(g['durations_s'],g['valid_prefix_s'])):
            yy=i+(repeat-1)*.15; color=colors[g['algorithm']]; ax.plot([0,prefix],[yy,yy],color=color,alpha=.3,lw=3); ax.scatter(prefix,yy,color=color,s=24)
            if duration!=prefix: ax.scatter(duration,yy,facecolors='none',edgecolors=color,s=35)
    ax.axvline(7.36,color='#555555',ls='--',lw=1); ax.set(yticks=range(6),yticklabels=labels,xlim=(0,7.6),xlabel='Physical time (s);filled=valid prefix,open=first invalid executed end')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); ax.set_title('All18 frozen-Actor cases:three numerical repeats per group;full reference7.36s')
    fig.text(.5,.015,'Missing future is censored. Durations are not full-task success or ego/generalization evidence. No training updates.',ha='center',fontsize=9); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out/'all18_durations.png',dpi=160); plt.close(fig)
    for pin in inputs: verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='pen_frozen_cost_long_all18_summary:1',producer=pp,review=rp,inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),rows=rows,groups=groups,initial_pairs=initial_pairs,common_valid_horizons=common,
        checks=checks,passed=sum(checks.values()),total=len(checks),csv=file_pin(out/'all18.csv'),figure=file_pin(out/'all18_durations.png'),native_integrations_in_producer=producer['actual_native_integrations'],actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='All frozen actors/repeats retained,including physics censoring and actual task flags. Numeric backend evaluation of old weights,not new training/world-model/ego/W32 or acceleration admission.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],groups=groups)),flush=True)
    if not all(checks.values()): raise ValueError('All18 summary integrity mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); main(p.parse_args())
