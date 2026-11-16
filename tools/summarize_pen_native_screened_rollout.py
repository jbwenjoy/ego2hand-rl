"""Immutable18/36/54-case descriptive summaries with all declared denominators."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.pen_screened_summary import case_metrics,aggregate
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main(args):
    vp=args.review_index.resolve();review=read(vp);reg=read(verify_file_pin(review['registration']));available=len(review['reviewed_cases']);count=(available//18)*18;out=ROOT/'runs'/f'pen_native_screened_summary_{count}_v1'
    if count not in [18,36,54] or out.exists():raise ValueError('Fresh complete-repetition18/36/54 summary required')
    if count==54 and (review.get('passed')!=review.get('total') or 'checks' not in review or not all(review['checks'].values())):raise ValueError('Final54 review with completed producer required')
    rows=[];inputs=[];null_modes=[]
    for slot,declared in zip(review['reviewed_cases'][:count],reg['cases'][:count]):
        if slot['name']!=declared['name']:raise ValueError('Fixed declared order changed')
        r=read(verify_file_pin(slot['report']));case=read(verify_file_pin(r['producer_case']));verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True);rows.append(case_metrics(case,r));inputs.append(dict(review=slot['report'],case=r['producer_case']));capsule={p['path']:p for p in r['input_file_capsule']['records_before_read']}
        if declared['model']=='RESPONSE_ONLY':
            errors=[];finite=True;nonfinite=[]
            for node in case['nodes']:
                path=Path(r['producer_case']['path']).parent/('P'+str(node['phase']).zfill(3))/'gradient.npz';g=dict(np.load(verify_file_pin(capsule[str(path.resolve())])));a,b=g['RAW'][:,0],g['RAW'][:,2];finite=finite and bool(np.isfinite(a).all() and np.isfinite(b).all())
                if np.isfinite(a).all() and np.isfinite(b).all():errors.append(float(abs(a-b).max()))
                else:nonfinite.append(node['phase'])
            null_modes.append(dict(name=declared['name'],all_finite=finite,declared_source_gradients=len(case['nodes']),finite_source_gradients=len(errors),nonfinite_source_phases=nonfinite,max_abs_OBS_FULL_RAW_on_finite=max(errors) if errors else None,scope='Same-source baseline-head null comparison. Distinct executed trajectories may differ numerically;not a ranking metric.'))
    summary=aggregate(reg['cases'],rows);checks=dict(all_declared54=len(reg['cases'])==54,complete_repetitions=count==len(rows) and count%18==0,all18_groups=len(summary['groups'])==18,exact_queries=summary['native_integrations']==8*summary['native_queries'],all_reviewed_case_counts=all(r['native_queries']>=r['selected_controls'] for r in rows),all_unreviewed_denominators=len(summary['unreviewed_case_names'])+count==54,no_optimizer_native=True)
    out.mkdir();source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)];receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    fields=['name','teacher','controller','model','score_mode','repetition','case_status','declared_controls','selected_controls','executed_selected_duration_s','valid_selected_prefix_s','stop_reason','full_task_success','native_queries','native_integrations','accepted_proposals','accepted_duplicate_BASE','invalid_BASE_queries','invalid_ALT_queries','case_wall_s','actor_wall_s','proposal_wall_s','restore_wall_s','capture_wall_s'];observed={r['name']:r for r in rows}
    with (out/'cases.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for d in reg['cases']:w.writerow(dict(observed.get(d['name'],d),declared_controls=368,case_status='REVIEWED' if d['name'] in observed else 'NOT_INCLUDED_IN_THIS_CHECKPOINT'))
    fig,axes=plt.subplots(1,2,figsize=(14,6.8));styles={'BASE_PHYSICAL_INVALID':('#bd3d38','x'),'TASK_FAILURE':('#d28b22','o'),'REFERENCE_COMPLETION':('#398959','*')}
    for ax,teacher in zip(axes,['RESIDUAL','SUPPORT']):
        groups=[g for g in summary['groups'] if g['teacher']==teacher];labels=[]
        for i,g in enumerate(groups):
            label=g['controller'] if g['model'] is None else g['model']+' / '+('OBS' if g['score_mode']=='OBSERVABLE' else 'FULL');labels.append(label)
            matching=[m for m in rows if m['teacher']==teacher and m['controller']==g['controller'] and m['model']==g['model'] and m['score_mode']==g['score_mode']]
            for m in matching:
                color,marker=styles[m['stop_reason']];ax.scatter(m['executed_selected_duration_s'],i+(m['repetition']-1)*.12,c=color,marker=marker,s=40,zorder=3)
        ax.axvline(7.36,color='.4',ls='--',lw=1);ax.set(xlim=(0,7.6),yticks=np.arange(9),yticklabels=labels,xlabel='Executed selected time including stopping control (s)',title=teacher+f' ({count//18}/3 numerical repeats reviewed)');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=9);ax.grid(axis='x',alpha=.2)
    handles=[Line2D([0],[0],color=c,marker=m,ls='',label=n) for n,(c,m) in [('Physical audit stop',styles['BASE_PHYSICAL_INVALID']),('Task failure',styles['TASK_FAILURE']),('Full reference complete',styles['REFERENCE_COMPLETION'])]];fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),ncol=3);fig.suptitle(f'All declared controllers: {count}/54 reviewed; full7.36s successes {summary["full_task_successes"]}',fontsize=14);fig.text(.5,.012,'Frozen models / one controlled reference. Physical and task stopping differ. No independent-seed or causal retention inference.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(out/'all_outcomes.png',dpi=170);plt.close(fig)
    verify_source_snapshot(receipt,live=True);result=dict(schema='pen_native_screened_descriptive_summary:1',review=file_pin(vp),registration=review['registration'],case_inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),cases=rows,summary=summary,baseline_same_source_null_modes=null_modes,checks=checks,passed=sum(checks.values()),total=len(checks),csv=file_pin(out/'cases.csv'),figure=file_pin(out/'all_outcomes.png'),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,complete_study=count==54)
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],reviewed=count,successes=summary['full_task_successes'],native_in_closed_cases=summary['native_integrations'])),flush=True)
    if not all(checks.values()):raise ValueError('Descriptive summary discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-index',type=Path,required=True);main(p.parse_args())
