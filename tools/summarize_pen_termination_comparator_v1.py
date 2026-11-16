"""All240 predeclared query outcomes, matched transitions and solver costs."""
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
from supermimic_research.strict_json import write_json


def read(pin): return json.loads(verify_file_pin(pin).read_text())


def distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)


def main(a):
    cp=a.config.resolve(); cfg=tomllib.loads(cp.read_text()); paths={k:(cp.parent/cfg[k]).resolve() for k in ['registration','output','reader_output']}
    # Explicit corrected reader version; immutable worker config retains V1 paths.
    paths['reader_output']=ROOT/'runs/pen_termination_comparator_review_v2'
    out=ROOT/'runs/pen_termination_comparator_summary_v1'
    if out.exists(): raise ValueError('Fresh complete summary required')
    regpin=file_pin(paths['registration']); reg=read(regpin); inputs=[regpin]; rows=[]; packets={}; ticks={}; roots={}
    for replica in ['A','B']:
        pp=file_pin(paths['output']/replica/'report.json'); rp=file_pin(paths['reader_output']/replica/'report.json')
        p,v=read(pp),read(rp); roots[replica]=dict(producer=pp,review=rp); inputs += [pp,rp]
        if p['passed']!=p['total'] or v['passed']!=v['total'] or not all(p['checks'].values()) or not all(v['checks'].values()) or v['producer']!=pp or p['actual_native_integrations']!=960 or v.get('reader_version')!=2:
            raise ValueError('Both complete independently reviewed replicas required')
        for slot in p['rows']:
            q=read(slot['report']); c=read(q['case']); d=q['declaration']; inputs += [slot['report'],q['case'],c['trace']]
            key=(replica,d['repeat'],d['row_index'],d['scope']); z=dict(np.load(verify_file_pin(c['trace'])))
            packets[(key,d['algorithm'])]=z
            row=dict(replica=replica,repeat=d['repeat'],row_index=d['row_index'],original_name=d['original_name'],algorithm=d['algorithm'],scope=d['scope'],
                historical_impulse_valid=q['original_impulse_valid'],physical_valid=q['new_physical_valid'],impulse_valid=q['new_impulse_valid'],
                peak_original_impulse_Ns=c['physical_audit']['decoder_metrics']['max_linear_impulse_residual_N_s'],terminated=c['terminated'],truncated=c['truncated'],
                position_failure=c['termination_reasons']['position'],shaft_failure=c['termination_reasons']['shaft'],source_reward=c['source_reward'],
                query_case_wall_s=c['wall_s'],source_restore_wall_s=q['source_restore']['wall_s'])
            if d['scope']=='PACKET_PLUS_SOLVE_EXIT':
                e=read(q['epoch_review']); inputs += [q['epoch_review'],q['solve_exit']]
                o=dict(np.load(verify_file_pin(q['solve_exit']))); niter=o['data.solver_niter'][:,0].astype(int)
                peak=e['peak_tick']; row.update(peak_iterations=int(niter[peak]),max_iterations=int(niter.max()),mean_iterations=float(niter.mean()),
                    cap_ticks=int((niter==50).sum()),peak_improvement=float(o['context.improvement'][peak,0]),
                    peak_scaled_gradient_projection=e['peak_signed_projections']['actual_scaled_gradient'])
                for field in ['grad_dot','grad_scale','newton_decrement','alpha']:
                    row['peak_'+field]=float(o['context.'+field][peak,0])
                for field in ['ls_exhausted','search_unchanged']:
                    row['peak_'+field]=bool(o['context.'+field][peak,0])
                for field in ['state_changed_count','quad_changed_count']:
                    row['peak_'+field]=int(o['context.'+field][peak,0])
                ticks.setdefault(d['algorithm'],[]).extend(niter.tolist())
                for k,vv in e['terminal_predicates'].items(): row[k+'_ticks']=sum(vv)
            rows.append(row)
    original='INSTALLED_ORIGINAL_OR'; candidate='DROP_IMPROVEMENT_ONLY'; matched=[]
    bykey={(r['replica'],r['repeat'],r['row_index'],r['scope'],r['algorithm']):r for r in rows}
    for key in sorted({k for k,algorithm in packets}):
        b=bykey[(*key,original)]; c=bykey[(*key,candidate)]; bz=packets[(key,original)]; cz=packets[(key,candidate)]
        matched.append(dict(replica=key[0],repeat=key[1],row_index=key[2],scope=key[3],
            original_valid=b['impulse_valid'],candidate_valid=c['impulse_valid'],original_physical_valid=b['physical_valid'],candidate_physical_valid=c['physical_valid'],
            original_peak_Ns=b['peak_original_impulse_Ns'],candidate_peak_Ns=c['peak_original_impulse_Ns'],
            reward_difference=c['source_reward']-b['source_reward'],
            post_qpos_max_abs=float(abs(cz['post_qpos'].astype(float)-bz['post_qpos'].astype(float)).max()),
            post_qvel_max_abs=float(abs(cz['post_qvel'].astype(float)-bz['post_qvel'].astype(float)).max()),
            held_PD_bytes_equal=bz['post_ctrl'].tobytes()==cz['post_ctrl'].tobytes(),
            task_flags_equal=(b['terminated'],b['truncated'],b['position_failure'],b['shaft_failure'])==(c['terminated'],c['truncated'],c['position_failure'],c['shaft_failure'])))
    groups=[]
    for ri in range(10):
        for algorithm in [original,candidate]:
            group=[r for r in rows if r['row_index']==ri and r['algorithm']==algorithm]; observed=[r for r in group if r['scope']=='PACKET_PLUS_SOLVE_EXIT']
            groups.append(dict(row_index=ri,name=group[0]['original_name'],algorithm=algorithm,queries=len(group),invalid_queries=sum(not r['impulse_valid'] for r in group),
                physically_invalid_queries=sum(not r['physical_valid'] for r in group),peak_impulse_Ns=distribution([r['peak_original_impulse_Ns'] for r in group]),
                cap_ticks=sum(r['cap_ticks'] for r in observed),observed_ticks=len(observed)*8,
                max_iterations=max(r['max_iterations'] for r in observed),task_failure_queries=sum(r['terminated'] for r in group)))
    scopes={}
    for algorithm in [original,candidate]:
        pairs=[]
        for replica in ['A','B']:
            for repeat in range(3):
                for ri in range(10):
                    b=packets[((replica,repeat,ri,'PACKET_BASELINE'),algorithm)]; c=packets[((replica,repeat,ri,'PACKET_PLUS_SOLVE_EXIT'),algorithm)]
                    br=bykey[(replica,repeat,ri,'PACKET_BASELINE',algorithm)]; cr=bykey[(replica,repeat,ri,'PACKET_PLUS_SOLVE_EXIT',algorithm)]
                    pairs.append(dict(bytes_equal=all(b[k].shape==c[k].shape and b[k].dtype==c[k].dtype and b[k].tobytes()==c[k].tobytes() for k in b),
                        masks_equal=br['impulse_valid']==cr['impulse_valid'],peak_abs_difference_Ns=abs(br['peak_original_impulse_Ns']-cr['peak_original_impulse_Ns'])))
        scopes[algorithm]=dict(pairs=len(pairs),byte_equal=sum(p['bytes_equal'] for p in pairs),impulse_masks_equal=sum(p['masks_equal'] for p in pairs),
            peak_abs_difference_Ns=distribution([p['peak_abs_difference_Ns'] for p in pairs]))
    transition=Counter((p['original_valid'],p['candidate_valid']) for p in matched)
    checks=dict(all240=len(rows)==240,all120_matched=len(matched)==120,all20_groups=len(groups)==20 and all(g['queries']==12 for g in groups),
        all120_observed=sum(len(v) for v in ticks.values())==960,all_PD_same=all(p['held_PD_bytes_equal'] for p in matched),all1920=sum(read(roots[r]['producer'])['actual_native_integrations'] for r in roots)==1920)
    for pin in inputs: verify_file_pin(pin)
    out.mkdir(); receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    for name,values in [('all240_queries.csv',rows),('all120_algorithm_pairs.csv',matched)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            cols=list(dict.fromkeys(k for v in values for k in v)); writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader(); writer.writerows(values)
    fig,axes=plt.subplots(1,2,figsize=(15,6)); y=np.arange(10)
    for offset,algorithm,color in [(-.18,original,'#b65c4f'),(.18,candidate,'#4181ac')]:
        g=[x for x in groups if x['algorithm']==algorithm]
        axes[0].barh(y+offset,[x['invalid_queries'] for x in g],height=.34,label=algorithm,color=color)
        axes[1].barh(y+offset,[x['cap_ticks'] for x in g],height=.34,label=algorithm,color=color)
    axes[0].set(yticks=y,yticklabels=[x['name'] for x in groups if x['algorithm']==original],xlim=(0,12.5),xlabel='Impulse-invalid queries / all12 per condition and algorithm')
    axes[1].set(yticks=y,yticklabels=[],xlim=(0,48.5),xlabel='Iteration-cap ticks / all48 observed ticks per condition and algorithm')
    for ax in axes: ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); ax.tick_params(axis='y',labelsize=8)
    axes[1].legend(loc='lower right',fontsize=8); fig.suptitle('All240 targeted queries; original gate retained; cap50 is not convergence')
    fig.text(.5,.015,'Both fresh processes, numerical repeats and observation scopes retained. Local solver comparison, not task/Actor/ego benefit or isolated speed benchmark.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(out/'all_conditions_and_caps.png',dpi=160); plt.close(fig)
    result=dict(schema='pen_termination_comparator_all240_summary:1',roots=roots,inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),
        reader_version=2, retained_reader_v1_failure=file_pin(ROOT/'runs/pen_termination_comparator_review_v1/A/report.json'),
        groups=groups,original_candidate_mask_transitions={str(k):v for k,v in transition.items()},scope_pairs=scopes,
        observed_iterations={k:distribution(v) for k,v in ticks.items()},observed_cap_ticks={k:sum(x==50 for x in v) for k,v in ticks.items()},
        failed_observed_peaks={algorithm:[r for r in rows if r['algorithm']==algorithm and r['scope']=='PACKET_PLUS_SOLVE_EXIT' and not r['impulse_valid']] for algorithm in [original,candidate]},
        all_matched_task_flags_equal=sum(p['task_flags_equal'] for p in matched),max_matched_qpos_abs=max(p['post_qpos_max_abs'] for p in matched),max_matched_qvel_abs=max(p['post_qvel_max_abs'] for p in matched),
        wall_s=dict(native_processes=sum(read(roots[r]['producer'])['wall_s'] for r in roots),independent_readers=sum(read(roots[r]['review'])['wall_s'] for r in roots),
            retained_failed_v1_reader=read(file_pin(ROOT/'runs/pen_termination_comparator_review_v1/A/report.json'))['wall_s']),
        timing_scope='Recorded native-process and reader walls include setup/audit/storage;failed V1 review retained. Registration,CPU tests and summary wall are not included;not complete adaptation cost or isolated solver speed.',
        csv=file_pin(out/'all240_queries.csv'),matched_csv=file_pin(out/'all120_algorithm_pairs.csv'),figure=file_pin(out/'all_conditions_and_caps.png'),
        native_integrations_in_producers=1920,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='All predeclared targeted conditions and matched transitions. No relaxed gate, convergence from cap, full task/Actor/ego benefit or isolated speed claim. New solver remains physically unqualified outside this comparator.')
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],transitions=result['original_candidate_mask_transitions'],caps=result['observed_cap_ticks'],native=0)),flush=True)
    if not all(checks.values()): raise ValueError('Complete comparator summary discrepancy')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); main(parser.parse_args())
