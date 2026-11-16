"""All120 registered queries after both independent saved-data reviews."""
import csv
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def distribution(a):
    a=np.asarray(a,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),max=float(a.max())) if len(a) else dict(n=0,min=None,median=None,max=None)


def main():
    out=ROOT/'runs/pen_solve_exit_summary_v2'
    if out.exists():raise ValueError('Fresh complete diagnostic summary required')
    roots={};rows=[];scope_pairs={};inputs=[];checks={}
    for replica in ['A','B']:
        pp=ROOT/f'runs/pen_solve_exit_replay_v2/{replica}/report.json';vp=ROOT/f'runs/pen_solve_exit_saved_review_v2/{replica}/report.json';p=read(pp);v=read(vp);roots[replica]=dict(producer=file_pin(pp),review=file_pin(vp))
        if p['passed']!=p['total'] or v['passed']!=v['total'] or not all(p['checks'].values()) or not all(v['checks'].values()) or v['producer']!=file_pin(pp):raise ValueError('Both complete reviewed replicas required')
        for kind,folder in [('worker',pp.parent),('reader',vp.parent)]:
            manifest=file_pin(folder/'context_layout_adapter.json');m=read(verify_file_pin(manifest));inputs.append(manifest)
            if m['kind']!=kind or m['body_error'] is not None or not m['bindings']['bindings_restored']:raise ValueError('Successful V2 adapter attestation required')
            for k in ['config','entrypoint','entrypoint_snapshot','adapter_source','layout_preflight','layout_tests']:verify_file_pin(m[k])
        checks[replica+'_native480']=p['actual_native_integrations']==p['attempted_native_integrations']==480 and not p['setup_native_unknown'] and p['setup_extra_native']==0
        for slot in p['rows']:
            r=read(verify_file_pin(slot['report']));d=r['declaration'];inputs.append(slot['report']);key=(replica,d['repeat'],d['original_name'])
            if 'differences_between_scopes' in r:scope_pairs[key]=all(x['bytes_equal'] for x in r['differences_between_scopes'].values())
            row=dict(replica=replica,teacher=d['teacher'],repeat=d['repeat'],row_index=d['row_index'],original_name=d['original_name'],scope=d['scope'],original_impulse_valid=r['original_impulse_valid'],new_impulse_valid=r['new_impulse_valid'],all_original_trace_bytes_equal=all(x['bytes_equal'] for x in r['differences_from_original'].values()),native_integrations=8)
            if d['scope']=='PACKET_PLUS_SOLVE_EXIT':
                e=read(verify_file_pin(r['epoch_review']));o=dict(np.load(verify_file_pin(r['solve_exit'])));balance=dict(np.load(verify_file_pin(r['balance'])));k=e['peak_tick'];total=balance['original_total'][k];n2=float(np.dot(total,total));pr=lambda x:float(np.dot(x,total)/n2) if n2 else None
                row.update(peak_tick=k,peak_impulse_Ns=e['peak_original_impulse_Ns'],iterations=int(o['data.solver_niter'][k,0]),improvement=float(o['context.improvement'][k,0]),grad_dot=float(o['context.grad_dot'][k,0]),grad_scale=float(o['context.grad_scale'][k,0]),**{k:v[e['peak_tick']] for k,v in e['terminal_predicates'].items()})
                for term in ['free_mass_chart','Ma_accumulation','actual_scaled_gradient','gradient_force_recovery','smooth_minus_gravity']:
                    row[term+'_norm_Ns']=e['peak_component_norms_Ns'][term];row[term+'_projection']=e['peak_signed_projections'][term]
                for term in ['velocity_update','constraint_contact','gravity_representation']:row[term+'_projection']=pr(balance['original_'+term][k])
            rows.append(row)
        verify_source_snapshot(read(verify_file_pin(v['source_receipt'])),live=True)
    for r in rows:r['between_scope_trace_bytes_equal']=scope_pairs[(r['replica'],r['repeat'],r['original_name'])]
    observed=[r for r in rows if r['scope']=='PACKET_PLUS_SOLVE_EXIT'];failed=[r for r in observed if not r['new_impulse_valid']];conditions=[]
    for ri in range(10):
        group=[r for r in rows if r['row_index']==ri];conditions.append(dict(row_index=ri,name=group[0]['original_name'],original_valid=group[0]['original_impulse_valid'],declared_queries=12,actual_queries=len(group),new_invalid=sum(not r['new_impulse_valid'] for r in group),observed_invalid=sum(not r['new_impulse_valid'] and r['scope']=='PACKET_PLUS_SOLVE_EXIT' for r in group)))
    checks.update(all120=len(rows)==120,all60_observed=len(observed)==60,all60_scope_pairs=len(scope_pairs)==60,all10_groups=len(conditions)==10 and all(c['actual_queries']==12 for c in conditions),exact_native=sum(r['native_integrations'] for r in rows)==960)
    for pin in inputs:verify_file_pin(pin)
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'all_queries.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(14,6));ax=axes[0];y=np.arange(10);ax.barh(y,[c['new_invalid'] for c in conditions],color=['#bd5555' if not c['original_valid'] else '#cb9234' for c in conditions]);ax.set(yticks=y,yticklabels=[c['name'] for c in conditions],xlim=(0,12.5),xlabel='New impulse-invalid queries / all12 per condition',title='All10 registered conditions, both processes/scopes');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',labelsize=8)
    terms=['actual_scaled_gradient','Ma_accumulation','gradient_force_recovery','smooth_minus_gravity','free_mass_chart'];ax=axes[1]
    for i,term in enumerate(terms):ax.scatter([r[term+'_projection'] for r in failed],[i]*len(failed),s=20,alpha=.5)
    ax.set(yticks=np.arange(len(terms)),yticklabels=terms,xlabel='Signed projection on full original impulse residual',title=f'All{len(failed)} failed observed-query peaks');ax.axvline(0,color='.6',lw=1);ax.axvline(1,color='.6',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
    fig.text(.5,.025,'Ten targeted historical conditions x3 numerical repeats x2 fresh processes x2 scopes.\nNo population incidence, unique historical cause, policy success or physical-gate change inferred.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'all_conditions_and_components.png',dpi=160);plt.close(fig)
    masks=Counter((r['original_impulse_valid'],r['new_impulse_valid']) for r in rows);predicates=Counter(tuple(r[k] for k in ['improvement_below_tolerance','gradient_below_tolerance','newton_decrement_below_tolerance','at_iteration_cap']) for r in failed)
    result=dict(schema='pen_solve_exit_all120_summary:2',roots=roots,query_inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),conditions=conditions,queries=120,observed_queries=60,new_impulse_failures=sum(not r['new_impulse_valid'] for r in rows),failed_observed_queries=len(failed),original_new_mask_counts={str(k):v for k,v in masks.items()},scope_pairs_byte_equal=sum(scope_pairs.values()),scope_pairs=60,old_trace_byte_equal=sum(r['all_original_trace_bytes_equal'] for r in rows),failed_peak_predicate_counts={str(k):v for k,v in predicates.items()},failed_peak_iterations=distribution([r['iterations'] for r in failed]),failed_peak_improvement=distribution([r['improvement'] for r in failed]),failed_peak_scaled_gradient_projection=distribution([r['actual_scaled_gradient_projection'] for r in failed]),failed_peak_Ma_projection=distribution([r['Ma_accumulation_projection'] for r in failed]),csv=file_pin(out/'all_queries.csv'),figure=file_pin(out/'all_conditions_and_components.png'),native_integrations_in_producers=960,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,scope='All120 targeted queries and all masks, both independently reviewed processes. Actual nonzero scaled solver gradient explains these observed impulse failures;zero improvement permits solver termination. No proof of the numerical origin of zero improvement or all historical cases;no engine fix/Actor/ego or speed admission.')
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],new_invalid=result['new_impulse_failures'],observed_invalid=len(failed),offline_native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Complete diagnostic summary discrepancy')


if __name__=='__main__':main()
