"""Read-only failure taxonomy, retaining every reviewed native query."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0,min=None,median=None,p95=None,max=None)


def main(a):
    ip=a.review_index.resolve();index=read(ip);count=len(index['reviewed_cases']);reg=read(verify_file_pin(index['registration']));out=ROOT/'runs'/f'pen_screened_failure_taxonomy_{count}_v1'
    if count not in [18,36,54] or out.exists():raise ValueError('Fresh complete-repetition taxonomy required')
    if count==54 and (index.get('passed')!=index.get('total') or not index.get('checks') or not all(index['checks'].values()) or not index['complete']):raise ValueError('Final54 independent review required')
    rows=[];inputs=[];checks={};stops=[];gate_counts=Counter();invalid_roles=Counter()
    for slot,decl in zip(index['reviewed_cases'],reg['cases'][:count]):
        r=read(verify_file_pin(slot['report']));case=read(verify_file_pin(r['producer_case']));inputs.extend([slot['report'],r['producer_case']])
        if slot['name']!=decl['name'] or r['passed']!=r['total'] or not all(r['checks'].values()) or not r['input_file_capsule']['all_bytes_verified_after']:raise ValueError('Exact complete V3 record ownership required')
        base_bad=[]
        for p in r['packets']:
            f=p['physical'];m=f['decoder_metrics'];res=m['max_linear_impulse_residual_N_s'];reasons=[k for k,v in f['checks'].items() if not v]
            if not f['geometry_legal']:reasons.append('GEOMETRY_GATE')
            if not f['online_quality']:reasons.append('ONLINE_QUALITY')
            if not f['valid'] and not reasons:reasons=['PACKET_QUALITY_OTHER']
            role='BASE' if p['name'].endswith('_BASE') else 'ALT'
            valid=bool(f['online_quality'] and f['packet_quality'] and f['geometry_legal'] and all(f['checks'].values()))
            checks[p['name']+'_physical_mask']=f['valid']==valid
            checks[p['name']+'_original_linear_gate']=np.isfinite(res) and (res<=5e-7)==f['checks']['EFC_linear_impulse']
            if not f['valid']:
                gate_counts.update(reasons);invalid_roles[role]+=1
                if role=='BASE':base_bad.append(p['name'])
            rows.append(dict(case=case['name'],query=p['name'],teacher=decl['teacher'],controller=decl['controller'],model=decl['model'],score=decl['score_mode'],repetition=decl['repetition'],role=role,physical_valid=f['valid'],terminated=p['terminated'],truncated=p['truncated'],failure_reasons=';'.join(reasons),max_linear_impulse_residual_Ns=res,multiple_of_original_linear_gate=res/5e-7,object_penetration_m=f['geometry'][0],self_penetration_m=f['geometry'][1],joint_excursion_rad=f['geometry'][2]))
        stop=case['stop_reason'];checks[case['name']+'_BASE_stopping']=((len(base_bad)==1 and base_bad[0]==case['name']+'_P'+str(case['nodes'][-1]['phase']).zfill(3)+'_BASE') if stop=='BASE_PHYSICAL_INVALID' else len(base_bad)==0)
        stops.append(dict(case=case['name'],reason=stop,invalid_BASE=base_bad))
    checks['all_queries']=len(rows)==sum(x['summary']['native_queries'] for x in index['reviewed_cases'])
    checks['all_case_prefix']=len(stops)==count==len(index['reviewed_cases'])
    for pin in inputs+[file_pin(ip)]:verify_file_pin(pin)
    invalid=[r for r in rows if not r['physical_valid']];censored=[r for r in invalid if r['role']=='BASE' and not r['terminated'] and not r['truncated']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    with (out/'all_queries.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(schema='pen_screened_stopping_taxonomy:1',index=file_pin(ip),case_inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),csv=file_pin(out/'all_queries.csv'),checks=checks,passed=sum(checks.values()),total=len(checks),reviewed_cases=count,declared_cases=54,queries=len(rows),valid_queries=len(rows)-len(invalid),invalid_queries=len(invalid),invalid_roles=dict(invalid_roles),failure_gates_nonexclusive=dict(gate_counts),stop_counts=dict(Counter(s['reason'] for s in stops)),invalid_query_terminal_flags=dict(Counter(str((r['terminated'],r['truncated'])) for r in invalid)),physically_stopped_but_no_task_flag=len(censored),all_query_linear_impulse_Ns=stats([r['max_linear_impulse_residual_Ns'] for r in rows]),invalid_linear_impulse_Ns=stats([r['max_linear_impulse_residual_Ns'] for r in invalid]),original_linear_gate_Ns=5e-7,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,scope='Reclassifies already independently decoded packets;does not re-run native physics or alter masks. Physical audit stopping without task flags censors future outcomes;not proof of object fall or eventual success. No root-cause,relaxed tolerance,Actor/ego or ranking inference.')
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],invalid=len(invalid),censored=len(censored),native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Stopping taxonomy discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-index',type=Path,required=True);main(p.parse_args())
