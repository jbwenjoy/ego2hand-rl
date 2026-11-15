"""Audit exact shared inputs/budgets and report within-profile replay outcomes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    if a.output.exists():raise ValueError('Fresh matched-profile comparison')
    runs=[json.loads(p.read_text()) for p in [a.baseline,a.tight]]
    reviews=[json.loads(p.read_text()) for p in [a.baseline_review,a.tight_review]]
    if [r['ccd_profile'] for r in runs]!=['BASELINE','TIGHT']:raise ValueError('Fixed original and tighter profiles')
    receipts=[json.loads(verify_file_pin(r['source_receipt']).read_text()) for r in runs]
    for m in receipts:verify_source_snapshot(m)
    source_maps=[{e['observed']['path']:(e['observed']['bytes'],e['observed']['sha256']) for e in m['records']} for m in receipts]
    contracts=dict(original_run_exact=runs[0]['original_run']==runs[1]['original_run'],original_review_exact=runs[0]['original_review']==runs[1]['original_review'],
        active_source_bytes_exact=source_maps[0]==source_maps[1],CCD_cap_fixed=all(r['collision_numerics']['ccd_iterations']==50 for r in runs),
        declared_tolerances=[r['collision_numerics']['ccd_tolerance'] for r in runs]==[1e-6,1e-7],
        budget_candidate_ticks=runs[0]['candidate_physics_ticks']==runs[1]['candidate_physics_ticks']==6840,
        budget_instrumented_ticks=runs[0]['instrumented_physics_ticks']==runs[1]['instrumented_physics_ticks']==120)
    input_checks=[]
    base_records={(e['mode'],e['dt'],e['k'],e['repeat']):e for e in runs[0]['candidate_replays']}
    tight_records={(e['mode'],e['dt'],e['k'],e['repeat']):e for e in runs[1]['candidate_replays']}
    if set(base_records)!=set(tight_records) or len(base_records)!=21:raise ValueError('Exact21 matched searches/profile')
    for key,b in base_records.items():
        t=tight_records[key];bp=np.load(verify_file_pin(b['trace']));tp=np.load(verify_file_pin(t['trace']))
        checks=dict(original_source_pin=b['source']==t['source'],original_target_pin=b['original_prediction']==t['original_prediction'],
            original_selected_action=b['original_selected']==t['original_selected'],all57_executed_PD_exact=np.array_equal(bp['U'],tp['U']),
            all57_clock_exact=np.array_equal(bp['T'],tp['T']),native_budget=b['native_physics_ticks']==t['native_physics_ticks'],
            measured_boundary_timing=all(k in e['search_timing'] for e in [b,t] for k in ['candidate_cuda_ms','submit_and_wait_wall_s','readback_and_host_copy_wall_s','total_search_wall_s']))
        input_checks.append(dict(mode=key[0],dt=key[1],k=key[2],repeat=key[3],checks=checks))
    profiles=[]
    for path,run,review in zip([a.baseline,a.tight],runs,reviews):
        if verify_file_pin(review['run'])!=path.resolve():raise ValueError('Review belongs to a different report')
        relevant=[e for e in review['comparisons'] if e['used_for_profile_qualification']]+review['all57_repeats']
        expected_qualified=review['contract_passed']==review['contract_total'] and review['EFC_counts'][0]==review['EFC_counts'][1] and all(e['passed'] for e in relevant)
        if expected_qualified!=review['within_profile_qualified']:raise ValueError('Qualification not reconstructed from all declared gates')
        timings=[]
        for dt in [.005,.0025]:
            rows=[e for e in run['candidate_replays'] if e['dt']==dt]
            timings.append(dict(dt=dt,searches=len(rows),mean_candidate_cuda_ms=float(np.mean([e['candidate_cuda_ms'] for e in rows])),
                mean_submit_wait_wall_ms=1000*float(np.mean([e['search_timing']['submit_and_wait_wall_s'] for e in rows])),
                mean_readback_copy_wall_ms=1000*float(np.mean([e['search_timing']['readback_and_host_copy_wall_s'] for e in rows]))))
        failed=[e for e in relevant if not e['passed']]
        profiles.append(dict(profile=run['ccd_profile'],qualified=expected_qualified,source_contracts=[run['passed'],run['total']],
            saved_contracts=[review['contract_passed'],review['contract_total']],EFC=review['EFC_counts'],comparison_counts=review['comparison_counts'],
            all57=[review['all57_repeat_passed'],review['all57_repeat_total']],failed_required_comparisons=len(failed),
            failed_gate_counts={k:sum(not e['checks'][k] for e in failed) for k in failed[0]['checks']} if failed else {},
            anchors=review['anchors'],timings=timings,setup_timings=run['setup_timings'],collector_main_wall_s=run['collector_main_wall_s']))
    passed=sum(contracts.values())+sum(sum(e['checks'].values()) for e in input_checks);total=len(contracts)+sum(len(e['checks']) for e in input_checks)
    a.output.mkdir();result=dict(schema='matched_saved_source_CCD_comparison:1',baseline=file_pin(a.baseline),tight=file_pin(a.tight),
        baseline_review=file_pin(a.baseline_review),tight_review=file_pin(a.tight_review),source=file_pin(__file__),
        contracts=contracts,input_checks=input_checks,passed=passed,total=total,profiles=profiles,native_integrations=0,neural_updates=0,
        matched_experiment_native_integrations=2*(6840+120),
        scope='Same saved7 anchors,57 PD candidates,3 repeats/path, active source bytes and integration budget. Within-profile replay evidence, not final equal-budget learned-policy A1/A2 or real-ego/force-parameter calibration. Old-reference numerical response differences excluded from candidate qualification. Timing includes diagnostic overhead and is not training acceleration.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[passed,total],profiles=[dict(profile=p['profile'],qualified=p['qualified'],EFC=p['EFC'],all57=p['all57'],counts=p['comparison_counts']) for p in profiles])))
    if passed!=total:raise ValueError('Matched input/budget contract failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--tight',type=Path,required=True)
    p.add_argument('--baseline-review',type=Path,required=True);p.add_argument('--tight-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
