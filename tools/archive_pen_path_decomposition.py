"""Close frozen derivative attribution and the separate live-input regression."""
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/'runs/pen_path_decomposition_archive_v1';archive=ROOT/'provenance/pen_path_decomposition_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh decomposition archive required')
    roots=dict(analysis=ROOT/'runs/pen_path_decomposition_v1/report.json',registration=ROOT/'runs/pen_path_decomposition_registration_v1/registration.json',
        review=ROOT/'runs/pen_path_decomposition_review_v1/report.json',inputs=ROOT/'runs/pen_live_decision_inputs_v1/report.json')
    r={k:read(p) for k,p in roots.items()};analysis,review,inputs=[r[k] for k in ['analysis','review','inputs']];arrays=dict(np.load(verify_file_pin(analysis['arrays'])))
    checks=dict(analysis73=analysis['passed']==analysis['total']==73 and all(analysis['checks'].values()),review73=review['passed']==review['total']==73 and all(review['checks'].values()) and review['analysis']==file_pin(roots['analysis']),
        inputs4=inputs['passed']==inputs['total']==4 and all(inputs['checks'].values()),all36_source_contexts=len(inputs['rows'])==36 and all(all(x['checks'].values()) for x in inputs['rows']),
        all30_actual_input_bytes=sum(x['observed_original_BASE'] for x in inputs['rows'])==30,
        zero_native_optimizer_normalizer=all(x['actual_native_integrations']==x['optimizer_steps']==x['normalizer_fits']==0 for x in [analysis,review,inputs]))
    attribution={};FD_summary={}
    for role,c in analysis['cohorts'].items():
        attribution[role]={};FD_summary[role]={}
        for name,m in c['models'].items():
            attribution[role][name]=m['attribution_summary'];FD_summary[role][name]={mode:obj['aggregate'] for mode,obj in m['modes'].items()}
            for k,summary in enumerate(m['attribution_summary']):
                # The independent review already recomputed every source's
                # scalar dot/norm. Aggregate those verified sourcewise values;
                # vectorized reductions can change acos near1 by~1e-6 degrees.
                rows=[a[k] for a in m['source_attribution']];ratios=[a['path_to_full_norm'] for a in rows if a['path_to_full_norm'] is not None];angles=[a['full_observable_angle_deg'] for a in rows if a['full_observable_angle_deg'] is not None]
                expected=dict(sources=len(rows),defined_norm_ratios=len(ratios),median_path_to_full_norm=float(np.median(ratios)) if len(ratios) else None,max_path_to_full_norm=float(max(ratios)) if len(ratios) else None,
                    median_full_observable_angle_deg=float(np.median(angles)) if len(angles) else None,max_full_observable_angle_deg=float(max(angles)) if len(angles) else None)
                checks[role+'_'+name+'_'+str(k)+'_norm_angle_summary']=all(summary[n]==v or (v is not None and np.isclose(summary[n],v,rtol=1e-10,atol=1e-10)) for n,v in expected.items())
    tests=[ROOT/'runs/pen_path_decomposition_tests546_v1.log',ROOT/'runs/pen_live_decision_tests549_v1.log']
    for count,p in zip([546,549],tests):checks['tests'+str(count)]=f'Ran {count} tests' in p.read_text() and '\nOK\n' in p.read_text()
    design=ROOT/'configs/pen_native_screened_rollout_design_v1.toml';cfg=tomllib.loads(design.read_text())
    checks['unlaunched_design_budget']=cfg['status']=='DESIGN_ONLY_NOT_REGISTERED_OR_LAUNCHED' and cfg['cases_per_repetition']==2*(3*2+3)==18 and cfg['native_budget_per_repetition_max']==16*368*2*8+2*368*8==100096 and cfg['native_budget_all_repetitions_max']==3*100096==300288
    receipts=[obj['source_receipt'] for obj in r.values()]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Decomposition archive discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-path-decomposition-results-2026-11-16.md','research-plan-v1.42.md','pen-native-screened-rollout-design-v1.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[design,Path(__file__),ROOT/'tools/review_pen_path_decomposition.py',ROOT/'tools/qualify_pen_live_decision_inputs.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],analysis_checks=[73,73],independent_review_checks=[73,73],independent_arithmetic_checks=review['arithmetic_checks'],
        attribution=attribution,native_FD_comparisons=FD_summary,analysis_wall_s=analysis['wall_s'],differences=analysis['differences'],
        input_regression_checks=[4,4],source_contexts=36,existing_input_byte_comparisons=30,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        rollout_design_registered=False,rollout_design_launched=False,
        scope='Within-frozen-model gradient attribution and historical owned-input equivalence only. No native observable-only or new multi-control execution evidence. Full original scope/risks remain active.')
    write_json(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_native_suggestions_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    diagnostic=ROOT/'runs/pen_path_decomposition_archive_angle_diagnostic_v1/report.json'
    todo=[file_pin(p) for p in list(roots.values())+docs+tests+[design,out/'independent_review.json',out/'source_receipt.json',prior_path,diagnostic]]+receipts;allowed={p.parent for p in roots.values()}|{out,diagnostic.parent}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write_json(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write_json(archive/'summary.json',summary)
    write_json(archive/'ledger.json',dict(schema='pen_path_decomposition_results_ledger:1',prior_evidence=file_pin(prior_path),roots={k:file_pin(p) for k,p in roots.items()},review=file_pin(out/'independent_review.json'),summary=file_pin(archive/'summary.json'),
        documents=[file_pin(p) for p in docs],rollout_design=file_pin(design),source_receipts=receipts+[file_pin(out/'source_receipt.json')],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),native=0,optimizer=0)),flush=True)


if __name__=='__main__':main()
