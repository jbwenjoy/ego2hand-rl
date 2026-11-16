"""Declared-denominator and same-source comparisons of fixed native directions."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_path_response import ARMS
from supermimic_research.learning.pen_native_suggestions import GOALS,EPSILONS
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    rp=ROOT/'runs/pen_native_suggestions_review_v1/report.json';review=read(rp);out=ROOT/'runs/pen_native_suggestions_comparison_v1'
    if out.exists() or review['passed']!=review['total'] or not all(review['checks'].values()):raise ValueError('Fresh completed physical review required')
    verify_source_snapshot(read(verify_file_pin(review['source_receipt'])),live=True);rows={r['name']:r for r in review['rows']};all_groups=review['groups'];cells=[];checks={}
    for role in ['TRAIN','VALIDATE']:
        for goal in GOALS:
            k=7+GOALS.index(goal)
            for epsilon in EPSILONS:
                groups=[g for g in all_groups if (g['role'],g['goal'],g['epsilon'])==(role,goal,epsilon)];names=sorted({g['source_name'] for g in groups});by_source={s:{g['model']:g for g in groups if g['source_name']==s} for s in names}
                if any(set(v)!=set(ARMS) for v in by_source.values()):raise ValueError('All models/declared sources required')
                common=[s for s in names if all(by_source[s][m]['complete'] and by_source[s][m]['result']['eligible'] for m in ARMS)];model_records={}
                for model in ARMS:
                    own=[by_source[s][model] for s in names];own_eligible=[g for g in own if g['complete'] and g['result']['eligible']]
                    verified=sum(g['result']['improvement_exceeds_observed_repeat_envelope'] for g in own_eligible);cg=[by_source[s][model] for s in common]
                    model_records[model]=dict(declared_sources=len(names),executed_sources=sum(g['complete'] for g in own),eligible_sources=len(own_eligible),verified_improving_sources=verified,
                        verified_improving_fraction_of_declared=verified/len(names),common_eligible_sources=len(common),common_mean_gain=None if not common else float(np.mean([g['result']['positive_gain'] for g in cg])),
                        common_verified_improving_sources=sum(g['result']['improvement_exceeds_observed_repeat_envelope'] for g in cg))
                paired={}
                for model in ARMS[1:]:
                    records=[]
                    for s in common:
                        candidate,baseline=by_source[s][model],by_source[s][ARMS[0]];cp=np.array([rows[n]['native_components'][k] for n in candidate['cases'][2:4]]);bp=np.array([rows[n]['native_components'][k] for n in baseline['cases'][2:4]])
                        difference=float(cp.mean()-bp.mean());envelope=float(abs(cp-cp.mean()).max()+abs(bp-bp.mean()).max())
                        records.append(dict(source_name=s,gain_difference=difference,positive_repeat_difference_envelope=envelope,better_than_baseline_above_envelope=difference>envelope,worse_than_baseline_above_envelope=difference< -envelope))
                    values=np.array([r['gain_difference'] for r in records]);paired[model]=dict(records=records,sources=len(records),mean_gain_difference=None if not len(records) else float(values.mean()),
                        median_gain_difference=None if not len(records) else float(np.median(values)),better_sources=sum(r['better_than_baseline_above_envelope'] for r in records),worse_sources=sum(r['worse_than_baseline_above_envelope'] for r in records))
                key=role+'_'+goal+'_'+str(epsilon);checks[key+'_all_sources']=len(groups)==3*len(names) and len(names)==(24 if role=='TRAIN' else 12)
                cells.append(dict(role=role,goal=goal,epsilon=epsilon,common_source_names=common,models=model_records,paired_against_RESPONSE_ONLY=paired))
    if not all(checks.values()):raise ValueError('Fixed comparison denominator discrepancy')
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    result=dict(schema='pen_native_suggestions_common_source_comparison:1',physical_review=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),cells=cells,checks=checks,passed=sum(checks.values()),total=len(checks),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='Primary verified-improvement/DECLARED fractions keep missing and physics-invalid proposals unverified. Secondary same-source intersection requires ALL3 models physically eligible. All masks/counts retained;repeats are not IID/confidence bounds. No best-model/amplitude selection or Actor/trajectory/ego claim.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],validation=[c for c in cells if c['role']=='VALIDATE'])),flush=True)


if __name__=='__main__':main()
