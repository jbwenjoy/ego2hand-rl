"""Seal negative GPU admission evidence without converting it into qualification."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_stable_friction_qualification_checkpoint_v1'; dest=ROOT/'provenance/pen_stable_friction_qualification_v1'
    if out.exists() or dest.exists(): raise ValueError('Fresh negative qualification checkpoint required')
    prior=file_pin(ROOT/'provenance/pen_paired_ray_results_v1/ledger.json'); parent=json.loads(verify_file_pin(prior).read_text())
    manifests=parent['inherited_verified_manifests']+[parent['additional_verified_manifest']]
    identity=lambda p:json.dumps(p,sort_keys=True); known=set()
    for p in manifests: known.update(identity(q) for q in pins(json.loads(verify_file_pin(p).read_text())))
    names={f'failed_qualification_v{i}':f'pen_stable_friction_backend_qualification_v{i}/report.json' for i in [1,2,3]}
    names.update(layout_diagnostic='pen_stable_friction_compile_layout_v1/report.json',saved_review='pen_stable_friction_arithmetic_saved_review_v1/report.json')
    roots={k:file_pin(ROOT/'runs'/v) for k,v in names.items()}; reports={k:json.loads(verify_file_pin(p).read_text()) for k,p in roots.items()}
    checks={f'v{i}_retained_failed_gate':(reports[f'failed_qualification_v{i}']['passed'],reports[f'failed_qualification_v{i}']['total'])==expected for i,expected in [(1,(9,14)),(2,(12,13)),(3,(12,13))]}
    for i in [1,2,3]:
        q=reports[f'failed_qualification_v{i}']
        checks[f'v{i}_all_inputs_and_stop_retained']=q['rows']==139160 and q['fallback']==2016 and q['checks']['input_bytes_unchanged'] and q['checks']['unchanged_original_stopping'] and q['factory_binding']['bindings_restored']
    review=reports['saved_review']; checks['independent_negative_review']=review['passed']==review['total']==24 and all(review['checks'].values())
    checks['separate_kernels_keep_failure']=reports['layout_diagnostic']['kernel_launches']==4 and all(not v['fallback_bytes_equal'] for v in reports['layout_diagnostic']['comparisons'].values())
    checks['zero_native_optimizer_normalizer']=all(d.get('actual_native_integrations',0)==d.get('optimizer_steps',0)==d.get('normalizer_fits',0)==0 for d in reports.values())
    logs=[ROOT/'runs'/f'pen_stable_friction_tests{i}_v1.log' for i in [627,628,629]]
    logs += [ROOT/'runs'/f'pen_stable_friction_backend_qualification_v{i}.log' for i in [1,2,3]]
    logs += [ROOT/'runs'/name for name in ['pen_stable_friction_compile_layout_v1.log','pen_stable_friction_arithmetic_saved_review_v1.log']]
    checks['full629_tests']='Ran 629 tests in 16.253s' in logs[2].read_text(encoding='utf-8-sig') and '\nOK\n' in logs[2].read_text(encoding='utf-8-sig')
    if not all(checks.values()): raise ValueError('Negative evidence checkpoint mismatch')
    documents=[ROOT/'docs'/n for n in ['pen-stable-friction-qualification-results-2026-11-17.md','research-plan-v1.55.md']]
    tools=sorted((ROOT/'tools').glob('*pen_stable_friction*.py'))
    tests=sorted((ROOT/'tests').glob('test_pen_stable_friction*.py'))
    out.mkdir(); dest.mkdir()
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+tools+tests+documents,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    todo=list(roots.values())+[file_pin(out/'source_receipt.json')]+[file_pin(p) for p in logs]; verified={}; reached=set()
    while todo:
        p=todo.pop(); key=identity(p); path=Path(p['path']).resolve()
        if path==ROOT/'runs/pen_native_screened_rollout_v1/work.json': raise ValueError('Mutable historical progress cannot enter evidence checkpoint')
        if key in known: reached.add(key); continue
        if key in verified: continue
        verify_file_pin(p); verified[key]=p
        if path.suffix=='.json' and path.is_relative_to(ROOT/'runs'): todo.extend(pins(json.loads(path.read_text())))
    for p in list(roots.values())+manifests+[prior]+[file_pin(p) for p in logs]: verify_file_pin(p)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),inherited_reached=len(reached),scope='New pins cold-verified;historical byte sets inherited via exact manifest identities.'))
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    write_json(dest/'ledger.json',dict(schema='pen_stable_friction_failed_qualification_checkpoint:1',prior_evidence=prior,inherited_verified_manifests=manifests,
        roots=roots,documents=[file_pin(p) for p in documents],retained_logs=[file_pin(p) for p in logs],source_receipt=file_pin(out/'source_receipt.json'),
        review=file_pin(out/'review.json'),additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),additional_verified_pins=len(verified),inherited_pins_reached=len(reached),
        fixed_input_GPU_kernel_launches=7,candidate_factory_CUDA_compiles=3,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='All three GPU admission failures and separate-kernel negative evidence retained. Green checkpoint/review certifies evidence integrity only;no cost backend/native/Actor/task/ego/speed admission. Full research goal remains active and incomplete.'))
    print(json.dumps(dict(ledger=file_pin(dest/'ledger.json'),checks=[sum(checks.values()),len(checks)],new_pins=len(verified),inherited_reached=len(reached))),flush=True)


if __name__=='__main__': main()
