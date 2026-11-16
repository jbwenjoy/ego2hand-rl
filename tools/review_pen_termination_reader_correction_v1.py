"""Verify V2 changed only the120 JSON-audit outcomes on identical saved inputs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_termination_reader_correction_review_v1'
    if out.exists(): raise ValueError('Fresh correction review required')
    pp=[file_pin(ROOT/f'runs/pen_termination_comparator_review_v{v}/A/report.json') for v in [1,2]]
    old,new=[json.loads(verify_file_pin(p).read_text()) for p in pp]
    changed=[k for k in old['checks'] if old['checks'][k]!=new['checks'][k]]
    checks=dict(same_producer=old['producer']==new['producer'],same_check_inventory=old['checks'].keys()==new['checks'].keys(),
        same_raw_input_capsule=old['input_file_capsule']==new['input_file_capsule'],
        same_all_arithmetic_elements=old['recomputed_array_elements']==new['recomputed_array_elements']==70560,
        exactly120_json_audits=len(changed)==120 and all(k.endswith('_metadata_audit') and not old['checks'][k] and new['checks'][k] for k in changed),
        old_expected_failure=old['passed']==3373 and old['total']==3493,
        new_all_pass=new['passed']==new['total']==3493 and all(new['checks'].values()),
        no_native_or_updates=all(d[k]==0 for d in [old,new] for k in ['actual_native_integrations','optimizer_steps','normalizer_fits']))
    if not all(checks.values()): raise ValueError('Reader correction changed unrelated evidence')
    out.mkdir(); receipt=freeze_sources([Path(__file__)]+sorted((ROOT/'src').rglob('*.py')),out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    for pin in pp: verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='pen_termination_reader_correction_review:1',reviews=pp,
        source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),changed_checks=changed,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='Exact failed/corrected saved-reader A input capsules and check outcomes. Only120 JSON-audit comparisons changed;no native replay,typed-state or numerical/physical gate relaxation.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],native=0)))


if __name__=='__main__': main()
