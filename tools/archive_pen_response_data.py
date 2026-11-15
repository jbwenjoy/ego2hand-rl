"""Compact tracked evidence; detailed packet verification remains a pinned local artifact."""
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot


def main():
    out=ROOT/'provenance/pen_candidate_response_results';analysis=ROOT/'runs/pen_candidate_response_analysis_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence')
    reports=dict(parents=ROOT/'runs/pen_response_parents_v1/report.json',prepared=ROOT/'runs/pen_candidate_responses_v1/report.json',
        readback=ROOT/'runs/pen_candidate_responses_readback_v1/report.json',decoder_parity=ROOT/'runs/pen_parent_decoder_parity_v1/report.json')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();analysis.mkdir()
    manifest=freeze_sources([Path(__file__),ROOT/'tools/archive_pen_source_ccd_evidence.py',ROOT/'tools/review_pen_candidate_data.py',
        ROOT/'tests/test_candidate_response_targets.py',ROOT/'tests/test_response_parent_contract.py',ROOT/'tests/test_response_label_kernel.py'],analysis/'snapshot',ROOT,revision)
    write(analysis/'source_receipt.json',manifest)
    prior=ROOT/'provenance/pen_matched_amplitude_results/source_resolutions.json';aliases={json.dumps(e['original'],sort_keys=True):e for e in json.loads(prior.read_text())['records']}
    receipt_paths=[analysis/'source_receipt.json']
    for folder in {p.parent for p in reports.values()}:receipt_paths+=list(folder.glob('*source_receipt.json'))
    for path in receipt_paths:
        receipt=json.loads(path.read_text());verify_source_snapshot(receipt)
        for e in receipt['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact observed-to-immutable source receipt'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    logs=[ROOT/f'runs/{name}' for name in ['pen_response_parents_tests221.log','pen_candidate_data_tests222.log','pen_candidate_fit_first100_dryrun_v1.json']]
    todo=[file_pin(p) for p in list(reports.values())+receipt_paths+logs+[ROOT/'docs/research-plan-v0.86.md',ROOT/'configs/pen_candidate_fit_first100_v1.toml']]
    allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    p=json.loads(reports['prepared'].read_text());readback=json.loads(reports['readback'].read_text());parents=json.loads(reports['parents'].read_text());parity=json.loads(reports['decoder_parity'].read_text())
    if p['passed']!=p['total'] or readback['passed']!=readback['total'] or parity['passed']!=parity['total']:raise ValueError('Preparation/readback contracts incomplete')
    budget={key:p[key] for key in ['parent_primary_candidate_ticks','parent_actual_ticks','parent_extra_response_ticks','native_audit_ticks']}
    if sum(budget.values())!=95664 or len(p['rows'])!=6156:raise ValueError('Collection budget changed')
    summary=dict(schema='fresh_parent_candidate_data_summary:1',roots={k:file_pin(v) for k,v in reports.items()},
        physics=p['physics'],roles={role:dict(parents=sum(e['role']==role for e in p['parents_detail']),rows=sum(e['rows'] for e in p['parents_detail'] if e['role']==role),valid=sum(e['valid'] for e in p['parents_detail'] if e['role']==role)) for role in ['TRAIN','CHECK']},
        attempted_rows=len(p['rows']),base_valid=sum(e['base_valid'] for e in p['rows']),group_valid=sum(e['group_valid'] for e in p['rows']),
        EFC_failed_flags=dict(Counter(key for e in p['rows'] for key,val in e['EFC_checks'].items() if not val)),
        fidelity_failed_flags=dict(Counter(key for e in p['rows'] for key,val in e['fidelity'].items() if not val)),
        target_failed_flags=dict(Counter(key for e in p['rows'] for key,val in e['checks'].items() if not val)),
        readback=[readback['passed'],readback['total']],readback_errors=readback['max_target_errors'],closure_identity=p['closure_errors'],
        parent_sources=p['parents_detail'],parent_tasks=[dict(mode=c['name'],dt=c['dt'],duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified']) for c in parents['cases']],
        budgets=budget,total_native_integrations=sum(budget.values()),outputs=p['outputs'],scales=p['scales'],head_scales=p['head_scales'],
        scope='Masks/normalization verified, rejected rows retained. Development same-clip parent/phase holdout, not experts/unseen trajectories/ego/gradient or acceleration qualification. No neural updates.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions)
    for path in logs:shutil.copyfile(path,out/path.name)
    # Detailed packet pins are already in the large root manifests. Preserve the
    # complete traversal as a local pinned artifact rather than duplicating it in Git.
    write(analysis/'verified_pins.json',dict(records=list(verified.values()),scope='Nested cohort pins verified; historical inputs exact at boundary'))
    write(out/'ledger.json',dict(schema='fresh_candidate_data_evidence:1',analysis_revision=revision,roots={key:file_pin(value) for key,value in reports.items()},
        summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),previous_resolutions=file_pin(prior),source_receipts=[file_pin(p) for p in receipt_paths],
        verified_pins=len(verified),verified_manifest=file_pin(analysis/'verified_pins.json'),exact_source_aliases=len(aliases),total_native_integrations=95664,neural_updates=0,
        scope='Detailed packet manifests kept local and hash-pinned; compact verified evidence tracked. Failed labels/parent tasks preserved, no training, policy, physical-gradient or ego success claim.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),native_integrations=95664)))


if __name__=='__main__':main()
