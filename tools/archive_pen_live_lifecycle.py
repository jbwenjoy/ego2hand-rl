"""Close transport/lifecycle prerequisites without claiming full54 execution."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/'runs/pen_live_lifecycle_archive_v1';archive=ROOT/'provenance/pen_live_lifecycle_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh lifecycle archive required')
    roots={k:ROOT/'runs'/v/'report.json' for k,v in dict(plain='pen_snapshot_transport_v1',shared='pen_shared_snapshot_transport_v1',node='pen_live_node_qualification_v1',node_review='pen_live_node_review_v1',lifecycle='pen_terminal_lifecycle_v1',lifecycle_review='pen_terminal_lifecycle_review_v1',proposal='pen_screened_proposal_qa_v1').items()};r={k:read(p) for k,p in roots.items()};counts=dict(plain=4,shared=5,node=12,node_review=7,lifecycle=12,lifecycle_review=7,proposal=5)
    checks={k+'_complete':v['passed']==v['total']==counts[k] and all(v['checks'].values()) for k,v in r.items()}
    checks.update(node_review_parent=r['node_review']['run']==file_pin(roots['node']),lifecycle_review_parent=r['lifecycle_review']['run']==file_pin(roots['lifecycle']),new_native_exact=r['node']['actual_native_integrations']==r['node']['attempted_native_integrations']==128 and r['lifecycle']['actual_native_integrations']==r['lifecycle']['attempted_native_integrations']==2248,
        no_optimizer=all(v['optimizer_steps']==0 for v in r.values()),all_zero_native_auxiliary=all(r[k]['actual_native_integrations']==0 for k in ['plain','shared','node_review','lifecycle_review','proposal']),node_all16_physical=len(r['node_review']['rows'])==16 and all(v['physics']['valid'] for v in r['node_review']['rows']),lifecycle_281_rows=len(r['lifecycle_review']['rows'])==281 and r['lifecycle_review']['physical_valid_rows']==279,
        original2_lifecycle_failures=[x['name'] for x in r['lifecycle_review']['rows'] if not x['physics']['valid']]==['SUPPORT_CLIPPED','SUPPORT_P058'],full54_not_registered_or_launched=not (ROOT/'runs/pen_native_screened_rollout_registration_v1').exists() and not (ROOT/'runs/pen_native_screened_rollout_v1').exists())
    tests=[ROOT/'runs'/n for n in ['pen_snapshot_transport_tests552_v1.log','pen_shared_snapshot_tests554_v1.log','pen_native_screening_tests560_v1.log']]
    for n,p in zip([552,554,560],tests):checks['tests'+str(n)]=f'Ran {n} tests' in p.read_text() and '\nOK\n' in p.read_text()
    receipts=[v['source_receipt'] for v in r.values()]
    for k in ['node','lifecycle']:
        reg=r[k]['registration'];receipts.append(read(verify_file_pin(reg))['source_receipt'])
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Lifecycle archive discrepancy '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-live-lifecycle-results-2026-11-17.md','research-plan-v1.43.md','pen-native-screened-rollout-protocol-v1.md']];config=ROOT/'configs/pen_native_screened_rollout_v1.toml';tools=[ROOT/'tools'/n for n in ['qualify_pen_snapshot_transport.py','qualify_pen_shared_snapshot_transport.py','qualify_pen_live_node.py','review_pen_live_node.py','qualify_pen_terminal_lifecycle.py','review_pen_terminal_lifecycle.py','qualify_pen_screened_proposals.py','run_pen_native_screened_rollout.py']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+tools+[config,Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    records=[x['transport'] for x in r['shared']['rows']]+[c['transport'] for a in r['lifecycle']['arms'] for c in a['captures']];md=max(x['payload']['compressed']['bytes'] for x in records);ms=max(x['shared']['compressed']['bytes'] for x in records);projected=2*md*57408+2*ms*54+5*2**30
    summary=dict(checks=[sum(checks.values()),len(checks)],plain_checks=[4,4],shared_checks=[5,5],node_checks=[12,12],node_review_checks=[7,7],lifecycle_checks=[12,12],lifecycle_review_checks=[7,7],proposal_QA_checks=[5,5],actual_native_integrations=2376,optimizer_steps=0,normalizer_fits=0,successful_restores=18,bad_transport_denials=4,terminal_pending_denials=2,
        physical_rows=297,physical_valid_rows=295,original_physics_flags=5940,original_reward_flags=4158,lifecycle_failures=['SUPPORT_CLIPPED','SUPPORT_P058'],empirical_full54_projected_bytes=projected,reserve_bytes=32*2**30,full54_registered=False,full54_launched=False,
        scope='Transport and runtime lifecycle prerequisites,not complete policy/WM/ego/hardware task or true A1/A2. All prior failures retained;observed future differences require SHAM.')
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_path_decomposition_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    todo=[file_pin(p) for p in list(roots.values())+docs+tools+tests+[config,out/'review.json',out/'source_receipt.json',prior_path]]+receipts;allowed={p.parent for p in roots.values()}|{out,ROOT/'runs/pen_live_node_registration_v1',ROOT/'runs/pen_terminal_lifecycle_registration_v1'}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write_json(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write_json(archive/'summary.json',summary)
    write_json(archive/'ledger.json',dict(schema='pen_live_lifecycle_results_ledger:1',prior_evidence=file_pin(prior_path),roots={k:file_pin(p) for k,p in roots.items()},review=file_pin(out/'review.json'),summary=file_pin(archive/'summary.json'),documents=[file_pin(p) for p in docs],full54_config=file_pin(config),source_receipts=receipts+[file_pin(out/'source_receipt.json')],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),actual_native_integrations=2376,optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),native=2376,optimizer=0)),flush=True)


if __name__=='__main__':main()
