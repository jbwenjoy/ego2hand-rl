"""Retain all owned-state source versions, failed copy and final readbacks."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/owned_pen_state_results';analysis=ROOT/'runs/owned_pen_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable owned-state evidence')
    prior_path=ROOT/'provenance/native_collection_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    old=set();cursor=prior;prior_counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());prior_counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if 'prior_evidence' not in cursor:break
        # Stop at the independently verified portable ledger; earlier branches
        # remain linked there rather than recursively inventing a new audit.
        if cursor['schema']=='pen_portable_state_evidence_ledger:1':break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    reports={p.parent.name:p for p in sorted((ROOT/'runs').glob('owned_pen_W*/report.json'))}
    reports.update({p.parent.name:p for p in sorted((ROOT/'runs').glob('owned_pen_review_v*/report.json'))})
    data={k:json.loads(p.read_text()) for k,p in reports.items()}
    final=[f'owned_pen_W{W}_{mode}_v6' for W in [1,32] for mode in ['producer','restorer']]
    for name in final:
        value=data[name];expected=40 if value['mode']=='producer' else 10
        if [value['passed'],value['total']]!=[expected,expected] or [value['audit']['array_fields'],value['audit']['scalar_fields']]!=[833,804]:raise ValueError('Final owned worker outcome changed')
    if [data['owned_pen_review_v3']['passed'],data['owned_pen_review_v3']['total']]!=[67,67]:raise ValueError('Independent final review changed')
    failures=sorted((ROOT/'runs').glob('owned_pen_W*/failure.json'));failure_data=[json.loads(p.read_text()) for p in failures]
    total=sum(v['actual_native_integrations'] for v in data.values())+sum(v['actual_native_integrations'] for v in failure_data)
    if total!=6410 or len(failures)!=1 or failure_data[0]['actual_native_integrations']!=1184 or any(v['optimizer_steps'] for v in list(data.values())+failure_data):raise ValueError('Retained failed/iteration budget changed')
    log=ROOT/'runs/owned_pen_tests318_v1.log';test=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 318 tests in ' not in test or '\nOK\n' not in test:raise ValueError('Final318-test log missing')
    docs=[ROOT/'docs'/n for n in ['owned-pen-state-results-2026-11-16.md','research-plan-v1.02.md']];config=ROOT/'configs/owned_pen_state_v1.toml'
    tools=[ROOT/'tools'/n for n in ['qualify_owned_pen_state.py','review_owned_pen_state.py','archive_owned_pen_state.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+[config,ROOT/'README.md',ROOT/'AGENTS.md']+tools
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[p.parent/'source_receipt.json' for p in failures]+[analysis/'source_receipt.json']
    aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact complete original source pin, including pre-fix/failure versions; no current-source or basename fallback.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    failed_artifacts=[file_pin(p) for f in failures for p in sorted(f.parent.glob('*.pt'))]
    logs=sorted({p for pattern in ['owned_array_unit*.log','owned_pen_tests*.log','owned_pen_W*_console_v*.log','owned_pen_review_console_v*.log'] for p in (ROOT/'runs').glob(pattern)})
    todos=[file_pin(p) for p in list(reports.values())+failures+receipts+docs+logs+[config,prior_path]]+failed_artifacts
    allowed={p.parent for p in reports.values()}|{p.parent for p in failures}|{analysis};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    review=data['owned_pen_review_v3'];summary=dict(schema='owned_pen_state_evidence_summary:1',final_workers={k:file_pin(reports[k]) for k in final},independent=file_pin(reports['owned_pen_review_v3']),
        final_worker_checks=[sum(data[k]['passed'] for k in final),sum(data[k]['total'] for k in final)],independent_checks=[67,67],array_fields=833,scalar_fields=804,
        retained_outcomes={k:dict(checks=[v['passed'],v['total']],actual_native_integrations=v['actual_native_integrations']) for k,v in data.items()},
        failures=[dict(evidence=file_pin(p),**value) for p,value in zip(failures,failure_data)],failed_artifacts=failed_artifacts,
        final_future_single_tick=[dict(worlds=c['worlds'],observations=c['future_single_tick']) for c in review['cases']],
        actual_native_integrations=6410,final_worker_integrations=sum(data[k]['actual_native_integrations'] for k in final),optimizer_steps=0,unit_tests=318,
        scope='Native4 owned cache/metric/native/Python/RNG bytes and partial resets qualified; no training2 payload, new learned policy, end-to-end speedup, deterministic W32 future, physical WM gradients, support benefit or ego/full Goal success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='owned_pen_state_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        failure_roots=[file_pin(p) for p in failures],source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],
        prior_evidence=file_pin(prior_path),prior_manifest_counts=prior_counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),
        verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),actual_native_integrations=6410,optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),aliases=len(aliases),prior_reverified=len(old))))


if __name__=='__main__':main()
