"""Freeze registration/source banks while response collectors are still live.

This ledger deliberately excludes changing progress/log/response files and
cannot be read as a final collection/preparation/performance result.
"""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot
from supermimic_research.learning.pen_interventions import validate_full_rank_cohort
import tomllib


def main():
    out=ROOT/'provenance/pen_fullrank_registration_v1';analysis=ROOT/'runs/pen_fullrank_registration_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable registration required')
    prior_path=ROOT/'provenance/pen_cached_adoption_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    roots={k:ROOT/'runs'/('pen_response_fullrank_'+k+'_v1') for k in ['train','validation']};configs=[ROOT/'configs'/('pen_response_fullrank_'+k+'_v1.toml') for k in roots]+[ROOT/'configs/pen_fullrank_data_review_v1.toml'];banks={k:p/'source_bank.json' for k,p in roots.items()};receipts=[p/'source_receipt.json' for p in roots.values()]
    configurations={k:tomllib.loads(configs[i].read_text()) for i,k in enumerate(roots)}
    for cfg in configurations.values():validate_full_rank_cohort(cfg)
    data={k:json.loads(p.read_text()) for k,p in banks.items()}
    if any(len(data[k]['sources'])!=configurations[k]['expected_sources'] or not all(s['qualified'] for s in data[k]['sources']) for k in roots):raise ValueError('Observed registered source bank differs')
    if sum(d['actual_native_integrations'] for d in data.values())!=4352:raise ValueError('Source producer cost differs')
    log=ROOT/'runs/pen_fullrank_tests385_v1.log';txt=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 385 tests in ' not in txt or '\nOK\n' not in txt:raise ValueError('Full385-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-response-fullrank-protocol-v1.md','pen-fullrank-data-review-protocol-v1.md','research-plan-v1.14.md']]
    scripts=[ROOT/'tools'/n for n in ['collect_trained_pen_responses.py','prepare_pen_train_coverage.py','review_pen_full_rank_data.py','register_pen_full_rank_study.py','archive_pen_source_ccd_evidence.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+configs+docs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt);receipts+=[analysis/'source_receipt.json']
    aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source,live=True)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact full path/bytes/SHA;full20 split/protocol registration and completed producer banks,not final response results.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    todo=[file_pin(p) for p in list(banks.values())+receipts+configs+docs+[prior_path,log]];allowed=set(roots.values())|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_fullrank_registration_summary:1',collection_status='RUNNING_AT_REGISTRATION_NOT_FINAL',source_banks={k:file_pin(p) for k,p in banks.items()},qualified_sources={k:len(d['sources']) for k,d in data.items()},configs=[file_pin(p) for p in configs],declared_cases={k:configurations[k]['expected_cases'] for k in roots},declared_max_native_integrations=51008,producer_native_integrations=4352,response_native_integrations=None,optimizer_steps=0,unit_tests=385,source_coverage='Requested rank20/condition1;effective qualified physical rank pending independent decode',scope='Preregistered data-only source/split/budget contracts plus completed producer-bank evidence. Response collection is still running;no final row masks/rank/WM/model selection/Actor/task/ego or throughput claim.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_fullrank_registration_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],configs=[file_pin(p) for p in configs],source_banks=summary['source_banks'],prior_evidence=file_pin(prior_path),prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),producer_native_integrations=4352,optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
