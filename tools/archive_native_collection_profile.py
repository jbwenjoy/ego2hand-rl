"""Compact pins for actual profiling, optional cache and independent review."""
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
    out=ROOT/'provenance/native_collection_results';analysis=ROOT/'runs/native_collection_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable collection evidence')
    prior_path=ROOT/'provenance/resident_contact_results/ledger.json';prior=json.loads(prior_path.read_text())
    previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());ppo=json.loads(verify_file_pin(prior['prior_evidence']).read_text());portable=json.loads(verify_file_pin(ppo['prior_evidence']).read_text())
    old=set();prior_counts=[]
    for ledger in [prior,ppo,portable]:
        m=json.loads(verify_file_pin(ledger['verified_manifest']).read_text());prior_counts.append(len(m['records']))
        for e in m['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
    reports={key:ROOT/'runs'/folder/'report.json' for key,folder in dict(profile='native_collection_profile_v1',cache='native_reference_cache_v1',review='native_collection_review_v1').items()}
    data={k:json.loads(p.read_text()) for k,p in reports.items()}
    for k,v in dict(profile=[7,7],cache=[9,9],review=[125,125]).items():
        if [data[k]['passed'],data[k]['total']]!=v:raise ValueError('Declared outcome changed '+k)
    if sum(data[k].get('actual_native_integrations',data[k].get('native_integrations',0)) for k in data)!=278528 or data['profile']['discarded_optimizer_steps']!=256:raise ValueError('Cost changed')
    parity=json.loads(verify_file_pin(data['cache']['parity']).read_text())
    if len(parity['cases'])!=372 or max(c['max_error'] for c in parity['cases'])!=0 or parity['live_max_error']!=0 or parity['live_same_input_calls']!=142:raise ValueError('Exact parity scope changed')
    log=ROOT/'runs/native_collection_tests305_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 305 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Complete test evidence required')
    docs=[ROOT/'docs'/n for n in ['native-collection-profile-results-2026-11-16.md','research-plan-v1.01.md']]
    configs=[ROOT/'configs'/n for n in ['native_collection_profile_v1.toml','native_reference_cache_v1.toml']]
    tools=[ROOT/'tools'/n for n in ['profile_native_collection.py','qualify_native_reference_cache.py','review_native_collection_profile.py','archive_native_collection_profile.py','archive_pen_source_ccd_evidence.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+tools+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir()
    receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Full original path/size/hash; exact GPU-run source version, not current-source substitution.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    logs=[ROOT/'runs'/n for n in ['native_collection_profile_console_v1.log','native_reference_cache_console_v1.log','native_collection_review_console_v1.log','native_collection_tests305_v1.log']]
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]];allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='native_collection_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks={k:[v['passed'],v['total']] for k,v in data.items()},
        profile_medians=data['profile']['medians'],cache_medians=data['cache']['medians'],observed_collection_time_reduction_fraction=data['review']['observed_collection_time_reduction_fraction'],
        parity=dict(cases=372,max_error=0,mixed_error=parity['mixed_error'],live_calls=142,live_max_error=0),resident_cache_bytes=data['cache']['cache']['resident_bytes'],
        timed_rollout_reset_accounting=data['review']['accounting'],fixed_input_feature=[dict(mode=k,ATen_ops=sum(v['operation_count']['operations'].values()),index_ops=v['operation_count']['operations'].get('aten.index.Tensor',0),
            median_host_us_per_call=v['median_host_us_per_call']) for k,v in data['review']['fixed_input_micro'].items()],
        actual_native_integrations=278528,discarded_optimizer_steps=256,retained_policy_optimizer_steps=0,unit_tests=305,
        scope='Opt-in fixed-reference computational cache. Same-input exact observed feature parity, short controlled collection timings; reset counts/W32 trajectories differ, cold guards excluded. No default checkpoint3 migration, new policy/task/WM-gradient/ego or full training speedup claim.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='native_collection_evidence_ledger:1',analysis_revision=revision,roots={k:file_pin(p) for k,p in reports.items()},summary=file_pin(out/'summary.json'),
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=prior_counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=278528,discarded_optimizer_steps=256,retained_policy_optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),aliases=len(aliases),prior_reverified=len(old))))


if __name__=='__main__':main()
