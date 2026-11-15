"""Retain native metric evidence and exact historical source versions."""
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
    out=ROOT/'provenance/resident_contact_results';analysis=ROOT/'runs/resident_contact_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence roots')
    prior_path=ROOT/'provenance/pen_native_ppo_results/ledger.json';prior=json.loads(prior_path.read_text())
    previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());pm=json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    portable=json.loads(verify_file_pin(prior['prior_evidence']).read_text());portable_manifest=json.loads(verify_file_pin(portable['verified_manifest']).read_text())
    already=set()
    for manifest in [pm,portable_manifest]:
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in already:verify_source_pin(entry['observed'],previous);already.add(key)
    reports={k:ROOT/'runs'/folder/'report.json' for k,folder in dict(replay='resident_contact_replay_v1',live='resident_contact_native_v2',
        profile_minimal='resident_contact_profile_v1',profile_native='resident_contact_profile_v2').items()}
    values={k:json.loads(p.read_text()) for k,p in reports.items()}
    for key,expected in dict(replay=[81,81],live=[63,63],profile_minimal=[10,10],profile_native=[10,10]).items():
        if [values[key]['passed'],values[key]['total']]!=expected:raise ValueError('Observed checks changed '+key)
    if values['live']['actual_native_integrations']!=4224 or values['replay']['packets']!=648 or values['replay']['world_samples']!=2632:raise ValueError('Metric study budget changed')
    log=ROOT/'runs/resident_contact_tests299_v1.log';test=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 299 tests in ' not in test or '\nOK\n' not in test:raise ValueError('Full tests missing')
    analysis.mkdir();failed=ROOT/'runs/resident_contact_native_v1'
    failure=dict(schema='posthoc_native_metric_failed_run_record:1',console=file_pin(ROOT/'runs/resident_contact_native_console_v1.log'),
        source_receipt=file_pin(failed/'source_receipt.json'),terminal_evidence='Process exited1: adapter rejected another step while pending terminal reset.',
        physical_integrations_exact=None,physical_integrations_lower_bound=0,physical_integrations_upper_bound=128,optimizer_steps=0,
        reason='First qualification did not persist its counter on exception; do not infer an exact count from later runs. W1 failed before W32 was constructed; predeclared16x8W1 bounds additional steps.',
        scope='Posthoc record from retained source/log, not recovered missing native trajectories/counter.')
    write(analysis/'failed_live_v1.json',failure)
    docs=[ROOT/'docs'/n for n in ['research-plan-v1.00.md','resident-contact-metrics-results-2026-11-16.md']]
    configs=[ROOT/'configs'/n for n in ['resident_contact_replay_v1.toml','resident_contact_native_v1.toml','resident_contact_profile_v1.toml']]
    tools=[ROOT/'tools'/n for n in ['archive_resident_contact_metrics.py','archive_pen_source_ccd_evidence.py','qualify_resident_contact_replay.py','qualify_resident_contact_native.py','profile_resident_contact_metrics.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+tools+[ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[failed/'source_receipt.json',analysis/'source_receipt.json']
    aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact full original source pin; preserve both failed and successful versions.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    logs=[ROOT/'runs'/n for n in ['resident_contact_unit8_v1.log','resident_contact_unit8_v2.log','resident_contact_unit8_v3.log','resident_contact_unit8_v4.log',
        'resident_contact_unit12_v1.log','resident_contact_tests299_v1.log','resident_contact_replay_console_v1.log','resident_contact_native_console_v1.log',
        'resident_contact_native_console_v2.log','resident_contact_profile_console_v1.log','resident_contact_profile_console_v2.log']]
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[analysis/'failed_live_v1.json',prior_path]]
    allowed={p.parent for p in reports.values()}|{analysis,failed};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='resident_contact_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks={k:[v['passed'],v['total']] for k,v in values.items()},
        replay_packets=648,replay_world_samples=2632,confirmed_physical_integrations=4224,additional_failed_physical_integrations=dict(exact=None,lower=0,upper=128),optimizer_steps=0,
        source_capture=dict(model_fields=404,data_fields=169,native_indices=[4,5,6],role='New diagnostic sources, not historical W32 divergence sources'),
        native_profile=[{k:c[k] for k in ['worlds','global_active_contacts','launched_contact_capacity','median_GPU_us_per_sample','median_host_dispatch_us_per_sample']} for c in values['profile_native']['cases']],
        unit_tests=299,failed_live=file_pin(analysis/'failed_live_v1.json'),scope='Read-only resident arithmetic/live fold qualification only. No new policy learning, end-to-end speedup, W32 physical repeatability, WM gradients, CHORD/ego or full Goal success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='resident_contact_evidence_ledger:1',analysis_revision=revision,roots={k:file_pin(p) for k,p in reports.items()},summary=file_pin(out/'summary.json'),
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],
        prior_evidence=file_pin(prior_path),prior_reverified_pins=len(already),prior_PPO_manifest_pins=len(pm['records']),prior_portable_manifest_pins=len(portable_manifest['records']),
        verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),
        confirmed_physical_integrations=4224,unrecorded_failed_integrations_upper_bound=128,optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),aliases=len(aliases),prior_reverified=len(already))))


if __name__=='__main__':main()
