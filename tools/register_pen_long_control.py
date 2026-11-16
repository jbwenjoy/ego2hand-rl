"""Register immutable startup evidence; never close or total a running job."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import (
    freeze_sources, verify_source_pin, verify_source_snapshot,
)


def main():
    destination = ROOT / 'provenance/pen_long_control_registration_v1'
    analysis = ROOT / 'runs/pen_long_control_registration_v1'
    if destination.exists() or analysis.exists():
        raise ValueError('Fresh registration paths required')
    prior_path = ROOT / 'provenance/pen_fn_response_results/ledger.json'
    prior = json.loads(prior_path.read_text())
    resolutions = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    prior_manifest = json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for entry in prior_manifest['records']:
        resolved = verify_source_pin(entry['observed'], resolutions)
        if file_pin(resolved) != entry['resolved']:
            raise ValueError('Prior manifest resolution changed')
    for alias in resolutions['records']:
        verify_source_pin(alias['original'], resolutions)

    qualification_path = ROOT / 'runs/pen_long_control_qualification_v1/report.json'
    run = ROOT / 'runs/pen_long_control_training_v1'
    if (run / 'failure.json').exists() or (run / 'report.json').exists():
        raise ValueError('This registration is only for an unfinished nonfailed job')
    health_path = run / 'RESIDUAL/health_0001.json'
    qualification = json.loads(qualification_path.read_text())
    health = json.loads(health_path.read_text())
    if (qualification['passed'], qualification['total'],
            qualification['actual_native_integrations'], qualification['optimizer_steps']) != (75, 75, 4096, 0):
        raise ValueError('Qualification work/checks differ')
    if (health['iteration'], health['optimizer_steps'], health['training_transitions'],
            health['normalizer_count'], health['collector']['passed'], health['collector']['total']) != (1, 32, 2048, 288, 25, 25):
        raise ValueError('First optimized checkpoint differs')
    reload = health['first_checkpoint_reload']
    if not all(v for v in reload.values() if type(v) is bool):
        raise ValueError('First ordinary reload failed')
    native = reload['native']
    if not native['all_array_bytes_equal'] or not native['all_scalar_values_equal'] or not native['pointers_preserved']:
        raise ValueError('Native immediate readback failed')

    receipts = [Path(qualification['source_receipt']['path']), run / 'source_receipt.json']
    for path in receipts:
        verify_source_snapshot(json.loads(path.read_text()), live=True)
    analysis.mkdir()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source = freeze_sources([Path(__file__)], analysis / 'source_snapshot', ROOT, revision)
    write(analysis / 'source_receipt.json', source)
    receipts.append(analysis / 'source_receipt.json')
    aliases = {json.dumps(e['original'], sort_keys=True): e for e in resolutions['records']}
    for path in receipts:
        for record in json.loads(path.read_text())['records']:
            aliases.setdefault(json.dumps(record['observed'], sort_keys=True), dict(
                original=record['observed'], archived=record['snapshot'],
                reason='Exact path, bytes and SHA at long-control startup registration.'))
    resolutions = dict(schema='explicit_source_relocations:1', records=list(aliases.values()))
    roots = dict(qualification=file_pin(qualification_path), first_health=file_pin(health_path),
                 initialization=file_pin(run / 'RESIDUAL/initialization.json'))
    todo = list(roots.values()) + [file_pin(p) for p in receipts] + [file_pin(prior_path)]
    # Only these immutable JSON roots may expand. Never traverse active progress logs.
    expand = {p['path'] for p in roots.values()} | {str(p) for p in receipts}
    verified = {}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, resolutions)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if pin['path'] in expand:
            todo.extend(pins(json.loads(resolved.read_text())))
    for path in receipts:
        verify_source_snapshot(json.loads(path.read_text()), live=True)
    summary = dict(
        schema='pen_long_control_startup_registration:1',
        status='RUNNING_AT_REGISTRATION_NOT_FINAL', roots=roots,
        registered_per_arm=dict(iterations=2000, native_integrations=32768000,
                                optimizer_steps=64000, transitions=4096000),
        registered_total=dict(native_integrations=65536000, optimizer_steps=128000),
        separate_qualification=dict(actual_native_integrations=4096, optimizer_steps=0),
        first_checkpoint=dict(optimizer_steps=32, training_transitions=2048,
                              derived_native_budget=16384, normalizer_count=288,
                              reload=reload, reload_wall_s=health['first_checkpoint_reload_wall_s']),
        later_training_actual_work=None, final_actual_work=None,
        scope='Immutable qualification and first optimized reload evidence only. '
              'Running session was separately polled by the parent agent. '
              'This record is not current process liveness or final work accounting. '
              'No active console/progress log is pinned. Immediate reload byte equality '
              'does not establish future trajectory determinism. No WM Actor updates; '
              'model-free training metrics are not strict task success or ego results.')
    destination.mkdir()
    write(destination / 'summary.json', summary)
    write(destination / 'source_resolutions.json', resolutions)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    ledger = dict(schema='pen_long_control_registration_ledger:1', analysis_revision=revision,
                  summary=file_pin(destination / 'summary.json'), roots=roots,
                  source_receipts=[file_pin(p) for p in receipts],
                  source_resolutions=file_pin(destination / 'source_resolutions.json'),
                  verified_manifest=file_pin(analysis / 'verified_pins.json'),
                  verified_pins=len(verified), prior_evidence=file_pin(prior_path),
                  prior_manifest_reverified=len(prior_manifest['records']),
                  status=summary['status'], final_actual_work=None, scope=summary['scope'])
    write(destination / 'ledger.json', ledger)
    print(json.dumps(dict(ledger=file_pin(destination / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
