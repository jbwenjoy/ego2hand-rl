"""Audit/close derived collision evidence, including silent negatives and failures."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot, verify_source_pin


def read(path): return json.loads(Path(path).read_text())


def main():
    out = ROOT / 'provenance/pen_retained_epa_diagnosis_v1'
    analysis = ROOT / 'runs/pen_retained_epa_archive_v1'
    if out.exists() or analysis.exists(): raise ValueError('Fresh diagnosis archive required')
    prior_path = ROOT / 'provenance/pen_first_overflow_diagnostic_v1/ledger.json'
    prior = read(prior_path); aliases = read(verify_file_pin(prior['source_resolutions']))
    for row in read(verify_file_pin(prior['verified_manifest']))['records']:
        if file_pin(verify_source_pin(row['observed'], aliases)) != row['resolved']: raise ValueError('Prior resolution changed')
    names = dict(scene_registration='pen_retained_collision_scene_registration_v1/registration.json',
        scene='pen_retained_collision_scene_v1/report.json', pair_registration_V1='pen_epa_pair_capacity_registration_v1/registration.json',
        pair_failure_V1='pen_epa_pair_capacity_v1/failure.json', pair_registration_V2='pen_epa_pair_capacity_registration_v2/registration.json',
        pair='pen_epa_pair_capacity_v2/report.json', distance_preflight_V1_source='pen_epa_native_distance_review_v1/source_receipt.json',
        distance='pen_epa_native_distance_review_v2/report.json', translation_registration='pen_epa_translation_registration_v1/registration.json',
        translation='pen_epa_translation_v1/report.json', branch_trace='pen_epa_degenerate_exit_trace_v1/report.json')
    roots = {k: file_pin(ROOT / 'runs' / name) for k, name in names.items()}
    r = {k: read(verify_file_pin(p)) for k, p in roots.items()}
    scene = r['scene']; pair = r['pair']; translation = r['translation']; trace = r['branch_trace']
    capacity = read(verify_file_pin(pair['capacity_results'])); isolated = read(verify_file_pin(pair['isolated_candidates']))
    failed = read(verify_file_pin(pair['failing_pairs'])); trans = read(verify_file_pin(translation['rows']))
    raw_parent = read(Path(roots['scene']['path']).parent / 'PARENT_RAW_geometry.json')
    derived_parent = read(Path(roots['scene']['path']).parent / 'PARENT_DERIVED_geometry.json')
    scene_by = {s['label']: s for s in scene['rows']}
    modes = {mode: [v for v in trans if v['placement'] == mode] for mode in ['WORLD', 'GEOM1_ORIGIN', 'MIDPOINT']}
    checks = dict(
        temporal_boundaries_admitted=all(v['admitted_for_derived_query'] for v in scene['temporal_boundaries']),
        raw_parent_geometry_exact=all(raw_parent['all_bytes_equal'].values()),
        derived_parent_error_bounded=max(derived_parent['max_errors'].values()) <= 1e-6,
        five_full_scenes=(scene['actual_FK_builds'], scene['actual_batched_collision_queries']) == (5, 5),
        derived63_EPA_world31=scene_by['ORIGINAL_ROW63_DERIVED']['raw_mask'] == [0] * 31 + [256],
        other_scene_controls_no_overflow=all(not any(v['raw_mask']) for k, v in scene_by.items() if k != 'ORIGINAL_ROW63_DERIVED'),
        all12_candidates_isolated=len(isolated) == 12,
        sole_physical_pair=len(failed) == 1 and failed[0]['geoms'] == [1, 55] and failed[0]['types'] == [5, 7] and failed[0]['pairid'] == [-1, -1],
        V1_failure_work_preserved=r['pair_failure_V1']['counts'] == dict(full_attempted=1, full_completed=1, pair_attempted=12, pair_completed=12, capacity_attempted=1, capacity_completed=1, FK_attempted=0, FK_completed=0),
        V2_counts=pair['counts'] == dict(full_attempted=1, full_completed=1, pair_attempted=12, pair_completed=12, capacity_attempted=36, capacity_completed=36, FK_attempted=8, FK_completed=8),
        capacity_four_overflows24=sum(v['raw_mask'][31] == 256 for v in capacity if v['capacity'] == 24) == 4,
        capacity_larger_silent_four_missing=all(sum(v['nacon'] == 0 and not any(v['raw_mask']) for v in capacity if v['capacity'] == cap) == 4 for cap in [48, 96]),
        all_larger_raw_flags_clear=all(not any(v['raw_mask']) for v in capacity if v['capacity'] in [48, 96]),
        all_48_96_contact_bytes_equal=all(next(v for v in capacity if v['scene'] == label and v['capacity'] == 48)['contact_bytes_sha256'] == next(v for v in capacity if v['scene'] == label and v['capacity'] == 96)['contact_bytes_sha256'] for label in {v['scene'] for v in capacity}),
        positive_capacity_controls_equal=all(len({v['contact_bytes_sha256'] for v in capacity if v['scene'] == label}) == 1 and all(v['nacon'] == 1 for v in capacity if v['scene'] == label) for label in ['PARENT_RAW', 'ORIGINAL_ROW61_DERIVED', 'ORIGINAL_ROW62_DERIVED']),
        native24_distances_all_overlap=r['distance']['actual_native_distance_queries'] == 24 and all(q['signed_distance_m'] < 0 and q['finite'] for v in r['distance']['rows'] for q in v['native_FP64_queries']),
        native_50_200_equal=all(v['native_FP64_queries'][0]['signed_distance_m'] == v['native_FP64_queries'][1]['signed_distance_m'] for v in r['distance']['rows']),
        translation36_pair_and_distance=translation['actual_GPU_pair_queries'] == translation['actual_native_distance_queries'] == 36,
        local_modes_silent_missing=all(sum(v['nacon'] == 0 for v in modes[mode]) == 1 and not any(any(v['raw_mask']) for v in modes[mode]) for mode in ['GEOM1_ORIGIN', 'MIDPOINT']),
        translation_distance_all_overlap=all(v['native_signed_distance_m'] < 0 and v['finite'] for v in trans),
        translation_relative_error_under_nanometer=max(v['relative_position_max_rounding_m'] for v in trans) < 1e-9,
        trace_four_reference_counts_equal=trace['actual_pair_queries'] == 4 and all(v['reference_flag_and_count_equal'] for v in trace['rows']),
        all_new_native_and_Adam_zero=all(r[name]['actual_native_integrations'] == r[name]['optimizer_steps'] == 0 for name in ['scene', 'pair_failure_V1', 'pair', 'distance', 'translation', 'branch_trace']),
        full445_tests='Ran 445 tests in 9.652s' in (ROOT / 'runs/pen_retained_collision_tests445_v1.log').read_text() and '\nOK\n' in (ROOT / 'runs/pen_retained_collision_tests445_v1.log').read_text())
    console = ROOT / 'runs/pen_epa_degenerate_exit_trace_console_v1.log'
    log = console.read_text()
    segments = {label: log.split('TRACE_CASE_BEGIN ' + label, 1)[1].split('TRACE_CASE_END ', 1)[0] for label in [v['label'] for v in trace['rows']]}
    checks['base48_degenerate_and_invalid_observed'] = all(token in segments['BASE48_SILENT_MISSING'] for token in ['TRACE_EPA_DEGENERATE_FACE world=31 face=256 horizon=34', 'TRACE_EPA_INVALID_RESULT world=31 idx=-1'])
    checks['local_missing_not_same_logged_exit'] = 'TRACE_EPA_DEGENERATE_FACE' not in segments['LOCAL24_SILENT_MISSING'] and 'TRACE_EPA_INVALID_RESULT' not in segments['LOCAL24_SILENT_MISSING']
    if not all(checks.values()): raise ValueError('Independent observed diagnosis differs: ' + repr(checks))
    docs = [ROOT / 'docs' / name for name in ['pen-retained-collision-and-epa-results-2026-11-16.md', 'research-plan-v1.24.md']]
    scripts = [Path(__file__)] + [ROOT / 'tools' / name for name in ['probe_retained_pen_collision.py', 'probe_pen_epa_pair_capacity.py', 'probe_pen_epa_translation.py', 'review_pen_epa_signed_distance.py', 'trace_pen_epa_degenerate_exit.py']]
    analysis.mkdir()
    receipt = freeze_sources(scripts + docs + [ROOT / 'README.md'], analysis / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', receipt)
    folders = {Path(pin['path']).parent for pin in roots.values()}
    receipts = sorted({p / 'source_receipt.json' for p in folders}) + [analysis / 'source_receipt.json']
    historical = {'pen_epa_pair_capacity_registration_v1', 'pen_epa_pair_capacity_v1', 'pen_epa_native_distance_review_v1'}
    merged = {json.dumps(v['original'], sort_keys=True): v for v in aliases['records']}
    source_reviews = []
    for path in receipts:
        manifest = read(path); live = path.parent.name not in historical
        verify_source_snapshot(manifest, live=live); source_reviews.append(dict(receipt=file_pin(path), live=live))
        for v in manifest['records']:
            merged.setdefault(json.dumps(v['observed'], sort_keys=True), dict(original=v['observed'], archived=v['snapshot'], reason='Exact full-path/bytes/SHA retained collision evidence source.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    logs = [ROOT / 'runs' / name for name in ['pen_retained_collision_scene_console_v1.log', 'pen_epa_pair_capacity_console_v1.log',
        'pen_epa_pair_capacity_console_v2.log', 'pen_epa_native_distance_review_console_v1.log', 'pen_epa_native_distance_review_console_v2.log',
        'pen_epa_translation_console_v1.log', 'pen_epa_degenerate_exit_trace_console_v1.log', 'pen_retained_collision_tests445_v1.log']]
    preflight_log = logs[3].read_text()
    checks['preflight_v1_original_index_failure_retained'] = 'IndexError: index 31 is out of bounds for axis 0 with size 1' in preflight_log
    if not checks['preflight_v1_original_index_failure_retained']: raise ValueError('Preserve failed pre-query source/trace')
    todo = list(roots.values()) + [file_pin(p) for p in receipts + docs + scripts + logs + [prior_path]]
    todo += [file_pin(p) for folder in folders for p in folder.glob('*.json')]
    allowed = folders | {analysis}; verified = {}
    while todo:
        pin = todo.pop(); key = json.dumps(pin, sort_keys=True)
        if key in verified: continue
        resolved = verify_source_pin(pin, aliases); verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(p) for p in allowed): todo.extend(pins(read(resolved)))
    out.mkdir()
    write(out / 'summary.json', dict(roots=roots, checks=checks, passed=sum(checks.values()), total=len(checks), source_reviews=source_reviews,
        added_work=dict(native_integrations=0, optimizer_steps=0, full_collision_batches=7, world_full_collision_queries=224,
            isolated_pair_queries=101, native_distance_queries=60, FK_batches=13, world_FK_builds=416),
        distance_V1_prequery_failure_scope='Zero distance queries inferred from exact frozen pre-loop source and original traceback;runtime counter/wall unavailable. CPU construction work not zero.',
        current_live_training=False, derived_fixture_only=True, production_capacity_fix_admitted=False, production_translation_fix_admitted=False,
        original_SUPPORT206_cause_proven=False, scope='Contact false negatives remain. No integrations/Adam added here;all prior incurred training cost stays in prior chain. Zero integrations is not zero compute.'))
    write(out / 'source_resolutions.json', aliases); write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='retained_EPA_diagnosis_ledger:1', prior_evidence=file_pin(prior_path), roots=roots,
        summary=file_pin(out / 'summary.json'), documents=[file_pin(p) for p in docs], source_receipts=[file_pin(p) for p in receipts],
        source_resolutions=file_pin(out / 'source_resolutions.json'), verified_manifest=file_pin(analysis / 'verified_pins.json'),
        verified_pins=len(verified), actual_native_integrations=0, optimizer_steps=0, full_collision_batches=7,
        isolated_pair_queries=101, native_distance_queries=60, FK_batches=13, current_live_training=False))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), checks=[sum(checks.values()), len(checks)], verified_pins=len(verified))), flush=True)


if __name__ == '__main__': main()
