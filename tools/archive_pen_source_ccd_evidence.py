"""Freeze and verify the completed v0.81 cohort without rewriting raw reports."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_pin, verify_source_snapshot


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def pins(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            yield {k: value[k] for k in ['path', 'bytes', 'sha256']}
        else:
            for item in value.values():
                yield from pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from pins(item)


def main():
    destination = ROOT / 'provenance/pen_source_ccd_results'
    analysis = ROOT / 'runs/pen_source_ccd_analysis_sources_v1'
    if destination.exists() or analysis.exists():
        raise ValueError('Fresh evidence destination required; no replacement of past records')
    names = {
        'pilot': 'pen_visited_ccd_baseline_v1',
        'pilot_review': 'pen_visited_ccd_baseline_review_v1',
        'baseline': 'pen_visited_ccd_baseline_v2',
        'baseline_review': 'pen_visited_ccd_baseline_review_v2',
        'tight': 'pen_visited_ccd_tight_v1',
        'tight_review': 'pen_visited_ccd_tight_review_v1',
        'comparison': 'pen_visited_ccd_comparison_v1',
        'figure': 'pen_visited_ccd_figure_v1',
    }
    reports = {k: ROOT / 'runs' / v / 'report.json' for k, v in names.items()}
    for key in ['pilot_review', 'baseline_review', 'tight_review']:
        reports[key + '_EFC'] = reports[key].parent / 'EFC/report.json'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    analysis.mkdir()
    receipt = freeze_sources([Path(__file__), ROOT / 'tools/compare_pen_source_ccd.py', ROOT / 'tools/plot_pen_source_ccd.py'], analysis / 'snapshot', ROOT, revision)
    write(analysis / 'source_receipt.json', receipt)
    previous_path = ROOT / 'provenance/pen_contact_stages_results/source_resolutions.json'
    previous = json.loads(previous_path.read_text())
    aliases = {json.dumps(r['original'], sort_keys=True): r for r in previous['records']}
    receipt_paths = [analysis / 'source_receipt.json']
    for folder in names.values():
        receipt_paths += sorted((ROOT / 'runs' / folder).glob('*source_receipt.json'))
    for path in receipt_paths:
        manifest = json.loads(path.read_text())
        verify_source_snapshot(manifest, live=False)
        for record in manifest['records']:
            aliases.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'], archived=record['snapshot'], reason='Exact observed-to-snapshot receipt; no basename or hash-only substitution'))
    resolutions = dict(schema='explicit_source_relocations:1', records=list(aliases.values()))
    for record in resolutions['records']:
        verify_source_pin(record['original'], resolutions)
    # Follow every nested pin inside this cohort, but do not recursively reinterpret
    # historical input reports. Their exact bytes are verified as boundary inputs.
    roots = list(reports.values()) + receipt_paths
    roots += [ROOT / 'runs/pen_visited_ccd_tests208.log', ROOT / 'docs/research-plan-v0.81.md']
    allowed = [ROOT / 'runs' / name for name in names.values()] + [analysis]
    todo = [file_pin(p) for p in roots]
    verified = {}
    while todo:
        pin = todo.pop()
        identity = json.dumps(pin, sort_keys=True)
        if identity in verified:
            continue
        resolved = verify_source_pin(pin, resolutions)
        verified[identity] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(folder) for folder in allowed):
            todo.extend(pins(json.loads(resolved.read_text())))
    compare = json.loads(reports['comparison'].read_text())
    figure = json.loads(reports['figure'].read_text())
    if compare['passed'] != compare['total'] or compare['total'] != 154:
        raise ValueError('Matched input audit failed')
    if [s['state_failed'] for s in figure['summaries']] != [6, 0] or any(p['qualified'] for p in compare['profiles']):
        raise ValueError('Unexpected qualification: review the science before archiving')
    budget = {}
    for key in ['pilot', 'baseline', 'tight']:
        raw = json.loads(reports[key].read_text())
        budget[key] = raw['candidate_physics_ticks'] + raw['instrumented_physics_ticks']
    if list(budget.values()) != [6960, 6960, 6960]:
        raise ValueError('Native cost accounting changed')
    destination.mkdir()
    copied = {}
    for key, path in reports.items():
        target = destination / (key + '.json')
        shutil.copyfile(path, target)
        if file_pin(path)['sha256'] != file_pin(target)['sha256']:
            raise ValueError('Evidence copy changed bytes')
        copied[key] = file_pin(target)
    for name, path in {'summary.png': reports['figure'].parent / 'summary.png', 'tests208.log': ROOT / 'runs/pen_visited_ccd_tests208.log'}.items():
        shutil.copyfile(path, destination / name)
        if file_pin(path)['sha256'] != file_pin(destination / name)['sha256']:
            raise ValueError('Figure/log copy changed bytes')
        copied[name] = file_pin(destination / name)
    write(destination / 'source_resolutions.json', resolutions)
    write(destination / 'ledger.json', dict(schema='pen_source_ccd_evidence:1', analysis_revision=revision,
        roots={k: file_pin(p) for k, p in reports.items()}, copied=copied,
        source_receipts=[file_pin(p) for p in receipt_paths], previous_resolutions=file_pin(previous_path),
        source_resolutions=file_pin(destination / 'source_resolutions.json'), verified=list(verified.values()),
        verified_pins=len(verified), exact_source_aliases=len(aliases), native_integrations=budget,
        formal_integrations=budget['baseline'] + budget['tight'], total_integrations=sum(budget.values()), neural_updates=0,
        scope='All nested file pins in this cohort verified; exact historical input reports verified at boundary, not an independent rerun of all historical experiments. Source superset snapshots are not imported-module execution proof. Both profiles unqualified; no task/gradient/ego/acceleration promotion.'))
    print(json.dumps(dict(ledger=file_pin(destination / 'ledger.json'), verified_pins=len(verified), aliases=len(aliases), budget=budget)))


if __name__ == '__main__':
    main()
