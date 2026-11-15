"""Compact exact evidence for scheduled native horizon diagnostics, failures kept."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_horizon_results';analysis=ROOT/'runs/pen_horizon_analysis_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence')
    folders=dict(pilot='pen_horizon_prefix_audit_v1',H1='pen_native_horizon_H1_v1',H5='pen_native_horizon_H5_v1',
        H1_review='pen_native_horizon_H1_review_v1',H5_review='pen_native_horizon_H5_review_v1',matrix='pen_horizon_matrix_review_v1',video='pen_horizon_video_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in folders.items()}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();analysis.mkdir()
    tool_names=['archive_pen_horizon_results.py','archive_pen_source_ccd_evidence.py','audit_pen_horizon_prefix.py',
                'review_pen_horizon.py','compare_pen_native_ccd.py','render_pen_native_ccd.py','review_pen_native_search.py']
    files=sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools'/n for n in tool_names]+[
        ROOT/'tests/test_pen_horizon.py',ROOT/'tests/test_controlled_pen_records.py']
    receipt=freeze_sources(files,analysis/'snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    legacy=['pen_horizon_legacy_sources_v1','pen_horizon_legacy_tools_v1','pen_horizon_render_legacy_sources_v1','pen_horizon_compare_legacy_sources_v1']
    receipt_paths=[ROOT/'runs'/name/'source_receipt.json' for name in legacy]+[analysis/'source_receipt.json']
    for name in folders.values():receipt_paths+=sorted((ROOT/'runs'/name).glob('*source_receipt.json'))
    prior=ROOT/'provenance/pen_fit_failure_diagnosis_results/source_resolutions.json'
    aliases={json.dumps(e['original'],sort_keys=True):e for e in json.loads(prior.read_text())['records']}
    for path in receipt_paths:
        m=json.loads(path.read_text());verify_source_snapshot(m)
        for e in m['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(
            original=e['observed'],archived=e['snapshot'],reason='Exact observed source-byte receipt, no basename/hash-only fallback'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    docs=[ROOT/'docs'/n for n in ['research-plan-v0.91.md','research-plan-v0.92.md','research-plan-v0.93.md','pen-horizon-results-2026-11-15.md']]
    logs=[ROOT/'runs'/n for n in ['pen_horizon_tests238.log','pen_horizon_final_tests238.log','pen_horizon_warm_guard_tests238.log',
        'pen_horizon_prefix_console_v1.log','pen_native_horizon_H1_console_v1.log','pen_native_horizon_H5_console_v1.log','pen_horizon_video_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipt_paths+docs+logs];verified={}
    allowed={p.parent for p in reports.values()}|{p.parent for p in receipt_paths}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):
            todo+=list(pins(json.loads(resolved.read_text())))
    r={key:json.loads(p.read_text()) for key,p in reports.items()}
    if r['matrix']['passed']!=r['matrix']['total']:raise ValueError('Matrix implementation/provenance failure')
    for name in ['H1_review','H5_review']:
        if r[name]['contract_passed']!=r[name]['contract_total']:raise ValueError('Controller/score contract failure')
    total=r['pilot']['native_integrations']+r['matrix']['actual_integrations']+r['matrix']['candidate_integrations']
    if total!=390000:raise ValueError('Declared integration accounting changed')
    timing={}
    for name in ['H1','H5']:
        timing[name]=[dict(mode=c['name'],dt=c['dt'],duration=c['summary']['metrics']['actual_duration_s'],case_wall_s=c['wall_s'],
            candidate_CUDA_ms=sum(d['candidate_cuda_ms'] for d in c['planning']),
            readback_and_copy_wall_s=sum(d['search_timing']['readback_and_host_copy_wall_s'] for d in c['planning']),
            provider_with_artifact_wall_s=sum(d['total_with_artifact_wall_s'] for d in c['planning']),
            native_candidate_ticks=c['candidate_physics_ticks'],actual_ticks=c['physics_steps']) for c in r[name]['cases']]
    summary=dict(schema='native_horizon_evidence_summary:1',roots={k:file_pin(v) for k,v in reports.items()},
        pilot_counts=r['pilot']['counts'],matrix_counts=[r['matrix']['passed'],r['matrix']['total']],matrix_cells=r['matrix']['cells'],
        actual_reviews={k:dict(contracts=[r[k]['contract_passed'],r[k]['contract_total']],EFC=r[k]['EFC_counts'],
            prefix=r[k]['selected_prefix_counts'],cases=r[k]['cases']) for k in ['H1_review','H5_review']},
        timings=timing,videos=[{k:v[k] for k in ['dt','video','fps','speed','frames','physical_durations_s','decoded_frames']} for v in r['video']['videos']],
        budgets=dict(pilot=r['pilot']['native_integrations'],candidate=r['matrix']['candidate_integrations'],actual=r['matrix']['actual_integrations'],total=total),
        optimizer_updates=0,scope='All original tasks fail; graph/independent dynamics and actual EFC have failures. H5 longer survival in unequal-compute same-clip diagnostic, not qualified control/Actor/ego/acceleration benefit. Full Goal remains active/incomplete.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions)
    shutil.copyfile(logs[2],out/logs[2].name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='native_horizon_evidence_ledger:1',analysis_revision=revision,
        roots={k:file_pin(v) for k,v in reports.items()},summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),
        previous_resolutions=file_pin(prior),source_receipts=[file_pin(p) for p in receipt_paths],documents=[file_pin(p) for p in docs],
        verified_pins=len(verified),verified_manifest=file_pin(analysis/'verified_pins.json'),exact_source_aliases=len(aliases),
        native_integrations=total,optimizer_updates=0,test_log=file_pin(out/logs[2].name),console_logs=[file_pin(p) for p in logs],
        scope='Current cohort nested pins verified; older reports/data remain exact boundary inputs backed by preceding ledgers. Historical source bytes preserved, failed physics/task gates not promoted to success.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),native_integrations=total)))


if __name__=='__main__':main()
