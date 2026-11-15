"""Freeze matched-source failure evidence, contrast audit and installed-source review."""
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
    out=ROOT/'provenance/pen_matched_amplitude_results';analysis=ROOT/'runs/pen_matched_amplitude_analysis_sources_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence')
    reports=dict(raw=ROOT/'runs/pen_matched_amplitude_sources_v1/report.json',review=ROOT/'runs/pen_matched_amplitude_review_v1/report.json',
        EFC=ROOT/'runs/pen_matched_amplitude_review_v1/EFC/report.json',contrasts=ROOT/'runs/pen_matched_amplitude_contrasts_v2/report.json',
        serialization_failure=ROOT/'runs/pen_matched_amplitude_contrast_setup_v1/record.json')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();analysis.mkdir()
    manifest=freeze_sources([Path(__file__),ROOT/'tools/archive_pen_source_ccd_evidence.py',ROOT/'tools/audit_pen_matched_contrasts.py',
        ROOT/'src/supermimic_research/learning/pen_native_response.py',ROOT/'tests/test_observed_native_data.py',ROOT/'tests/test_repeated_contrast_spread.py'],analysis/'snapshot',ROOT,revision)
    write(analysis/'source_receipt.json',manifest)
    prior=ROOT/'provenance/pen_native_amplitude_results/source_resolutions.json';aliases={json.dumps(e['original'],sort_keys=True):e for e in json.loads(prior.read_text())['records']}
    receipt_paths=[analysis/'source_receipt.json']
    for folder in {p.parent for p in reports.values()}:
        receipt_paths+=list(folder.glob('*source_receipt.json'))
    for path in receipt_paths:
        receipt=json.loads(path.read_text());verify_source_snapshot(receipt)
        for e in receipt['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact source pin observed in frozen receipt'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    todo=[file_pin(p) for p in list(reports.values())+receipt_paths+[ROOT/'runs/pen_matched_amplitude_tests216.log',ROOT/'runs/pen_matched_amplitude_tests218.log',ROOT/'docs/research-plan-v0.85.md']]
    allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    raw=json.loads(reports['raw'].read_text());review=json.loads(reports['review'].read_text());contrast=json.loads(reports['contrasts'].read_text())
    if (sum(e['native_physics_ticks'] for e in raw['candidate_replays'])!=2736 or sum(e['native_physics_ticks'] for e in raw['cases'])!=48
            or review['contract_passed']!=review['contract_total'] or contrast['passed']!=contrast['total'] or contrast['input_role']!='DIAGNOSTIC_NEVER_FIT'):
        raise ValueError('Exact source/role/budget contract required; scientific fidelity failures remain allowed')
    installed=json.loads(Path(raw['installed_source_receipt']['path']).read_text());solver=next(e for e in installed['records'] if e['observed']['path'].endswith('\\_src\\solver.py'))
    lines=Path(solver['snapshot']['path']).read_text().splitlines();excerpts=[]
    for name,length in [('_create_solver_context',51),('_rescale',3),('_qfrc_constraint_from_grad',15),('_solve_done',34),('_solve',58)]:
        index=next(i for i,line in enumerate(lines) if line.startswith('def '+name+'('))
        excerpts.append(dict(function=name,line_start=index+1,lines=lines[index:index+length]))
    out.mkdir();copied={}
    for key in ['review','EFC','contrasts','serialization_failure']:
        target=out/(key+'.json');shutil.copyfile(reports[key],target)
        if file_pin(target)['sha256']!=file_pin(reports[key])['sha256']:raise ValueError('Copy changed bytes')
        copied[key]=file_pin(target)
    for name in ['216','218']:
        p=ROOT/f'runs/pen_matched_amplitude_tests{name}.log';target=out/f'tests{name}.log';shutil.copyfile(p,target);copied['tests'+name]=file_pin(target)
    write(out/'installed_solver_review.json',dict(schema='installed_native_solver_local_review:1',observed=solver['observed'],snapshot=solver['snapshot'],excerpts=excerpts,
        model_scope='Saved solver NEWTON(2),pyramidal cone(0),enableflags0,no SLEEP: original noncompact path. Constraint iteration/line-search caps50, unchanged.',
        facts=['Context workspace arrays are allocated by native code; full public Data snapshots do not instrument final private SolverContext stopping fields.',
            'Native Newton stop tests OR scaled cost improvement, gradient norm, model improvement, or iteration cap.',
            'Incremental Newton force recovery uses qfrc_constraint=Ma-qfrc_smooth-grad_scale*grad.',
            'Recorded sparse mass block and Ma consistency localize a force/acceleration residual. Final private gradient and exact stop reason were not recorded, so no specific termination cause or workspace-contamination claim is proven.'],
        native_integrations=0,neural_updates=0))
    write(out/'source_resolutions.json',resolutions)
    write(out/'ledger.json',dict(schema='pen_matched_observed_source_evidence:1',analysis_revision=revision,roots={k:file_pin(p) for k,p in reports.items()},copied=copied,
        source_receipts=[file_pin(p) for p in receipt_paths],previous_resolutions=file_pin(prior),source_resolutions=file_pin(out/'source_resolutions.json'),
        installed_solver_review=file_pin(out/'installed_solver_review.json'),verified_pins=len(verified),exact_source_aliases=len(aliases),verified=list(verified.values()),
        candidate_integrations=2736,instrumented_integrations=48,total_integrations=2784,neural_updates=0,
        scope='All nested pins in this cohort checked; historical input reports exact at boundary. Preserves failed fidelity/impulse and failed initial JSON serialization. Diagnostic contrast data never fit. No expert, global gradient/physics, policy, ego, hardware or speedup qualification.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),native_integrations=2784)))


if __name__=='__main__':main()
