"""Retain fresh-process failures and ready common fit data with exact sources."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_solver_observation_results';analysis=ROOT/'runs/pen_solver_observation_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence destination')
    prior_path=ROOT/'provenance/pen_train_coverage_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(observation_A='pen_train_solver_observation_A_v1',observation_B='pen_train_solver_observation_B_v1',review='pen_train_solver_observation_review_v1',common_data='pen_contact_response_common_data_v1')
    reports={key:ROOT/'runs'/name/'report.json' for key,name in names.items()};data={key:json.loads(p.read_text()) for key,p in reports.items()};expected=dict(observation_A=[266,266],observation_B=[266,266],review=[485,485],common_data=[45,45])
    if any([data[k]['passed'],data[k]['total']]!=expected[k] for k in expected) or sum(d['actual_native_integrations'] for d in data.values())!=768 or any(d['optimizer_steps'] for d in data.values()):raise ValueError('Recorded budget/contract differs')
    review=data['review'];common=data['common_data']
    if sum(s['impulse_failures'] for s in review['summaries'])!=39 or sum(s['impulse_failures'] for s in review['summaries'] if s['selected_role']=='MATCHED_ORIGINAL_PASS')!=0 or review['exact_scope_pairs']!=40 or sorted({t['solver_niter'] for t in review['failed_extra_field_ticks']})!=[2,4]:raise ValueError('Do not erase physical counterexamples/variation')
    if common['TRAIN_valid']!=808 or common['TRAIN_pairs']!=188 or common['CHECK_valid']!=324 or common['scale_input_role']!='VALID_TRAIN_ONLY' or common['neural_updates']:raise ValueError('Data role or fit budget differs')
    log=ROOT/'runs/pen_solver_common_data_tests360_v1.log';test_text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 360 tests in ' not in test_text or '\nOK\n' not in test_text:raise ValueError('Full360-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-train-solver-observation-protocol-v1.md','pen-contact-response-fit-protocol-v1.md','pen-solver-observation-results-2026-11-16.md','research-plan-v1.09.md']]
    configs=[ROOT/'configs'/n for n in ['pen_train_solver_observation_v1.toml','pen_contact_response_fit_v1.toml']];scripts=[ROOT/'tools'/n for n in ['observe_pen_train_solver.py','review_pen_train_solver.py','prepare_pen_contact_response_fit.py','archive_pen_solver_observation.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[verify_file_pin(d['source_receipt']) for d in data.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/SHA;historical failed/successful data and source revisions stay immutable.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[];seen=set()
    for case in data['observation_A']['cases']:
        key=json.dumps(case['source'],sort_keys=True)
        if key in seen:continue
        seen.add(key);state=torch.load(verify_file_pin(case['source']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'],live=True)
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        engines+=state['native']['engine_sources']
    logs=[log]+[ROOT/'runs'/n for n in ['pen_train_solver_observation_A_console_v1.log','pen_train_solver_observation_B_console_v1.log','pen_train_solver_observation_review_console_v1.log','pen_contact_response_common_data_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_solver_observation_common_data_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,
        physical_status=review['physical_status'],summaries=review['summaries'],extra_failed_ticks=len(review['failed_extra_field_ticks']),observed_failed_solver_niters=[2,4],iteration_cap=50,private_solver_context_captured=False,
        exact_scope_pairs=[40,48],TRAIN_valid=808,TRAIN_total=816,CHECK_valid=324,TRAIN_pairs=188,masked_TRAIN_rows=common['masked_TRAIN_rows'],scale_input_role=common['scale_input_role'],normalizer_fits=common['normalizer_fits'],
        actual_native_integrations=768,optimizer_steps=0,unit_tests=360,
        scope='Selected strict failures reproduce in two fresh processes before iteration cap;private stopping cause unresolved,observer/order variation retained. Common within-reference TRAIN/CHECK/pair data ready;no neural/Actor/task/ego/performance conclusion.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_solver_observation_common_data_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),
        actual_native_integrations=768,optimizer_steps=0,physical_qualification=review['physical_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
