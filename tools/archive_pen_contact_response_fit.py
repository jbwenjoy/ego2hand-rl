"""Retain the completed four-arm fit and its failed physical-gradient results."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_contact_response_fit_results';analysis=ROOT/'runs/pen_contact_response_fit_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable fit evidence')
    prior_path=ROOT/'provenance/pen_solver_observation_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    recovery_path=ROOT/'runs/pen_contact_fit_source_recovery_v1/report.json';recovery=json.loads(recovery_path.read_text());legacy=json.loads(verify_file_pin(recovery['proof_report']).read_text());legacy_pins=list(pins(legacy))
    for entry in recovery['records']:
        if entry['original'] not in legacy_pins or entry['original']['bytes']!=entry['archived']['bytes'] or entry['original']['sha256']!=entry['archived']['sha256']:raise ValueError('Exact historical CLI pin proof required')
        verify_file_pin(entry['archived']);previous['records'].append(entry)
    old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(fit='pen_contact_response_fit_v1',evaluation='pen_contact_response_evaluation_v1',diagnosis='pen_contact_response_diagnosis_v1',review='pen_contact_response_review_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()};fit=data['fit'];ev=data['evaluation'];diag=data['diagnosis'];review=data['review']
    if fit['update']!=2000 or fit['optimizer_steps']!=8016 or fit['clone_only_optimizer_steps']!=16 or fit['Actor_updates'] or fit['training_running'] or sum(d['actual_native_integrations'] for d in data.values()):raise ValueError('Registered work accounting differs')
    for key,pair in dict(evaluation=[108,108],diagnosis=[7,7],review=[152,152]).items():
        if [data[key]['passed'],data[key]['total']]!=pair:raise ValueError('Observed check scope differs '+key)
    for pin in fit['checkpoint_health'].values():
        h=json.loads(verify_file_pin(pin).read_text())
        if [h['passed'],h['total']]!=[87,87]:raise ValueError('Original checkpoint health differs')
    if review['physical_status']!='NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION' or any(review['actor_gradient_qualified'].values()) or diag['local_excitation']['declared_rank']!=4:raise ValueError('Do not promote failed gradient/rank evidence')
    log=ROOT/'runs/pen_contact_fit_tests362_v2.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 362 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full362 tests after final core edit required')
    docs=[ROOT/'docs'/n for n in ['pen-contact-response-fit-protocol-v1.md','pen-contact-response-fit-results-2026-11-16.md','research-plan-v1.10.md']];configs=[ROOT/'configs'/n for n in ['pen_contact_response_fit_v1.toml','pen_contact_response_evaluation_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['train_pen_contrasts.py','evaluate_pen_contact_response_fit.py','diagnose_pen_contact_response_fit.py','review_pen_contact_response_fit.py','archive_pen_contact_response_fit.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[verify_file_pin(d['source_receipt']) for d in data.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/SHA;executed fresh trainer and historical data sources preserved separately.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    logs=[log]+[ROOT/'runs'/n for n in ['pen_contact_response_fit_console_v1.log','pen_contact_response_evaluation_console_v1.log','pen_contact_response_diagnosis_console_v1.log','pen_contact_response_review_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path,recovery_path]];allowed={p.parent for p in reports.values()}|{analysis,recovery_path.parent};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='contact_pen_four_arm_fit_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checkpoint_health=[87,87],evaluation_checks=[108,108],diagnosis_checks=[7,7],saved_review_checks=[152,152],
        physical_status=review['physical_status'],CHECK_forward={k:v['CHECK'] for k,v in ev['forward_normalized_MSE'].items()},CHECK_response=ev['aggregate'],TRAIN_response=diag['TRAIN_response'],local_excitation=diag['local_excitation'],loss_diagnostics=diag['loss_diagnostics'],
        fit_and_QA_wall_s=fit['fit_and_QA_wall_s'],total_runner_wall_s=fit['total_runner_wall_s'],actual_native_integrations=0,optimizer_steps=8016,live_optimizer_steps=8000,clone_only_optimizer_steps=16,Actor_updates=0,unit_tests=362,
        scope='Registered fresh four-arm final2000 completed. TRAIN responses improve under response supervision but CHECK forward/contact/AD gates fail. Rank4 local excitation and response-loss scale are descriptive leads,not causal proof. No policy/video/ego/generalization or GPU-speed benefit claim.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='contact_pen_four_arm_fit_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],historical_CLI_recovery=file_pin(recovery_path),prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),
        actual_native_integrations=0,optimizer_steps=8016,physical_qualification=review['physical_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
