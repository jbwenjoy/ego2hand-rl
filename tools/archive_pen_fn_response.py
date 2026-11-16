"""Close the Fn6 study separately from the live long-control experiment."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot


def main():
    out=ROOT/'provenance/pen_fn_response_results';analysis=ROOT/'runs/pen_fn_response_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable Fn-response archive required')
    prior_path=ROOT/'provenance/pen_response_unit_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(fit='pen_fn_response_fit_v1',evaluation='pen_fn_response_evaluation_v1',diagnosis='pen_fn_response_diagnosis_v1',review='pen_fn_response_review_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()};fit=data['fit'];expected=dict(evaluation=[980,980],diagnosis=[11,11],review=[1243,1243])
    if any([data[k]['passed'],data[k]['total']]!=v for k,v in expected.items()) or sum(d['optimizer_steps'] for d in data.values())!=16032 or sum(d['actual_native_integrations'] for d in data.values())!=0:raise ValueError('Observed closed Fn checks/cost differ')
    health=[json.loads(verify_file_pin(p).read_text()) for p in fit['checkpoint_health'].values()]
    if fit['update']!=2000 or fit['live_models']!=8 or fit['clone_only_optimizer_steps']!=32 or fit['training_running'] or any(h['passed']!=h['total'] or h['total']!=187 for h in health) or not data['review']['control_replay_all_exact'] or data['review']['physical_status']!='NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION':raise ValueError('Retain replay/health and unqualified Actor status')
    log=ROOT/'runs/pen_fn_long_tests410_v1.log';txt=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 410 tests in ' not in txt or '\nOK\n' not in txt:raise ValueError('Full410-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-fn-response-fit-protocol-v1.md','pen-fn-response-results-2026-11-16.md','research-plan-v1.18.md','pen-long-control-protocol-v1.md']]
    configs=[ROOT/'configs'/n for n in ['pen_fn_response_fit_v1.toml','pen_fn_response_evaluation_v1.toml','pen_long_control_training_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['train_pen_contrasts.py','train_pen_support_pilot.py','evaluate_pen_contact_response_fit.py','diagnose_pen_contact_response_fit.py','review_pen_contact_response_fit.py','archive_pen_fn_response.py','archive_pen_source_ccd_evidence.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source,live=True)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact full path/bytes/SHA;explicit Fn6-addition tradeoff,all eight outputs and exact controls retained.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    logs=[log]+[ROOT/'runs'/n for n in ['pen_fn_response_tests407_v1.log','pen_fn_response_fit_dry_console_v1.log','pen_fn_response_fit_console_v1.log','pen_fn_response_evaluation_console_v1.log','pen_fn_response_diagnosis_console_v1.log','pen_fn_response_review_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]];allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_fn_response_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,checkpoint_health=[dict(update=h['update'],passed=h['passed'],total=h['total']) for h in health],forward=data['evaluation']['forward_normalized_MSE'],physical_response=data['evaluation']['aggregate'],TRAIN_response=data['diagnosis']['TRAIN_response'],TRAIN_component_curve=fit['response_components_curve'],control_replay_all_exact=True,control_replay=data['review']['control_replay'],physical_status=data['review']['physical_status'],actor_gradient_qualified=data['review']['actor_gradient_qualified'],live_optimizer_steps=16000,clone_only_optimizer_steps=32,optimizer_steps=16032,actual_native_integrations=0,Actor_updates=0,unit_tests=410,fit_and_QA_wall_s=fit['fit_and_QA_wall_s'],total_runner_wall_s=fit['total_runner_wall_s'],main_policy_promoted=False,scope='Closed Fn6-addition comparison only. Some contact gradients improve,but object/wrench/forward tradeoffs remain,no Actor admission. Ongoing long model-free control training is a separate cohort and is not counted here. Same inspected development data,not ego/final test.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_fn_response_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],configs=[file_pin(p) for p in configs],prior_evidence=file_pin(prior_path),prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),optimizer_steps=16032,actual_native_integrations=0,control_replay_all_exact=True,physical_status=summary['physical_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
