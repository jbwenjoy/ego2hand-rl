"""Archive final full20 data,explicit masks and matched TRAIN products."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot


def main():
    out=ROOT/'provenance/pen_fullrank_data_results';analysis=ROOT/'runs/pen_fullrank_data_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable full-rank data archive required')
    prior_path=ROOT/'provenance/pen_fullrank_registration_v1/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();counts=[];cursor=prior
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(train='pen_response_fullrank_train_v1',validation='pen_response_fullrank_validation_v1',train_preparation='pen_response_fullrank_train_preparation_v1',validation_preparation='pen_response_fullrank_validation_preparation_v1',review='pen_fullrank_data_review_v1',common='pen_fullrank_common_data_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()};expected=dict(train=[5,5],validation=[5,5],train_preparation=[81,81],validation_preparation=[51,51],review=[7312,7312],common=[9,9])
    if any([data[k]['passed'],data[k]['total']]!=v for k,v in expected.items()):raise ValueError('Observed full20 checks differ')
    native=sum(d['actual_native_integrations'] for d in data.values());updates=sum(d['optimizer_steps'] for d in data.values())
    if (native,updates)!=(51008,0) or data['common']['pair_counts']!={'rank4':189,'rank20':949}:raise ValueError('Actual work/pair counts differ')
    for k in ['train','validation']:
        if prior['source_banks'][k]!=data[k]['source_bank']:raise ValueError('Producer overlap must refer to the exact already-registered bank')
    if len(data['train_preparation']['rejected_rows'])!=22 or len(data['validation_preparation']['rejected_rows'])!=9 or data['train_preparation']['qualified_rows']!=3866 or data['validation_preparation']['qualified_rows']!=1935:raise ValueError('Preserve physical failures/masks')
    log=ROOT/'runs/pen_fullrank_dataset_tests390_v1.log';txt=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 390 tests in ' not in txt or '\nOK\n' not in txt:raise ValueError('Full390-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-response-fullrank-protocol-v1.md','pen-fullrank-data-review-protocol-v1.md','pen-fullrank-common-data-protocol-v1.md','pen-fullrank-fit-protocol-v1.md','pen-fullrank-data-results-2026-11-16.md','research-plan-v1.15.md']]
    configs=[ROOT/'configs'/n for n in ['pen_response_fullrank_train_v1.toml','pen_response_fullrank_validation_v1.toml','pen_fullrank_data_review_v1.toml','pen_fullrank_common_data_v1.toml','pen_fullrank_fit_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['archive_pen_fullrank_data.py','collect_trained_pen_responses.py','prepare_pen_train_coverage.py','review_pen_full_rank_data.py','prepare_pen_fullrank_common_data.py','archive_pen_source_ccd_evidence.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source,live=True)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact full path/bytes/SHA;final full20 raw/prepared/matched data,all masks and reserved-validation isolation.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    logs=[log]+[ROOT/'runs'/n for n in ['pen_response_fullrank_train_console_v1.log','pen_response_fullrank_validation_console_v1.log','pen_response_fullrank_train_preparation_console_v1.log','pen_response_fullrank_validation_preparation_console_v1.log','pen_fullrank_data_review_console_v1.log','pen_fullrank_common_data_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]];allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True);review=data['review'];coverage={k:dict(rows=c['rows'],qualified_rows=c['qualified_rows'],groups=c['groups'],qualified_groups=c['qualified_groups'],failed_gate_counts=c['failed_gate_counts'],physical_screen_direction_counts=c['physical_screen_direction_counts'],both_amplitude_ranks=[r['declared']['rank'] for r in c['source_ranks'] if r['epsilon']=='BOTH']) for k,c in review['cohorts'].items()}
    summary=dict(schema='pen_fullrank_data_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,coverage=coverage,pair_counts=data['common']['pair_counts'],actual_native_integrations=native,producer_overlap_with_prior_registration=4352,incremental_native_integrations_since_registration=46656,optimizer_steps=0,normalizer_fits=2,unit_tests=390,validation_rows_used_for_TRAIN=0,main_policy_promoted=False,training_fit_started=False,scope='Completed full20 controlled physical-response cohort;31 failed impulse rows retained/masked,physical-mask ranks17–20,matched rank4/rank20 pair subsets share one point bank. Same original reference/fixed actors;not model-gradient/task/ego/final-test or speed qualification. Cohort total includes4352 producers already registered.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_fullrank_data_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],configs=[file_pin(p) for p in configs],prior_evidence=file_pin(prior_path),prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),actual_native_integrations=native,producer_overlap_with_prior_registration=4352,incremental_native_integrations_since_registration=46656,optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
