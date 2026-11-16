"""Retain cached training,trained-weight fixtures and bounded timing evidence."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_cached_adoption_results';analysis=ROOT/'runs/pen_cached_adoption_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable adoption archive required')
    prior_path=ROOT/'provenance/pen_fixed_frame_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();counts=[];cursor=prior
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(features='pen_fixed_frame_qualification_v4',qualification='pen_cached_support_qualification_v1',training='pen_cached_support_training_v1',checkpoint_review='pen_cached_support_checkpoint_review_v1',comparison='pen_trained64_frame_comparison_v1',gradients='pen_fixed_frame_gradients_v3',review='pen_trained64_frame_review_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(features=[388,388],qualification=[75,75],training=[5,5],checkpoint_review=[18,18],comparison=[4,4],gradients=[9,9],review=[41,41])
    if any([data[k]['passed'],data[k]['total']]!=v for k,v in expected.items()):raise ValueError('Observed adoption checks differ')
    native=sum(d['actual_native_integrations'] for d in data.values());updates=sum(d['optimizer_steps'] for d in data.values());review=data['review']
    if (native,updates)!=(233472,448) or review['performance_status']!='OBSERVED_COMPONENT_TIMING_GAIN' or review['mechanism_status']!='EXACT_TESTED_FEATURES_GRADIENTS_AND_DECLARED_SOURCE_FORKS':raise ValueError('Keep work and bounded timing/mechanism status')
    initialization=json.loads(verify_file_pin(data['comparison']['initialization']).read_text())
    if initialization['full_model_arrays']!=404 or initialization['full_model_scalars']!=195 or not initialization['full_native_and_RNG_unchanged'] or initialization['old_native_optimizer_RNG_counters_loaded'] or initialization['normalizer_seed']['actor_count']!=129312:raise ValueError('Exact trained-weight seed and fresh-native context required')
    log=ROOT/'runs/pen_cached_adoption_tests379_v1.log';txt=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 379 tests in ' not in txt or '\nOK\n' not in txt:raise ValueError('Full379-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-cached-adoption-protocol-v1.md','pen-cached-adoption-results-2026-11-16.md','research-plan-v1.13.md']]
    configs=[ROOT/'configs'/n for n in ['pen_cached_support_training_v1.toml','pen_trained64_frame_comparison_v1.toml','pen_fixed_frame_qualification_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['archive_pen_cached_adoption.py','qualify_pen_support_pilot.py','train_pen_support_pilot.py','profile_ordinary_pen_ppo.py','qualify_pen_fixed_frames.py','check_pen_fixed_frame_gradients.py','review_pen_fixed_frames.py','review_pen_cached_adoption.py','archive_pen_source_ccd_evidence.py']]
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for path in receipts:
        source=json.loads(path.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/SHA;preserve optional cached configuration,explicit training5 normalizer seed and current-source GPU evidence.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    fixtures=[];engines=[]
    for pin,direct in [(data['features']['native_source'],True),(data['qualification']['common_initial'],False),(data['comparison']['fixture'],False)]:
        path=verify_file_pin(pin);fixtures.append(pin);state=torch.load(path,map_location='cpu',weights_only=False);state=state if direct else state['native'];verify_source_snapshot(state['sources'])
        for engine in state['engine_sources']:verify_source_pin(engine,resolutions)
        engines+=state['engine_sources']
    logs=[log]+[ROOT/'runs'/n for n in ['pen_fixed_frame_qualification_console_v4.log','pen_cached_support_qualification_console_v1.log','pen_cached_support_training_console_v1.log','pen_cached_support_checkpoint_review_console_v1.log','pen_trained64_frame_comparison_console_v1.log','pen_fixed_frame_gradients_console_v3.log','pen_trained64_frame_review_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+fixtures+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={};skip={'trace.json','torch_cpu_trace.json','python_calls.json','torch_cpu_operators.json','key_averages.json'}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and resolved.name not in skip and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_cached_adoption_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,initialization=initialization,timing=review['timing'],collect_throughput_ratio=review['collect_throughput_ratio'],collect_wall_reduction_fraction=review['collect_wall_reduction_fraction'],whole_phase_throughput_ratio=review['whole_phase_throughput_ratio'],performance_status=review['performance_status'],mechanism_status=review['mechanism_status'],default_mode='LIVE',new_training_config_mode='CACHED',main_policy_promoted=False,earlier_cohorts_pooled=False,unit_tests=379,actual_native_integrations=native,optimizer_steps=updates,scope='Qualified optional cached32 training/checkpoint integration and trained64 NN/norm-only W32 timing. New normalizer seed history is explicit;no old native/Adam/RNG/work counters imported. Not universal/full-training speed,long-trajectory equivalence,physical WM gradients,main-policy/task/ego success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_cached_adoption_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),actual_native_integrations=native,optimizer_steps=updates,performance_status=review['performance_status'],mechanism_status=review['mechanism_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
