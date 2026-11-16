"""Retain ordinary CPU/GPU profiles,failed attempts and negative optimization."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_ordinary_performance_results';analysis=ROOT/'runs/pen_ordinary_performance_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable ordinary-performance evidence')
    prior_path=ROOT/'provenance/pen_contact_response_fit_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(failed='pen_ordinary_ppo_profile_v1',monitored='pen_ordinary_ppo_profile_v2',isolated='pen_ordinary_ppo_profile_v3',comparison='pen_empty_signal_comparison_v1',review='pen_ordinary_performance_review_v1',gpu='pen_ordinary_gpu_timeline_v1',gpu_review_initial='pen_ordinary_gpu_timeline_review_v1',gpu_review='pen_ordinary_gpu_timeline_review_v2')
    reports={k:ROOT/'runs'/v/('failure.json' if k=='failed' else 'report.json') for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(monitored=[4,4],isolated=[4,4],comparison=[4,4],review=[37,37],gpu=[7,7],gpu_review_initial=[18,19],gpu_review=[19,19])
    if any([data[k]['passed'],data[k]['total']]!=v for k,v in expected.items()) or sum(d['actual_native_integrations'] for d in data.values())!=450560 or sum(d['optimizer_steps'] for d in data.values())!=864:raise ValueError('Observed checks/work differ;retain failed scopes')
    review=data['review'];gpu=data['gpu_review']
    if review['performance_status']!='NO_CLEAN_SPEEDUP_OBSERVED' or review['equivalence_status']!='NOT_ESTABLISHED_OUTSIDE_OBSERVED_BASE_ENVELOPES' or review['adoption']!='UPSTREAM_DEFAULT_RETAINED' or gpu['forward_frequency']['total']!=698 or len(gpu['CCD_groups'])!=5:raise ValueError('Do not promote negative speed/equivalence result')
    log=ROOT/'runs/pen_empty_signal_tests366_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 366 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full366-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-ordinary-ppo-profile-protocol-v1.md','pen-ordinary-ppo-profile-protocol-v2.md','pen-empty-signal-comparison-protocol-v1.md','pen-ordinary-gpu-timeline-protocol-v1.md','pen-ordinary-performance-results-2026-11-16.md','research-plan-v1.11.md']]
    configs=[ROOT/'configs'/n for n in ['pen_ordinary_ppo_profile_v1.toml','pen_ordinary_ppo_profile_v2.toml','pen_empty_signal_comparison_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['profile_ordinary_pen_ppo.py','review_ordinary_pen_performance.py','capture_ordinary_pen_gpu_timeline.py','analyze_ordinary_pen_gpu_timeline.py','archive_ordinary_pen_performance.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/SHA;preserve failed logger,monitor-contaminated profiler,isolated/optimized source revisions and failed initial trace grouping.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[];fixtures=[]
    for key in ['failed','monitored','isolated','comparison']:
        path=reports[key].parent/'fixture.pt';fixtures.append(file_pin(path));state=torch.load(path,map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'])
        for pin in state['native']['engine_sources']:verify_source_pin(pin,resolutions)
        engines+=state['native']['engine_sources']
    logs=[log]+[ROOT/'runs'/n for n in ['pen_ordinary_ppo_profile_console_v1.log','pen_ordinary_ppo_profile_console_v2.log','pen_ordinary_ppo_profile_console_v3.log','pen_empty_signal_comparison_console_v1.log','pen_ordinary_performance_review_console_v1.log','pen_ordinary_gpu_timeline_console_v1.log','pen_ordinary_gpu_timeline_review_console_v1.log','pen_ordinary_gpu_timeline_review_console_v2.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+fixtures+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    skip_json={'trace.json','torch_cpu_trace.json','python_calls.json','torch_cpu_operators.json','key_averages.json'}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and resolved.name not in skip_json and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='ordinary_pen_performance_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,failed_attempt=data['failed'],timing=review['timing'],BASE_over_FAST_median_collect_ratio=review['BASE_over_FAST_median_collect_ratio'],
        performance_status=review['performance_status'],equivalence_status=review['equivalence_status'],adoption='UPSTREAM_DEFAULT_RETAINED',operators=review['operators'],isolated_python_calls=review['isolated_python_calls'],
        GPU_phases=gpu['phases'],GPU_forward_frequency=gpu['forward_frequency'],CCD_groups=gpu['CCD_groups'],CCD_sum_ms=gpu['CCD_sum_ms'],CCD_fraction_of_collect_kernel_sum=gpu['CCD_fraction_of_collect_kernel_sum'],
        actual_native_integrations=450560,optimizer_steps=864,main_policy_promoted=False,unit_tests=366,
        scope='Actual ordinary32-world profiling and an opt-in negative optimization trial. Query removal is not faster,repeat equivalence incomplete;CPU/GPU trace identifies next fixed-frame/reference/reset targets. Diagnostic PPO updates are not main-policy/ego/task or speedup results.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='ordinary_pen_performance_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),
        actual_native_integrations=450560,optimizer_steps=864,physical_qualification=review['equivalence_status'],performance_status=review['performance_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
