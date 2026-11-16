"""Close strict attribution and the unintegrated full-rehash candidate separately."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import pstats
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/pen_backend_probe_profile_archive_v1';archive=ROOT/'provenance/pen_backend_probe_profile_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh profiling archive required')
    roots=dict(profile=ROOT/'runs/pen_backend_probe_profile_v1/report.json',review=ROOT/'runs/pen_backend_probe_profile_review_v1/report.json',
        verifier=ROOT/'runs/full_rehash_verifier_benchmark_v1/report.json',profile_registration=ROOT/'runs/pen_backend_probe_profile_registration_v1/registration.json',
        verifier_registration=ROOT/'runs/full_rehash_verifier_registration_v1/registration.json')
    r={k:read(p) for k,p in roots.items()};profile,review,bench=[r[k] for k in ['profile','review','verifier']]
    checks=dict(profile6=profile['passed']==profile['total']==6 and all(profile['checks'].values()),
        review6=review['passed']==review['total']==6 and all(review['checks'].values()) and review['profile']==file_pin(roots['profile']),
        original_physics_all40=review['valid_rows']==review['total_rows']==40 and review['row_flags']==[800,800],
        verifier26=bench['passed']==bench['total']==26 and all(bench['checks'].values()),
        exact320_native=profile['actual_native_integrations']==profile['attempted_native_integrations']==320 and profile['restore_calls']==186,
        zero_new_optimizer=all(x['optimizer_steps']==0 for x in [profile,review,bench]),
        no_epoch_or_default_change=not profile['source_epoch_used'] and not bench['default_verifier_changed'] and not bench['native_integration_tested'])
    for mode,entry in review['stages'].items():
        rows=[c for c in profile['cases'] if c['profile_mode']==mode];whole=sum(c['branch_complete_wall_s'] for c in rows)
        sums={k:sum(c['stages'][k] for c in rows) for k in rows[0]['stages']}
        checks[mode+'_stage_sums_exact']=entry['stages_s']==sums and entry['whole_branch_s']==whole and entry['restore_fraction']==sums['strict_restore_and_observation_s']/whole
    for key,pin in profile['profiles'].items():
        obj=pstats.Stats(str(verify_file_pin(pin)));label='restore_profile' if key=='strict_restore.prof' else 'branch_profile'
        expected={(a['file'],a['line'],a['function']):(a['calls'],a['self_s'],a['cumulative_s']) for a in profile[label]['all_functions']}
        actual={k:(v[1],v[2],v[3]) for k,v in obj.stats.items()}
        checks[key+'_serialized_profile_exact']=expected==actual
    checks['all108_restore_times']=len(profile['restore_measurements'])==108 and sum(x['wall_s'] for x in profile['restore_measurements'])==review['restore_timing']['sum_s']
    total_calls=0
    for catalog in bench['catalogs']:
        name=catalog['name'];n=len(catalog['pins']);total_calls+=14*n
        for mode in ['ORIGINAL','SINGLE_RESOLVE']:
            times=[x['wall_s'] for x in bench['timings'] if x['catalog']==name and x['mode']==mode]
            checks[name+'_'+mode+'_median_exact']=len(times)==5 and float(np.median(times))==bench['summary'][name]['median_wall_s'][mode]
        a,b=[bench['profiles'][name+'_'+mode] for mode in ['ORIGINAL','SINGLE_RESOLVE']]
        checks[name+'_full_rehash_count_parity']=a['resolve_calls']==2*n and b['resolve_calls']==n and a['full_file_reads']==b['full_file_reads']>0
    checks['all22848_complete_verifications']=total_calls==bench['complete_hash_verifications']==22848
    tests=ROOT/'runs/pen_full_rehash_tests519_v1.log';checks['full519_tests']='Ran 519 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[x['source_receipt'] for x in r.values()]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Profile archive discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-backend-probe-profile-results-2026-11-16.md','research-plan-v1.37.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tools/review_backend_probe_profile.py'],out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],stages=review['stages'],restore_timing=review['restore_timing'],
        profile_wall_s=profile['wall_s'],restore_calls=186,profile_native_integrations=320,independent_review_native_integrations=0,
        optimizer_steps=0,physical_rows=[40,40],row_flags=[800,800],verifier_microbenchmark=bench['summary'],
        verifier_complete_hash_calls=22848,default_verifier_changed=False,native_candidate_integration_qualified=False,
        scope='New-backend strict restore/recording attribution and full-rehash standalone candidate only. No source epoch,speedup of native collection/training,Actor/task/ego claim. Old source-epoch anomaly stays unresolved.')
    write(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_RAW_reward_gradient_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    todo=[file_pin(p) for p in list(roots.values())+docs+[out/'independent_review.json',out/'source_receipt.json',prior_path,tests]]+receipts
    allowed={p.parent for p in roots.values()}|{out}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='backend_probe_profile_results_ledger:1',prior_evidence=file_pin(prior_path),
        roots={k:file_pin(p) for k,p in roots.items()},summary=file_pin(archive/'summary.json'),review=file_pin(out/'independent_review.json'),
        source_receipts=receipts+[file_pin(out/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),profile_native_integrations=320,
        added_review_native_integrations=0,optimizer_steps=0,default_changed=False))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),physical_rows=summary['physical_rows'])),flush=True)


if __name__=='__main__':main()
