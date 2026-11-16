"""Independently review fixed CCD candidate evidence and retain confounded costs."""
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def read(path):return json.loads(Path(path).read_text())


def main():
    out=ROOT/'provenance/pen_ccd_recovery_candidate_v1';analysis=ROOT/'runs/pen_ccd_recovery_candidate_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh candidate archive required')
    prior_path=ROOT/'provenance/pen_retained_epa_diagnosis_v1/ledger.json';prior=read(prior_path);aliases=read(verify_file_pin(prior['source_resolutions']))
    for record in read(verify_file_pin(prior['verified_manifest']))['records']:
        if file_pin(verify_source_pin(record['observed'],aliases))!=record['resolved']:raise ValueError('Prior source resolution changed')
    names=dict(entry_trace='pen_ccd_entry_trace_v1/report.json',recovery='pen_ccd_recovery_variants_v1/report.json',predicate='pen_ccd_predicate_variants_v1/report.json',
        kernel_isolation='pen_ccd_kernel_isolation_v1/report.json',axis='pen_ccd_axis_retry_v1/report.json',multifinger='pen_ccd_multifinger_validation_v1/report.json',
        tolerance='pen_ccd_tolerance_validation_v1/report.json',face_normal='pen_ccd_face_normal_validation_v1/report.json')
    for key,name in list(names.items()):
        if key=='entry_trace':continue
        parent={'recovery':'pen_ccd_recovery_variants_registration_v1','predicate':'pen_ccd_predicate_variants_registration_v1','kernel_isolation':'pen_ccd_kernel_isolation_registration_v1',
            'axis':'pen_ccd_axis_retry_registration_v1','multifinger':'pen_ccd_multifinger_registration_v1','tolerance':'pen_ccd_tolerance_registration_v1','face_normal':'pen_ccd_face_normal_registration_v1'}[key]
        names[key+'_registration']=parent+'/registration.json'
    roots={k:file_pin(ROOT/'runs'/name) for k,name in names.items()};r={k:read(verify_file_pin(p)) for k,p in roots.items()}
    face=r['face_normal'];fr=read(verify_file_pin(face['results']));candidate=[v for v in fr if v['variant']=='FACE_NORMAL'];controls=[v for v in fr if v['variant']=='CONTROL']
    by_key=lambda v:(v['scene'],*v['geoms']);control={by_key(v):v for v in controls}
    if len(candidate)!=484 or len(control)!=484:raise ValueError('All484 paired candidate cases required')
    changed=collections.Counter();independent=[];maximum=collections.defaultdict(float)
    for row in candidate:
        old=control[by_key(row)];a=np.load(verify_file_pin(row['artifact']));b=np.load(verify_file_pin(old['artifact']));oracle=np.load(verify_file_pin(row['reference']))
        if row['native_distance_m']!=old['native_distance_m'] or a['native_witness'].tobytes()!=oracle['native_witness'].tobytes():raise ValueError('Oracle identity changed')
        for field in ['overflow','worldid','geom','dist','pos','frame','dim']:
            if a[field].shape!=b[field].shape or a[field].dtype!=b[field].dtype or a[field].tobytes()!=b[field].tobytes():changed[field]+=1
        count=len(a['dist']);distance=row['native_distance_m'];w=a['native_witness'];scope='OVERLAP' if distance < -1e-6 else 'SEPARATED' if distance>1e-6 else 'BOUNDARY'
        flags=dict(finite=all(np.isfinite(a[k]).all() for k in a.files),no_overflow=not bool(a['overflow'].any()),max_one=count<=1,
            overlap=scope!='OVERLAP' or count==1,separation=scope!='SEPARATED' or count==0)
        if count:
            frame=a['frame'][0].astype(np.float64);orth=float(np.linalg.norm(frame@frame.T-np.eye(3),ord=np.inf));det=float(np.linalg.det(frame))
            flags.update(identity=a['worldid'].tolist()==[31] and a['geom'].tolist()==[row['geoms']],frame=orth<3e-4 and det>0.9999)
            maximum['frame_entry_error']=max(maximum['frame_entry_error'],float(np.max(np.abs(frame@frame.T-np.eye(3)))))
            if scope=='OVERLAP':
                n=frame[0];v=(w[3:]-w[:3])/distance;angle=math.degrees(math.acos(max(-1.,min(1.,float(n@v)/(float(np.linalg.norm(n))*float(np.linalg.norm(v)))))))
                depth=abs(float(a['dist'][0])-distance);mid=float(np.linalg.norm(a['pos'][0].astype(np.float64)-(w[:3]+w[3:])/2))
                flags.update(depth=depth<=5e-6,normal=angle<=5.,midpoint=mid<=1e-4)
                maximum['distance_error_m']=max(maximum['distance_error_m'],depth);maximum['normal_angle_deg']=max(maximum['normal_angle_deg'],angle);maximum['midpoint_error_m']=max(maximum['midpoint_error_m'],mid)
        independent.append(dict(scene=row['scene'],geoms=row['geoms'],scope=scope,checks=flags))
    ids=read(verify_file_pin(r['kernel_isolation']['kernel_identities']))['records']
    tol=read(verify_file_pin(r['tolerance']['results']));multi=read(verify_file_pin(r['multifinger']['results']))
    tol_pass={str(t):sum(v['passed']==v['total'] for v in tol if v['tolerance']==t) for t in [1e-6,1e-7,1e-8]}
    trace_log=ROOT/'runs/pen_ccd_entry_trace_console_v1.log';text=trace_log.read_text()
    sections={name:text.split('TRACE_CASE_BEGIN '+name,1)[1].split('TRACE_CASE_END ',1)[0] for name in ['BASE24','BASE48','LOCAL24','PARENT24']}
    paths={key:ROOT/'runs'/name for key,name in dict(axis='pen_ccd_axis_retry_console_v1.log',multifinger='pen_ccd_multifinger_validation_console_v1.log',
        tolerance='pen_ccd_tolerance_validation_console_v1.log',face='pen_ccd_face_normal_console_v1.log').items()}
    retry_counts={key:path.read_text().count('TRACE_AXIS_RETRY world=') for key,path in paths.items()}
    counts=[r['entry_trace']['actual_pair_queries'],r['recovery']['actual_GPU_pair_queries'],r['predicate']['actual_GPU_pair_queries'],r['kernel_isolation']['actual_GPU_pair_queries'],
        r['axis']['actual_pair_queries'],r['multifinger']['actual_GPU_pair_queries'],r['tolerance']['actual_GPU_pair_queries'],r['face_normal']['actual_GPU_pair_queries']]
    checks=dict(face_capacity_distinguished='TRACE_ATTACH_FACE_CAPACITY nface=256 capacity=256' in sections['BASE48'] and 'TRACE_ATTACH_FACE_PROJECTION' not in sections['BASE48'],
        local_status12_observed='status=12' in sections['LOCAL24'],trace_reference4=all(v['reference_equal'] for v in r['entry_trace']['rows']),
        eight_factories_and_kernels_distinct=len(ids)==len({v['factory_name'] for v in ids})==len({v['kernel_object_id'] for v in ids})==8,
        fixed_query_counts=counts==[4,220,220,352,88,968,1452,968],all_new_native_and_Adam_zero=all(r[k]['actual_native_integrations']==r[k]['optimizer_steps']==0 for k in ['entry_trace','recovery','predicate','kernel_isolation','axis','multifinger','tolerance','face_normal']),
        fixed_reference_queries=r['recovery']['actual_native_distance_queries']==44 and r['multifinger']['actual_native_distance_queries']==484,
        extra_retry_work=retry_counts==dict(axis=19,multifinger=19,tolerance=39,face=38),candidate484_all_worker_checks=all(v['passed']==v['total'] for v in candidate),
        candidate484_all_independent_checks=all(all(v['checks'].values()) for v in independent),
        exact_scenario_classification=collections.Counter(v['scope'] for v in independent)==dict(OVERLAP=175,SEPARATED=308,BOUNDARY=1),
        control_prior_reproduced=all(v['prior_bytes_equal'] for v in controls),control_retains_three_failures=sum(v['passed']!=v['total'] for v in controls)==3,
        candidate_only_frame_changes=dict(changed)==dict(frame=175),tolerance_failure_retained=tol_pass=={'1e-06':481,'1e-07':482,'1e-08':474},
        tolerance_baseline_bytes=all(v['prior_bytes_equal'] for v in tol if v['tolerance']==1e-6),
        source_data_multiple_pairs=len({tuple(v['geoms']) for v in candidate})==11,
        original_pair_ids_unchanged=all(v['pairids']==[-1,-1] for v in read(verify_file_pin(r['multifinger']['pairs']))),
        test457='Ran 457 tests in 9.700s' in (ROOT/'runs/pen_ccd_recovery_tests457_v1.log').read_text() and '\nOK\n' in (ROOT/'runs/pen_ccd_recovery_tests457_v1.log').read_text())
    from mujoco_warp._src import warp_util
    cache_source=Path(warp_util.__file__).read_text()
    checks['cache_confound_source_retained']='hash(func.__name__)' in cache_source and 'key = tuple(_hash_arg(a) for a in args)' in cache_source
    if not all(checks.values()):raise ValueError('Independent candidate review differs: '+repr(checks))
    docs=[ROOT/'docs'/n for n in ['pen-ccd-recovery-candidate-results-2026-11-16.md','research-plan-v1.25.md']]
    tools=[Path(__file__)]+[ROOT/'tools'/n for n in ['trace_pen_ccd_entry.py','probe_pen_ccd_recovery_variants.py','probe_pen_ccd_predicate_variants.py','probe_pen_ccd_kernel_isolation.py',
        'probe_pen_ccd_axis_retry.py','validate_pen_ccd_multifinger.py','validate_pen_ccd_tolerances.py','validate_pen_ccd_face_normal.py']]
    analysis.mkdir();receipt=freeze_sources(tools+docs+[ROOT/'README.md'],analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(analysis/'source_receipt.json',receipt);write(analysis/'independent_contacts.json',independent)
    folders={Path(p['path']).parent for p in roots.values()};receipts=sorted({p/'source_receipt.json' for p in folders})+[analysis/'source_receipt.json']
    merged={json.dumps(v['original'],sort_keys=True):v for v in aliases['records']}
    for p in receipts:
        manifest=read(p);verify_source_snapshot(manifest,live=True)
        for record in manifest['records']:merged.setdefault(json.dumps(record['observed'],sort_keys=True),dict(original=record['observed'],archived=record['snapshot'],reason='Exact full-path/bytes/SHA CCD candidate evidence.'))
    aliases=dict(schema='explicit_source_relocations:1',records=list(merged.values()))
    logs=[ROOT/'runs'/n for n in ['pen_ccd_entry_trace_console_v1.log','pen_ccd_recovery_variants_console_v1.log','pen_ccd_predicate_variants_console_v1.log','pen_ccd_kernel_isolation_console_v1.log',
        'pen_ccd_axis_retry_console_v1.log','pen_ccd_multifinger_validation_console_v1.log','pen_ccd_tolerance_validation_console_v1.log','pen_ccd_face_normal_console_v1.log','pen_ccd_kernel_tests453_v1.log','pen_ccd_recovery_tests457_v1.log']]
    todo=list(roots.values())+[file_pin(p) for p in receipts+docs+tools+logs+[prior_path,analysis/'independent_contacts.json']]
    todo += [file_pin(p) for folder in folders for p in folder.glob('*.json')]
    allowed=folders|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,aliases);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo.extend(pins(read(resolved)))
    out.mkdir();write(out/'source_resolutions.json',aliases);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(roots=roots,checks=checks,passed=sum(checks.values()),total=len(checks),candidate_maximum=dict(maximum),only_changed_contact_fields=dict(changed),retry_helper_counts=retry_counts,
        actual_added_work=dict(GPU_pair_queries=sum(counts),native_distance_queries=528,extra_CCD_helpers=sum(retry_counts.values()),native_integrations=0,optimizer_steps=0,FK_builds=0),
        cache_confounded_algorithm_queries=440,cache_confounded_algorithm_comparison_admitted=False,
        candidate_static_fixture_checks_passed=True,production_backend_or_state_admitted=False,integrated_dynamics_admitted=False,current_live_training=False,
        scope='Static derived-data numerical candidate only. All old costs remain in prior chain;no task/ego/gradient/acceleration claim. Previous ambiguous face-degeneracy label superseded by explicit capacity trace.'))
    write(out/'ledger.json',dict(schema='CCD_recovery_candidate_ledger:1',prior_evidence=file_pin(prior_path),roots=roots,summary=file_pin(out/'summary.json'),documents=[file_pin(p) for p in docs],
        source_receipts=[file_pin(p) for p in receipts],source_resolutions=file_pin(out/'source_resolutions.json'),verified_manifest=file_pin(analysis/'verified_pins.json'),independent_contacts=file_pin(analysis/'independent_contacts.json'),
        verified_pins=len(verified),actual_native_integrations=0,optimizer_steps=0,GPU_pair_queries=4272,native_distance_queries=528,extra_CCD_helpers=115,current_live_training=False,
        candidate_static_admitted=True,production_backend_or_state_admitted=False))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified),maximum=dict(maximum))),flush=True)


if __name__=='__main__':main()
