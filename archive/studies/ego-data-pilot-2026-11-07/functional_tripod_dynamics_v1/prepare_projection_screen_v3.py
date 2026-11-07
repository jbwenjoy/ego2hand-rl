"""One new actual projection metadata screen; no native or NPZ-value reads."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent;P=H.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed pin '+r['path'])
    return Path(r['path'])
def write(p,j):
    with p.open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
def main():
    report=P/'functional_tripod_projection_v1/completed_run1/report.json'
    if pin(report)['sha256']!='74a9d659a86954a25df2ab7a43bc9428857dfabb5971ec67c611b94f5492622f':raise ValueError('Root fixed projection report')
    process=H/'projection_metadata_v3';process.mkdir()
    protocol=H/'protocol_projection_CPUFalse_v3.json';output=H/'projection_completed_run1'
    builder=H/'build_protocol_v3.py';runner=H/'run_functional_tripod_dynamics_v3.py'
    command=[sys.executable,'-X','utf8',str(builder),'--candidate-report',str(report),'--protocol',str(protocol),'--output',str(output)]
    b=subprocess.run(command,capture_output=True,text=True,encoding='utf8')
    write(process/'builder_terminal.json',dict(command=command,returncode=b.returncode,stdout=b.stdout,stderr=b.stderr,source=pin(builder)))
    if b.returncode:raise RuntimeError('Builder failed; no retry')
    cfg=read(protocol)
    for p in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:checked(p)
    command=[sys.executable,'-X','utf8',str(runner),'--protocol',str(protocol)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.perf_counter()
    with (process/'stdout.log').open('x',encoding='utf8') as so,(process/'stderr.log').open('x',encoding='utf8') as se:
        child=subprocess.Popen(command,stdout=so,stderr=se);rc=child.wait()
    terminal=dict(schema='functional_tripod_projection_metadata_terminal:3',pid=child.pid,returncode=rc,command=command,
        started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-clock,
        runner=pin(runner),protocol=pin(protocol),stdout=pin(process/'stdout.log'),stderr=pin(process/'stderr.log'),automatic_retry=False)
    write(process/'terminal.json',terminal)
    if rc:raise RuntimeError('Metadata failed; no retry')
    result=read(process/'stdout.log');write(process/'metadata_result.json',result)
    parent=read(report);checked(parent['protocol']);checked(parent['parent_report']);checked(parent['model_mjb'])
    candidate_reviews=[]
    for entry in parent['candidates']:
        c=read(checked(entry['candidate']));checked(entry['state_npz']);g=c['geometry']
        candidate_reviews.append(dict(candidate_id=c['candidate_id'],candidate=entry['candidate'],state_npz=entry['state_npz'],
            geometry_passed=g['passed'],fresh_final_evaluation=g['fresh_final_evaluation'],unknown=g['unknown'],
            phone_pair_count=g['phone_pair_count'],self_pair_count=g['self_pair_count'],table_complete=g['table_complete'],
            pad_bindings_count=len(c['pad_bindings']),minimum_saved_constraint_margin=min(g['constraint_values']),violations=g['violations']))
    expected=['exact_source__source_middle_anchor_nearest_phone__fixed_phone_q_projection','source1mm__index_material_toward_COM_30mm__fixed_phone_q_projection']
    enrolled=[x['candidate_id'] for x in result['screening'] if x['eligible_geometry']]
    checks=dict(all_source_and_candidate_state_pins_current=True,CPU_false_and_no_run_command=cfg['CPU_run_authorized'] is False and '--run' not in command,
        all_four_screen_records_retained=len(result['screening'])==len(candidate_reviews)==4,
        exactly_two_predeclared_geometric_candidates_enrolled=enrolled==expected and result['geometry_candidates']==2,
        maximum_twelve_cases_115200_steps=result['maximum_cases']==12 and result['maximum_steps']==115200,
        rejected_candidates_retained_with_reasons=sum(bool(x['reasons']) for x in result['screening'])==2,
        actual_dynamic_output_remains_absent=not output.exists(),terminal_exit0_no_stderr=rc==0 and terminal['stderr']['bytes']==0)
    out=dict(schema='functional_tripod_projection_actual_metadata_review:3',status='PASS_TWO_GEOMETRY_CANDIDATES_METADATA_ONLY_NO_DYNAMICS' if all(checks.values()) else 'FAILED_METADATA_REVIEW',
        passed=sum(checks.values()),total=len(checks),checks=checks,source=pin(__file__),protocol=pin(protocol),projection_report=pin(report),
        metadata_result=pin(process/'metadata_result.json'),terminal=pin(process/'terminal.json'),candidate_reviews=candidate_reviews,
        eligible_candidate_ids=enrolled,maximum_dynamic_cases=12,maximum_dynamic_steps=115200,
        counts=dict(metadata_preflight_attempted=1,metadata_preflight_returned=1,backend_imports=0,MJB_model_loads=0,NPZ_value_reads=0,native_forward=0,integration_steps=0,holding_evaluations=0),
        scope='Geometry enrollment only. All3 predeclared inverse-PD controls are still untested for native reachability/actual initial force. Root owns any future authorized3routes×2dt physical evaluation; no holding or force-support success is inferred.')
    write(process/'report.json',out)
    print(json.dumps(dict(protocol=pin(protocol),report=pin(process/'report.json'),passed=out['passed'],total=out['total'],enrolled=enrolled)))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
