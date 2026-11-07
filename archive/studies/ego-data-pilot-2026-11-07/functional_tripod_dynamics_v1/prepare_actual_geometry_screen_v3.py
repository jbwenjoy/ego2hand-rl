"""One actual-JSON metadata screen, no backend/model/array-value execution."""
from pathlib import Path
import datetime,hashlib,json,math,subprocess,sys,time
H=Path(__file__).resolve().parent;P=H.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Pin changed '+r['path'])
    return Path(r['path'])
def write(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    out=H/'actual_geometry_metadata_v3';out.mkdir()
    geometry=P/'functional_tripod_support_v1/completed_v1/report.json'
    protocol=H/'protocol_actual_geometry_CPUFalse_v3.json';declared_output=H/'actual_geometry_screen_v3'
    builder=H/'build_protocol_v3.py';runner=H/'run_functional_tripod_dynamics_v3.py'
    command=[sys.executable,'-X','utf8',str(builder),'--candidate-report',str(geometry),'--protocol',str(protocol),'--output',str(declared_output)]
    b=subprocess.run(command,capture_output=True,text=True,encoding='utf8')
    write(out/'builder_terminal.json',dict(command=command,returncode=b.returncode,stdout=b.stdout,stderr=b.stderr,source=pin(builder)))
    if b.returncode:raise RuntimeError('Builder failed; preserved without retry')
    c=read(protocol)
    if c['CPU_run_authorized'] is not False:raise ValueError('Metadata only')
    for q in [*c['inputs'].values(),*c['source_pins'].values()]:checked(q)
    cmd=[sys.executable,'-X','utf8',str(runner),'--protocol',str(protocol)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.perf_counter()
    with (out/'stdout.log').open('x',encoding='utf8') as stdout,(out/'stderr.log').open('x',encoding='utf8') as stderr:
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);code=child.wait()
    terminal=dict(schema='functional_tripod_actual_metadata_terminal:3',command=cmd,pid=child.pid,returncode=code,
        started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-clock,
        source=pin(runner),protocol=pin(protocol),stdout=pin(out/'stdout.log'),stderr=pin(out/'stderr.log'),automatic_retry=False)
    write(out/'terminal.json',terminal)
    if code:raise RuntimeError('Metadata failed; no retry')
    result=json.loads((out/'stdout.log').read_text());write(out/'metadata_result.json',result)
    g=read(geometry);checked(g['protocol']);checked(g['model_mjb']);reviews=[]
    for e in g['candidates']:
        d=read(checked(e['candidate']));checked(e['state_npz']);gg=d['geometry'];vals=gg['constraint_values'];labels=gg['constraint_labels']
        if len(vals)!=216 or len(labels)!=216 or not all(type(v) in (int,float) and math.isfinite(v) for v in vals):raise ValueError('All saved constraints')
        negative=[dict(label=k,margin=v) for k,v in zip(labels,vals) if v<0]
        if negative!=gg['violations']:raise ValueError('Saved violated-constraint list inconsistent')
        reviews.append(dict(candidate_id=d['candidate_id'],candidate=e['candidate'],state_npz=e['state_npz'],
            geometry_passed=gg['passed'],fresh_final_evaluation=gg['fresh_final_evaluation'],unknown=gg['unknown'],
            pad_bindings_count=len(d['pad_bindings']),phone_pair_count=gg['phone_pair_count'],self_pair_count=gg['self_pair_count'],
            table_complete=gg['table_complete'],violations=negative,minimum_saved_margin=min(vals),
            stages=d['stages'],scope='Saved source geometry margins only, no independent shape query/reconstruction.'))
    checks=dict(source_and_input_output_pins_current=True,CPU_authorized_false=True,
        all_four_failed_candidates_retained=len(reviews)==4 and len(result['screening'])==4,
        actual_geometry_none_enrolled=result['geometry_candidates']==0 and all(not v['eligible_geometry'] for v in result['screening']) and all(v['geometry_passed'] is False for v in reviews),
        complete_fresh_geometry_preserved=all(v['fresh_final_evaluation'] is True and v['unknown']==[] and v['phone_pair_count']==22 and v['self_pair_count']==189 and v['table_complete'] is True and v['pad_bindings_count']==3 for v in reviews),
        declared_dynamic_output_not_created=not declared_output.exists(),
        metadata_returned_zero_budget=result['maximum_cases']==0 and result['maximum_steps']==0,
        no_native_command_executed='--run' not in cmd and code==0 and terminal['stderr']['bytes']==0)
    report=dict(schema='functional_tripod_actual_geometry_metadata_review:3',status='PASS_ZERO_ELIGIBLE_METADATA_ONLY_NO_HOLDING_EVALUATION' if all(checks.values()) else 'FAILED_METADATA_REVIEW',
        passed=sum(checks.values()),total=len(checks),checks=checks,source=pin(__file__),protocol=pin(protocol),geometry_report=pin(geometry),
        runner=pin(runner),metadata_result=pin(out/'metadata_result.json'),terminal=pin(out/'terminal.json'),candidate_reviews=reviews,
        counts=dict(metadata_preflight_attempted=1,metadata_preflight_returned=1,eligible_candidates=0,backend_imports=0,MJB_model_loads=0,
            NPZ_value_reads=0,native_forward=0,integration_steps=0,holding_evaluations=0),
        scope='Actual geometry search failed all4 under its frozen constraints. No dynamic --run empty experiment and no holding result. This does not prove that other tripod configurations are infeasible. New candidate protocols must be separate.')
    write(out/'report.json',report)
    print(json.dumps(dict(report=pin(out/'report.json'),protocol=pin(protocol),passed=report['passed'],total=report['total'],counts=report['counts'])))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
