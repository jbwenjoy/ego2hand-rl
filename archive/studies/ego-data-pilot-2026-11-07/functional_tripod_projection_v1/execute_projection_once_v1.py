"""Root reviewed one four-state projection run, no retries or integration."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
out=H/'completed_run1';process=H/'process_run1';protocol=H/'protocol_run1.json'
assert not any(p.exists() for p in [out,process,protocol])
pr=dict(schema='functional_tripod_saved_geometry_plane_projection_protocol:1',
    sources={'runner':pin(H/'project_saved_candidates_v1.py'),'launcher':pin(__file__),
        'frozen_geometry_solver':pin(P/'functional_tripod_support_v1/solve_tripod_geometry_v2.py'),
        'plane_helper':pin(P/'functional_tripod_support_v1/refinement_v1/separating_plane_refinement_v1.py')},
    inputs={'original_geometry_protocol':pin(P/'functional_tripod_support_v1/protocol_CPUTrue_run1.json'),
        'original_geometry_report':pin(P/'functional_tripod_support_v1/completed_v1/report.json'),
        'original_geometry_terminal':pin(P/'functional_tripod_support_v1/process_geometry_run1/terminal.json'),
        'helper_source_contract':pin(P/'functional_tripod_support_v1/refinement_v1/SOURCE_CONTRACT_v1.json')},
    output=str(out),CPU_static_run_authorized=True,neural_training_or_integration_authorized=False,
    budget=dict(global_wall_s=240,entry_wall_s=45,final_wall_s=8,iterations_per_entry=20,full_queries_per_entry=80,heavy_queries_per_entry=3000),
    proposal=dict(joint_step_trust_rad=.02,interior_target_margin_m=2e-6,line_search_fractions=[1.,.5,.25,.125,.0625]),
    admission='All four previous failed candidates retained. Fixed each saved phone pose/q12..19. Actual geometry gate remains three pad[-.1,.2]mm and all nondesignated/self nonpenetration, full22/189/table, original35mm/30deg trust; fresh final required. Interior2um is a search target, not a new qualification gate.',
    derivative_scope='Only approximate current phone support planes guide q0..11. Unknown/far/self proposal matrix rows are unmodelled, never asserted zero physics derivatives. True full geometry validates every trial; no ideal force LP or force claim.',
    automatic_retry=False,unseen_or_real_ego_validation=False)
write(protocol,pr);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',str(H/'project_saved_candidates_v1.py'),'--protocol',str(protocol)]
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se)
    print(json.dumps(dict(pid=child.pid,output=str(out),process=str(process))),flush=True);rc=child.wait()
terminal=dict(schema='bounded_geometry_projection_terminal:1',pid=child.pid,returncode=rc,command=command,
    started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(H/'project_saved_candidates_v1.py'),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',terminal);print(json.dumps(dict(terminal=pin(process/'terminal.json'),returncode=rc)))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
