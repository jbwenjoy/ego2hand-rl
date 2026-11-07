"""Finalize pinned run protocol and execute metadata-only CLI, never --run."""
from pathlib import Path
import hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
false_path=H/'protocol_CPUFalse_v1.json';p=read(false_path)
assert p['CPU_run_authorized'] is False and p['training_or_GPU_authorized'] is False
proof=read(H/'source_preservation_v11.json')
assert proof['passed']==proof['total']==47 and all(proof['checks'].values())
review=H.parent/'hand2_source_mapping_dynamics_v11_review_v1/report.json'
r=read(review)
assert pin(H/'run_source_mapping_dynamics_v11.py')['sha256']=='fe0cb9074bc5d3df768aba7b8cef697efe41d22904fba2ae91672ca1901de1f2'
assert len(p['selections'])==2
p['CPU_run_authorized']=True
p['source_pins']['independent_source_review']=pin(review)
p['source_pins']['readiness_source']=pin(Path(__file__))
p['prepared_CPUFalse_protocol']=pin(false_path)
p['metric_semantics']={
    'state_phase':'POST integration integer tick+1; contact/energy/geometry are solver PRE integer tick',
    'tail_linear_velocity_RMS_m_s':'Original preserved metric is root world translational freejoint qvel, not offset COM velocity.',
    'tail_angular_velocity_RMS_rad_s':'Original preserved metric is the norm of freejoint angular qvel, in body coordinates; norm is frame invariant.',
    'optional_true_COM_velocity':'May be derived separately from saved qpos/qvel and body_ipos; does not change original gates.'}
p['selected_control_provenance']=[dict(branch=s['branch'],record=s['record'],control_origin=s['control_origin'],
    source_native_probe_ctrl_origin=read(s['record']['path']).get('initial_native_probe_ctrl_source'),
    ideal_flags_preserved=True) for s in p['selections']]
out=H/'protocol_CPUTrue_v1.json';write(out,p)
cmd=[sys.executable,str(H/'run_source_mapping_dynamics_v11.py'),'--protocol',str(out)]
completed=subprocess.run(cmd,capture_output=True,text=True)
stdout=H/'metadata_CPUTrue_v1.stdout.json';stderr=H/'metadata_CPUTrue_v1.stderr.log'
with stdout.open('x') as f:f.write(completed.stdout)
with stderr.open('x') as f:f.write(completed.stderr)
assert completed.returncode==0,(completed.returncode,completed.stderr)
result=json.loads(completed.stdout)
assert result['status']=='METADATA_VALID_NO_MODEL_OR_STEPS' and len(result['variants'])==2 and result['maximum_steps']==38400
receipt=dict(schema='source_mapping_dynamic_readiness:11',status='READY_SOURCE_AND_SAVED_PREPARATION_ONLY_NOT_EXECUTED',
    source=pin(Path(__file__)),runner=pin(H/'run_source_mapping_dynamics_v11.py'),
    CPUFalse_protocol=pin(false_path),CPUTrue_protocol=pin(out),
    prepared_states_receipt=pin(H/'prepared_states_v1/prepared_selections.json'),
    source_derivation=pin(H/'source_derivation_v11.json'),physics_preservation=pin(H/'source_preservation_v11.json'),
    independent_source_review=pin(review),metadata_stdout=pin(stdout),metadata_stderr=pin(stderr),
    metadata_command=cmd,metadata_exit_code=completed.returncode,selections=p['selections'],
    physical_execution_performed=False,new_MJB_model_FK_LP_forward_or_steps=0,
    scope='CPUTrue is the pinned Root-authorized prospective execution protocol. This readiness process never passes --run. Source/metadata approval is not holding success.')
write(H/'readiness_v11.json',receipt)
print(json.dumps({'readiness':pin(H/'readiness_v11.json'),'CPUTrue_protocol':pin(out),'metadata_exit':completed.returncode,'physical_execution_performed':False}))
