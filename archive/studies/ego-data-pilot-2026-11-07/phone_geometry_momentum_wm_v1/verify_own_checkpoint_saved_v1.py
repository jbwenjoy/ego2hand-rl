"""One additional CPU read of our own checkpoint; no model/FW/optimizer/native."""
from pathlib import Path
import os,hashlib,json,math
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
H=Path(__file__).resolve().parent; R=H/'cpu_run1'; O=H/'root_saved_payload_review_v1'
def pin(p):
    p=Path(p).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
health=read(R/'health_0250.json'); result=read(R/'result.json'); terminal=read(H/'process_run1/terminal.json')
checkpoint=Path(health['checkpoint']['path']); assert pin(checkpoint)==health['checkpoint']==result['checkpoint']
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
p=read(H/'protocol_CPUTrue_run1.json'); rows=[json.loads(v) for v in (R/'updates.jsonl').read_text(encoding='utf8').splitlines()]
weights=ck['model_state']; opt=ck['optimizer_state']; norm=ck['normalizers']
checks=dict(terminal_exit0=terminal['returncode']==0 and terminal['stderr']['bytes']==0,
    actual_source_protocol_pins=terminal['protocol']==pin(H/'protocol_CPUTrue_run1.json')==ck['protocol'] and terminal['source']==p['sources']['trainer'],
    saved_artifact_iteration_phase=ck['iteration']==250 and ck['tag']=='AFTER_OPTIMIZER_250' and ck['gradient_snapshot']['tag']=='PRE_OPTIMIZER_250' and ck['gradient_snapshot']['weights_after_updates']==249,
    exact_model_shapes=[list(v.shape) for v in weights.values()]==[[128,366],[128],[128,128],[128],[222,128],[222]],
    all92126_parameters_CPU_finite=sum(v.numel() for v in weights.values())==92126 and all(v.device.type=='cpu' and torch.isfinite(v).all().item() for v in weights.values()),
    six_saved_Adam_states_step250=len(opt['state'])==6 and all(float(v['step'])==250 and torch.isfinite(v['exp_avg']).all().item() and torch.isfinite(v['exp_avg_sq']).all().item() for v in opt['state'].values()),
    fixed_recipe_fresh_initial_lineage=ck['recipe']==p['recipe'] and ck['recipe']['seed']==1731 and ck['resume_parent_checkpoint'] is None and ck['fresh_initial_checkpoint']==pin(R/'fresh_initial_state.pt'),
    saved_fit_normalizer_context_finite=all(torch.isfinite(v).all().item() for v in norm.values()) and all(torch.isfinite(v).all().item() for v in ck['physical_context'].values()),
    input_source_lineage_all_pins_unchanged=ck['inputs']==p['inputs'] and ck['sources']==p['sources'] and all(pin(v['path'])==v for v in [*p['inputs'].values(),*p['sources'].values()]),
    exactly250_finite_logged_updates=[v['iteration'] for v in rows]==list(range(1,251)) and all(math.isfinite(v['clip_pre_norm']) and v['update_seconds']>0 and all(math.isfinite(q) for q in v['losses'].values()) for v in rows),
    durable_log_prefix_pin=pin(R/'updates.jsonl')['sha256']==ck['log_prefix']['sha256'] and pin(R/'updates.jsonl')['bytes']==ck['log_prefix']['bytes'] and ck['log_prefix']['records']==250,
    worker_health_result_consistent=health['passed']==health['total']==15 and all(health['checks'].values()) and result['health']==pin(R/'health_0250.json') and result['iteration']==250 and health['process_stops_here'] is True,
    worker_stop_and_counts=result['status']=='STOPPED_AT_FIRST_HEALTHY_250_RESUMABLE_TARGET5000_NOT_COMPLETE' and result['counts']==dict(training_FW=250,backward=250,optimizer_steps=250,validation_FW=1,health_FW=2,NN_constructors=2,own_checkpoint_loads=1,native_FK_forward_steps=0) and not (R/'checkpoint_1000.pt').exists() and not (R/'checkpoint_5000.pt').exists(),
    no_GPU_context=not torch.cuda.is_initialized())
report=dict(schema='phone_geometry_momentum_root_saved_payload_review:1',checks=checks,passed=sum(checks.values()),total=len(checks),checkpoint=pin(checkpoint),
    source=pin(__file__),protocol=pin(H/'protocol_CPUTrue_run1.json'),health=pin(R/'health_0250.json'),terminal=pin(H/'process_run1/terminal.json'),
    additional_operation_counts=dict(own_CPU_checkpoint_payload_loads=1,NN_constructors=0,NN_FW=0,backward=0,optimizer_steps=0,native=0,GPU=0,target_NPZ_reads=0),
    losses_at_PRE_OPT1=rows[0]['losses'],losses_at_PRE_OPT250=rows[-1]['losses'],
    scope='Saved own CPU payload and file lineage review only; does not rerun training, restore optimizer stepping, test gradients against native, or qualify ego/holding performance.')
O.mkdir()
with (O/'report.json').open('x',encoding='utf8') as f: json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(O/'report.json'))))
assert all(checks.values())
