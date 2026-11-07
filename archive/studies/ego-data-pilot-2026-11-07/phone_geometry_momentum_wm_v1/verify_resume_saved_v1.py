"""Saved resume payload/log/readback verification, zero NN execution."""
from pathlib import Path
import os,hashlib,json,math
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
H=Path(__file__).resolve().parent;R=H/'cpu_resume1';O=H/'resume_saved_review_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
protocol=read(H/'protocol_CPUTrue_resume1.json');parent=read(H/'protocol_CPUTrue_run1.json');result=read(R/'result.json');terminal=read(H/'process_resume1/terminal.json')
healths={i:read(R/f'health_{i:04d}.json') for i in [1000,5000]};ck=torch.load(R/'checkpoint_5000.pt',map_location='cpu',weights_only=False)
rows=[json.loads(v) for v in (R/'updates.jsonl').read_text(encoding='utf8').splitlines()]
rawreadbacks={}
for i in [1000,5000]:
    hp=healths[i];assert pin(hp['saved_readback_evidence']['path'])==hp['saved_readback_evidence']
    with np.load(hp['saved_readback_evidence']['path'],allow_pickle=False) as z:
        rawreadbacks[str(i)]=bool(z['before_reload_raw222_FP32'].tobytes()==z['after_reload_raw222_FP32'].tobytes() and z['Torch_RNG_before'].tobytes()==z['Torch_RNG_after'].tobytes())
logpin=pin(R/'updates.jsonl');lp=ck['log_prefix'];states=ck['optimizer_state']['state'];names=ck['ordered_parameter_names'];weights=ck['model_state']
checks=dict(terminal_exit0_no_stderr=terminal['returncode']==0 and terminal['stderr']['bytes']==0,
    both_healthy_pinned_checkpoints=all(h['iteration']==i and h['passed']==h['total']==15 and all(h['checks'].values()) and pin(h['checkpoint']['path'])==h['checkpoint'] for i,h in healths.items()),
    fixed_final5000_selected_in_advance=result['iteration']==5000 and result['checkpoint']==pin(R/'checkpoint_5000.pt')==healths[5000]['checkpoint'] and result['status']=='COMPLETED_FIXED5000_UPDATES_NOT_POLICY_QUALIFICATION',
    same_recipe_source_input_parent=ck['recipe']==protocol['recipe']==parent['recipe'] and ck['sources']==protocol['sources']==parent['sources'] and ck['inputs']==protocol['inputs']==parent['inputs'] and ck['resume_parent_checkpoint']==protocol['resume_checkpoint']==pin(H/'cpu_run1/checkpoint_0250.pt'),
    healthy_parent_binding=protocol['resume_health']==pin(H/'cpu_run1/health_0250.json'),
    payload_phase5000=ck['iteration']==5000 and ck['tag']=='AFTER_OPTIMIZER_5000' and ck['gradient_snapshot']['tag']=='PRE_OPTIMIZER_5000' and ck['gradient_snapshot']['weights_after_updates']==4999,
    finite92126_CPU_parameters=len(names)==len(weights)==6 and sum(v.numel() for v in weights.values())==92126 and all(v.device.type=='cpu' and torch.isfinite(v).all().item() for v in weights.values()),
    Adam6_step5000=len(states)==6 and all(float(v['step'])==5000 and torch.isfinite(v['exp_avg']).all().item() and torch.isfinite(v['exp_avg_sq']).all().item() for v in states.values()),
    finite_PRE5000_gradients=all(torch.isfinite(v).all().item() for group in ['pre_clip','post_clip'] for v in ck['gradient_snapshot'][group].values()),
    finite_FIT_only_normalizers_and_context=all(torch.isfinite(v).all().item() for v in ck['normalizers'].values()) and all(torch.isfinite(v).all().item() for v in ck['physical_context'].values()) and ck['normalizer_scope']=='2118FIT absolute/1434FIT response only',
    log_exact251_to5000=[v['iteration'] for v in rows]==list(range(251,5001)) and all(v['update_seconds']>0 and math.isfinite(v['clip_pre_norm']) and all(math.isfinite(q) for q in v['losses'].values()) for v in rows),
    final_durable_log_prefix=lp['sha256']==logpin['sha256'] and lp['bytes']==logpin['bytes'] and lp['records']==4750 and lp['global_iteration_range']==[251,5000],
    both_raw_prediction_RNG_saved_readbacks=all(rawreadbacks.values()),
    final_worker_mixed_scope_counts=result['counts']==dict(training_FW=5000,backward=5000,optimizer_steps=5000,validation_FW=2,health_FW=4,NN_constructors=3,own_checkpoint_loads=3,native_FK_forward_steps=0),
    no_GPU_initialized=not torch.cuda.is_initialized())
report=dict(schema='phone_geometry_momentum_resume_saved_payload_review:1',checks=checks,passed=sum(checks.values()),total=len(checks),
    source=pin(__file__),checkpoint=pin(R/'checkpoint_5000.pt'),protocol=pin(H/'protocol_CPUTrue_resume1.json'),terminal=pin(H/'process_resume1/terminal.json'),result=pin(R/'result.json'),
    health={str(i):pin(R/f'health_{i:04d}.json') for i in healths},raw_readbacks_exact=rawreadbacks,
    actual_this_resume_worker=dict(training_FW=4750,backward=4750,optimizer_steps=4750,health_FW=4,validation_FW=2,NN_constructors=3,own_checkpoint_loads=3,native=0,GPU=0),
    lineage_training_counts=dict(training_FW=5000,backward=5000,optimizer_steps=5000),
    additional_saved_review_counts=dict(own_CPU_checkpoint_loads=1,health_NPZ_reads=2,target_NPZ_reads=0,NN_FW=0,NN_constructors=0,native=0),
    first_record_PRE251=rows[0]['losses'],last_record_PRE5000=rows[-1]['losses'],
    elapsed_external_resume_s=terminal['elapsed_s'],elapsed_update_sum_s=sum(v['update_seconds'] for v in rows),
    scope='Saved payload/log/2readbacks. No fresh optimizer replay or native gradient/ego benefit. Resume fresh_initial_state is the pre-load reconstruction template; actual start is the pinned own250 parent.')
O.mkdir()
with (O/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(O/'report.json'))));assert all(checks.values())
