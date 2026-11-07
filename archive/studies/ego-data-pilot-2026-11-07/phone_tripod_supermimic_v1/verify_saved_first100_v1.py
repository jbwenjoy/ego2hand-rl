"""Independent own CPU payload/file review after worker exits; zero NN/FW."""
from pathlib import Path
import os,hashlib,json,math
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
H=Path(__file__).resolve().parent;R=H/'gpu_run1';O=H/'root_saved_first100_review_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
protocol=read(H/'protocol_CPU_GPU_True_run1.json');health=read(R/'health_0100.json');result=read(R/'result.json');terminal=read(H/'process_run1/terminal.json')
ck=torch.load(R/'paired_checkpoint_0100.pt',map_location='cpu',weights_only=False)
initial=torch.load(R/'initial_modules_not_learning_checkpoint.pt',map_location='cpu',weights_only=False)
arms=ck['arms'];modes=['A1_tracking','A2_engineered_material_proxy'];opt=ck['optimizers'];grads=ck['gradient_snapshots']
rows=[json.loads(line) for line in (R/'paired_updates.jsonl').read_text(encoding='utf8').splitlines()];lp=ck['log_prefix'];b=(R/'paired_updates.jsonl').read_bytes()
changes={}
for mode in modes:
    changes[mode]={}
    for role in ['actor','critic','target_critic']:
        state=arms[mode][role];prev=initial['arms'][mode][role]
        changes[mode][role]=float(torch.sqrt(sum((v.double()-prev[k].double()).square().sum() for k,v in state.items() if k.startswith('net.'))))
with np.load(R/'health_readback_0100.npz',allow_pickle=False) as z:
    pairs={key[:-7]:z[key].tobytes()==z[key[:-7]+'_after'].tobytes() for key in z.files if key.endswith('_before')}
allstate=[ck['shared_WM'],*[a[role] for a in arms.values() for role in ['actor','critic','target_critic']]]
checks=dict(worker_exit0_empty_stderr=terminal['returncode']==0 and terminal['stderr']['bytes']==0,
    exact_final_protocol_source_dataset=ck['protocol']==pin(H/'protocol_CPU_GPU_True_run1.json') and ck['recipe']==protocol['recipe'] and ck['sources']==protocol['sources'] and ck['inputs']==protocol['inputs'] and all(pin(v['path'])==v for v in [*protocol['sources'].values(),*protocol['inputs'].values()]),
    proper_first_paired_checkpoint_schema=ck['schema']=='phone_tripod_sharedWM_A1A2_training_checkpoint:1' and ck['paired_round']==100 and ck['shared_WM_updates']==200,
    worker_health_result_same_checkpoint=health['checkpoint']==result['checkpoint']==pin(R/'paired_checkpoint_0100.pt') and health['passed']==health['total'] and all(health['checks'].values()),
    all_weights_and_norm_buffers_finite=all(torch.isfinite(v).all().item() for state in allstate for v in state.values()),
    matched_initial_policies_and_critics=equal(initial['arms'][modes[0]]['actor'],initial['arms'][modes[1]]['actor']) and equal(initial['arms'][modes[0]]['critic'],initial['arms'][modes[1]]['critic']),
    Actor_Critic_weight_updates_observed=all(changes[mode][role]>0 for mode in modes for role in ['actor','critic']),
    fixed_empirical_norms_still_matched=all(equal({k:v for k,v in arms[mode][role].items() if k.startswith('norm.')},{k:v for k,v in initial['arms'][mode][role].items() if k.startswith('norm.')}) for mode in modes for role in ['actor','critic','target_critic']),
    five_Adam_states_correct_steps=set(opt)=={'WM',*[m+'_'+r for m in modes for r in ['Actor','Critic']]} and all(len(v['state'])==(6 if key=='WM' else 8 if key.endswith('Actor') else 10) and all(float(s['step'])==(200 if key=='WM' else 100) and torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item() for s in v['state'].values()) for key,v in opt.items()),
    finite_PRE_update_gradient_snapshots=set(grads)==set(opt) and all(torch.isfinite(v).all().item() for g in grads.values() for side in ['pre_clip','post_clip'] for v in g[side].values()),
    gradient_snapshot_phases=grads['WM']['tag']=='PRE_SHARED_WM_UPDATE_200' and all(grads[mode+'_'+role]['tag']==f'PRE_{mode}_{role.upper()}_ROUND_100' for mode in modes for role in ['Actor','Critic']),
    exact100_round_log_and_durable_prefix=[v['paired_round'] for v in rows]==list(range(1,101)) and [v['shared_WM_updates'] for v in rows]==list(range(101,201)) and lp['bytes']==len(b) and lp['sha256']==hashlib.sha256(b).hexdigest() and lp['round_range']==[1,100] and lp['records']==100,
    finite_logged_losses=all(math.isfinite(v['WM_loss']) and v['round_seconds']>0 and all(math.isfinite(x) for p in v['modes'].values() for x in p.values()) for v in rows),
    five_raw_prediction_readbacks=len(pairs)==5 and all(pairs.values()),
    GPU_run_and_positive_peak=health['device']=='cuda:0' and health['GPU_peak_allocated_bytes']>0 and protocol['GPU_run_authorized'] is True,
    first100_exit_not_completed1000=result['status']=='STOPPED_AT_FIRST_HEALTHY_PAIRED100_RESUMABLE_TARGET1000' and health['process_stops_here'] is True and not (R/'paired_checkpoint_0500.pt').exists() and not (R/'paired_checkpoint_1000.pt').exists(),
    remaining_900_round_ETA=math.isclose(health['estimated_remaining_training_s'],health['last25_mean_round_seconds']*900,rel_tol=1e-12),
    review_CPU_only_no_new_GPU=not torch.cuda.is_initialized())
report=dict(schema='phone_tripod_matched_first100_root_saved_review:1',checks=checks,passed=sum(checks.values()),total=len(checks),checkpoint=pin(R/'paired_checkpoint_0100.pt'),health=pin(R/'health_0100.json'),protocol=pin(H/'protocol_CPU_GPU_True_run1.json'),terminal=pin(H/'process_run1/terminal.json'),source=pin(__file__),weight_L2_changes_from_initial=changes,raw_prediction_pairs=pairs,
    root_additional_operations=dict(own_CPU_checkpoint_payload_loads=2,saved_health_NPZ_reads=1,NN_constructors=0,NN_FW=0,backward=0,optimizer_steps=0,native=0,GPU=0),
    scope='Own saved payload/log/health artifact review. No new policy/native evaluation, no optimizer continuation replay or measured real-ego/holding/CHORD benefit.')
O.mkdir()
with (O/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(O/'report.json'))));assert all(checks.values())
