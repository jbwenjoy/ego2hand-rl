"""Fresh CUDA response baseline; unconditionally stop at first healthy checkpoint.

No Actor, native simulator or geometry runs in the update loop. CHECK/RESPONSE
are read-only diagnostics and never fit scales, sampled updates or selection.
"""
from pathlib import Path
import argparse,json,os,random,subprocess,sys,time,tomllib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.compound_response import CompoundResponseWM,fit_scales,INPUT_SCHEMA,OUTPUT_SCHEMA
from supermimic_research.learning.world_model import capture_rng,restore_rng,atomic_checkpoint,raw_tree_equal
from supermimic_research.learning.checkpoint import _finite_tree,_adam_valid


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def gpu_snapshot():
    try:return subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None


def main(args):
    import torch
    config_path=args.config.resolve();config=tomllib.loads(config_path.read_text());packed_path=(config_path.parent/config['packed_report']).resolve();packed=json.loads(packed_path.read_text())
    if (packed['passed'],packed['total'])!=(7,7) or config['input_schema']!=INPUT_SCHEMA or config['target_schema']!=OUTPUT_SCHEMA or config['fit_role']!='TRAIN' or config['check_roles']!=['CHECK','RESPONSE']:raise ValueError('Reviewed explicit roles and new input/output schemas')
    if config['first_checkpoint_update']!=10 or not config['stop_at_first_healthy_checkpoint'] or config['planned_updates']<=10 or config['allow_cpu_fallback']:raise ValueError('Standing first healthy checkpoint stop rule')
    raw={role:np.load(verify_file_pin(pin)) for role,pin in packed['outputs'].items()}
    if not args.run:
        print(json.dumps(dict(config=file_pin(config_path),packed=file_pin(packed_path),rows={r:len(z['x']) for r,z in raw.items()},training_started=False)));return
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    device=torch.device(config['device']);seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scales=fit_scales(raw['TRAIN']['x'],raw['TRAIN']['y']);model=CompoundResponseWM(scales,width=config['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=True)
    cache={role:{k:torch.as_tensor(z[k],dtype=torch.float32,device=device) for k in ['x','y']} for role,z in raw.items()}
    sampler=torch.Generator(device=device);sampler.manual_seed(seed+1);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    sources=[file_pin(ROOT/p) for p in ['tools/train_mug_response_first10.py','src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_response_CUDA_first_checkpoint_intent:1',sources=sources,config=file_pin(config_path),packed=file_pin(packed_path),
        dataset_pins=packed['outputs'],updates_executed_before_stop=10,planned_updates=config['planned_updates'],batch_size=config['batch_size'],device=str(device),
        model='Fresh102->512x3->87 response MLP; no legacy checkpoint/Actor/policy',dataset_GPU_resident=True,normalizers_fitted_from='TRAIN_ONLY',native_or_geometry_calls_in_updates=0,
        per_update_host_scalar_checks=0,health_checks_at_checkpoint=True,neural_stop_rule='Stop immediately after healthy first10 readback/ETA; application Goal pause by caller',nvidia_before=gpu_snapshot()))
    timers=[];losses=[];gradient_snapshot=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    for update in range(1,11):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        indices=torch.randint(len(raw['TRAIN']['x']),(config['batch_size'],),generator=sampler,device=device)
        optimizer.zero_grad(set_to_none=True);loss=model.loss(cache['TRAIN']['x'][indices],cache['TRAIN']['y'][indices]);loss.backward()
        before={k:p.grad.detach().clone() for k,p in model.named_parameters()} if update==10 else None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=False)
        if update==10:gradient_snapshot=dict(pre_clip=before,post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone())
        optimizer.step();end.record();timers.append((begin,end));losses.append(loss.detach())
    torch.cuda.synchronize(device);training_wall=time.perf_counter()-start;milliseconds=[a.elapsed_time(b) for a,b in timers];loss_values=torch.stack(losses).cpu().tolist()
    payload=dict(schema='compound_H20_response_checkpoint:1',update=10,planned_updates=config['planned_updates'],config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
        rng=capture_rng(device=device),sampler=sampler.get_state(),normalizer_fit_role='TRAIN',scales=scales,dataset_pins=packed['outputs'],packed=file_pin(packed_path),
        sources=sources,gradient_snapshot=gradient_snapshot,device=str(device),torch_version=torch.__version__,model_input_schema=INPUT_SCHEMA,model_output_schema=OUTPUT_SCHEMA)
    checkpoint=out/'checkpoint_0010.pt';atomic_checkpoint(checkpoint,payload)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    restored=CompoundResponseWM(saved['scales'],width=saved['config']['width']).to(device);restored.load_state_dict(saved['model'])
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=True);restored_optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'],device=device);rng_equal=raw_tree_equal(saved['rng'],capture_rng(device=device));restored_sampler=torch.Generator(device=device);restored_sampler.set_state(saved['sampler'])
    a=torch.randint(len(raw['TRAIN']['x']),(config['batch_size'],),generator=sampler,device=device);b=torch.randint(len(raw['TRAIN']['x']),(config['batch_size'],),generator=restored_sampler,device=device)
    sampler_equal=bool(torch.equal(a,b));sampler.set_state(saved['sampler'])
    with torch.no_grad():
        probe=model.prediction(cache['CHECK']['x']);clone=restored.prediction(cache['CHECK']['x']);diagnostic_error=(probe-cache['CHECK']['y'])/model.y_scale
        check_metrics={name:float(diagnostic_error[:,lo:hi].square().mean()) for name,lo,hi in [('delta52_NMSE',0,52),('next_warm26_NMSE',52,78),('wrench6_NMSE',78,84),('Fn3_NMSE',84,87)]}
        response_prediction=model.prediction(cache['RESPONSE']['x']);response_target=cache['RESPONSE']['y'];pairs=[]
        rr=raw['RESPONSE']
        for key in sorted(set(zip(rr['k'].tolist(),rr['dt'].tolist(),rr['joint'].tolist()))):
            matching=np.flatnonzero((rr['k']==key[0])&(rr['dt']==key[1])&(rr['joint']==key[2]));minus=int(matching[rr['sign'][matching]==-1][0]);plus=int(matching[rr['sign'][matching]==1][0]);pairs.append((minus,plus))
        lo_idx=torch.tensor([p[0] for p in pairs],device=device);hi_idx=torch.tensor([p[1] for p in pairs],device=device)
        actual=(response_target[hi_idx]-response_target[lo_idx])/model.y_scale;predicted=(response_prediction[hi_idx]-response_prediction[lo_idx])/model.y_scale
        response_metrics={name:float((predicted[:,lo:hi]-actual[:,lo:hi]).square().sum()/actual[:,lo:hi].square().sum()) for name,lo,hi in [('delta52_response_vs_zero_NMSE',0,52),('contact9_response_vs_zero_NMSE',78,87)]}
    checks=dict(schema_and_cursor=saved['schema']=='compound_H20_response_checkpoint:1' and saved['update']==10,
        complete_model_raw_equal=raw_tree_equal(model.state_dict(),restored.state_dict()),complete_optimizer_raw_equal=raw_tree_equal(optimizer.state_dict(),restored_optimizer.state_dict()),
        all_Adam_steps10=_adam_valid(optimizer,model,10,torch) and _adam_valid(restored_optimizer,restored,10,torch),
        model_optimizer_and_gradients_finite=_finite_tree([model.state_dict(),optimizer.state_dict(),gradient_snapshot],torch),
        all_losses_finite=bool(np.isfinite(loss_values).all()),parameters_changed=not raw_tree_equal(initial,model.state_dict()),
        RNG_raw_restore=rng_equal,next_sampler_batch_exact=sampler_equal,next_CHECK_prediction_raw_exact=raw_tree_equal(probe,clone),
        finite_nonnegative_native_Fn=bool(torch.isfinite(probe).all() and (probe[:,-3:]>=0).all()),no_CHECK_fitting=saved['normalizer_fit_role']=='TRAIN' and saved['dataset_pins']==packed['outputs'])
    checks={k:bool(v) for k,v in checks.items()};remaining=config['planned_updates']-10;estimate=remaining*np.median(milliseconds[5:])/1000
    report=dict(schema='compound_response_first_checkpoint_health:1',checkpoint=file_pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),
        update=10,training_running=False,stopped_at_first_checkpoint=True,pause_goal_requested=True,model_only_no_Actor=True,
        losses=loss_values,check_metrics=check_metrics,response_metrics=response_metrics,response_pairs=len(pairs),response_zero_prior_NMSE=1.,
        training_wall_s=training_wall,update_CUDA_ms=milliseconds,remaining_updates=remaining,remaining_compute_estimate_s=float(estimate),
        estimate_scope='Conditional on resuming same1000-update recipe; last5 of only10 updates, excludes startup/checkpoint/native collection/validation. No job continues in background.',
        device_name=torch.cuda.get_device_name(device),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),nvidia_after=gpu_snapshot(),
        sources=sources,config=file_pin(config_path),packed=file_pin(packed_path),
        scope='Healthy structural saved-state/RNG/sampler/forward readback only. First response WM warmup on378 new TRAIN transitions,193 heldout-seed/dt CHECK and48 paired response cases. Not a policy, validated contact gradient/reward, full trajectory, independent RGB or generalization/acceleration result.')
    for pin in sources:verify_file_pin(pin)
    write(out/'health_0010.json',report);print(json.dumps(dict(health=file_pin(out/'health_0010.json'),checkpoint=report['checkpoint'],checks=checks,remaining_compute_estimate_s=estimate,check=check_metrics,response=response_metrics,nvidia=report['nvidia_after'])))
    if not all(checks.values()):raise ValueError('First checkpoint unhealthy; retain evidence and do not continue training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');main(p.parse_args())
