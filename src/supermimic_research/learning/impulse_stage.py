"""Matched independent/impulse-head fit with explicit new checkpoint semantics."""
import json,random,time,tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from .impulse_response import ImpulseResponseWM,HEAD_SCHEMA
from .world_model import capture_rng,restore_rng,raw_tree_equal,atomic_checkpoint
from .checkpoint import _finite_tree,_adam_valid
from .response_diagnostics import BLOCKS,paired_metrics,probe_metrics
from .paired_response import intervention_metrics
from .response_rollout import load_rollout_cases,rollout_metrics
from .gpu_monitor import GPUMonitor


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def update_impulse(arm,cache,config,*,capture_gradient=False):
    model,opt,sampler=[arm[k] for k in ['model','optimizer','sampler']]
    indices=torch.randint(len(cache['x']),(config['batch_size'],),generator=sampler,device=cache['x'].device)
    opt.zero_grad(set_to_none=True);loss,terms=model.supervised_loss(cache['x'][indices],cache['y'][indices],cache[arm['mode']][indices],auxiliary_weight=config['auxiliary_weight']);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=False)
    gradient=dict(post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    opt.step();return torch.cat([loss.detach()[None],terms]),indices,gradient


def restore_impulse_arm(payload,device):
    model=ImpulseResponseWM(payload['scales'],payload['head_scales'],payload['physics'],width=payload['width'],coupled=payload['coupled']).to(device);model.load_state_dict(payload['model'],strict=True)
    config=payload['config'];opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=torch.device(device).type=='cuda');opt.load_state_dict(payload['optimizer'])
    sampler=torch.Generator(device=device);sampler.set_state(payload['sampler']);restore_rng(payload['rng'],device=device)
    return dict(model=model,optimizer=opt,sampler=sampler,mode=payload['mode'])


def impulse_payload(arm,config,scales,head_scales,physics,*,update,device):
    return dict(schema='compound_impulse_arm:1',mode=arm['mode'],coupled=arm['model'].coupled,width=config['width'],config=config,scales=scales,head_scales=head_scales,physics=physics,update=update,
        model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),sampler=arm['sampler'].get_state(),rng=capture_rng(device=device),head_schema=HEAD_SCHEMA if arm['model'].coupled else 'original_delta52_warm26_wrench6_Fn3:1')


def verify_impulse_step(arm,payload,cache,config,device):
    clone=restore_impulse_arm(payload,device);a=update_impulse(arm,cache,config,capture_gradient=True);b=update_impulse(clone,cache,config,capture_gradient=True)
    checks=dict(losses_exact=raw_tree_equal(a[0],b[0]),batch_exact=raw_tree_equal(a[1],b[1]),gradients_exact=raw_tree_equal(a[2],b[2]),model_exact=raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()),
        optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),sampler_exact=raw_tree_equal(arm['sampler'].get_state(),clone['sampler'].get_state()))
    return checks,a[0]


def run_impulse_stage(root,config_path,out,*,run=False):
    root,config_path,out=Path(root),Path(config_path).resolve(),Path(out).resolve();config=tomllib.loads(config_path.read_text());prepared_path=(config_path.parent/config['prepared']).resolve();prep=json.loads(prepared_path.read_text())
    if (prep['passed'],prep['total'])!=(6,6) or config['first_checkpoint_update']!=100 or config['planned_updates']!=2000 or not config['stop_at_first_healthy_checkpoint']:raise ValueError('Reviewed preparation and first100 boundary')
    verify_file_pin(prep['config'])
    for pin in prep['sources']:verify_file_pin(pin)
    docs={k:json.loads(verify_file_pin(pin).read_text()) for k,pin in prep['inputs'].items()}
    for doc in docs.values():
        for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)
    warm,review,old,probes=[docs[k] for k in ['warm_health','intervention_review','old_packed','old_probes']]
    new={r:dict(np.load(verify_file_pin(p))) for r,p in review['outputs'].items()};old_data={r:dict(np.load(verify_file_pin(p))) for r,p in old['outputs'].items()};probe_data=dict(np.load(verify_file_pin(probes['probes'])))
    head_scales=dict(np.load(verify_file_pin(prep['head_scales'])));heads={r:dict(np.load(verify_file_pin(pin)))['heads'] for r,pin in prep['targets'].items()};warm_path=verify_file_pin(warm['checkpoint'])
    if not run:return dict(training_started=False,TRAIN_rows=len(heads['TRAIN']),CHECK_rows=len(heads['CHECK']),first_checkpoint=100,warm=warm['checkpoint'],optimizer_reset_both=True)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    saved=torch.load(warm_path,map_location='cpu',weights_only=False);config={**config,'width':saved['config']['width']};scales=saved['scales'];cases,physics,rollout_pins=load_rollout_cases(review);device=torch.device(config['device'])
    if saved['update']!=1000 or saved['dataset_pins']!=old['outputs']:raise ValueError('Predeclared original single1000 parent')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(config['sampler_seed']);np.random.seed(config['sampler_seed']);random.seed(config['sampler_seed'])
    sources=[file_pin(root/p) for p in ['tools/train_mug_impulse_response.py','src/supermimic_research/learning/impulse_stage.py','src/supermimic_research/learning/impulse_response.py',
        'src/supermimic_research/mechanics/impulse_codec.py','src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/world_model.py',
        'src/supermimic_research/learning/checkpoint.py','src/supermimic_research/learning/paired_response.py','src/supermimic_research/learning/response_diagnostics.py',
        'src/supermimic_research/learning/response_rollout.py','src/supermimic_research/mechanics/response_decode.py','src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/learning/gpu_monitor.py']]
    arm_scales={'DIRECT':dict(head_mean=scales['y_mean'],head_scale=scales['y_scale']),'IMPULSE':head_scales};arms={};net={k.removeprefix('net.'):v for k,v in saved['model'].items() if k.startswith('net.')}
    for mode in ['DIRECT','IMPULSE']:
        model=ImpulseResponseWM(scales,arm_scales[mode],physics,width=config['width'],coupled=mode=='IMPULSE').to(device);model.net.load_state_dict(net,strict=True)
        with torch.no_grad():model.net[-1].weight[46:52].zero_();model.net[-1].bias[46:52].zero_()
        opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=True);sampler=torch.Generator(device=device).manual_seed(config['sampler_seed'])
        arms[mode]=dict(model=model,optimizer=opt,sampler=sampler,mode=mode)
    initial_equal=raw_tree_equal(arms['DIRECT']['model'].net.state_dict(),arms['IMPULSE']['model'].net.state_dict())
    initial_net={k:v.detach().cpu().clone() for k,v in arms['DIRECT']['model'].net.state_dict().items()}
    if not initial_equal:raise ValueError('Matched NN initialization differs')
    x=np.concatenate([old_data['TRAIN']['x'],new['TRAIN']['x']]);y=np.concatenate([old_data['TRAIN']['y'],new['TRAIN']['y']]);cache=dict(x=torch.as_tensor(x,dtype=torch.float32,device=device),y=torch.as_tensor(y,dtype=torch.float64,device=device),
        DIRECT=torch.as_tensor(y,dtype=torch.float64,device=device),IMPULSE=torch.as_tensor(heads['TRAIN'],dtype=torch.float64,device=device))
    out.mkdir();write(out/'intent.json',dict(schema='compound_impulse_comparison:1',config=file_pin(config_path),sources=sources,preparation=file_pin(prepared_path),warm_checkpoint=warm['checkpoint'],rollout_inputs=rollout_pins,
        initial_NN_equal=initial_equal,optimizer_reset_both=True,velocity_head_rows46_52_reset_both=True,original_scales_frozen=True,defect_scales_fitted_from='TRAIN_ONLY',
        difference='Direct velocity versus measured-wrench/learned-native-defect impulse decoding and corresponding auxiliary labels; same NN parameters/rows/batches/step budget.',
        first_checkpoint=100,planned_updates=2000,extra_clone_optimizer_updates_for_verification=2,geometry_native_or_host_health_in_update_loop=False,policy_training=False))
    parity={};logs={mode:[] for mode in arms};verification_start=time.perf_counter()
    for mode,arm in arms.items():
        payload=impulse_payload(arm,config,scales,arm_scales[mode],physics,update=0,device=device);parity[mode],values=verify_impulse_step(arm,payload,cache,config,device);logs[mode].append(values)
        if not all(parity[mode].values()):write(out/'failed_first_update.json',parity);raise ValueError('Exact initial optimizer replay failed')
    torch.cuda.synchronize(device);verification_wall=time.perf_counter()-verification_start;write(out/'first_update_verification.json',parity)
    monitor=GPUMonitor();monitor.start();timers=[];gradients={};torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    try:
        for update in range(2,101):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                values,_,gradient=update_impulse(arm,cache,config,capture_gradient=update==100);logs[mode].append(values)
                if gradient is not None:gradients[mode]=gradient
            end.record();timers.append((begin,end))
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    milliseconds=[a.elapsed_time(b) for a,b in timers];logs={m:torch.stack(v).cpu().tolist() for m,v in logs.items()};children={}
    for mode,arm in arms.items():
        children[mode]=impulse_payload(arm,config,scales,arm_scales[mode],physics,update=100,device=device);children[mode]['gradient_snapshot']=gradients[mode]
    checkpoint=out/'impulse_checkpoint_0100.pt';atomic_checkpoint(checkpoint,dict(schema='compound_impulse_comparison_checkpoint:1',update=100,planned_updates=2000,arms=children,config=file_pin(config_path),preparation=file_pin(prepared_path),sources=sources,warm_checkpoint=warm['checkpoint'],logs=logs,first_update_verification=parity))
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);checks={};errors={};response={};rollouts={}
    for mode,arm in arms.items():
        payload=read['arms'][mode];clone=restore_impulse_arm(payload,device);model=arm['model'];rng_equal=raw_tree_equal(payload['rng'],capture_rng(device=device));a=torch.randint(len(x),(config['batch_size'],),generator=arm['sampler'],device=device);b=torch.randint(len(x),(config['batch_size'],),generator=clone['sampler'],device=device)
        same_batch=bool(torch.equal(a,b));arm['sampler'].set_state(payload['sampler'])
        with torch.no_grad():
            probe=torch.as_tensor(new['CHECK']['x'],dtype=torch.float32,device=device);prediction=model.prediction(probe);copy_prediction=clone['model'].prediction(probe);target=torch.as_tensor(new['CHECK']['y'],dtype=torch.float64,device=device)
            errors[mode]={name:float(((prediction[:,lo:hi]-target[:,lo:hi])/model.y_scale[lo:hi]).square().mean()) for name,(lo,hi) in BLOCKS.items()}
        local=dict(model_exact=raw_tree_equal(model.state_dict(),clone['model'].state_dict()),optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),Adam100=_adam_valid(arm['optimizer'],model,100,torch) and _adam_valid(clone['optimizer'],clone['model'],100,torch),
            model_optimizer_gradients_finite=_finite_tree([model.state_dict(),arm['optimizer'].state_dict(),gradients[mode]],torch),losses_finite=bool(np.isfinite(logs[mode]).all()),RNG_exact=rng_equal,next_batch_exact=same_batch,CHECK_prediction_exact=raw_tree_equal(prediction,copy_prediction),
            frozen_output_scales_exact=all(raw_tree_equal(saved['model'][k],model.state_dict()[k]) for k in scales),physics_head_scales_exact=raw_tree_equal(payload['head_scales'],arm_scales[mode]) and all(raw_tree_equal(torch.tensor(arm_scales[mode][k],dtype=torch.float32),model.state_dict()[k]) for k in ['head_mean','head_scale']),
            parameters_changed=not raw_tree_equal(initial_net,model.net.state_dict()),finite_nonnegative_CHECK_Fn=bool(torch.isfinite(prediction).all() and (prediction[:,-3:]>=0).all()))
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()});checks.update({mode+'_first_update_'+k:v for k,v in parity[mode].items()})
        metric,arrays=intervention_metrics(model,new['CHECK'],device);np.savez_compressed(out/(mode+'_new_CHECK.npz'),**arrays)
        old_metric,arrays=probe_metrics(model,probe_data,device);np.savez_compressed(out/(mode+'_old_probes.npz'),**arrays);response[mode]=dict(new_CHECK=metric,old_probes=old_metric,old_RESPONSE=paired_metrics(model,old_data['RESPONSE'],device))
        rollouts[mode],arrays=rollout_metrics(model,cases,physics,device);np.savez_compressed(out/(mode+'_rollouts.npz'),**arrays)
    checks['matched_sampler_cursor']=raw_tree_equal(read['arms']['DIRECT']['sampler'],read['arms']['IMPULSE']['sampler']);checks['fixed_checkpoint_cursor']=read['update']==100 and read['planned_updates']==2000
    samples=[s for s in monitor.samples if s['start']<=start+wall and s['end']>=start];telemetry=dict(samples=samples,errors=monitor.errors,scope='Device-wide query windows overlapping99 regular two-arm updates, excludes first-update restore checks and later diagnostics.')
    if samples:
        gpu=np.array([s['values'] for s in samples]);telemetry.update(GPU_utilization_median_percent=float(np.median(gpu[:,0])),GPU_utilization_max_percent=float(gpu[:,0].max()),VRAM_used_max_MiB=float(gpu[:,1].max()),VRAM_total_MiB=float(gpu[0,2]))
    eta=1900*float(np.median(milliseconds[-50:]))/1000
    result=dict(schema='compound_impulse_comparison_health:1',checkpoint=file_pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),sources=sources,config=file_pin(config_path),preparation=file_pin(prepared_path),warm_checkpoint=warm['checkpoint'],
        errors=errors,response=response,recorded_control_rollouts=rollouts,logs=logs,update=100,planned_updates=2000,remaining_updates=1900,remaining_two_arm_compute_estimate_s=eta,
        regular99_update_wall_s=wall,first_update_verification_wall_s=verification_wall,extra_clone_updates=2,CUDA_ms_per_two_arm_update=milliseconds,GPU_telemetry=telemetry,
        peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(device),training_running=False,pause_goal_requested=True,
        scope='New representation and auxiliary-target matched experiment; same external chart, not legacy checkpoint resume, Actor/A1A2, geometry/contact truth, holding or RGB benefit. Decoder consistency by construction does not prove correct physical actions.',
        estimate_scope='Conditional remaining1900 paired updates from last50 CUDA-event median, excludes startup/save/diagnostics/data/native/retry costs; no background continuation.')
    for pin in sources+prep['sources']+[warm['checkpoint']]:verify_file_pin(pin)
    write(out/'health_0100.json',result)
    printout=dict(health=file_pin(out/'health_0100.json'),passed=result['passed'],total=result['total'],regular99_wall_s=wall,verification_wall_s=verification_wall,remaining_compute_estimate_s=eta,
        small_response={mode:{name:response[mode]['new_CHECK']['by_amplitude']['0.04'][name]['AD'] for name in ['delta52','object12','contact9']} for mode in arms},H10_rollout={mode:rollouts[mode]['by_horizon']['10'] for mode in arms},GPU_samples=len(samples),GPU_telemetry={k:v for k,v in telemetry.items() if k!='samples'})
    if not all(checks.values()):raise ValueError('Retain unhealthy first impulse checkpoint; no further updates')
    return printout
