"""Exact fixed-budget impulse study continuation using shared updates/decoders."""
import json,time,tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from .impulse_stage import restore_impulse_arm,impulse_payload,verify_impulse_step,update_impulse
from .world_model import capture_rng,raw_tree_equal,atomic_checkpoint
from .checkpoint import _finite_tree,_adam_valid
from .response_diagnostics import BLOCKS,paired_metrics,probe_metrics
from .paired_response import intervention_metrics
from .response_rollout import load_rollout_cases,rollout_metrics
from .response_strata import response_strata
from .gpu_monitor import GPUMonitor


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def verify_sources(doc):
    for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)


def run_impulse_continuation(root,config_path,out,*,run=False):
    root,config_path,out=Path(root),Path(config_path).resolve(),Path(out).resolve();resume=tomllib.loads(config_path.read_text());parent_path=(config_path.parent/resume['parent_health']).resolve();parent=json.loads(parent_path.read_text())
    if (parent['passed'],parent['total'],parent['update'],parent['planned_updates'])!=(38,38,100,2000) or resume['end_update']!=2000 or not resume['stop_at_first_new_checkpoint']:raise ValueError('Healthy fixed first100 and original2000 endpoint')
    verify_sources(parent);config=tomllib.loads(verify_file_pin(parent['config']).read_text());prep_path=verify_file_pin(parent['preparation']);prep=json.loads(prep_path.read_text());verify_sources(prep)
    if (prep['passed'],prep['total'])!=(6,6):raise ValueError('Reviewed original head labels/scales')
    docs={k:json.loads(verify_file_pin(pin).read_text()) for k,pin in prep['inputs'].items()}
    for doc in docs.values():verify_sources(doc)
    review,old,probes=[docs[k] for k in ['intervention_review','old_packed','old_probes']]
    new={r:dict(np.load(verify_file_pin(p))) for r,p in review['outputs'].items()};old_data={r:dict(np.load(verify_file_pin(p))) for r,p in old['outputs'].items()};probe_data=dict(np.load(verify_file_pin(probes['probes'])))
    heads={r:dict(np.load(verify_file_pin(pin)))['heads'] for r,pin in prep['targets'].items()};verify_file_pin(prep['head_scales']);cases,physics,rollout_pins=load_rollout_cases(review);checkpoint_in=verify_file_pin(parent['checkpoint'])
    if not run:return dict(training_started=False,start_update=100,end_update=2000,TRAIN_rows=len(heads['TRAIN']),CHECK_rows=len(heads['CHECK']),read_only_rollout_cases=len(cases),all20_joint_components=True)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    saved=torch.load(checkpoint_in,map_location='cpu',weights_only=False)
    if saved['schema']!='compound_impulse_comparison_checkpoint:1' or saved['update']!=100 or saved['planned_updates']!=2000 or set(saved['arms'])!={'DIRECT','IMPULSE'}:raise ValueError('Original checkpoint schema/cursor/modes')
    device=torch.device(config['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=[file_pin(root/p) for p in ['tools/resume_mug_impulse_response.py','src/supermimic_research/learning/impulse_continuation.py','src/supermimic_research/learning/response_strata.py',
        'src/supermimic_research/learning/impulse_stage.py','src/supermimic_research/learning/impulse_response.py','src/supermimic_research/mechanics/impulse_codec.py',
        'src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py',
        'src/supermimic_research/learning/paired_response.py','src/supermimic_research/learning/response_diagnostics.py','src/supermimic_research/learning/response_rollout.py',
        'src/supermimic_research/mechanics/response_decode.py','src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/learning/gpu_monitor.py']]
    arms={};initial={}
    for mode in ['DIRECT','IMPULSE']:
        payload=saved['arms'][mode]
        if payload['update']!=100 or payload['mode']!=mode or payload['coupled']!=(mode=='IMPULSE'):raise ValueError('Typed arm cursor/representation')
        if any(payload['config'][k]!=config[k] for k in config):raise ValueError('Unchanged recipe required')
        arm=restore_impulse_arm(payload,device);initial[mode]=dict(model_exact=raw_tree_equal(payload['model'],arm['model'].state_dict()),optimizer_exact=raw_tree_equal(payload['optimizer'],arm['optimizer'].state_dict()),
            RNG_exact=raw_tree_equal(payload['rng'],capture_rng(device=device)),sampler_exact=raw_tree_equal(payload['sampler'],arm['sampler'].get_state()),Adam100=_adam_valid(arm['optimizer'],arm['model'],100,torch));arms[mode]=arm
    if not all(v for values in initial.values() for v in values.values()):raise ValueError('Initial restored state differs')
    x=np.concatenate([old_data['TRAIN']['x'],new['TRAIN']['x']]);y=np.concatenate([old_data['TRAIN']['y'],new['TRAIN']['y']]);cache=dict(x=torch.as_tensor(x,dtype=torch.float32,device=device),y=torch.as_tensor(y,dtype=torch.float64,device=device),
        DIRECT=torch.as_tensor(y,dtype=torch.float64,device=device),IMPULSE=torch.as_tensor(heads['TRAIN'],dtype=torch.float64,device=device))
    out.mkdir();write(out/'intent.json',dict(schema='compound_impulse_exact_continuation:1',sources=sources,resume=file_pin(config_path),parent_health=file_pin(parent_path),parent_checkpoint=parent['checkpoint'],
        config=parent['config'],preparation=parent['preparation'],rollout_inputs=rollout_pins,initial_checks=initial,start_update=100,end_update=2000,optimizer_reset=False,sampler_reseed=False,scales_refit=False,
        additional_clone_updates=2,stop='First new checkpoint2000 after full readback, original diagnostics and predeclared component/joint response reports; then caller pauses Goal.',policy_training=False))
    parity={};logs={mode:[] for mode in arms};verification_start=time.perf_counter()
    for mode,arm in arms.items():
        parity[mode],values=verify_impulse_step(arm,saved['arms'][mode],cache,saved['arms'][mode]['config'],device);logs[mode].append(values)
        if not all(parity[mode].values()):write(out/'failed_next_update.json',parity);raise ValueError('Exact next optimizer/gradient replay failed')
    torch.cuda.synchronize(device);verification_wall=time.perf_counter()-verification_start;write(out/'next_update_verification.json',parity)
    monitor=GPUMonitor();monitor.start();timers=[];gradients={};torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    try:
        for update in range(102,2001):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                values,_,gradient=update_impulse(arm,cache,saved['arms'][mode]['config'],capture_gradient=update==2000);logs[mode].append(values)
                if gradient is not None:gradients[mode]=gradient
            end.record();timers.append((begin,end))
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    milliseconds=[a.elapsed_time(b) for a,b in timers];logs={mode:torch.stack(v).cpu().tolist() for mode,v in logs.items()};children={}
    for mode,arm in arms.items():
        p=saved['arms'][mode];children[mode]=impulse_payload(arm,p['config'],p['scales'],p['head_scales'],p['physics'],update=2000,device=device);children[mode]['gradient_snapshot']=gradients[mode]
    checkpoint=out/'impulse_checkpoint_2000.pt';atomic_checkpoint(checkpoint,dict(schema=saved['schema'],update=2000,planned_updates=2000,arms=children,config=parent['config'],preparation=parent['preparation'],
        sources=sources,parent_checkpoint=parent['checkpoint'],resume=file_pin(config_path),logs=logs,next_update_verification=parity))
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);checks={};errors={};response={};strata={};rollouts={}
    datasets={'old_TRAIN':old_data['TRAIN'],'new_TRAIN':new['TRAIN'],'old_CHECK':old_data['CHECK'],'new_CHECK':new['CHECK']}
    for mode,arm in arms.items():
        payload=read['arms'][mode];clone=restore_impulse_arm(payload,device);model=arm['model'];rng_equal=raw_tree_equal(payload['rng'],capture_rng(device=device));a=torch.randint(len(x),(config['batch_size'],),generator=arm['sampler'],device=device);b=torch.randint(len(x),(config['batch_size'],),generator=clone['sampler'],device=device)
        same_batch=bool(torch.equal(a,b));arm['sampler'].set_state(payload['sampler'])
        with torch.no_grad():
            probe=torch.as_tensor(new['CHECK']['x'],dtype=torch.float32,device=device);prediction=model.prediction(probe);copy_prediction=clone['model'].prediction(probe);errors[mode]={}
            for role,data in datasets.items():
                p=model.prediction(torch.as_tensor(data['x'],dtype=torch.float32,device=device));target=torch.as_tensor(data['y'],dtype=torch.float64,device=device)
                errors[mode][role]={name:float(((p[:,lo:hi]-target[:,lo:hi])/model.y_scale[lo:hi]).square().mean()) for name,(lo,hi) in BLOCKS.items()}
        local=dict(model_exact=raw_tree_equal(model.state_dict(),clone['model'].state_dict()),optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),Adam2000=_adam_valid(arm['optimizer'],model,2000,torch) and _adam_valid(clone['optimizer'],clone['model'],2000,torch),
            model_optimizer_gradients_finite=_finite_tree([model.state_dict(),arm['optimizer'].state_dict(),gradients[mode]],torch),losses_finite=bool(np.isfinite(logs[mode]).all()),RNG_exact=rng_equal,next_batch_exact=same_batch,CHECK_prediction_exact=raw_tree_equal(prediction,copy_prediction),
            physics_head_scales_and_buffers_frozen=all(raw_tree_equal(saved['arms'][mode]['model'][k],model.state_dict()[k]) for k in model.state_dict() if not k.startswith('net.')),
            parameters_changed=not raw_tree_equal(saved['arms'][mode]['model'],model.state_dict()),finite_nonnegative_CHECK_Fn=bool(torch.isfinite(prediction).all() and (prediction[:,-3:]>=0).all()))
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()});checks.update({mode+'_next_update_'+k:v for k,v in parity[mode].items()})
        metric,arrays=intervention_metrics(model,new['CHECK'],device);np.savez_compressed(out/(mode+'_new_CHECK.npz'),**arrays);strata[mode]=response_strata(new['CHECK'],arrays)
        old_metric,arrays=probe_metrics(model,probe_data,device);np.savez_compressed(out/(mode+'_old_probes.npz'),**arrays);response[mode]=dict(new_CHECK=metric,old_probes=old_metric,old_RESPONSE=paired_metrics(model,old_data['RESPONSE'],device))
        rollouts[mode],arrays=rollout_metrics(model,cases,physics,device);np.savez_compressed(out/(mode+'_rollouts.npz'),**arrays)
    checks['matched_sampler_cursor']=raw_tree_equal(read['arms']['DIRECT']['sampler'],read['arms']['IMPULSE']['sampler']);checks['fixed_checkpoint_cursor']=read['update']==2000 and read['planned_updates']==2000
    write(out/'response_strata.json',strata)
    samples=[s for s in monitor.samples if s['start']<=start+wall and s['end']>=start];telemetry=dict(samples=samples,errors=monitor.errors,scope='Device-wide query windows overlapping1899 regular two-arm updates, excludes real step101/clone verification and final diagnostics.')
    if samples:
        gpu=np.array([s['values'] for s in samples]);telemetry.update(GPU_utilization_median_percent=float(np.median(gpu[:,0])),GPU_utilization_max_percent=float(gpu[:,0].max()),VRAM_used_max_MiB=float(gpu[:,1].max()),VRAM_total_MiB=float(gpu[0,2]))
    result=dict(schema='compound_impulse_completion_health:1',checkpoint=file_pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),sources=sources,resume=file_pin(config_path),parent_health=file_pin(parent_path),parent_checkpoint=parent['checkpoint'],config=parent['config'],preparation=parent['preparation'],
        errors=errors,response=response,stratified_response=file_pin(out/'response_strata.json'),recorded_control_rollouts=rollouts,logs=logs,update=2000,planned_updates=2000,remaining_updates=0,initial_checks=initial,next_update_verification=parity,
        regular1899_update_wall_s=wall,next_update_verification_wall_s=verification_wall,extra_clone_updates=2,CUDA_ms_per_two_arm_update=milliseconds,GPU_telemetry=telemetry,
        peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(device),training_running=False,pause_goal_requested=True,
        timing_scope='The fixed2000 budget is complete.1899 regular updates plus retained step101/clone verification reported separately; excludes startup, saving, validation, native acquisition and prior stages. No full research ETA.',
        scope='Exact fixed-budget matched representation continuation; CHECK remains read-only. Numerical momentum consistency is largely algebraic; native response and recorded-control errors must be inspected. No Actor/A1A2, reference tracking, holding, geometry/contact truth, independent RGB or full Goal success.')
    for pin in sources+parent['sources']+prep['sources']+[parent['checkpoint']]:verify_file_pin(pin)
    for doc in docs.values():verify_sources(doc)
    write(out/'health_2000.json',result)
    if not all(checks.values()):raise ValueError('Retain unhealthy completion checkpoint; no more training')
    return dict(health=file_pin(out/'health_2000.json'),passed=result['passed'],total=result['total'],regular1899_wall_s=wall,verification_wall_s=verification_wall,
        small_response={mode:{name:response[mode]['new_CHECK']['by_amplitude']['0.04'][name]['AD'] for name in ['delta52','object12','contact9']} for mode in arms},
        small_components={mode:{name:values['AD'] for name,values in strata[mode]['groups']['amplitude0.04/all']['components'].items() if name in ['COM_velocity3','omega3','force3','torque3']} for mode in arms},
        H10_rollout={mode:rollouts[mode]['by_horizon']['10'] for mode in arms},GPU_samples=len(samples),GPU_telemetry={k:v for k,v in telemetry.items() if k!='samples'})
