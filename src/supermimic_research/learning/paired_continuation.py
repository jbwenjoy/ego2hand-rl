"""Reusable exact continuation/readback for the fixed paired response experiment."""
import json,time,tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from .response_training import restore_response,response_payload
from .paired_response import paired_cache,paired_update,intervention_metrics
from .response_diagnostics import BLOCKS,paired_metrics,probe_metrics
from .response_rollout import load_rollout_cases,rollout_metrics
from .world_model import raw_tree_equal,capture_rng,atomic_checkpoint
from .checkpoint import _adam_valid,_finite_tree
from .gpu_monitor import GPUMonitor


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def verify_next_update(arm,payload,absolute,paired,stage,device):
    """Advance the active branch ONCE and an expendable restore ONCE; compare all."""
    clone,opt,sampler=restore_response(payload,device)
    a=paired_update(arm['model'],arm['optimizer'],arm['sampler'],absolute,paired,stage,coefficient=arm['coefficient'],capture_gradient=True)
    b=paired_update(clone,opt,sampler,absolute,paired,stage,coefficient=arm['coefficient'],capture_gradient=True)
    checks=dict(losses_exact=raw_tree_equal(a[0],b[0]),absolute_batch_exact=raw_tree_equal(a[1],b[1]),pair_batch_exact=raw_tree_equal(a[2],b[2]),
        gradients_exact=raw_tree_equal(a[3],b[3]),model_exact=raw_tree_equal(arm['model'].state_dict(),clone.state_dict()),
        optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),opt.state_dict()),sampler_cursor_exact=raw_tree_equal(arm['sampler'].get_state(),sampler.get_state()))
    return checks,a[0]


def _verify_sources(doc):
    for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)


def run_continuation(root,config_path,out,*,run=False):
    root,config_path,out=Path(root),Path(config_path).resolve(),Path(out).resolve();resume=tomllib.loads(config_path.read_text())
    parent_path=(config_path.parent/resume['parent_health']).resolve();parent=json.loads(parent_path.read_text());_verify_sources(parent)
    if parent['passed']!=parent['total'] or parent['paired_stage_updates']!=100 or resume['end_stage_update']!=2000 or not resume['stop_at_first_new_checkpoint']:raise ValueError('Healthy100 checkpoint and unchanged fixed2000 boundary')
    stage_path=verify_file_pin(parent['stage']);stage=tomllib.loads(stage_path.read_text());evidence={key:verify_file_pin(pin) for key,pin in parent['evidence'].items()};docs={k:json.loads(v.read_text()) for k,v in evidence.items()}
    for doc in docs.values():_verify_sources(doc)
    review,old,probes=[docs[k] for k in ['intervention_review','old_packed','old_probes']]
    new={r:dict(np.load(verify_file_pin(p))) for r,p in review['outputs'].items()};old_raw={r:dict(np.load(verify_file_pin(p))) for r,p in old['outputs'].items()};probe_raw=dict(np.load(verify_file_pin(probes['probes'])))
    cases,physics,rollout_pins=load_rollout_cases(review);checkpoint_in=verify_file_pin(parent['checkpoint'])
    if not run:return dict(training_started=False,start_stage=100,end_stage=2000,absolute_rows=len(new['TRAIN']['x'])+len(old_raw['TRAIN']['x']),TRAIN_pairs=len(new['TRAIN']['pairs']),rollout_CHECK_cases=len(cases))
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    saved=torch.load(checkpoint_in,map_location='cpu',weights_only=False)
    if saved['schema']!='compound_paired_response_checkpoint:1' or saved['stage_update']!=100 or saved['planned_stage_updates']!=2000 or set(saved['arms'])!={'ABS','CF'}:raise ValueError('Exact matched checkpoint schema/cursor')
    device=torch.device(stage['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=[file_pin(root/p) for p in ['tools/resume_mug_paired_response.py','src/supermimic_research/learning/paired_continuation.py',
        'src/supermimic_research/learning/paired_response.py','src/supermimic_research/learning/response_training.py','src/supermimic_research/learning/response_diagnostics.py',
        'src/supermimic_research/learning/response_rollout.py','src/supermimic_research/mechanics/response_decode.py','src/supermimic_research/mechanics/transition_records.py',
        'src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py','src/supermimic_research/learning/gpu_monitor.py']]
    arms={};initial={};dataset_pins=dict(old=old['outputs'],new=review['outputs'])
    for mode in ['ABS','CF']:
        child=saved['arms'][mode]
        if child['dataset_pins']!=dataset_pins or child['stage_update']!=100 or child['update']!=1100 or child['config']['coefficient']!=stage['coefficient_'+mode]:raise ValueError('Frozen arm lineage/budget/coefficient')
        model,opt,sampler=restore_response(child,device)
        initial[mode]=dict(model_exact=raw_tree_equal(child['model'],model.state_dict()),optimizer_exact=raw_tree_equal(child['optimizer'],opt.state_dict()),
            RNG_exact=raw_tree_equal(child['rng'],capture_rng(device=device)),sampler_exact=raw_tree_equal(child['sampler'],sampler.get_state()),Adam1100=_adam_valid(opt,model,1100,torch))
        arms[mode]=dict(model=model,optimizer=opt,sampler=sampler,coefficient=child['config']['coefficient'])
    if not all(v for c in initial.values() for v in c.values()):raise ValueError('Before-update restoration differs')
    absolute={key:torch.as_tensor(np.concatenate([old_raw['TRAIN'][key],new['TRAIN'][key]]),dtype=torch.float32,device=device) for key in ['x','y']};cache=paired_cache(new['TRAIN'],arms['ABS']['model'],device)
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='paired_response_exact_continuation:1',sources=sources,resume=file_pin(config_path),parent_health=file_pin(parent_path),parent_checkpoint=parent['checkpoint'],
        stage=parent['stage'],evidence=parent['evidence'],rollout_inputs=rollout_pins,initial_checks=initial,start_stage=100,end_stage=2000,optimizer_reset=False,scales_refitted=False,
        additional_restore_diagnostic_updates=2,diagnostic_scope='An extra cloned TRAIN update per arm is charged as verification, never selected or fitted into the retained trajectory.',
        stop='First new checkpoint2000 plus full structural readback, fixed read-only diagnostics and timing report; then caller pauses Goal.'))
    parity={};logs={m:[] for m in arms};verification_start=time.perf_counter()
    for mode,arm in arms.items():
        parity[mode],values=verify_next_update(arm,saved['arms'][mode],absolute,cache,stage,device);logs[mode].append(values)
        if not all(parity[mode].values()):write(out/'failed_next_update.json',parity);raise ValueError('Exact next optimizer/gradient replay failed')
    torch.cuda.synchronize(device);verification_wall=time.perf_counter()-verification_start
    write(out/'next_update_verification.json',parity)
    monitor=GPUMonitor();monitor.start();timers=[];gradients={};torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    try:
        for update in range(102,2001):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                values,_,_,gradient=paired_update(arm['model'],arm['optimizer'],arm['sampler'],absolute,cache,stage,coefficient=arm['coefficient'],capture_gradient=update==2000)
                logs[mode].append(values)
                if gradient is not None:gradients[mode]=gradient
            end.record();timers.append((begin,end))
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    elapsed=[a.elapsed_time(b) for a,b in timers];logs={m:torch.stack(v).cpu().tolist() for m,v in logs.items()};children={}
    for mode,arm in arms.items():
        child=response_payload(arm['model'],arm['optimizer'],arm['sampler'],saved['arms'][mode],update=3000,device=device)
        child.update(stage_update=2000,stage=parent['stage'],gradient_snapshot=gradients[mode],sources=sources);children[mode]=child
    checkpoint=out/'paired_checkpoint_2000.pt';atomic_checkpoint(checkpoint,dict(schema=saved['schema'],stage_update=2000,planned_stage_updates=2000,arms=children,sources=sources,stage=parent['stage'],
        parent_checkpoint=parent['checkpoint'],resume=file_pin(config_path),evidence=parent['evidence'],logs=logs,next_update_verification=parity))
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);checks={};errors={};metrics={};rollouts={};datasets={'old_TRAIN':old_raw['TRAIN'],'new_TRAIN':new['TRAIN'],'old_CHECK':old_raw['CHECK'],'new_CHECK':new['CHECK']}
    for mode,arm in arms.items():
        child=read['arms'][mode];clone,opt,sampler=restore_response(child,device);m=arm['model'];rng_equal=raw_tree_equal(child['rng'],capture_rng(device=device));same_batches=True
        for count,size in [(len(absolute['x']),stage['batch_size']),(len(cache['du']),stage['pair_batch_size'])]:
            a=torch.randint(count,(size,),generator=arm['sampler'],device=device);b=torch.randint(count,(size,),generator=sampler,device=device);same_batches=bool(torch.equal(a,b)) and same_batches
        arm['sampler'].set_state(child['sampler'])
        with torch.no_grad():
            probe=torch.as_tensor(old_raw['CHECK']['x'],dtype=torch.float32,device=device);p=m.prediction(probe);cp=clone.prediction(probe);errors[mode]={}
            for role,data in datasets.items():
                pred=m.prediction(torch.as_tensor(data['x'],dtype=torch.float32,device=device));target=torch.as_tensor(data['y'],dtype=torch.float32,device=device)
                errors[mode][role]={name:float(((pred[:,a:b]-target[:,a:b])/m.y_scale[a:b]).square().mean()) for name,(a,b) in BLOCKS.items()}
        local=dict(model_exact=raw_tree_equal(m.state_dict(),clone.state_dict()),optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),opt.state_dict()),Adam3000=_adam_valid(opt,clone,3000,torch) and _adam_valid(arm['optimizer'],m,3000,torch),
            finite_model_optimizer_gradients=_finite_tree([m.state_dict(),arm['optimizer'].state_dict(),gradients[mode]],torch),finite_losses=bool(np.isfinite(logs[mode]).all()),
            parameters_changed=not raw_tree_equal(saved['arms'][mode]['model'],m.state_dict()),RNG_exact=rng_equal,next_absolute_and_pair_batches_exact=same_batches,CHECK_prediction_exact=raw_tree_equal(p,cp),
            frozen_scales_exact=raw_tree_equal(saved['arms'][mode]['scales'],child['scales']) and all(raw_tree_equal(saved['arms'][mode]['model'][k],m.state_dict()[k]) for k in saved['arms'][mode]['scales']),
            dataset_lineage_exact=child['dataset_pins']==dataset_pins,finite_CHECK_predictions=bool(torch.isfinite(p).all() and (p[:,-3:]>=0).all()) and all(np.isfinite(v) for values in errors[mode].values() for v in values.values()))
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()});checks.update({mode+'_next_update_'+k:v for k,v in parity[mode].items()})
        new_metrics,arrays=intervention_metrics(m,new['CHECK'],device);np.savez_compressed(out/(mode+'_new_CHECK.npz'),**arrays)
        old_metrics,arrays=probe_metrics(m,probe_raw,device);np.savez_compressed(out/(mode+'_old_probes.npz'),**arrays)
        metrics[mode]=dict(new_CHECK=new_metrics,old_RESPONSE=paired_metrics(m,old_raw['RESPONSE'],device),old_probes=old_metrics)
        rollouts[mode],arrays=rollout_metrics(m,cases,physics,device);np.savez_compressed(out/(mode+'_recorded_control_rollouts.npz'),**arrays)
    checks['identical_arm_sampler_cursor']=raw_tree_equal(read['arms']['ABS']['sampler'],read['arms']['CF']['sampler']);checks['fixed_stage_boundary']=read['stage_update']==2000 and read['planned_stage_updates']==2000
    samples=[s for s in monitor.samples if s['start']<=start+wall and s['end']>=start];telemetry=dict(samples=samples,errors=monitor.errors,scope='Device-wide query windows overlapping1899 regular paired updates, excludes two real step101/restore-check updates; other desktop workloads included.')
    if samples:
        gpu=np.array([s['values'] for s in samples]);telemetry.update(GPU_utilization_median_percent=float(np.median(gpu[:,0])),GPU_utilization_max_percent=float(gpu[:,0].max()),VRAM_used_max_MiB=float(gpu[:,1].max()),VRAM_total_MiB=float(gpu[0,2]))
    result=dict(schema='compound_paired_response_completion_health:1',checkpoint=file_pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),sources=sources,parent_health=file_pin(parent_path),parent_checkpoint=parent['checkpoint'],
        resume=file_pin(config_path),stage=parent['stage'],evidence=parent['evidence'],rollout_inputs=rollout_pins,errors=errors,response_metrics=metrics,recorded_control_rollouts=rollouts,
        logs=logs,paired_stage_updates=2000,new_paired_updates=1900,Adam_steps_per_arm=3000,planned_stage_remaining_updates=0,initial_checks=initial,next_update_verification=parity,
        regular1899_update_wall_s=wall,next_update_verification_wall_s=verification_wall,additional_clone_diagnostic_updates=2,CUDA_ms_per_two_arm_update=elapsed,
        timing_scope='Fixed2000 stage is complete; zero updates remain. Regular1899 timing excludes actual step101/clone parity, startup/checkpoint/diagnostics/native acquisition; not total research ETA.',
        GPU_telemetry=telemetry,peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(device),training_running=False,pause_goal_requested=True,
        scope='Fixed matched response supervision completed with full state restore. Read-only20..200ms recorded-control extrapolation/momentum checks, not a policy, reference tracking, contact reward/A1A2/holding success, independent RGB or full research completion.')
    for pin in sources+[parent['checkpoint']]:verify_file_pin(pin)
    _verify_sources(parent)
    for doc in docs.values():_verify_sources(doc)
    write(out/'health_2000.json',result)
    if not all(checks.values()):raise ValueError('Retain unhealthy checkpoint; no more training')
    return dict(health=file_pin(out/'health_2000.json'),passed=result['passed'],total=result['total'],regular1899_wall_s=wall,verification_wall_s=verification_wall,
        new_CHECK_small_response={mode:{name:metrics[mode]['new_CHECK']['by_amplitude']['0.04'][name]['AD'] for name in ['delta52','object12','contact9']} for mode in arms},
        H10_rollout={mode:rollouts[mode]['by_horizon']['10'] for mode in arms},GPU_telemetry={k:v for k,v in telemetry.items() if k!='samples'},GPU_samples=len(samples))
