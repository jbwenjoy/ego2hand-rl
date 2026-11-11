"""Matched ABS/CF response study; stop at first paired checkpoint and readback."""
from pathlib import Path
import argparse,json,os,sys,time,tomllib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.response_training import restore_response,response_payload
from supermimic_research.learning.response_diagnostics import BLOCKS,paired_metrics,probe_metrics
from supermimic_research.learning.paired_response import paired_cache,paired_update,intervention_metrics
from supermimic_research.learning.gpu_monitor import GPUMonitor
from supermimic_research.learning.world_model import capture_rng,atomic_checkpoint,raw_tree_equal
from supermimic_research.learning.checkpoint import _finite_tree,_adam_valid


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    stage_path=args.config.resolve();stage=tomllib.loads(stage_path.read_text());out=args.output.resolve()
    files={k:(stage_path.parent/stage[k]).resolve() for k in ['parent_health','intervention_review','old_packed','old_probes']};docs={k:json.loads(p.read_text()) for k,p in files.items()}
    parent,review,old,probes=[docs[k] for k in ['parent_health','intervention_review','old_packed','old_probes']]
    if (parent['passed'],parent['total'],parent['update'])!=(12,12,1000) or (review['passed'],review['total'])!=(8,8):raise ValueError('Healthy fixed parent and raw intervention review')
    if stage['first_checkpoint_update']!=100 or stage['planned_stage_updates']!=2000 or not stage['stop_at_first_healthy_checkpoint'] or (stage['coefficient_ABS'],stage['coefficient_CF'])!=(0,1):raise ValueError('Declared matched study and stop boundary')
    for doc in [parent,review,old,probes]:
        for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)
    new={r:np.load(verify_file_pin(p)) for r,p in review['outputs'].items()};old_raw={r:np.load(verify_file_pin(p)) for r,p in old['outputs'].items()};probe_raw=np.load(verify_file_pin(probes['probes']))
    parent_path=verify_file_pin(parent['checkpoint'])
    if not args.run:
        print(json.dumps(dict(training_started=False,TRAIN_absolute_rows=len(old_raw['TRAIN']['x'])+len(new['TRAIN']['x']),TRAIN_pairs=len(new['TRAIN']['pairs']),CHECK_pairs=len(new['CHECK']['pairs']),parent=parent['checkpoint'])));return
    if out.exists() or not out.is_relative_to(ROOT/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    saved=torch.load(parent_path,map_location='cpu',weights_only=False);device=torch.device(stage['device'])
    if saved['update']!=1000 or saved['dataset_pins']!=old['outputs'] or saved['normalizer_fit_role']!='TRAIN':raise ValueError('Frozen original model/data/scales parent')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=[file_pin(ROOT/p) for p in ['tools/train_mug_paired_response.py','src/supermimic_research/learning/paired_response.py','src/supermimic_research/learning/gpu_monitor.py',
        'src/supermimic_research/learning/response_training.py','src/supermimic_research/learning/response_diagnostics.py','src/supermimic_research/learning/compound_response.py',
        'src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py']]
    arms={};initial_checks={}
    for mode in ['ABS','CF']:
        m,opt,sampler=restore_response(saved,device);sampler.manual_seed(stage['sampler_seed'])
        initial_checks[mode]=dict(model_exact=raw_tree_equal(saved['model'],m.state_dict()),Adam_exact=raw_tree_equal(saved['optimizer'],opt.state_dict()),Adam1000=_adam_valid(opt,m,1000,torch))
        arms[mode]=dict(model=m,optimizer=opt,sampler=sampler,coefficient=stage['coefficient_'+mode])
    if not all(v for checks in initial_checks.values() for v in checks.values()):raise ValueError('Warm-start state drift')
    absolute={key:torch.as_tensor(np.concatenate([old_raw['TRAIN'][key],new['TRAIN'][key]]),dtype=torch.float32,device=device) for key in ['x','y']}
    cached=paired_cache(new['TRAIN'],arms['ABS']['model'],device)
    dataset_pins=dict(old=old['outputs'],new=review['outputs']);out.mkdir(parents=True)
    write(out/'intent.json',dict(schema='mug_matched_ABS_CF_stage:1',stage=file_pin(stage_path),sources=sources,evidence={k:file_pin(v) for k,v in files.items()},
        parent_checkpoint=parent['checkpoint'],dataset_pins=dataset_pins,initial_checks=initial_checks,sampling='Same reset sampler seed per arm; identical uniform3258 absolute-row batch256 and1440-pair batch64 draws.',
        normalizers_refitted=False,optimizer_reset=False,stage_optimizer_steps_start=1000,planned_stage_updates=2000,first_checkpoint=100,
        only_difference='Secant loss coefficient0 versus1; both execute the same pair forwards and update count.',policy_training=False,
        stop='First healthy paired checkpoint + read-only diagnostics/ETA, then caller pauses Goal. No CHECK/model selection or further updates.'))
    monitor=GPUMonitor();monitor.start();timers=[];logs={mode:[] for mode in arms};gradients={};torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.perf_counter()
    try:
        for update in range(1,stage['first_checkpoint_update']+1):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                values,_,_,gradient=paired_update(arm['model'],arm['optimizer'],arm['sampler'],absolute,cached,stage,coefficient=arm['coefficient'],capture_gradient=update==stage['first_checkpoint_update'])
                logs[mode].append(values)
                if gradient is not None:gradients[mode]=gradient
            end.record();timers.append((begin,end))
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    milliseconds=[a.elapsed_time(b) for a,b in timers];logs={m:torch.stack(v).cpu().tolist() for m,v in logs.items()};children={}
    for mode,arm in arms.items():
        child=response_payload(arm['model'],arm['optimizer'],arm['sampler'],saved,update=1100,device=device)
        child.update(planned_updates=3000,config={**saved['config'],**{k:stage[k] for k in ['pair_batch_size','response_reference_rad']},'coefficient':arm['coefficient'],'planned_updates':3000,'first_checkpoint_update':1100},
            stage_update=100,stage=file_pin(stage_path),dataset_pins=dataset_pins,gradient_snapshot=gradients[mode],normalizer_fit_role='ORIGINAL_TRAIN_FROZEN',sources=sources)
        children[mode]=child
    checkpoint=out/'paired_checkpoint_0100.pt';atomic_checkpoint(checkpoint,dict(schema='compound_paired_response_checkpoint:1',stage_update=100,planned_stage_updates=2000,arms=children,
        sources=sources,stage=file_pin(stage_path),parent_checkpoint=parent['checkpoint'],evidence={k:file_pin(v) for k,v in files.items()},logs=logs))
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);checks={};errors={};metrics={};datasets={'old_TRAIN':old_raw['TRAIN'],'new_TRAIN':new['TRAIN'],'old_CHECK':old_raw['CHECK'],'new_CHECK':new['CHECK']}
    for mode,arm in arms.items():
        child=read['arms'][mode];clone,opt,sampler=restore_response(child,device);rng_equal=raw_tree_equal(child['rng'],capture_rng(device=device));a=torch.randint(len(absolute['x']),(stage['batch_size'],),generator=arm['sampler'],device=device);b=torch.randint(len(absolute['x']),(stage['batch_size'],),generator=sampler,device=device)
        sampler_equal=torch.equal(a,b);arm['sampler'].set_state(child['sampler']);m=arm['model']
        with torch.no_grad():
            probe=torch.tensor(old_raw['CHECK']['x'],dtype=torch.float32,device=device);p=m.prediction(probe);cp=clone.prediction(probe)
            errors[mode]={}
            for role,data in datasets.items():
                prediction=m.prediction(torch.as_tensor(data['x'],dtype=torch.float32,device=device));target=torch.as_tensor(data['y'],dtype=torch.float32,device=device)
                errors[mode][role]={name:float(((prediction[:,a:b]-target[:,a:b])/m.y_scale[a:b]).square().mean()) for name,(a,b) in BLOCKS.items()}
        local=dict(model_exact=raw_tree_equal(m.state_dict(),clone.state_dict()),Adam_exact=raw_tree_equal(arm['optimizer'].state_dict(),opt.state_dict()),Adam1100=_adam_valid(opt,clone,1100,torch) and _adam_valid(arm['optimizer'],m,1100,torch),
            finite_model_optimizer_gradients=_finite_tree([m.state_dict(),arm['optimizer'].state_dict(),gradients[mode]],torch),finite_losses=bool(np.isfinite(logs[mode]).all()),parameters_changed=not raw_tree_equal(saved['model'],m.state_dict()),
            RNG_exact=rng_equal,next_sampler_batch_exact=bool(sampler_equal),CHECK_prediction_exact=raw_tree_equal(p,cp),frozen_scales_exact=raw_tree_equal(saved['scales'],child['scales']),dataset_lineage_exact=child['dataset_pins']==dataset_pins,
            finite_CHECK_predictions=bool(torch.isfinite(p).all() and (p[:,-3:]>=0).all()) and all(np.isfinite(v) for values in errors[mode].values() for v in values.values()))
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()})
        new_metrics,arrays=intervention_metrics(m,new['CHECK'],device);np.savez_compressed(out/(mode+'_new_CHECK.npz'),**arrays)
        old_physical,old_arrays=probe_metrics(m,probe_raw,device);np.savez_compressed(out/(mode+'_old_probes.npz'),**old_arrays)
        metrics[mode]=dict(new_CHECK=new_metrics,old_RESPONSE=paired_metrics(m,old_raw['RESPONSE'],device),old_probes=old_physical)
    checks['identical_arm_sampler_cursor']=raw_tree_equal(read['arms']['ABS']['sampler'],read['arms']['CF']['sampler']);checks['fixed_stage_boundary']=read['stage_update']==100 and read['planned_stage_updates']==2000
    remaining=stage['planned_stage_updates']-stage['first_checkpoint_update'];eta=remaining*float(np.median(milliseconds[-50:]))/1000
    samples=[s for s in monitor.samples if s['start']<=start+wall and s['end']>=start];telemetry=dict(samples=samples,errors=monitor.errors,scope='Device-wide one-shot query windows overlapping the update loop; other desktop workloads included, not process-only occupancy.')
    if samples:
        gpu=np.array([s['values'] for s in samples]);telemetry.update(GPU_utilization_median_percent=float(np.median(gpu[:,0])),GPU_utilization_max_percent=float(gpu[:,0].max()),VRAM_used_max_MiB=float(gpu[:,1].max()),VRAM_total_MiB=float(gpu[0,2]))
    result=dict(schema='compound_paired_response_health:1',checkpoint=file_pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),sources=sources,stage=file_pin(stage_path),
        evidence={k:file_pin(v) for k,v in files.items()},errors=errors,response_metrics=metrics,logs=logs,paired_stage_updates=100,Adam_steps_per_arm=1100,planned_stage_remaining_updates=remaining,
        training_wall_s=wall,CUDA_ms_per_two_arm_update=milliseconds,remaining_two_arm_compute_estimate_s=eta,estimate_scope='Conditional1900 paired updates at last50 CUDA-event median; excludes native data, startup, saving and diagnostics. No background continuation.',
        GPU_telemetry=telemetry,peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(device),training_running=False,pause_goal_requested=True,
        scope='First warm-start matched ABS/CF response checkpoint, not a policy/A1A2/CHORD benefit experiment, full trajectory holding, independent RGB or generalization. Structural save health does not certify physical gradients.')
    for doc in [parent,review,old,probes]:
        for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)
    for pin in sources+[parent['checkpoint']]:verify_file_pin(pin)
    write(out/'health_0100.json',result)
    print(json.dumps(dict(health=file_pin(out/'health_0100.json'),passed=result['passed'],total=result['total'],wall_s=wall,remaining_compute_estimate_s=eta,
        new_CHECK_small_response={mode:{name:metrics[mode]['new_CHECK']['by_amplitude']['0.04'][name]['AD'] for name in ['delta52','object12','contact9']} for mode in arms},
        GPU_telemetry={k:v for k,v in telemetry.items() if k!='samples'},GPU_sample_count=len(samples))))
    if not all(checks.values()):raise ValueError('Retain unhealthy checkpoint, no further updates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');main(p.parse_args())
