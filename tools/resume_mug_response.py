"""Continue the fixed 1000-update recipe; checkpoint and stop at stage boundary."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,tomllib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.response_training import restore_response,response_update,response_payload
from supermimic_research.learning.response_diagnostics import BLOCKS,paired_metrics,probe_metrics
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal,capture_rng
from supermimic_research.learning.checkpoint import _finite_tree,_adam_valid


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    stage_path=args.config.resolve();stage=tomllib.loads(stage_path.read_text());parent_health_path=(stage_path.parent/stage['parent_health']).resolve();health=json.loads(parent_health_path.read_text())
    probes_path=(stage_path.parent/stage['native_probe_report']).resolve();probes=json.loads(probes_path.read_text());out=args.output.resolve()
    if health['passed']!=health['total'] or health['update']!=10 or stage['end_update']!=1000 or stage['checkpoint_update']!=1000 or not stage['stop_at_first_stage_checkpoint'] or stage['fit_role']!='TRAIN' or stage['allow_cpu_fallback']:raise ValueError('Fixed reviewed continuation and checkpoint boundary')
    if probes['role']!='CHECK_ONLY_NEVER_FIT_OR_SELECT' or len(probes['barriers'])!=6 or not all(x['passed'] for x in probes['barriers']):raise ValueError('Read-only native probes require replay barriers')
    for pin in health['sources']+probes['sources']+[health['config'],health['packed']]:verify_file_pin(pin)
    checkpoint_in=verify_file_pin(health['checkpoint']);packed=json.loads(verify_file_pin(health['packed']).read_text());raw={k:np.load(verify_file_pin(v)) for k,v in packed['outputs'].items()};probe_raw=np.load(verify_file_pin(probes['probes']))
    if not args.run:
        print(json.dumps(dict(training_started=False,parent=health['checkpoint'],end_update=1000,TRAIN_rows=len(raw['TRAIN']['x']),CHECK_rows=len(raw['CHECK']['x']),probe_cases=len(probe_raw['x']))));return
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    saved=torch.load(checkpoint_in,map_location='cpu',weights_only=False);config=saved['config'];device=torch.device(saved['device'])
    if saved['update']!=10 or saved['planned_updates']!=1000 or saved['dataset_pins']!=packed['outputs'] or saved['normalizer_fit_role']!='TRAIN':raise ValueError('Exact original recipe/data/scale lineage')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,optimizer,sampler=restore_response(saved,device)
    before_checks=dict(model_raw_equal=raw_tree_equal(saved['model'],model.state_dict()),optimizer_raw_equal=raw_tree_equal(saved['optimizer'],optimizer.state_dict()),
        RNG_exact=raw_tree_equal(saved['rng'],capture_rng(device=device)),sampler_exact=raw_tree_equal(saved['sampler'],sampler.get_state()),Adam10=_adam_valid(optimizer,model,10,torch))
    if not all(before_checks.values()):raise ValueError('Resume state differs before first update')
    cache={role:{k:torch.as_tensor(z[k],dtype=torch.float32,device=device) for k in ['x','y']} for role,z in raw.items()}
    sources=[file_pin(ROOT/p) for p in ['tools/resume_mug_response.py','src/supermimic_research/learning/response_training.py','src/supermimic_research/learning/response_diagnostics.py',
        'src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_response_resumed_stage:1',sources=sources,stage=file_pin(stage_path),parent_health=file_pin(parent_health_path),parent_checkpoint=health['checkpoint'],
        probes=file_pin(probes_path),start_update=11,end_update=1000,first_stage_checkpoint=1000,normalizers_refitted=False,fit_role='TRAIN',resume_checks=before_checks,
        stop='First resumed-stage checkpoint after structural readback, read-only response checks and timing report; caller pauses Goal.',
        full_research_complete=False,policy_training=False,geometry_or_native_in_update_loop=False))
    telemetry_path=out/'gpu_samples.csv';telemetry=None
    try:
        telemetry=subprocess.Popen(['nvidia-smi','--id=0','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits','-lms','100','-f',str(telemetry_path)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except OSError:pass
    timers=[];losses=[];torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);started=time.perf_counter()
    try:
        for update in range(11,1001):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            loss,_,gradient=response_update(model,optimizer,sampler,cache['TRAIN']['x'],cache['TRAIN']['y'],config,capture_gradient=update==1000)
            end.record();timers.append((begin,end));losses.append(loss)
        torch.cuda.synchronize(device);training_wall=time.perf_counter()-started
    finally:
        if telemetry is not None:
            if telemetry.poll() is None:telemetry.terminate()
            telemetry.wait(timeout=10)
    elapsed_ms=[a.elapsed_time(b) for a,b in timers];loss_values=torch.stack(losses).cpu().tolist()
    payload=response_payload(model,optimizer,sampler,saved,update=1000,device=device);payload.update(sources=sources,parent_checkpoint=health['checkpoint'],parent_health=file_pin(parent_health_path),stage=file_pin(stage_path),gradient_snapshot=gradient)
    checkpoint=out/'checkpoint_1000.pt';atomic_checkpoint(checkpoint,payload)
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);clone,clone_opt,clone_sampler=restore_response(read,device)
    rng_exact=raw_tree_equal(read['rng'],capture_rng(device=device));a=torch.randint(len(raw['TRAIN']['x']),(config['batch_size'],),generator=sampler,device=device);b=torch.randint(len(raw['TRAIN']['x']),(config['batch_size'],),generator=clone_sampler,device=device)
    sampler_exact=torch.equal(a,b);sampler.set_state(read['sampler'])
    with torch.no_grad():
        predictions={role:model.prediction(c['x']) for role,c in cache.items()};clone_prediction=clone.prediction(cache['CHECK']['x'])
        errors={role:{name:float(((predictions[role][:,lo:hi]-cache[role]['y'][:,lo:hi])/model.y_scale[lo:hi]).square().mean()) for name,(lo,hi) in BLOCKS.items()} for role in raw}
    response=paired_metrics(model,raw['RESPONSE'],device);physical,derivatives=probe_metrics(model,probe_raw,device);np.savez_compressed(out/'derivatives.npz',**derivatives)
    checks=dict(schema_and_cursor=read['schema']==saved['schema'] and read['update']==1000,complete_model_raw_equal=raw_tree_equal(model.state_dict(),clone.state_dict()),
        complete_optimizer_raw_equal=raw_tree_equal(optimizer.state_dict(),clone_opt.state_dict()),all_Adam_steps1000=_adam_valid(optimizer,model,1000,torch) and _adam_valid(clone_opt,clone,1000,torch),
        model_optimizer_gradients_finite=_finite_tree([model.state_dict(),optimizer.state_dict(),gradient],torch),all_losses_finite=bool(np.isfinite(loss_values).all()),
        parameters_changed=not raw_tree_equal(saved['model'],model.state_dict()),RNG_raw_restore=rng_exact,next_sampler_batch_exact=bool(sampler_exact),
        next_CHECK_prediction_raw_exact=raw_tree_equal(predictions['CHECK'],clone_prediction),finite_nonnegative_Fn=all(bool(torch.isfinite(p).all() and (p[:,-3:]>=0).all()) for p in predictions.values()),
        original_scales_and_roles_unchanged=raw_tree_equal(saved['scales'],read['scales']) and read['dataset_pins']==saved['dataset_pins'] and read['normalizer_fit_role']=='TRAIN')
    samples=[]
    if telemetry_path.exists():
        for line in telemetry_path.read_text().splitlines():
            cols=line.split(',')
            try:samples.append([float(v.strip()) for v in cols[-3:]])
            except ValueError:pass
    telemetry_summary=dict(samples=len(samples),scope='Device-wide nvidia-smi samples during update loop, includes other desktop workloads; not process-only kernel occupancy.')
    if samples:
        gpu=np.array(samples);telemetry_summary.update(GPU_utilization_percent_median=float(np.median(gpu[:,0])),GPU_utilization_percent_max=float(gpu[:,0].max()),used_VRAM_MiB_max=float(gpu[:,1].max()),total_VRAM_MiB=float(gpu[0,2]))
    result=dict(schema='compound_response_resume_checkpoint_health:1',checkpoint=file_pin(checkpoint),checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),
        update=1000,resume_checks=before_checks,sources=sources,parent_health=file_pin(parent_health_path),parent_checkpoint=health['checkpoint'],stage=file_pin(stage_path),probes=file_pin(probes_path),
        errors=errors,response_pairs=response,physical_action_secants=physical,derivatives=file_pin(out/'derivatives.npz'),losses=loss_values,
        executed_updates=990,planned_updates_remaining=0,training_wall_s=training_wall,update_CUDA_ms=elapsed_ms,median_last100_update_ms=float(np.median(elapsed_ms[-100:])),
        peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(device),GPU_telemetry=telemetry_summary,
        training_running=False,first_resumed_stage_checkpoint_stop=True,pause_goal_requested=True,
        scope='Fixed one-step supervised WM recipe completed; no model selection or CHECK fitting. Structural health is separate from physical response validity. No Actor, contact reward benefit, successful full native trajectory, RGB reconstruction or generalization claim.')
    for pin in sources+health['sources']+probes['sources']+[health['checkpoint']]:verify_file_pin(pin)
    write(out/'health_1000.json',result)
    print(json.dumps(dict(health=file_pin(out/'health_1000.json'),passed=result['passed'],total=result['total'],errors=errors,response=response,
        physical_probe_AD_delta52={k:v['delta52']['AD'] for k,v in physical['by_amplitude'].items()},wall_s=training_wall,GPU_telemetry=telemetry_summary)))
    if not all(checks.values()):raise ValueError('Retain unhealthy checkpoint; no further fitting')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');main(p.parse_args())
