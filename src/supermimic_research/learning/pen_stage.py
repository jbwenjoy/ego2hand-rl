"""Matched pen Fn6 response models, stopping at the first checkpoint boundary."""
import json,random,time,tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from .pen_response import (PenResponseWM,INPUT_SCHEMA,OUTPUT_SCHEMA,head_schema,pen_families,
                          model_from_arm,training_cache,training_head_scales)
from .impulse_stage import update_impulse
from .world_model import capture_rng,restore_rng,raw_tree_equal,atomic_checkpoint
from .checkpoint import _finite_tree,_adam_valid
from .gpu_monitor import GPUMonitor


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def metrics(model,x,y):
    with torch.no_grad():
        p=model.prediction(x)
        blocks=dict(delta52=(0,52),object12=(40,52),warm26=(52,78),wrench6=(78,84),Fn6=(84,90))
        result={name:float(((p[:,lo:hi]-y[:,lo:hi])/model.y_scale[lo:hi]).square().mean()) for name,(lo,hi) in blocks.items()}
    return result,p


def payload(arm,cfg,scales,head_scales,physics,device,gradient):
    version=cfg.get('schema',1)
    return dict(schema=f'controlled_pen_response_arm:{version}',mode=arm['mode'],coupled=arm['model'].coupled,width=cfg['width'],
        **(dict(pose_baseline=arm['model'].pose_baseline) if version==2 else {}),
        config=cfg,scales=scales,head_scales=head_scales,physics=physics,update=100,input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,
        head_schema=head_schema(arm['mode']),
        model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),sampler=arm['sampler'].get_state(),
        rng=capture_rng(device=device),gradient_snapshot=gradient)


def restore_arm(saved,device):
    m=model_from_arm(saved,device);c=saved['config']
    opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],fused=True);opt.load_state_dict(saved['optimizer'])
    sampler=torch.Generator(device=device);sampler.set_state(saved['sampler']);restore_rng(saved['rng'],device=device)
    return dict(model=m,optimizer=opt,sampler=sampler,mode=saved['mode'])


def run_pen_stage(root,config_path,out,*,run=False):
    root,cp,out=Path(root).resolve(),Path(config_path).resolve(),Path(out).resolve();cfg=tomllib.loads(cp.read_text())
    modes=pen_families(cfg);version=cfg.get('schema',1)
    pp=(cp.parent/cfg['prepared']).resolve();rp=(cp.parent/cfg['review']).resolve();prep=json.loads(pp.read_text());review=json.loads(rp.read_text())
    if not all(prep['checks'].values()) or review['passed']!=review['total'] or verify_file_pin(review['prepared']).resolve()!=pp:
        raise ValueError('Healthy prepared data and independent indexed/frame review required')
    if cfg['first_checkpoint_update']!=100 or cfg['planned_updates']!=2000 or not cfg['stop_at_first_healthy_checkpoint']:
        raise ValueError('Standing first100 checkpoint boundary')
    for pin in prep['sources']+[review['source']]:verify_file_pin(pin)
    data={role:dict(np.load(verify_file_pin(pin))) for role,pin in prep['outputs'].items()}
    scales=dict(np.load(verify_file_pin(prep['scales'])));coupled_scales=dict(np.load(verify_file_pin(prep['head_scales'])))
    if not run:return dict(training_started=False,TRAIN_valid=int(data['TRAIN']['valid'].sum()),CHECK_valid=int(data['CHECK']['valid'].sum()),first_checkpoint=100,planned_updates=2000,Actor_updates=0)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    hardware=dict(GPU_name=torch.cuda.get_device_name(device),Torch_version=torch.__version__,device=str(device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['sampler_seed']);np.random.seed(cfg['sampler_seed']);torch.manual_seed(cfg['sampler_seed'])
    sources=[file_pin(root/q) for q in ['tools/train_pen_response.py','src/supermimic_research/learning/pen_stage.py','src/supermimic_research/learning/pen_response.py',
        'src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/impulse_response.py','src/supermimic_research/learning/impulse_stage.py',
        'src/supermimic_research/learning/world_model.py','src/supermimic_research/learning/checkpoint.py','src/supermimic_research/learning/gpu_monitor.py','src/supermimic_research/mechanics/impulse_codec.py']]
    out.mkdir();write(out/'intent.json',dict(schema=f'controlled_pen_paired_WM_stage:{version}',config=file_pin(cp),prepared=file_pin(pp),review=file_pin(rp),sources=sources,
        first_checkpoint_update=100,planned_updates=2000,stop_at_first_healthy_checkpoint=True,Actor_updates=0,hardware=hardware,
        difference=('DIRECT versus DIRECT_CV additive H20 kinematic pose prior with TRAIN residual head mean; same parameter count/scales/data/sampler/budget.' if version==2 else
            'DIRECT independent velocity heads versus measured source-BODY wrench/learned WORLD numerical-defect impulse closure, corresponding auxiliary labels; same NN weights/rows/batches/optimizer budget.'),
        scope='Fresh single-clip nonexpert controlled-dynamics component experiment; not Actor A1/A2, expert imitation, successful retargeting or ego generalization.'))
    valid=data['TRAIN']['valid'];x=data['TRAIN']['x'][valid];y=data['TRAIN']['y'][valid];heads=data['TRAIN']['heads'][valid]
    cache=training_cache(data['TRAIN'],device,modes)
    check=data['CHECK'];mask=check['valid'];check_x=torch.as_tensor(check['x'][mask],dtype=torch.float32,device=device);check_y=torch.as_tensor(check['y'][mask],dtype=torch.float64,device=device)
    head_scales=training_head_scales(scales,coupled_scales,x,y,modes)
    arms={};initial_net=None;initial_check={}
    for mode in modes:
        model=PenResponseWM(scales,head_scales[mode],prep['physics'],width=cfg['width'],coupled=mode=='IMPULSE',
                            pose_baseline='CONSTANT_VELOCITY' if mode=='DIRECT_CV' else 'ZERO').to(device)
        if initial_net is None:initial_net={k:v.detach().cpu().clone() for k,v in model.net.state_dict().items()}
        else:model.net.load_state_dict(initial_net,strict=True)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
        sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed'])
        arms[mode]=dict(model=model,optimizer=opt,sampler=sampler,mode=mode);initial_check[mode],_=metrics(model,check_x,check_y)
    if not raw_tree_equal(arms[modes[0]]['model'].net.state_dict(),arms[modes[1]]['model'].net.state_dict()):raise ValueError('Matched NN initialization')
    logs={m:[] for m in arms};timers=[];gradients={};sample_last={};monitor=GPUMonitor();monitor.start();torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    try:
        for update in range(1,101):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                value,indices,gradient=update_impulse(arm,cache,cfg,capture_gradient=update==100);logs[mode].append(value)
                if update==100:gradients[mode]=gradient;sample_last[mode]=indices.detach().clone()
            end.record();timers.append((begin,end))
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    milliseconds=[a.elapsed_time(b) for a,b in timers];training_peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
    logs={mode:torch.stack(v).cpu().tolist() for mode,v in logs.items()}
    saved_arms={mode:payload(arm,cfg,scales,head_scales[mode],prep['physics'],device,gradients[mode]) for mode,arm in arms.items()}
    checkpoint=out/'pen_response_checkpoint_0100.pt'
    atomic_checkpoint(checkpoint,dict(schema=f'controlled_pen_response_pair_checkpoint:{version}',update=100,planned_updates=2000,arms=saved_arms,
        config=file_pin(cp),prepared=file_pin(pp),review=file_pin(rp),sources=sources,dataset_pins=prep['outputs'],logs=logs))
    # No live optimizer runs after the saved first100 boundary.
    read=torch.load(checkpoint,map_location='cpu',weights_only=False);checks={};errors={};gradmetrics={};parity={}
    for mode,arm in arms.items():
        saved=read['arms'][mode];clone=restore_arm(saved,device);prediction_error,prediction=metrics(arm['model'],check_x,check_y);_,restored_prediction=metrics(clone['model'],check_x,check_y)
        errors[mode]=prediction_error
        sample1=torch.randint(len(x),(cfg['batch_size'],),generator=arm['sampler'],device=device);sample2=torch.randint(len(x),(cfg['batch_size'],),generator=clone['sampler'],device=device)
        next_batch_exact=torch.equal(sample1,sample2);arm['sampler'].set_state(saved['sampler'])
        check_input=check_x.detach().clone().requires_grad_(True);P=clone['model'].prediction(check_input)
        g=torch.autograd.grad(P[:,78:84].square().sum(),check_input)[0][:,81:101]
        gradmetrics[mode]=dict(finite=bool(torch.isfinite(g).all()),mean_abs=float(g.abs().mean()),max_abs=float(g.abs().max()),scope='Network derivative only; physical intervention/gradient accuracy not yet established.')
        local=dict(model_exact=raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()),optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),
            Adam100=_adam_valid(arm['optimizer'],arm['model'],100,torch) and _adam_valid(clone['optimizer'],clone['model'],100,torch),
            finite_model_optimizer_gradients=_finite_tree([arm['model'].state_dict(),arm['optimizer'].state_dict(),gradients[mode]],torch),
            finite_loss_logs=bool(np.isfinite(logs[mode]).all()),parameters_changed=not raw_tree_equal(initial_net,arm['model'].net.state_dict()),
            next_batch_exact=next_batch_exact,RNG_exact=raw_tree_equal(saved['rng'],capture_rng(device=device)),
            CHECK_prediction_exact=raw_tree_equal(prediction,restored_prediction),CHECK_Fn6_finite_nonnegative=bool(torch.isfinite(prediction).all() and (prediction[:,84:]>=0).all()),
            CHECK_PD_gradient_finite=bool(torch.isfinite(g).all()),
            frozen_scales_exact=all(raw_tree_equal(torch.tensor(v,dtype=torch.float32),arm['model'].state_dict()[k]) for k,v in scales.items()),
            frozen_physics_exact=all(raw_tree_equal(torch.tensor(v,dtype=torch.float64),arm['model'].state_dict()[k]) for k,v in dict(mass=prep['physics']['mass'],inertia_BODY=prep['physics']['inertia_BODY'],inverse_inertia_BODY=np.linalg.inv(prep['physics']['inertia_BODY']),gravity=prep['physics']['gravity']).items()),
            head_scales_exact=all(raw_tree_equal(torch.tensor(v,dtype=torch.float32),arm['model'].state_dict()[k]) for k,v in head_scales[mode].items()))
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()})
        # Two restored clones get one update each; live model remains at100.
        a,b=restore_arm(saved,device),restore_arm(saved,device)
        la,ia,ga=update_impulse(a,cache,cfg,capture_gradient=True);lb,ib,gb=update_impulse(b,cache,cfg,capture_gradient=True)
        parity[mode]=dict(loss_exact=raw_tree_equal(la,lb),batch_exact=raw_tree_equal(ia,ib),gradients_exact=raw_tree_equal(ga,gb),
            model_exact=raw_tree_equal(a['model'].state_dict(),b['model'].state_dict()),optimizer_exact=raw_tree_equal(a['optimizer'].state_dict(),b['optimizer'].state_dict()),sampler_exact=raw_tree_equal(a['sampler'].get_state(),b['sampler'].get_state()))
        checks.update({mode+'_next_update_'+k:v for k,v in parity[mode].items()})
        del clone,a,b,P,g,check_input
    checks.update(matched_last_batch=raw_tree_equal(sample_last[modes[0]],sample_last[modes[1]]),matched_sampler_cursor=raw_tree_equal(read['arms'][modes[0]]['sampler'],read['arms'][modes[1]]['sampler']),
                  saved_cursor=read['update']==100 and read['planned_updates']==2000)
    if version==2:
        checks.update(matched_parameter_count=len({sum(p.numel() for p in a['model'].parameters()) for a in arms.values()})==1,
            matched_physical_loss_scales=raw_tree_equal(arms[modes[0]]['model'].y_scale,arms[modes[1]]['model'].y_scale))
    eta=1900*float(np.median(milliseconds[-50:]))/1000
    samples=[s for s in monitor.samples if s['start']<=start+wall and s['end']>=start]
    telemetry=dict(samples=samples,errors=monitor.errors,scope='Device-wide query windows overlapping this short100-paired-update compute interval; not whole-pipeline utilization.')
    for pin in sources+prep['sources']+[file_pin(pp),file_pin(rp)]:verify_file_pin(pin)
    result=dict(schema=f'controlled_pen_first100_health:{version}',checkpoint=file_pin(checkpoint),config=file_pin(cp),prepared=file_pin(pp),review=file_pin(rp),sources=sources,
        checks=checks,passed=sum(checks.values()),total=len(checks),initial_CHECK_metrics=initial_check,CHECK_metrics=errors,network_PD_gradients=gradmetrics,
        first_next_update_parity=parity,logs=logs,update=100,planned_updates=2000,remaining_updates=1900,paired_compute_wall_s=wall,
        CUDA_ms_per_paired_update=milliseconds,remaining_paired_compute_estimate_s=eta,training_peak_bytes=training_peak,GPU_telemetry=telemetry,hardware=hardware,
        extra_clone_optimizer_updates=4,Actor_optimizer_updates=0,training_running=False,pause_goal_requested=version==1,
        estimate_scope='Last50 CUDA-event median times1900 paired updates; excludes startup, collection, packing, checkpoint/diagnostic IO and later policy/gradient/rollout work.',
        scope=f'First checkpoint of matched {modes} Fn6 WM components only. Finite network gradients do not validate physical action gradients, predictive accuracy, full task success, Actor A1/A2 benefit or ego generalization.')
    write(out/'health_0100.json',result)
    if not all(checks.values()):raise ValueError('Retain unhealthy first checkpoint; no further updates or Goal pause')
    return dict(training_started=True,training_running=False,checkpoint=file_pin(checkpoint),health=file_pin(out/'health_0100.json'),
        passed=result['passed'],total=result['total'],remaining_paired_compute_estimate_s=eta,paired_compute_wall_s=wall,
        CHECK_metrics=errors,pause_goal_requested=version==1)
