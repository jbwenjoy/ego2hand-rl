"""Four-arm point/finite-response ablation; preserved frozen V2 baseline."""
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_pin,verify_file_pin
from .checkpoint import _adam_valid,_finite_tree
from .gpu_monitor import GPUMonitor
from .pen_branch_fit import arm_payload,validate_fit_contract
from .pen_contrast_update import update_contrast,restore_contrast_arm
from .pen_response import PenResponseWM
from .pen_stage import metrics,write
from .world_model import atomic_checkpoint,capture_rng,raw_tree_equal


def payload(arm,cfg,scales,heads,physics,update,device,gradient):
    return dict(**arm_payload(arm,cfg,scales,heads,physics,update,device,gradient),
        pair_sampler=arm['pair_sampler'].get_state(),contrast_weight=arm['contrast_weight'])


def checkpoint_health(path,arms,cache,cfg,device,initial_buffers,initial_net,baseline,update):
    read=torch.load(path,map_location='cpu',weights_only=False);checks={}
    for name,arm in arms.items():
        saved=read['arms'][name];clone=restore_contrast_arm(saved,device)
        with torch.no_grad():p=arm['model'].prediction(cache['probe']);q=clone['model'].prediction(cache['probe'])
        local=dict(model_exact=raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()),
            Adam_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),
            sampler_exact=raw_tree_equal(arm['sampler'].get_state(),clone['sampler'].get_state()),
            pair_sampler_exact=raw_tree_equal(arm['pair_sampler'].get_state(),clone['pair_sampler'].get_state()),
            RNG_exact=raw_tree_equal(saved['rng'],capture_rng(device=device)),
            Adam_cursor=_adam_valid(clone['optimizer'],clone['model'],update,torch),
            prediction_exact=raw_tree_equal(p,q),Fn6_finite_nonnegative=bool(torch.isfinite(p).all() and (p[:,84:]>=0).all()),
            finite_model_optimizer_gradient=_finite_tree([saved['model'],saved['optimizer'],saved['gradient_snapshot']],torch),
            finite_logs=bool(np.isfinite(read['logs'][name]).all()),cursor=saved['update']==update and len(read['logs'][name])==update,
            frozen_buffers=all(raw_tree_equal(v,initial_buffers[name][k]) for k,v in arm['model'].named_buffers()),
            parameters_changed=not raw_tree_equal(initial_net,arm['model'].net.state_dict()))
        checks.update({name+'_'+k:bool(v) for k,v in local.items()})
        a,b=restore_contrast_arm(saved,device),restore_contrast_arm(saved,device)
        la,ia,pa,ga=update_contrast(a,cache,cfg,capture_gradient=True);lb,ib,pb,gb=update_contrast(b,cache,cfg,capture_gradient=True)
        parity=dict(loss=raw_tree_equal(la,lb),point_batch=raw_tree_equal(ia,ib),pair_batch=raw_tree_equal(pa,pb),gradient=raw_tree_equal(ga,gb),
            model=raw_tree_equal(a['model'].state_dict(),b['model'].state_dict()),Adam=raw_tree_equal(a['optimizer'].state_dict(),b['optimizer'].state_dict()),
            point_sampler=raw_tree_equal(a['sampler'].get_state(),b['sampler'].get_state()),pair_sampler=raw_tree_equal(a['pair_sampler'].get_state(),b['pair_sampler'].get_state()))
        checks.update({name+'_clone_next_'+k:bool(v) for k,v in parity.items()})
        if name.startswith('POINT_'):
            original=baseline['arms'][arm['mode']]
            reference=dict(model=raw_tree_equal(saved['model'],original['model']),Adam=raw_tree_equal(saved['optimizer'],original['optimizer']),
                sampler=raw_tree_equal(saved['sampler'],original['sampler']),gradient=raw_tree_equal(saved['gradient_snapshot'],original['gradient_snapshot']),
                point_logs=raw_tree_equal([r[:3] for r in read['logs'][name]],baseline['logs'][arm['mode']]))
            checks.update({name+'_frozen_V2_'+k:bool(v) for k,v in reference.items()})
        del clone,a,b
    samplers=[a['sampler'].get_state() for a in arms.values()]
    checks['matched_all_point_samplers']=all(raw_tree_equal(samplers[0],s) for s in samplers[1:])
    checks['matched_response_pair_samplers']=raw_tree_equal(arms['RESPONSE_DIRECT']['pair_sampler'].get_state(),arms['RESPONSE_IMPULSE']['pair_sampler'].get_state())
    checks['cursor']=read['update']==update and read['planned_updates']==2000
    return checks


def run_fit(root,config_path,out,*,run=False):
    root,cp,out=[Path(p).resolve() for p in [root,config_path,out]];spec=tomllib.loads(cp.read_text())
    paths={k:(cp.parent/spec[k]).resolve() for k in ['base_fit','contrasts','contrast_readback']}
    base,contrast,readback=[json.loads(p.read_text()) for p in paths.values()]
    if (base['schema']!='pen_branch_paired_fit_result:1' or base['update']!=2000 or base['training_running']
            or contrast['schema']!='pen_TRAIN_finite_action_contrasts:1' or contrast['input_role']!='NEW_TRAIN_ONLY'
            or not all(contrast['checks'].values()) or not all(readback['checks'].values())
            or verify_file_pin(readback['preparation'])!=paths['contrasts']
            or spec['contrast_weight']!=.1 or spec['pair_batch_size']!=32 or spec['pair_sampler_seed']!=7751):
        raise ValueError('Fixed predeclared response ablation and reviewed TRAIN-only contrasts')
    cfg={**tomllib.loads(verify_file_pin(base['config']).read_text()),**{k:spec[k] for k in ['contrast_weight','pair_batch_size','pair_sampler_seed']}}
    pp=verify_file_pin(base['prepared']);prep=json.loads(pp.read_text());review=json.loads(verify_file_pin(base['review']).read_text())
    validate_fit_contract(cfg,prep,review)
    if verify_file_pin(contrast['prepared'])!=pp or verify_file_pin(contrast['original_scales'])!=verify_file_pin(prep['scales']):raise ValueError('Same augmented rows/scales')
    for pin in base['sources']+prep['sources']+[review['source']]+contrast['sources']:verify_file_pin(pin)
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()};pairs=dict(np.load(verify_file_pin(contrast['output'])))
    if len(pairs['epsilon'])!=26:raise ValueError('Predeclared26 finite-contrast pairs')
    if not run:return dict(training_started=False,TRAIN_rows=len(data['TRAIN']['x']),TRAIN_pairs=26,models=4,planned_updates_per_model=2000,Actor_updates=0)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['sampler_seed']);np.random.seed(cfg['sampler_seed']);torch.manual_seed(cfg['sampler_seed'])
    scales=dict(np.load(verify_file_pin(prep['scales'])));head_scales=dict(np.load(verify_file_pin(prep['head_scales'])))
    heads=dict(DIRECT=dict(head_mean=scales['y_mean'],head_scale=scales['y_scale']),IMPULSE=head_scales)
    arms={};initial=None;buffers={}
    for package in ['POINT','RESPONSE']:
        for mode in ['DIRECT','IMPULSE']:
            name=package+'_'+mode;model=PenResponseWM(scales,heads[mode],prep['physics'],width=cfg['width'],coupled=mode=='IMPULSE').to(device)
            if initial is None:initial={k:v.detach().cpu().clone() for k,v in model.net.state_dict().items()}
            else:model.net.load_state_dict(initial,strict=True)
            buffers[name]={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
            arms[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True),
                sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed']),pair_sampler=torch.Generator(device=device).manual_seed(cfg['pair_sampler_seed']),
                mode=mode,contrast_weight=spec['contrast_weight'] if package=='RESPONSE' else 0.)
    if not all(raw_tree_equal(initial,arm['model'].net.state_dict()) for arm in arms.values()):raise ValueError('Identical four-arm NN initialization')
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    d=data['TRAIN'];valid=d['valid'];cache=dict(x=tensor(d['x'][valid],torch.float32),y=tensor(d['y'][valid],torch.float64),
        DIRECT=tensor(d['y'][valid],torch.float64),IMPULSE=tensor(d['heads'][valid],torch.float64),
        x_plus=tensor(pairs['x_plus'],torch.float32),x_minus=tensor(pairs['x_minus'],torch.float32),
        finite_contrast=tensor(pairs['finite_contrast'],torch.float64),epsilon=tensor(pairs['epsilon'],torch.float64))
    evaluation={r:(tensor(d['x'][d['valid']],torch.float32),tensor(d['y'][d['valid']],torch.float64)) for r,d in data.items()}
    cache['probe']=evaluation['CHECK'][0]
    sources=[file_pin(root/p) for p in ['tools/train_pen_contrasts.py','src/supermimic_research/learning/pen_contrast_fit.py',
        'src/supermimic_research/learning/pen_contrast_update.py']]+base['sources']
    lineage=dict(config=file_pin(cp),base_fit=file_pin(paths['base_fit']),contrasts=file_pin(paths['contrasts']),contrast_readback=file_pin(paths['contrast_readback']),
        prepared=base['prepared'],review=base['review'],dataset_pins=base['dataset_pins'],sources=sources)
    out.mkdir();write(out/'intent.json',dict(schema='pen_response_loss_ablation:1',**lineage,planned_updates_per_model=2000,models=list(arms),
        matched_initial_NN=True,point_batch=256,response_pair_batch=32,response_weight=.1,pair_seed=7751,
        response_loss='mean object12 + mean source-BODY wrench6 squared finite-action contrast error, divided by unchanged frozen V2 output scales',
        scope='Same point streams and optimizer counts, additional64 pair endpoints/response-arm update; not equal FLOPs. No CHECK fitting/selection or Actor training.'))
    logs={n:[] for n in arms};gradients={};curve={};healths={};timers=[];monitor=GPUMonitor();monitor.start();start=time.perf_counter()
    try:
        for update in range(1,2001):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for name,arm in arms.items():
                value,ids,pair_ids,gradient=update_contrast(arm,cache,cfg,capture_gradient=update in [100,2000]);logs[name].append(value)
                if gradient is not None:gradients[name]=gradient
            end.record();timers.append((begin,end))
            if update in [100,500,1000,2000]:curve[str(update)]={n:{r:metrics(a['model'],x,y)[0] for r,(x,y) in evaluation.items()} for n,a in arms.items()}
            if update in [100,2000]:
                path=out/f'pen_response_loss_checkpoint_{update:04d}.pt'
                atomic_checkpoint(path,dict(schema='pen_response_loss_ablation_checkpoint:1',**lineage,update=update,planned_updates=2000,
                    arms={n:payload(a,cfg,scales,heads[a['mode']],prep['physics'],update,device,gradients[n]) for n,a in arms.items()},
                    logs={n:torch.stack(v).cpu().tolist() for n,v in logs.items()}))
                bh=json.loads(verify_file_pin(base['checkpoint_health'][str(update)]).read_text())
                baseline=torch.load(verify_file_pin(bh['checkpoint']),map_location='cpu',weights_only=False)
                checks=checkpoint_health(path,arms,cache,cfg,device,buffers,initial,baseline,update)
                health=dict(schema='pen_response_loss_checkpoint_health:1',checkpoint=file_pin(path),checks=checks,passed=sum(checks.values()),total=len(checks),
                    update=update,live_cursor_per_model=update,clone_only_optimizer_steps_at_boundary=8,Actor_updates=0,
                    remaining_all_four_compute_estimate_s=(2000-update)*float(np.median([a.elapsed_time(b) for a,b in timers[-50:]]))/1000)
                hp=out/f'health_{update:04d}.json';write(hp,health);healths[str(update)]=file_pin(hp)
                if not all(checks.values()):raise ValueError('Preserve failed restore/baseline check; no more updates')
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    for pin in sources:verify_file_pin(pin)
    result=dict(schema='pen_response_loss_ablation_result:1',**lineage,checkpoint=file_pin(path),checkpoint_health=healths,curve=curve,update=2000,
        live_models=4,live_updates_per_model=2000,clone_only_optimizer_steps=16,Actor_updates=0,training_running=False,
        point_rows_drawn_per_model=2000*256,response_pairs_drawn_per_response_model=2000*32,
        extra_pair_endpoint_forwards_per_response_model=2000*64,fit_and_QA_wall_s=wall,
        CUDA_ms_per_four_model_update=[a.elapsed_time(b) for a,b in timers],GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors),
        hardware=dict(GPU_name=torch.cuda.get_device_name(device),Torch_version=torch.__version__),
        scope='Fixed predeclared four-arm finite-response loss ablation only. All CHECK curves development evaluation, no model/weight selection. No physical derivative/task/Actor/ego qualification by checkpoint health.')
    write(out/'report.json',result);return dict(report=file_pin(out/'report.json'),checkpoint=file_pin(path),health=healths,fit_and_QA_wall_s=wall)
