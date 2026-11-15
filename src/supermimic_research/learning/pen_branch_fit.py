"""Matched fixed-budget Fn6 fit on explicitly augmented TRAIN, frozen old runs."""
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
from .impulse_stage import update_impulse
from .pen_stage import metrics,restore_arm,write
from .pen_response import PenResponseWM,INPUT_SCHEMA,OUTPUT_SCHEMA,HEAD_SCHEMA
from .world_model import atomic_checkpoint,capture_rng,raw_tree_equal


def validate_fit_contract(cfg,prep,review):
    if (prep['schema']!='pen_branch_response_preparation:1'
            or review['schema']!='pen_branch_preparation_saved_review:1'
            or not prep['checks'] or not all(prep['checks'].values())
            or review['passed']!=review['total'] or not review['total']
            or set(prep['outputs'])!={'TRAIN','CHECK','NEW_TRAIN','SEQUENCE_CHECK'}
            or prep['input_schema']!=INPUT_SCHEMA or prep['output_schema']!=OUTPUT_SCHEMA
            or cfg['first_checkpoint_update']!=100 or cfg['planned_updates']!=2000
            or not cfg['continue_after_first_healthy_checkpoint']):
        raise ValueError('Reviewed explicit augmented Fn6 dataset, fixed100/2000 stages')


def arm_payload(arm,cfg,scales,head_scales,physics,update,device,gradient):
    return dict(schema='controlled_pen_response_arm:1',mode=arm['mode'],coupled=arm['model'].coupled,
        width=cfg['width'],config=cfg,scales=scales,head_scales=head_scales,physics=physics,update=update,
        input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,head_schema=HEAD_SCHEMA if arm['model'].coupled else OUTPUT_SCHEMA,
        model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),sampler=arm['sampler'].get_state(),
        rng=capture_rng(device=device),gradient_snapshot=gradient)


def verify_checkpoint(path,arms,cache,cfg,device,update,initial,scales):
    saved=torch.load(path,map_location='cpu',weights_only=False);checks={}
    for mode,arm in arms.items():
        payload=saved['arms'][mode];clone=restore_arm(payload,device)
        with torch.no_grad():a=arm['model'].prediction(cache['probe']);b=clone['model'].prediction(cache['probe'])
        local=dict(model_exact=raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()),
            optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),
            sampler_exact=raw_tree_equal(arm['sampler'].get_state(),clone['sampler'].get_state()),
            Adam_cursor=_adam_valid(clone['optimizer'],clone['model'],update,torch),
            finite_model_optimizer_gradient=_finite_tree([payload['model'],payload['optimizer'],payload['gradient_snapshot']],torch),
            finite_logs=bool(np.isfinite(saved['logs'][mode]).all()),prediction_exact=raw_tree_equal(a,b),
            RNG_exact=raw_tree_equal(payload['rng'],capture_rng(device=device)),
            finite_nonnegative_Fn6=bool(torch.isfinite(a).all() and (a[:,84:]>=0).all()),
            parameters_changed=not raw_tree_equal(initial,arm['model'].net.state_dict()),
            frozen_buffers=all(raw_tree_equal(v,scales[mode][k]) for k,v in arm['model'].named_buffers()),
            cursor=payload['update']==update and len(saved['logs'][mode])==update)
        checks.update({mode+'_'+k:bool(v) for k,v in local.items()})
        x,y=restore_arm(payload,device),restore_arm(payload,device)
        la,ia,ga=update_impulse(x,cache,cfg,capture_gradient=True);lb,ib,gb=update_impulse(y,cache,cfg,capture_gradient=True)
        parity=dict(loss=raw_tree_equal(la,lb),batch=raw_tree_equal(ia,ib),gradient=raw_tree_equal(ga,gb),
            model=raw_tree_equal(x['model'].state_dict(),y['model'].state_dict()),optimizer=raw_tree_equal(x['optimizer'].state_dict(),y['optimizer'].state_dict()),
            sampler=raw_tree_equal(x['sampler'].get_state(),y['sampler'].get_state()))
        checks.update({mode+'_clone_next_'+k:v for k,v in parity.items()})
        del clone,x,y
    checks['matched_sampler']=raw_tree_equal(saved['arms']['DIRECT']['sampler'],saved['arms']['IMPULSE']['sampler'])
    checks['pair_cursor']=saved['update']==update and saved['planned_updates']==2000
    return checks


def run_fit(root,config_path,out,*,run=False):
    root,cp,out=[Path(p).resolve() for p in [root,config_path,out]];cfg=tomllib.loads(cp.read_text())
    pp=(cp.parent/cfg['prepared']).resolve();rp=(cp.parent/cfg['review']).resolve()
    prep,review=[json.loads(p.read_text()) for p in [pp,rp]];validate_fit_contract(cfg,prep,review)
    if verify_file_pin(review['prepared'])!=pp:raise ValueError('Exact reviewed preparation')
    pins=prep['sources']+[review['source'],prep['scales'],prep['head_scales']]+list(prep['inputs'].values())
    for pin in pins:verify_file_pin(pin)
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()}
    if not run:return dict(training_started=False,counts=prep['counts'],first_checkpoint=100,planned_updates=2000,Actor_updates=0)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA fit')
    device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['sampler_seed']);np.random.seed(cfg['sampler_seed']);torch.manual_seed(cfg['sampler_seed'])
    scale=dict(np.load(verify_file_pin(prep['scales'])));hs=dict(np.load(verify_file_pin(prep['head_scales'])))
    head_scales=dict(DIRECT=dict(head_mean=scale['y_mean'],head_scale=scale['y_scale']),IMPULSE=hs)
    arms={};initial=None;buffers={}
    for mode in ['DIRECT','IMPULSE']:
        model=PenResponseWM(scale,head_scales[mode],prep['physics'],width=cfg['width'],coupled=mode=='IMPULSE').to(device)
        if initial is None:initial={k:v.detach().cpu().clone() for k,v in model.net.state_dict().items()}
        else:model.net.load_state_dict(initial,strict=True)
        buffers[mode]={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        arms[mode]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True),
            sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed']),mode=mode)
    if not raw_tree_equal(arms['DIRECT']['model'].net.state_dict(),arms['IMPULSE']['model'].net.state_dict()):raise ValueError('Matched initial NN')
    tensor=lambda v,dtype:torch.as_tensor(v,dtype=dtype,device=device)
    train=data['TRAIN'];valid=train['valid'];cache=dict(x=tensor(train['x'][valid],torch.float32),y=tensor(train['y'][valid],torch.float64),
        DIRECT=tensor(train['y'][valid],torch.float64),IMPULSE=tensor(train['heads'][valid],torch.float64))
    evaluation={r:(tensor(d['x'][d['valid']],torch.float32),tensor(d['y'][d['valid']],torch.float64)) for r,d in data.items()}
    cache['probe']=evaluation['CHECK'][0];sources=[file_pin(root/p) for p in [
        'tools/train_pen_branches.py','src/supermimic_research/learning/pen_branch_fit.py','src/supermimic_research/learning/pen_response.py',
        'src/supermimic_research/learning/pen_stage.py','src/supermimic_research/learning/impulse_stage.py',
        'src/supermimic_research/learning/impulse_response.py','src/supermimic_research/learning/compound_response.py',
        'src/supermimic_research/mechanics/impulse_codec.py','src/supermimic_research/learning/world_model.py',
        'src/supermimic_research/learning/checkpoint.py','src/supermimic_research/learning/gpu_monitor.py']]
    lineage=dict(config=file_pin(cp),prepared=file_pin(pp),review=file_pin(rp),sources=sources,dataset_pins=prep['outputs'])
    out.mkdir();write(out/'intent.json',dict(schema='pen_branch_paired_fit:1',**lineage,counts=prep['counts'],matched_NN_initialization=True,
        optimizer_initialization='Fresh matched arms; original v1 checkpoint unchanged. New declared TRAIN scales only.',
        policy_updates=0,CHECK_for_optimizer_or_selection=False,planned_updates=2000,checkpoint_updates=[100,2000],
        difference='DIRECT versus measured-wrench/numerical-defect IMPULSE representation+auxiliary package, not Actor support-only isolation.'))
    logs={m:[] for m in arms};curve={};gradient={};healths={};times=[];monitor=GPUMonitor();monitor.start();start=time.perf_counter()
    curve['0']={m:{r:metrics(arm['model'],x,y)[0] for r,(x,y) in evaluation.items()} for m,arm in arms.items()}
    try:
        for update in range(1,2001):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for mode,arm in arms.items():
                value,indices,g=update_impulse(arm,cache,cfg,capture_gradient=update in [100,2000]);logs[mode].append(value)
                if g is not None:gradient[mode]=g
            end.record();times.append((begin,end))
            if update in [100,500,1000,2000]:
                curve[str(update)]={m:{r:metrics(arm['model'],x,y)[0] for r,(x,y) in evaluation.items()} for m,arm in arms.items()}
            if update in [100,2000]:
                full_logs={m:torch.stack(v).cpu().tolist() for m,v in logs.items()}
                path=out/f'pen_branch_checkpoint_{update:04d}.pt'
                payloads={m:arm_payload(arm,cfg,scale,head_scales[m],prep['physics'],update,device,gradient[m]) for m,arm in arms.items()}
                atomic_checkpoint(path,dict(schema='pen_branch_response_pair_checkpoint:1',**lineage,update=update,planned_updates=2000,arms=payloads,logs=full_logs))
                checks=verify_checkpoint(path,arms,cache,cfg,device,update,initial,buffers)
                health=dict(schema='pen_branch_fit_checkpoint_health:1',checkpoint=file_pin(path),update=update,checks=checks,passed=sum(checks.values()),total=len(checks),
                    live_cursor=update,extra_clone_updates_at_this_boundary=4,Actor_updates=0,remaining_updates=2000-update,
                    remaining_paired_compute_estimate_s=(2000-update)*float(np.median([a.elapsed_time(b) for a,b in times[-50:]]))/1000,
                    scope='Checkpoint numerical/readback health only; no native response, predictive accuracy or task readiness claim.')
                write(out/f'health_{update:04d}.json',health);healths[str(update)]=file_pin(out/f'health_{update:04d}.json')
                if not all(checks.values()):raise ValueError('Retain failed checkpoint; no further live updates')
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    for pin in sources+pins:verify_file_pin(pin)
    result=dict(schema='pen_branch_paired_fit_result:1',**lineage,checkpoint=file_pin(path),checkpoint_health=healths,curve=curve,
        update=2000,live_updates_per_arm=2000,extra_clone_optimizer_updates=8,Actor_updates=0,training_running=False,
        fit_and_checkpoint_QA_wall_s=wall,CUDA_ms_per_paired_update=[a.elapsed_time(b) for a,b in times],
        GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors),hardware=dict(GPU_name=torch.cuda.get_device_name(device),Torch_version=torch.__version__),
        scope='Fresh349-row augmented TRAIN fixed-budget model fit. New64 action-direction CHECK and old80 sequence CHECK only for evaluation. Not an Actor/support objective isolation, real ego result or total-cost acceleration.')
    write(out/'report.json',result);return dict(report=file_pin(out/'report.json'),checkpoint=file_pin(path),checkpoint_health=healths,fit_and_checkpoint_QA_wall_s=wall,metrics=curve['2000'])
