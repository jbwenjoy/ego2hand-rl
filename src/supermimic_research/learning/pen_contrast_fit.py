"""Shared historical/fresh four-arm and full20 six-arm contrast trainer."""
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .checkpoint import _adam_valid,_finite_tree
from .gpu_monitor import GPUMonitor
from .pen_branch_fit import arm_payload,validate_fit_contract
from .pen_contrast_update import update_contrast,restore_contrast_arm
from .pen_response import PenResponseWM
from .pen_response_dataset import common_response_data
from .pen_fullrank_study import model_config as fullrank_model_config,load_common as load_fullrank_common,pair_contract,evaluation_roles
from .pen_stage import metrics,write
from .world_model import atomic_checkpoint,capture_rng,raw_tree_equal


def payload(arm,cfg,scales,heads,physics,update,device,gradient):
    result=dict(**arm_payload(arm,cfg,scales,heads,physics,update,device,gradient),
        pair_sampler=arm['pair_sampler'].get_state(),contrast_weight=arm['contrast_weight'])
    if 'pair_contract' in arm:result['pair_contract']=arm['pair_contract'].copy()
    return result


def checkpoint_health(path,arms,cache,cfg,device,initial_buffers,initial_net,baseline,update,*,arm_caches=None):
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
        if 'pair_contract' in arm:checks[name+'_pair_contract_exact']=arm['pair_contract']==saved['pair_contract']==clone['pair_contract']
        a,b=restore_contrast_arm(saved,device),restore_contrast_arm(saved,device)
        selected=cache if arm_caches is None else arm_caches[name]
        la,ia,pa,ga=update_contrast(a,selected,cfg,capture_gradient=True);lb,ib,pb,gb=update_contrast(b,selected,cfg,capture_gradient=True)
        parity=dict(loss=raw_tree_equal(la,lb),point_batch=raw_tree_equal(ia,ib),pair_batch=raw_tree_equal(pa,pb),gradient=raw_tree_equal(ga,gb),
            model=raw_tree_equal(a['model'].state_dict(),b['model'].state_dict()),Adam=raw_tree_equal(a['optimizer'].state_dict(),b['optimizer'].state_dict()),
            point_sampler=raw_tree_equal(a['sampler'].get_state(),b['sampler'].get_state()),pair_sampler=raw_tree_equal(a['pair_sampler'].get_state(),b['pair_sampler'].get_state()))
        checks.update({name+'_clone_next_'+k:bool(v) for k,v in parity.items()})
        if baseline is not None and name.startswith('POINT_'):
            original=baseline['arms'][arm['mode']]
            reference=dict(model=raw_tree_equal(saved['model'],original['model']),Adam=raw_tree_equal(saved['optimizer'],original['optimizer']),
                sampler=raw_tree_equal(saved['sampler'],original['sampler']),gradient=raw_tree_equal(saved['gradient_snapshot'],original['gradient_snapshot']),
                point_logs=raw_tree_equal([r[:3] for r in read['logs'][name]],baseline['logs'][arm['mode']]))
            checks.update({name+'_frozen_V2_'+k:bool(v) for k,v in reference.items()})
        del clone,a,b
    samplers=[a['sampler'].get_state() for a in arms.values()]
    checks['matched_all_point_samplers']=all(raw_tree_equal(samplers[0],s) for s in samplers[1:])
    responses=[a['pair_sampler'].get_state() for a in arms.values() if a['contrast_weight']!=0]
    checks['matched_response_pair_samplers']=all(raw_tree_equal(responses[0],s) for s in responses[1:])
    checks['cursor']=read['update']==update and read['planned_updates']==2000
    return checks


def common_model_config(spec):
    """Study/data schema2 explicitly creates fresh model-family schema1 arms.

    Model schema2 means DIRECT_VS_DIRECT_CV elsewhere and is never reinterpreted.
    This adapter imports no historical network or optimizer.
    """
    if (spec.get('schema')!=2 or spec.get('models')!=['POINT_DIRECT','POINT_IMPULSE','RESPONSE_DIRECT','RESPONSE_IMPULSE']
            or spec.get('CHECK_role')!='DEVELOPMENT_NEVER_FIT_NOT_FINAL_TEST'
            or spec.get('first_checkpoint_update')!=100 or spec.get('planned_updates')!=2000
            or not spec.get('continue_after_first_healthy_checkpoint') or 'model_comparison' in spec
            or spec.get('contrast_weight')!=.1 or spec.get('pair_batch_size')!=32 or spec.get('pair_sampler_seed')!=7751):
        raise ValueError('Explicit fresh four-arm common-data study;no CV/schema reinterpretation')
    return {**spec,'schema':1,'study_schema':2,'dataset_schema':'contact_pen_common_response_data:2'}


def run_fit(root,config_path,out,*,run=False,common_data=None):
    entry_started=time.perf_counter()
    root,cp,out=[Path(p).resolve() for p in [root,config_path,out]];spec=tomllib.loads(cp.read_text())
    rank_study=spec.get('schema')==3;fresh=spec.get('schema',1) in [2,3];receipt=None;subsets=None
    if rank_study:
        cfg=fullrank_model_config(spec)
        if common_data is None:raise ValueError('Explicit full20 common-data report required')
        pp,prep,data,pairs,subsets=load_fullrank_common(cp,common_data);base=None
        source_paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (root/'src'/pkg).rglob('*.py')})+[root/'tools/train_pen_contrasts.py',cp,verify_file_pin(prep['protocol']),(cp.parent/spec['protocol']).resolve()]
        sources=[file_pin(p) for p in source_paths]
        lineage=dict(config=file_pin(cp),prepared=file_pin(pp),TRAIN_preparation=prep['TRAIN_preparation'],VALIDATE_preparation=prep['VALIDATE_preparation'],dataset_pins=prep['outputs'],pairs=prep['pairs'],pair_subsets=prep['pair_subsets'],scales=prep['scales'],head_scales=prep['head_scales'],sources=sources)
        expected_pairs=949
    elif fresh:
        cfg=common_model_config(spec)
        if common_data is None:raise ValueError('Explicit frozen common-data report required')
        pp=Path(common_data).resolve();prep=json.loads(pp.read_text());base=None
        if (prep['schema']!='contact_pen_common_response_data:2' or prep['config']!=file_pin(cp)
                or prep['TRAIN_preparation']!=file_pin((cp.parent/spec['train_preparation']).resolve())
                or prep['CHECK_review']!=file_pin((cp.parent/spec['check_review']).resolve())
                or prep['passed']!=prep['total'] or not all(prep['checks'].values()) or prep['TRAIN_valid']!=808
                or prep['CHECK_valid']!=324 or prep['TRAIN_pairs']!=188 or prep['scale_input_role']!='VALID_TRAIN_ONLY'
                or prep['CHECK_role']!=spec['CHECK_role']):raise ValueError('Exact role-checked registered808/324/188 common data')
        verify_source_snapshot(json.loads(verify_file_pin(prep['source_receipt']).read_text()))
        data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()};pairs=dict(np.load(verify_file_pin(prep['pairs'])))
        tr=json.loads(verify_file_pin(prep['TRAIN_preparation']).read_text());cr=json.loads(verify_file_pin(prep['CHECK_review']).read_text());directions=np.load(verify_file_pin(prep['directions']))['directions']
        rebuilt=common_response_data(tr,data['TRAIN'],cr,data['CHECK'],directions)
        for key in ['scales','head_scales','pairs']:
            saved=dict(np.load(verify_file_pin(prep[key])))
            if set(saved)!=set(rebuilt[key]) or any(not np.array_equal(v,saved[k]) for k,v in rebuilt[key].items()):raise ValueError('Frozen common data/scales differ from TRAIN-only reconstruction')
        source_paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (root/'src'/pkg).rglob('*.py')})+[root/'tools/train_pen_contrasts.py',cp,verify_file_pin(prep['protocol'])]
        sources=[file_pin(p) for p in source_paths]
        lineage=dict(config=file_pin(cp),prepared=file_pin(pp),TRAIN_preparation=prep['TRAIN_preparation'],CHECK_review=prep['CHECK_review'],dataset_pins=prep['outputs'],pairs=prep['pairs'],scales=prep['scales'],head_scales=prep['head_scales'],sources=sources)
        expected_pairs=188
    else:
        if spec.get('schema',1)!=1 or common_data is not None:raise ValueError('Historical study cannot silently accept common data')
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
        sources=[file_pin(root/p) for p in ['tools/train_pen_contrasts.py','src/supermimic_research/learning/pen_contrast_fit.py','src/supermimic_research/learning/pen_contrast_update.py']]+base['sources']
        lineage=dict(config=file_pin(cp),base_fit=file_pin(paths['base_fit']),contrasts=file_pin(paths['contrasts']),contrast_readback=file_pin(paths['contrast_readback']),prepared=base['prepared'],review=base['review'],dataset_pins=base['dataset_pins'],sources=sources)
        expected_pairs=26
    if len(pairs['epsilon'])!=expected_pairs:raise ValueError('Registered finite-contrast pair count differs')
    if not run:return dict(training_started=False,TRAIN_rows=len(data['TRAIN']['x']),TRAIN_valid=int(data['TRAIN']['valid'].sum()),TRAIN_pairs=expected_pairs,models=6 if rank_study else 4,planned_updates_per_model=2000,Actor_updates=0)
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh unified CUDA output')
    device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    init_seed=cfg.get('init_seed',cfg['sampler_seed']);random.seed(init_seed);np.random.seed(init_seed);torch.manual_seed(init_seed)
    scales=dict(np.load(verify_file_pin(prep['scales'])));head_scales=dict(np.load(verify_file_pin(prep['head_scales'])))
    heads=dict(DIRECT=dict(head_mean=scales['y_mean'],head_scale=scales['y_scale']),IMPULSE=head_scales)
    arms={};initial=None;buffers={}
    for package in (['POINT','RANK4','RANK20'] if rank_study else ['POINT','RESPONSE']):
        for mode in ['DIRECT','IMPULSE']:
            name=package+'_'+mode;model=PenResponseWM(scales,heads[mode],prep['physics'],width=cfg['width'],coupled=mode=='IMPULSE').to(device)
            if initial is None:initial={k:v.detach().cpu().clone() for k,v in model.net.state_dict().items()}
            else:model.net.load_state_dict(initial,strict=True)
            buffers[name]={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
            arms[name]=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True),
                sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed']),pair_sampler=torch.Generator(device=device).manual_seed(cfg['pair_sampler_seed']),
                mode=mode,contrast_weight=spec['contrast_weight'] if package!='POINT' else 0.)
    if not all(raw_tree_equal(initial,arm['model'].net.state_dict()) for arm in arms.values()):raise ValueError('Identical four-arm NN initialization')
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    d=data['TRAIN'];valid=d['valid'];cache=dict(x=tensor(d['x'][valid],torch.float32),y=tensor(d['y'][valid],torch.float64),
        DIRECT=tensor(d['y'][valid],torch.float64),IMPULSE=tensor(d['heads'][valid],torch.float64),
        x_plus=tensor(pairs['x_plus'],torch.float32),x_minus=tensor(pairs['x_minus'],torch.float32),
        finite_contrast=tensor(pairs['finite_contrast'],torch.float64),epsilon=tensor(pairs['epsilon'],torch.float64))
    evaluation={r:(tensor(d['x'][d['valid']],torch.float32),tensor(d['y'][d['valid']],torch.float64)) for r,d in data.items()}
    cache['probe']=evaluation['TRAIN'][0][:128] if rank_study else evaluation['CHECK'][0]
    arm_caches={name:cache for name in arms}
    if rank_study:
        for name,arm in arms.items():
            rank=0 if name.startswith('POINT_') else 4 if name.startswith('RANK4_') else 20
            indices=np.empty(0,dtype=np.int64) if rank==0 else subsets[f'rank{rank}'];selected=dict(cache)
            if rank:
                ids=tensor(indices,torch.long)
                for key in ['x_plus','x_minus','finite_contrast','epsilon']:selected[key]=cache[key].index_select(0,ids)
            contract=pair_contract(prep,rank,indices);arm['pair_contract']=contract;selected['pair_contract']=contract.copy();arm_caches[name]=selected
    out.mkdir()
    if fresh:
        import subprocess
        receipt=freeze_sources(source_paths,out/'source_snapshot',root,subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip());write(out/'source_receipt.json',receipt);lineage['source_receipt']=file_pin(out/'source_receipt.json')
        atomic_checkpoint(out/'initial_NN.pt',initial);lineage['initial_NN']=file_pin(out/'initial_NN.pt')
    study_version=3 if rank_study else 2 if fresh else 1
    intent=dict(schema=f'pen_response_loss_ablation:{study_version}',**lineage,planned_updates_per_model=2000,models=list(arms),
        matched_initial_NN=True,point_batch=256,response_pair_batch=32,response_weight=.1,pair_seed=7751,
        response_loss='mean object12 + mean source-BODY wrench6 squared finite-action contrast error;frozen valid-TRAIN-only scales' if fresh else 'mean object12 + mean source-BODY wrench6 squared finite-action contrast error, divided by unchanged frozen V2 output scales',
        scope='Same point streams and optimizer counts, additional64 pair endpoints/response-arm update; not equal FLOPs. No CHECK fitting/selection or Actor training.')
    if rank_study:intent.update(health_probe_role='TRAIN',health_probe_rows=128,validation_evaluation_updates=[2000],pair_contracts={n:a['pair_contract'] for n,a in arms.items()})
    write(out/'intent.json',intent)
    logs={n:[] for n in arms};gradients={};curve={};healths={};timers=[];monitor=GPUMonitor();monitor.start();start=time.perf_counter()
    try:
        for update in range(1,2001):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for name,arm in arms.items():
                value,ids,pair_ids,gradient=update_contrast(arm,arm_caches[name],cfg,capture_gradient=update in [100,2000]);logs[name].append(value)
                if gradient is not None:gradients[name]=gradient
            end.record();timers.append((begin,end))
            if update in [100,500,1000,2000]:curve[str(update)]={n:{r:metrics(a['model'],x,y)[0] for r,(x,y) in evaluation.items() if not rank_study or r in evaluation_roles(update)} for n,a in arms.items()}
            if update in [100,2000]:
                path=out/f'pen_response_loss_checkpoint_{update:04d}.pt'
                atomic_checkpoint(path,dict(schema=f'pen_response_loss_ablation_checkpoint:{study_version}',**lineage,update=update,planned_updates=2000,
                    arms={n:payload(a,cfg,scales,heads[a['mode']],prep['physics'],update,device,gradients[n]) for n,a in arms.items()},
                    logs={n:torch.stack(v).cpu().tolist() for n,v in logs.items()}))
                baseline=None
                if not fresh:
                    bh=json.loads(verify_file_pin(base['checkpoint_health'][str(update)]).read_text());baseline=torch.load(verify_file_pin(bh['checkpoint']),map_location='cpu',weights_only=False)
                checks=checkpoint_health(path,arms,cache,cfg,device,buffers,initial,baseline,update,arm_caches=arm_caches)
                health=dict(schema='pen_response_loss_checkpoint_health:1',checkpoint=file_pin(path),checks=checks,passed=sum(checks.values()),total=len(checks),
                    update=update,live_cursor_per_model=update,clone_only_optimizer_steps_at_boundary=2*len(arms),Actor_updates=0,
                    remaining_all_four_compute_estimate_s=(2000-update)*float(np.median([a.elapsed_time(b) for a,b in timers[-50:]]))/1000)
                if rank_study:health['remaining_all_six_compute_estimate_s']=health.pop('remaining_all_four_compute_estimate_s')
                hp=out/f'health_{update:04d}.json';write(hp,health);healths[str(update)]=file_pin(hp)
                print(json.dumps(dict(update=update,health=[health['passed'],health['total']],checkpoint=file_pin(path),remaining_compute_s=health.get('remaining_all_six_compute_estimate_s',health.get('remaining_all_four_compute_estimate_s')))),flush=True)
                if not all(checks.values()):raise ValueError('Preserve failed restore/baseline check; no more updates')
        torch.cuda.synchronize(device);wall=time.perf_counter()-start
    finally:monitor.stop()
    for pin in sources:verify_file_pin(pin)
    if receipt is not None:verify_source_snapshot(receipt,live=True)
    result=dict(schema=f'pen_response_loss_ablation_result:{study_version}',**lineage,checkpoint=file_pin(path),checkpoint_health=healths,curve=curve,update=2000,
        live_models=len(arms),live_updates_per_model=2000,clone_only_optimizer_steps=4*len(arms),Actor_updates=0,training_running=False,
        point_rows_drawn_per_model=2000*256,response_pairs_drawn_per_response_model=2000*32,
        extra_pair_endpoint_forwards_per_response_model=2000*64,fit_and_QA_wall_s=wall,
        CUDA_ms_per_four_model_update=[a.elapsed_time(b) for a,b in timers],GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors),
        hardware=dict(GPU_name=torch.cuda.get_device_name(device),Torch_version=torch.__version__),
        scope='Fixed predeclared four-arm finite-response loss ablation only. All CHECK curves development evaluation, no model/weight selection. No physical derivative/task/Actor/ego qualification by checkpoint health.')
    if fresh:result.update(total_runner_wall_s=time.perf_counter()-entry_started,loader_normalizer_recomputations=1,actual_native_integrations=0,optimizer_steps=2004*len(arms))
    if rank_study:
        result['CUDA_ms_per_six_model_update']=result.pop('CUDA_ms_per_four_model_update')
        result.update(validation_evaluation_updates=[2000],health_probe_role='TRAIN',health_probe_rows=128,pair_contracts={n:a['pair_contract'] for n,a in arms.items()},scope='Registered six-arm rank4/rank20 response supervision with one full20 TRAIN point bank. Validation only at final2000,no fit/scales/selection/early stopping;old CHECK unused. Same original reference/fixed actor seed,not final-test/ego/Actor/physical-gradient qualification.')
    write(out/'report.json',result);return dict(report=file_pin(out/'report.json'),checkpoint=file_pin(path),health=healths,fit_and_QA_wall_s=wall)
