"""Owned matched three-arm path supervision experiment,with resident CUDA data."""
import copy
import json
import random
import subprocess
import time
import tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .backend_reward_gradient import frozen_model_view
from .backend_response_fit import exact_context
from .checkpoint import _adam_valid,_finite_tree
from .gpu_monitor import GPUMonitor
from .world_model import atomic_checkpoint,capture_rng,restore_rng,raw_tree_equal
from .pen_response import INPUT_SCHEMA,OUTPUT_SCHEMA
from .pen_path_response import ARMS,PATH_SCHEMA,WEIGHTS,PathResponseWM,fit_path_units,update_path


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fixed_spec(cfg):
    expected=dict(schema=1,study='SAVED_INTERVAL_SOFT_GEOMETRY',models=ARMS,device='cuda:0',width=512,
        batch_size=256,pair_batch_size=32,learning_rate=.0003,weight_decay=.00001,gradient_clip_norm=1.,auxiliary_weight=.1,
        sampler_seed=7411,pair_sampler_seed=7751,init_seed=7411,planned_updates=2000,first_checkpoint_update=100,
        validation_evaluation_updates=[2000],validation_role='INSPECTED_DEVELOPMENT_NEVER_FIT_NOT_FINAL_TEST')
    if any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in expected.items()):raise ValueError('Fixed three-arm path study/budget required')
    return cfg


def data_view(paths):
    common,old_fit,old=frozen_model_view(paths['common_data'],paths['prior_fit']);recovery=read(paths['interval_targets']);ledger=read(paths['interval_ledger'])
    review=read(verify_file_pin(ledger['review']))
    if (recovery['schema']!='pen_saved_interval_targets:1' or recovery['passed']!=recovery['total'] or not all(recovery['checks'].values())
        or recovery['inputs']['common_data']!=file_pin(paths['common_data']) or ledger['recovery']!=file_pin(paths['interval_targets'])
        or review['recovery']!=ledger['recovery'] or review['passed']!=review['total'] or not all(review['checks'].values())
        or recovery['actual_native_integrations'] or recovery['optimizer_steps'] or recovery['normalizer_fits']):raise ValueError('Exact independently reviewed path labels required')
    for pin in [recovery['source_receipt'],review['source_receipt']]:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    data={k:dict(np.load(verify_file_pin(p))) for k,p in common['outputs'].items()};prep={k:read(verify_file_pin(p)) for k,p in common['preparations'].items()}
    path={k:dict(np.load(verify_file_pin(c['outputs']))) for k,c in recovery['cohorts'].items()}
    for role in data:
        if not np.array_equal(data[role]['valid'],path[role]['physical_valid']) or not np.array_equal(path[role]['case_indices'],[m['case_index'] for m in prep[role]['metadata']]):raise ValueError('Path/point row alignment differs')
        if path[role]['geometry_failure'].any() or path[role]['errors'].any() or path[role]['global_error'].any():raise ValueError('This observed-safe-only study cannot silently acquire hard-failure coverage')
    products={k:dict(np.load(verify_file_pin(p))) for k,p in common['products'].items()}
    return common,old_fit,old,recovery,data,prep,path,products


def new_arm(name,context,device):
    if name not in ARMS:raise ValueError('Declared path arm required')
    cfg=context['config'];model=PathResponseWM(context['scales'],context['physics'],context['path_units'],width=cfg['width']).to(device)
    return dict(name=name,model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=torch.device(device).type=='cuda'),
        sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed']),pair_sampler=torch.Generator(device=device).manual_seed(cfg['pair_sampler_seed']),contract=copy.deepcopy(context['contract']))


def save_arm(arm,context,update,device,gradient=None):
    return dict(schema='pen_interval_response_arm:1',name=arm['name'],context=copy.deepcopy(context),update=update,input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,path_schema=PATH_SCHEMA,
        weights=WEIGHTS[arm['name']],model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),sampler=arm['sampler'].get_state(),pair_sampler=arm['pair_sampler'].get_state(),rng=capture_rng(device=device),gradient=gradient)


def restore_arm(saved,context,device):
    if (saved.get('schema')!='pen_interval_response_arm:1' or saved['name'] not in ARMS or not exact_context(saved['context'],context)
        or saved['input_schema']!=INPUT_SCHEMA or saved['output_schema']!=OUTPUT_SCHEMA or saved['path_schema']!=PATH_SCHEMA
        or type(saved['update']) is not int or not 0<=saved['update']<=2000 or saved['weights']!=WEIGHTS[saved['name']]):
        raise ValueError('Own exact path/dataset/backend/units/schema required before construction')
    arm=new_arm(saved['name'],context,device)
    for key,buffer in arm['model'].named_buffers():
        if not raw_tree_equal(saved['model'][key].cpu(),buffer.cpu()):raise ValueError('Frozen physical/normalization/path units changed')
    arm['model'].load_state_dict(saved['model']);arm['optimizer'].load_state_dict(copy.deepcopy(saved['optimizer']))
    arm['sampler'].set_state(saved['sampler']);arm['pair_sampler'].set_state(saved['pair_sampler'])
    if saved['update'] and not _adam_valid(arm['optimizer'],arm['model'],saved['update'],torch):raise ValueError('Owned Adam cursor differs')
    if not _finite_tree([saved['model'],saved['optimizer'],saved['gradient']],torch):raise ValueError('Nonfinite path checkpoint')
    restore_rng(saved['rng'],device=device);return arm


def registered_context(cfg,paths,inputs,common,old_fit,old,products):
    reg=read(paths['registration'])
    if reg['config']!=file_pin(paths['config']) or reg['inputs']!=inputs or reg['models']!=ARMS or reg['initial_base_NN']!=old_fit['initial_NN']:raise ValueError('Registered path design changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    units={k:(float(v) if v.ndim==0 else v) for k,v in dict(np.load(verify_file_pin(reg['path_units']))).items()}
    if units['TRAIN_group_indices'].tolist()!=products['pairs']['TRAIN_group_index'].tolist():raise ValueError('Registered path pair indices differ')
    contract=dict(dataset=inputs['common_data'],intervals=inputs['interval_targets'],backend_id=old['backend']['backend_id'],pairs=common['products']['pairs'],path_units=reg['path_units'],response_units=old_fit['registration'])
    context=dict(config=cfg,backend=old['backend'],physics=old['physics'],scales=old['scales'],response_scale=old['response_scale'],path_units=units,contract=contract)
    return reg,context


def health(path,arms,context,cache,update,device,work):
    saved=torch.load(path,map_location='cpu',weights_only=False);checks={}
    for name,live in arms.items():
        s=saved['arms'][name];clone=restore_arm(s,context,device)
        with torch.no_grad():a=live['model'].full_prediction(cache['x'][:128]);b=clone['model'].full_prediction(cache['x'][:128])
        ch=dict(model=raw_tree_equal(live['model'].state_dict(),clone['model'].state_dict()),Adam=raw_tree_equal(live['optimizer'].state_dict(),clone['optimizer'].state_dict()),
            point_sampler=raw_tree_equal(live['sampler'].get_state(),clone['sampler'].get_state()),pair_sampler=raw_tree_equal(live['pair_sampler'].get_state(),clone['pair_sampler'].get_state()),
            predictions=raw_tree_equal(a,b),finite_nonnegative=bool(torch.isfinite(a[0]).all() and torch.isfinite(a[1]).all() and (a[0][:,84:]>=0).all() and (a[1]>=0).all()),
            cursor=s['update']==update and _adam_valid(clone['optimizer'],clone['model'],update,torch),RNG=raw_tree_equal(s['rng'],capture_rng(device=device)))
        left,right=restore_arm(s,context,device),restore_arm(s,context,device)
        work['QA_attempted_calls']+=1;la=update_path(left,cache,context['config'],capture_gradient=True);work['QA_completed_steps']+=1
        work['QA_attempted_calls']+=1;lb=update_path(right,cache,context['config'],capture_gradient=True);work['QA_completed_steps']+=1
        ch.update(next_loss_batches_gradient=raw_tree_equal(la,lb),next_model=raw_tree_equal(left['model'].state_dict(),right['model'].state_dict()),
            next_Adam=raw_tree_equal(left['optimizer'].state_dict(),right['optimizer'].state_dict()),next_point_sampler=raw_tree_equal(left['sampler'].get_state(),right['sampler'].get_state()),next_pair_sampler=raw_tree_equal(left['pair_sampler'].get_state(),right['pair_sampler'].get_state()),
            unchanged_frozen_buffers=all(raw_tree_equal(v,clone['model'].state_dict()[k]) for k,v in live['model'].named_buffers()))
        if name=='RESPONSE_ONLY':ch['zero_path_head']=all(not bool(p.any()) for p in live['model'].path_head.parameters())
        checks.update({name+'_'+k:bool(v) for k,v in ch.items()});del clone,left,right
    for stream in ['sampler','pair_sampler']:
        states=[a[stream].get_state() for a in arms.values()];checks['matched_'+stream]=all(raw_tree_equal(states[0],s) for s in states[1:])
    checks['checkpoint_cursor']=saved['update']==update and saved['planned_updates']==2000
    return checks


def run(root,config_path,*,register=False):
    started=time.perf_counter();root=Path(root);cp=Path(config_path).resolve();cfg=fixed_spec(tomllib.loads(cp.read_text()))
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']};common,old_fit,old,recovery,data,prep,path_data,products=data_view(paths)
    source_paths=sorted((root/'src').rglob('*.py'))+[cp,paths['protocol'],paths['evaluation_config'],root/'tools/train_pen_path_responses.py',root/'tools/evaluate_pen_path_rewards.py']
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',root,subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if register:
        units=fit_path_units(path_data['TRAIN']['geometry_loss'],data['TRAIN']['valid'],prep['TRAIN']['response_groups'],role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        if units['TRAIN_group_indices'].tolist()!=products['pairs']['TRAIN_group_index'].tolist():raise ValueError('Same779 original physical pairs required')
        altered=path_data['TRAIN']['geometry_loss'].copy();altered[~data['TRAIN']['valid']]+=1e6
        other=fit_path_units(altered,data['TRAIN']['valid'],prep['TRAIN']['response_groups'],role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        if not all(np.array_equal(v,other[k]) for k,v in units.items()):raise ValueError('Invalid TRAIN labels affected units')
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(root/'runs'):raise ValueError('Fresh path registration required')
        out.mkdir();freeze(out);np.savez_compressed(out/'TRAIN_path_units.npz',**units)
        write(paths['registration'],dict(schema='pen_path_fit_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),path_units=file_pin(out/'TRAIN_path_units.npz'),
            initial_base_NN=old_fit['initial_NN'],models=ARMS,updates_per_arm=2000,TRAIN_path_unit_fits=2,production_path_unit_fits=1,invalid_TRAIN_isolation_fits=1,
            point_normalizer_fits=0,actual_native_integrations=0,optimizer_steps=0,validation_rows_used_for_fit=False))
        return dict(registration=file_pin(paths['registration']))
    reg,context=registered_context(cfg,{**paths,'config':cp},inputs,common,old_fit,old,products);units=context['path_units'];contract=context['contract']
    out=paths['output']
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available():raise ValueError('Fresh CUDA path fit required')
    device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['init_seed']);np.random.seed(cfg['init_seed']);torch.manual_seed(cfg['init_seed'])
    initial=torch.load(verify_file_pin(reg['initial_base_NN']),map_location='cpu',weights_only=False);arms={n:new_arm(n,context,device) for n in ARMS}
    for a in arms.values():a['model'].net.load_state_dict(initial)
    states=[a['model'].state_dict() for a in arms.values()]
    if not all(raw_tree_equal(states[0],s) for s in states[1:]):raise ValueError('All initial model bytes must match')
    tensor=lambda v,dtype:torch.as_tensor(v,dtype=dtype,device=device);mask=data['TRAIN']['valid'];pairs=products['pairs'];targets=np.asarray(path_data['TRAIN']['geometry_loss'],np.float64)
    gp=units['TRAIN_group_indices'];path_contrast=np.asarray([(targets[prep['TRAIN']['response_groups'][i]['rows']['1']].mean()-targets[prep['TRAIN']['response_groups'][i]['rows']['-1']].mean())/(2*prep['TRAIN']['response_groups'][i]['epsilon']) for i in gp])
    cache=dict(x=tensor(data['TRAIN']['x'][mask],torch.float32),y=tensor(data['TRAIN']['y'][mask],torch.float64),path=tensor(targets[mask],torch.float64),
        x_plus=tensor(pairs['x_plus'],torch.float32),x_minus=tensor(pairs['x_minus'],torch.float32),epsilon=tensor(pairs['epsilon'],torch.float64),finite_contrast=tensor(pairs['finite_contrast'],torch.float64),
        response_scale=tensor(old['response_scale'],torch.float64),path_contrast=tensor(path_contrast,torch.float64),contract=contract)
    evaluation={r:(tensor(d['x'][d['valid']],torch.float32),tensor(d['y'][d['valid']],torch.float64),tensor(path_data[r]['geometry_loss'][d['valid']],torch.float64)) for r,d in data.items()}
    out.mkdir();receipt=freeze(out);write(out/'intent.json',dict(models=ARMS,matched_initial_model=True,registration=file_pin(paths['registration']),rows={r:[int(d['valid'].sum()),len(d['valid'])] for r,d in data.items()},pairs=len(gp),live_optimizer_budget=6000,QA_optimizer_budget=12,native_budget=0,
        scope='Matched point/pair streams and update budgets;all path terms computed in every arm,weights differ. Safe observed interval loss,not hard-risk classifier.'))
    work=dict(live_attempted_calls=0,live_completed_steps=0,QA_attempted_calls=0,QA_completed_steps=0);logs={n:[] for n in ARMS};grads={};healths={};curve={};events=[]
    monitor=GPUMonitor();monitor.start();fit_start=time.perf_counter()
    try:
        for update in range(1,2001):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            for name,arm in arms.items():
                work['live_attempted_calls']+=1;value,_,_,gradient=update_path(arm,cache,cfg,capture_gradient=update in [100,2000]);work['live_completed_steps']+=1;logs[name].append(value)
                if gradient is not None:grads[name]=gradient
            end.record();events.append((begin,end))
            if update in [100,2000]:
                curve[str(update)]={}
                with torch.no_grad():
                    for name,a in arms.items():
                        curve[str(update)][name]={}
                        for role,(x,y,target) in evaluation.items():
                            if role!='TRAIN' and update!=2000:continue
                            p,g=a['model'].full_prediction(x);curve[str(update)][name][role]=dict(forward_MSE=float(((p-y)/a['model'].y_scale).square().mean()),path_MSE=float(((g-target)/a['model'].path_point_scale).square().mean()))
                ck=out/f'checkpoint_{update:04d}.pt';atomic_checkpoint(ck,dict(schema='pen_path_fit_checkpoint:1',registration=file_pin(paths['registration']),source_receipt=file_pin(out/'source_receipt.json'),inputs=inputs,
                    update=update,planned_updates=2000,arms={n:save_arm(a,context,update,device,grads[n]) for n,a in arms.items()},logs={n:torch.stack(v).cpu().tolist() for n,v in logs.items()}))
                ch=health(ck,arms,context,cache,update,device,work);eta=(2000-update)*float(np.median([a.elapsed_time(b) for a,b in events[-50:]]))/1000
                hp=out/f'health_{update:04d}.json';write(hp,dict(checkpoint=file_pin(ck),checks=ch,passed=sum(ch.values()),total=len(ch),update=update,remaining_compute_s=eta,
                    estimate_scope='Last50 CUDA-event median excludes later checkpoint/QA/analysis/IO.'))
                healths[str(update)]=file_pin(hp);print(json.dumps(dict(update=update,health=[sum(ch.values()),len(ch)],remaining_compute_s=eta,checkpoint=file_pin(ck))),flush=True)
                if not all(ch.values()):raise ValueError('Owned path checkpoint/next-update mismatch')
        torch.cuda.synchronize(device)
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),work=work,partial_failed_call_possible=True,actual_native_integrations=0));raise
    finally:monitor.stop()
    verify_source_snapshot(receipt,live=True)
    report=dict(schema='pen_path_fit_result:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        checkpoint=file_pin(ck),checkpoint_health=healths,models=ARMS,update=2000,training_running=False,curve=curve,work=work,
        optimizer_steps=work['live_completed_steps']+work['QA_completed_steps'],actual_native_integrations=0,normalizer_fits=0,TRAIN_path_unit_fits=0,Actor_updates=0,
        CUDA_ms_per_three_arm_update=[a.elapsed_time(b) for a,b in events],fit_and_QA_wall_s=time.perf_counter()-fit_start,total_runner_wall_s=time.perf_counter()-started,GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors),
        scope='Observed-safe soft interval geometry comparison only. Registration owns the only production/isolation path-unit fits. Validation final2000 only,never fitted. No hard-failure,Actor,ego or speedup admission.')
    write(out/'report.json',report);return dict(report=file_pin(out/'report.json'),optimizer_steps=report['optimizer_steps'])
