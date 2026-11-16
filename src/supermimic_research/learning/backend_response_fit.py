"""Owned new-backend WM study,using maintained response models/update math."""
import copy
import hashlib
import json
import random
import subprocess
import time
import tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin, verify_file_pin
from ..frozen_sources import freeze_sources, verify_source_snapshot
from ..runtime.explicit_collision_backend import load_contract
from .backend_response_dataset import backend_response_data, ROLES
from .checkpoint import _adam_valid, _finite_tree
from .gpu_monitor import GPUMonitor
from .pen_contrast_update import update_contrast
from .pen_response import PenResponseWM, INPUT_SCHEMA, OUTPUT_SCHEMA, head_schema
from .pen_response_units import derivative_scale
from .pen_stage import metrics
from .world_model import atomic_checkpoint, capture_rng, restore_rng, raw_tree_equal

MODELS = [p+'_'+m for p in ['POINT','RESPONSE','CONTACT'] for m in ['DIRECT','IMPULSE']]


def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def exact_context(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact_context(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(exact_context(x,y) for x,y in zip(a,b))
    return raw_tree_equal(a,b)


def model_spec(spec):
    expected = dict(schema=1,study='BACKEND_RESPONSE_COMPONENTS',models=MODELS,device='cuda:0',
        width=512,batch_size=256,pair_batch_size=32,learning_rate=.0003,weight_decay=.00001,
        gradient_clip_norm=1.,auxiliary_weight=.1,response_weight=1.,response_floor_relative=.001,
        init_seed=7411,sampler_seed=7411,pair_sampler_seed=7751,planned_updates=2000,
        first_checkpoint_update=100,continue_after_first_healthy_checkpoint=True,
        evaluation_config='pen_backend_response_evaluation_v1.toml',
        validation_evaluation_updates=[2000],validation_role='INSPECTED_DEVELOPMENT_NEVER_FIT_NOT_FINAL_TEST')
    if any(spec.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact six-arm new-backend fixed-budget study required')
    return spec


def loss_spec(name):
    if name not in MODELS: raise ValueError('Declared new-backend WM arm required')
    package,mode = name.split('_')
    return dict(mode=mode,contrast_weight=0. if package=='POINT' else 1.,
                blocks=['object12','wrench6']+(['Fn6'] if package=='CONTACT' else []))


def new_arm(name,context,device):
    spec=loss_spec(name); cfg=context['config']; scales=context['scales']
    heads=context['head_scales'] if spec['mode']=='IMPULSE' else dict(head_mean=scales['y_mean'],head_scale=scales['y_scale'])
    model=PenResponseWM(scales,heads,context['physics'],width=cfg['width'],coupled=spec['mode']=='IMPULSE').to(device)
    return dict(name=name,model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],fused=torch.device(device).type=='cuda'),
        sampler=torch.Generator(device=device).manual_seed(cfg['sampler_seed']),
        pair_sampler=torch.Generator(device=device).manual_seed(cfg['pair_sampler_seed']),
        mode=spec['mode'],contrast_weight=spec['contrast_weight'],
        pair_contract=copy.deepcopy(context['pair_contract']),
        response_contract=dict(schema='backend_response_units:1',backend_id=context['backend']['backend_id'],
            dataset=context['dataset'],scale_sha256=hashlib.sha256(context['response_scale'].tobytes()).hexdigest(),
            weight=spec['contrast_weight'],blocks=spec['blocks']))


def saved_arm(arm,context,update,device,gradient=None):
    return dict(schema='backend_pen_response_arm:1',name=arm['name'],context=copy.deepcopy(context),
        input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,head_schema=head_schema(arm['mode']),
        update=update,mode=arm['mode'],contrast_weight=arm['contrast_weight'],
        model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),
        sampler=arm['sampler'].get_state(),pair_sampler=arm['pair_sampler'].get_state(),rng=capture_rng(device=device),
        pair_contract=copy.deepcopy(arm['pair_contract']),response_contract=copy.deepcopy(arm['response_contract']),
        gradient_snapshot=gradient)


def restore_backend_arm(saved,context,device):
    # Reject foreign provenance before NN construction or global RNG changes.
    if (saved.get('schema')!='backend_pen_response_arm:1' or not exact_context(saved['context'],context)
            or saved['name'] not in MODELS or saved['input_schema']!=INPUT_SCHEMA
            or saved['output_schema']!=OUTPUT_SCHEMA or type(saved['update']) is not int or not 0<=saved['update']<=2000):
        raise ValueError('Own exact backend/dataset/context required before model construction')
    spec=loss_spec(saved['name'])
    if saved['mode']!=spec['mode'] or saved['head_schema']!=head_schema(spec['mode']) or saved['contrast_weight']!=spec['contrast_weight']:
        raise ValueError('Declared model family/loss identity differs')
    arm=new_arm(saved['name'],context,device)
    if saved['pair_contract']!=arm['pair_contract'] or saved['response_contract']!=arm['response_contract']:
        raise ValueError('Owned pair/response identity differs')
    for name,buffer in arm['model'].named_buffers():
        if not raw_tree_equal(saved['model'][name].cpu(),buffer.detach().cpu()):
            raise ValueError('Frozen normalization or physical buffer differs')
    arm['model'].load_state_dict(saved['model'],strict=True)
    # Optimizer.load_state_dict can retain CPU tensor storage. Independent
    # restore clones must never mutate each other's moments or the input state.
    arm['optimizer'].load_state_dict(copy.deepcopy(saved['optimizer']))
    arm['sampler'].set_state(saved['sampler']); arm['pair_sampler'].set_state(saved['pair_sampler'])
    if saved['update'] and not _adam_valid(arm['optimizer'],arm['model'],saved['update'],torch):
        raise ValueError('Owned optimizer cursor/state differs')
    if not _finite_tree([saved['model'],saved['optimizer'],saved['gradient_snapshot']],torch):
        raise ValueError('Nonfinite owned checkpoint')
    restore_rng(saved['rng'],device=device)
    return arm


def load_dataset(cp,spec):
    pp=(cp.parent/spec['prepared_data']).resolve(); prep=read(pp)
    contract=load_contract((cp.parent/spec['backend_manifest']).resolve())
    if (prep.get('schema')!='backend_pen_response_common_data:1' or prep['collision_backend']!=contract
            or prep['passed']!=prep['total'] or not all(prep['checks'].values())
            or prep['scale_input_role']!='VALID_NEW_BACKEND_TRAIN_ONLY'
            or prep['validation_role']!=ROLES['VALIDATE'] or prep['validation_rows_used_for_TRAIN']
            or prep['CHECK_response_rows_used'] or prep['optimizer_steps'] or prep['actual_native_integrations']):
        raise ValueError('Reviewed explicit new-backend dataset required')
    verify_source_snapshot(read(verify_file_pin(prep['source_receipt'])),live=True)
    review=read(verify_file_pin(prep['inputs']['data_review']))
    if (review['preparations']!=prep['preparations'] or review['collision_backend']!=contract
            or review['passed']!=review['total'] or not all(review['checks'].values())):
        raise ValueError('Exact independent new-backend data review required')
    reports={r:read(verify_file_pin(p)) for r,p in prep['preparations'].items()}
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()}
    if not data['VALIDATE']['valid'].any():
        raise ValueError('No qualified validation rows for the registered complete evaluation')
    directions={r:np.load(verify_file_pin(p))['directions'] for r,p in prep['directions'].items()}
    built=backend_response_data(reports,data,directions,contract)
    for key in ['scales','head_scales','pairs','pair_subsets']:
        stored=dict(np.load(verify_file_pin(prep['products'][key])))
        if stored.keys()!=built[key].keys() or any(not np.array_equal(v,stored[k]) for k,v in built[key].items()):
            raise ValueError('New TRAIN-only reconstruction differs from frozen products')
    if built['pairs']['TRAIN_group_index'].tolist()!=review['TRAIN_eligible_pair_group_indices']['rank20']:
        raise ValueError('Pair bank differs from independent physical eligibility')
    units=derivative_scale(built['pairs']['finite_contrast'],built['scales']['y_scale'],role=ROLES['TRAIN'])
    pair_contract=dict(schema='backend_response_pair_bank:1',backend_id=contract['backend_id'],
        dataset=file_pin(pp),point_bank=prep['outputs']['TRAIN'],pair_bank=prep['products']['pairs'],
        index_sha256=hashlib.sha256(built['pairs']['TRAIN_group_index'].tobytes()).hexdigest(),
        rows=len(built['pairs']['epsilon']))
    context=dict(config=spec,backend=contract,dataset=file_pin(pp),physics=prep['physics'],
        scales=built['scales'],head_scales=built['head_scales'],response_scale=units['response_scale'],pair_contract=pair_contract)
    return pp,prep,data,built,context,units


def checkpoint_health(path,arms,context,caches,initial_buffers,update,device,work=None):
    saved=torch.load(path,map_location='cpu',weights_only=False); checks={}
    for name,arm in arms.items():
        state=saved['arms'][name]; clone=restore_backend_arm(state,context,device); cache=caches[name]
        with torch.no_grad(): a=arm['model'].prediction(cache['x'][:128]); b=clone['model'].prediction(cache['x'][:128])
        local=dict(model=raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()),
            Adam=raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()),
            point_sampler=raw_tree_equal(arm['sampler'].get_state(),clone['sampler'].get_state()),
            pair_sampler=raw_tree_equal(arm['pair_sampler'].get_state(),clone['pair_sampler'].get_state()),
            prediction=raw_tree_equal(a,b),finite_nonnegative_Fn6=bool(torch.isfinite(a).all() and (a[:,84:]>=0).all()),
            cursor=state['update']==update and _adam_valid(clone['optimizer'],clone['model'],update,torch),
            finite_logs=bool(np.isfinite(saved['logs'][name]).all()),
            frozen_buffers=all(raw_tree_equal(v.cpu(),initial_buffers[name][k]) for k,v in arm['model'].named_buffers()),
            RNG=raw_tree_equal(state['rng'],capture_rng(device=device)))
        left,right=restore_backend_arm(state,context,device),restore_backend_arm(state,context,device)
        if work is not None: work['QA_attempted_calls']+=1
        la,ia,pa,ga=update_contrast(left,cache,context['config'],capture_gradient=True)
        if work is not None: work['QA_completed_steps']+=1; work['QA_attempted_calls']+=1
        lb,ib,pb,gb=update_contrast(right,cache,context['config'],capture_gradient=True)
        if work is not None: work['QA_completed_steps']+=1
        local.update(next_loss=raw_tree_equal(la,lb),next_point_batch=raw_tree_equal(ia,ib),next_pair_batch=raw_tree_equal(pa,pb),
            next_gradient=raw_tree_equal(ga,gb),next_model=raw_tree_equal(left['model'].state_dict(),right['model'].state_dict()),
            next_Adam=raw_tree_equal(left['optimizer'].state_dict(),right['optimizer'].state_dict()),
            next_point_sampler=raw_tree_equal(left['sampler'].get_state(),right['sampler'].get_state()),
            next_pair_sampler=raw_tree_equal(left['pair_sampler'].get_state(),right['pair_sampler'].get_state()))
        checks.update({name+'_'+k:bool(v) for k,v in local.items()}); del clone,left,right
    samplers=[arm['sampler'].get_state() for arm in arms.values()]
    pair_samplers=[arm['pair_sampler'].get_state() for arm in arms.values() if arm['contrast_weight']]
    checks['all_point_streams_matched']=all(raw_tree_equal(samplers[0],s) for s in samplers[1:])
    checks['all_response_streams_matched']=all(raw_tree_equal(pair_samplers[0],s) for s in pair_samplers[1:])
    checks['checkpoint_cursor']=saved['update']==update and saved['planned_updates']==2000
    return checks


def run_fit(root,config_path,*,register=False):
    started=time.perf_counter(); root=Path(root).resolve(); cp=Path(config_path).resolve(); spec=model_spec(tomllib.loads(cp.read_text()))
    pp,prep,data,built,context,units=load_dataset(cp,spec)
    paths={k:(cp.parent/spec[k]).resolve() for k in ['registration','output','protocol']}
    evaluation_path=(cp.parent/spec['evaluation_config']).resolve(); evaluation=tomllib.loads(evaluation_path.read_text())
    if (evaluation['schema'],evaluation['model_AD_relative_error'],evaluation['model_AD_cosine'],evaluation['physical_screen'])!=(1,1.,.9,'REGISTERED_RESERVED_VALIDATION_PHYSICAL_SCREEN'):
        raise ValueError('Unchanged registered physical evaluation thresholds required')
    source_paths=sorted((root/'src').rglob('*.py'))+[cp,paths['protocol'],evaluation_path,
        root/'tools/train_backend_pen_responses.py',root/'tools/evaluate_backend_pen_responses.py']
    inputs=dict(config=file_pin(cp),dataset=file_pin(pp),protocol=file_pin(paths['protocol']),evaluation_config=file_pin(evaluation_path))
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',root,
            subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
        write(out/'source_receipt.json',receipt); return receipt
    if register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(root/'runs'): raise ValueError('Fresh model registration required')
        out.mkdir(); freeze(out); np.savez_compressed(out/'TRAIN_response_units.npz',**units)
        write(paths['registration'],dict(schema='backend_response_model_registration:1',inputs=inputs,
            source_receipt=file_pin(out/'source_receipt.json'),response_units=file_pin(out/'TRAIN_response_units.npz'),
            models=MODELS,updates_per_model=2000,health_updates=[100,2000],validation_evaluation_updates=[2000],
            actual_native_integrations=0,optimizer_steps=0,loader_normalizer_reconstructions=1,TRAIN_response_unit_computations=1))
        return dict(registration=file_pin(paths['registration']),training_started=False)
    reg=read(paths['registration'])
    if reg['inputs']!=inputs or reg['models']!=MODELS or reg['updates_per_model']!=2000:
        raise ValueError('Registered model/data design differs')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    stored=dict(np.load(verify_file_pin(reg['response_units'])))
    if stored.keys()!=units.keys() or any(not np.array_equal(v,stored[k]) for k,v in units.items()):
        raise ValueError('Registered valid TRAIN response units differ')
    out=paths['output']
    if out.exists() or not out.is_relative_to(root/'runs') or not torch.cuda.is_available(): raise ValueError('Fresh CUDA WM output required')
    device=torch.device(spec['device']); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(spec['init_seed']); np.random.seed(spec['init_seed']); torch.manual_seed(spec['init_seed'])
    arms={}; initial=None; buffers={}; caches={}
    for name in MODELS:
        arm=new_arm(name,context,device)
        if initial is None: initial={k:v.detach().cpu().clone() for k,v in arm['model'].net.state_dict().items()}
        else: arm['model'].net.load_state_dict(initial,strict=True)
        arms[name]=arm; buffers[name]={k:v.detach().cpu().clone() for k,v in arm['model'].named_buffers()}
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    train=data['TRAIN']; valid=train['valid']; pairs=built['pairs']
    cache=dict(x=tensor(train['x'][valid],torch.float32),y=tensor(train['y'][valid],torch.float64),
        DIRECT=tensor(train['y'][valid],torch.float64),IMPULSE=tensor(train['heads'][valid],torch.float64),
        x_plus=tensor(pairs['x_plus'],torch.float32),x_minus=tensor(pairs['x_minus'],torch.float32),
        epsilon=tensor(pairs['epsilon'],torch.float64),finite_contrast=tensor(pairs['finite_contrast'],torch.float64),
        response_scale=tensor(context['response_scale'],torch.float64),pair_contract=context['pair_contract'])
    for name,arm in arms.items(): caches[name]={**cache,'response_contract':arm['response_contract']}
    evaluation={r:(tensor(d['x'][d['valid']],torch.float32),tensor(d['y'][d['valid']],torch.float64)) for r,d in data.items()}
    if not all(raw_tree_equal(initial,a['model'].net.state_dict()) for a in arms.values()): raise ValueError('Matched initialization differs')
    out.mkdir(); receipt=freeze(out); atomic_checkpoint(out/'initial_NN.pt',initial)
    write(out/'intent.json',dict(schema='backend_response_fit_intent:1',inputs=inputs,registration=file_pin(paths['registration']),
        models=MODELS,matched_initial_NN=True,rows={r:[int(d['valid'].sum()),len(d['valid'])] for r,d in data.items()},
        pairs=len(pairs['epsilon']),point_draws_per_update=256,response_pairs_per_update=32,
        native_integrations=0,Actor_updates=0,scope='Matched Adam updates/point streams,not equal FLOPs. Validation final2000 only.'))
    logs={n:[] for n in arms}; gradients={}; curve={}; healths={}; timings=[]
    work=dict(live_attempted_calls=0,live_completed_steps=0,QA_attempted_calls=0,QA_completed_steps=0)
    monitor=GPUMonitor(); monitor.start(); train_start=time.perf_counter()
    try:
        for update in range(1,2001):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); begin.record()
            for name,arm in arms.items():
                work['live_attempted_calls']+=1
                value,_,_,gradient=update_contrast(arm,caches[name],spec,capture_gradient=update in [100,2000]); logs[name].append(value)
                work['live_completed_steps']+=1
                if gradient is not None: gradients[name]=gradient
            end.record(); timings.append((begin,end))
            if update in [100,500,1000,2000]:
                curve[str(update)]={name:{r:metrics(arm['model'],x,y)[0] for r,(x,y) in evaluation.items()
                    if r=='TRAIN' or update==2000} for name,arm in arms.items()}
            if update in [100,2000]:
                path=out/f'checkpoint_{update:04d}.pt'
                atomic_checkpoint(path,dict(schema='backend_response_fit_checkpoint:1',inputs=inputs,
                    registration=file_pin(paths['registration']),source_receipt=file_pin(out/'source_receipt.json'),
                    update=update,planned_updates=2000,arms={n:saved_arm(a,context,update,device,gradients[n]) for n,a in arms.items()},
                    logs={n:torch.stack(v).cpu().tolist() for n,v in logs.items()}))
                checks=checkpoint_health(path,arms,context,caches,buffers,update,device,work)
                eta=(2000-update)*float(np.median([a.elapsed_time(b) for a,b in timings[-50:]]))/1000
                hp=out/f'health_{update:04d}.json'; write(hp,dict(checkpoint=file_pin(path),checks=checks,passed=sum(checks.values()),
                    total=len(checks),update=update,clone_only_optimizer_steps_at_boundary=2*len(arms),remaining_compute_s=eta,
                    estimate_scope='Last50 CUDA-event median only;excludes later checkpoint/evaluation/IO/physical-gradient work.'))
                healths[str(update)]=file_pin(hp)
                print(json.dumps(dict(update=update,checks=[sum(checks.values()),len(checks)],remaining_compute_s=eta,checkpoint=file_pin(path))),flush=True)
                if not all(checks.values()): raise ValueError('Preserve failed owned restore/next-update check')
        torch.cuda.synchronize(device)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),work=work.copy(),
            completed_optimizer_steps=work['live_completed_steps']+work['QA_completed_steps'],
            failed_call_may_be_partial=True,actual_native_integrations=0,Actor_updates=0))
        raise
    finally: monitor.stop()
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='backend_response_fit_result:1',inputs=inputs,registration=file_pin(paths['registration']),
        source_receipt=file_pin(out/'source_receipt.json'),initial_NN=file_pin(out/'initial_NN.pt'),checkpoint=file_pin(path),
        checkpoint_health=healths,update=2000,models=MODELS,training_running=False,curve=curve,
        live_optimizer_steps=work['live_completed_steps'],clone_only_optimizer_steps=work['QA_completed_steps'],
        optimizer_steps=work['live_completed_steps']+work['QA_completed_steps'],work=work.copy(),
        actual_native_integrations=0,Actor_updates=0,validation_evaluation_updates=[2000],health_probe_role='TRAIN',
        loader_normalizer_reconstructions=1,TRAIN_response_unit_computations=1,
        CUDA_ms_per_six_arm_update=[a.elapsed_time(b) for a,b in timings],fit_and_QA_wall_s=time.perf_counter()-train_start,
        total_runner_wall_s=time.perf_counter()-started,GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors),
        scope='Fresh new-backend six-arm fixed-budget WM comparison only. All original masks/point streams preserved;development validation final-only and NEVER_FIT. Health is not physical-gradient/Actor/task/ego/speedup admission.')
    write(out/'report.json',result)
    return dict(report=file_pin(out/'report.json'),optimizer_steps=result['optimizer_steps'],health=healths)
