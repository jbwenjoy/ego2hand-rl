"""Fixed all-six one-step RAW observable reward diagnostics,no optimizer step."""
import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_response_fit import MODELS,restore_backend_arm
from supermimic_research.learning.backend_reward_gradient import (SCORES,frozen_model_view,batch_context,raw_reward_gradient,compare_reward_responses)
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.mechanics.pen_reward_chain import source_context,observable_proxy
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import perturbed_actor_raw


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if cfg['schema']!=1 or cfg['models']!=MODELS or cfg['mode']!='READ_ONLY_ONE_STEP_RAW_REWARD':raise ValueError('Fixed all-six one-step analysis required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['common_data','fit','native_reward_audit','objective_config','adapter_config','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']}
    common,fit,model_context=frozen_model_view(paths['common_data'],paths['fit'])
    native=read(paths['native_reward_audit'])
    if native['schema']!='backend_native_reward_chain_audit:1' or native['passed']!=native['total'] or not all(native['checks'].values()) or native['inputs']['common_data']!=inputs['common_data']:
        raise ValueError('Completed exact native controller/reward audit required')
    objective=tomllib.loads(paths['objective_config'].read_text());reward=tomllib.loads(paths['adapter_config'].read_text())['reward']
    if (native['inputs']['objective_config'],native['inputs']['adapter_config'])!=(inputs['objective_config'],inputs['adapter_config']):raise ValueError('Original reward constants changed')
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh RAW reward registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='backend_RAW_reward_gradient_registration:1',config=file_pin(cp),inputs=inputs,
            source_receipt=file_pin(out/'source_receipt.json'),models=MODELS,normalizer_fits=0,actual_native_integrations=0,optimizer_steps=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['models']!=MODELS:raise ValueError('Registered RAW reward design changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False)
    prepared={r:read(verify_file_pin(p)) for r,p in common['preparations'].items()}
    data={r:dict(np.load(verify_file_pin(p))) for r,p in common['outputs'].items()}
    cohorts={};raw_actions={};checks={};bases={};directions={};truth={}
    for role,p in prepared.items():
        raw=read(verify_file_pin(p['probe']))
        if native['inputs'][role]!=p['probe'] or native['inputs'][role+'_preparation']!=common['preparations'][role]:raise ValueError('Native reward/source rows differ')
        verify_source_snapshot(read(verify_file_pin(p['source_receipt'])),live=True)
        source_raw={};exemplar=next(c for c in raw['cases'] if c['available'])
        for entry in raw['sources']:
            state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(state,model_context['backend'])
            if not raw_tree_equal(contexts[entry['name']],source_context(state,exemplar)):raise ValueError('Verified controller context changed')
            source_raw[entry['name']]=state['base_raw']
        directions[role]=np.load(verify_file_pin(p['directions']))['directions'];rows=[];matches=[];bases[role]={}
        for i,m in enumerate(p['metadata']):
            c=raw['cases'][m['case_index']];profile=m['profile'];direction=np.zeros(20) if profile['direction']<0 else directions[role][profile['direction']]
            expected=perturbed_actor_raw(source_raw[m['source_name']],direction,profile['epsilon'],profile['sign'])
            with np.load(verify_file_pin(c['trace'])) as z:actual=z['actions20'][0]
            matches.append(c['name']==m['name'] and c['source_name']==m['source_name'] and np.array_equal(expected,actual));rows.append(actual)
            if profile['direction']<0 and profile['repeat']==0:bases[role][m['source_name']]=i
        raw_actions[role]=np.asarray(rows);truth[role]=np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values']
        checks[role+'_all_actual_RAW_actions']=all(matches) and len(rows)==p['total_rows']
        checks[role+'_masks_unchanged']=np.array_equal(data[role]['valid'],[m['valid'] for m in p['metadata']])
        cohorts[role]=dict(declared_sources=len(raw['sources']),observed_base_sources=len(bases[role]),models={})
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh one-step diagnostic output required')
    out.mkdir();receipt=freeze(out);checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    device=torch.device(model_context['config']['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output_arrays={};phase='model_restore'
    try:
        for name in MODELS:
            phase=name;arm=restore_backend_arm(checkpoint['arms'][name],model_context,device);model=arm['model'];model.eval()
            before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for role,p in prepared.items():
                metadata=p['metadata'];names=[m['source_name'] for m in metadata]
                x=torch.tensor(data[role]['x'],device=device,dtype=torch.float32);raw=torch.tensor(raw_actions[role],device=device,dtype=torch.float32)
                context=batch_context(contexts,names,device)
                with torch.no_grad():
                    prediction=model.prediction(x);scores=observable_proxy(x,prediction,raw,context,reward,objective)
                    values=torch.stack([scores[s] for s in SCORES],-1).cpu().numpy()
                base_names=list(bases[role]);base_ids=[bases[role][s] for s in base_names]
                grad=raw_reward_gradient(model,x[base_ids],raw[base_ids],batch_context(contexts,base_names,device),reward,objective)
                arrays={key:value.cpu().numpy() for key,value in grad.items()}
                per_direction={(source,j):arrays['RAW'][i].astype(np.float64)@directions[role][j] for i,source in enumerate(base_names) for j in range(20)}
                records,aggregate=compare_reward_responses(p,truth[role],values,per_direction,bases[role])
                checks[name+'_'+role+'_finite']=bool(np.isfinite(values).all() and all(np.isfinite(a).all() for a in arrays.values()))
                checks[name+'_'+role+'_quarter_gain']=bool(np.all(arrays['controller_diagonal']==.25))
                checks[name+'_'+role+'_PD_base_value']=bool(np.array_equal(arrays['computed_PD'],data[role]['x'][base_ids,81:101].astype(np.float32)))
                checks[name+'_'+role+'_all_groups']=len(records)==len(p['response_groups']) and [g['qualified'] for g in records]==[g['qualified'] for g in p['response_groups']]
                output_arrays[role+'_'+name+'_values']=values
                for key,a in arrays.items():output_arrays[role+'_'+name+'_'+key]=a
                cohorts[role]['models'][name]=dict(groups=records,aggregate=aggregate,base_source_order=base_names,base_row_indices=base_ids)
            checks[name+'_model_unchanged']=raw_tree_equal(before,model.state_dict());del arm,model
        np.savez_compressed(out/'model_scores_and_RAW_gradients.npz',**output_arrays);verify_source_snapshot(receipt,live=True)
        report=dict(schema='backend_one_step_RAW_reward_gradient:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,
            source_receipt=file_pin(out/'source_receipt.json'),checkpoint=fit['checkpoint'],arrays=file_pin(out/'model_scores_and_RAW_gradients.npz'),
            score_order=SCORES,cohorts=cohorts,checks=checks,passed=sum(checks.values()),total=len(checks),normalizer_fits=0,
            actual_native_integrations=0,optimizer_steps=0,Actor_updates=0,wall_s=time.perf_counter()-started,
            physical_status='OBSERVABLE_PROXY_ONLY_NOT_ADMITTED_FOR_ACTOR',
            scope='All six fixed final models/observed sources/original masks. One-step RAW chain at actual held-PD base value plus explicit RAW action cost. Both native observable/full rewards and their own scalar screens retained. Interval geometry not predicted;no Actor,full-reward/multistep gradient,task,ego or speedup admission.')
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Preserve RAW reward diagnostic discrepancy')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],physical_status=report['physical_status'])),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
