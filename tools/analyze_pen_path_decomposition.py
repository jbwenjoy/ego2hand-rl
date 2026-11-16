"""All-model within-network path attribution and unchanged native FD targets."""
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
from supermimic_research.learning.pen_path_fit import fixed_spec,data_view,registered_context,restore_arm
from supermimic_research.learning.pen_path_response import ARMS
from supermimic_research.learning.pen_path_decomposition import MODES,values,gradients
from supermimic_research.learning.backend_reward_gradient import batch_context,compare_reward_responses
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def gradient_close(a,b):
    budget=8*np.finfo(np.float32).eps*(1+abs(a)+abs(b))
    return bool(np.all(abs(a-b)<=budget)),float(abs(a-b).max())


def attribution(g):
    result=[]
    for i in range(len(g)):
        channels=[]
        for k in range(2):
            obs,path,full=g[i,:,k].astype(np.float64);no,np_,nf=map(float,map(np.linalg.norm,[obs,path,full]));cos=None if not no or not nf else float(np.clip(np.dot(obs,full)/(no*nf),-1,1))
            channels.append(dict(observable_norm=no,path_norm=np_,full_norm=nf,path_to_full_norm=None if not nf else np_/nf,full_observable_cosine=cos,
                full_observable_angle_deg=None if cos is None else float(np.degrees(np.arccos(cos))),unit_direction_difference=None if cos is None else float(np.linalg.norm(obs/no-full/nf))))
        result.append(channels)
    return result


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if cfg['schema']!=1 or cfg['mode']!='WITHIN_FROZEN_MODEL_PATH_SCORE_DECOMPOSITION' or cfg['models']!=ARMS:raise ValueError('Fixed all-model attribution required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['fit','evaluation','model_ledger','native_suggestion_ledger','protocol','registration','output']};inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']}
    fit=read(paths['fit']);evaluation=read(paths['evaluation']);ledger=read(paths['model_ledger']);native_ledger=read(paths['native_suggestion_ledger']);native_review=read(verify_file_pin(native_ledger['review']))
    if (ledger['fit']!=inputs['fit'] or ledger['evaluation']!=inputs['evaluation'] or evaluation['fit']!=inputs['fit'] or evaluation['passed']!=evaluation['total'] or not all(evaluation['checks'].values())
        or native_review['passed']!=native_review['total'] or not all(native_review['checks'].values())):raise ValueError('Exact completed model/native evidence required')
    fc=verify_file_pin(fit['config']);spec=fixed_spec(tomllib.loads(fc.read_text()));fpaths={k:(fc.parent/spec[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']}
    finputs={k:file_pin(p) for k,p in fpaths.items() if k not in ['registration','output']};common,old_fit,old,recovery,data,prep,path_data,products=data_view(fpaths)
    reg,context=registered_context(spec,{**fpaths,'config':fc},finputs,common,old_fit,old,products)
    for obj in [fit,evaluation]:verify_source_snapshot(read(verify_file_pin(obj['source_receipt'])),live=True)
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh decomposition registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_path_decomposition_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),models=ARMS,modes=MODES,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    registration=read(paths['registration'])
    if registration['config']!=file_pin(cp) or registration['inputs']!=inputs:raise ValueError('Registered attribution changed')
    verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True)
    native=read(verify_file_pin(evaluation['inputs']['native_reward']));contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False);prior=dict(np.load(verify_file_pin(evaluation['arrays'])))
    truth={role:np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values'] for role in prep};objective=tomllib.loads(fpaths['objective_config'].read_text());reward=tomllib.loads(fpaths['adapter_config'].read_text())['reward']
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh decomposition output required')
    out.mkdir();receipt=freeze(out);checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);device=torch.device(spec['device'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    arrays={};cohorts={r:dict(models={}) for r in prep};checks={};differences={};stage='models'
    try:
        for name in ARMS:
            stage=name;arm=restore_arm(checkpoint['arms'][name],context,device);model=arm['model'];model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for role,p in prep.items():
                old_result=evaluation['cohorts'][role]['models'][name];base_names=old_result['base_source_order'];base_ids=old_result['base_row_indices'];base_map=dict(zip(base_names,base_ids));prefix=role+'_'+name+'_'
                x=torch.tensor(data[role]['x'],device=device,dtype=torch.float32);raw=torch.tensor(prior[role+'_RAW20'],device=device,dtype=torch.float32);names=[m['source_name'] for m in p['metadata']]
                with torch.no_grad():v=values(model,x,raw,batch_context(contexts,names,device),reward,objective).cpu().numpy()
                grad=gradients(model,x[base_ids],raw[base_ids],batch_context(contexts,base_names,device),reward,objective);ga={k:t.cpu().numpy() for k,t in grad.items()}
                arrays[prefix+'values']=v
                for k,a in ga.items():arrays[prefix+('base_values' if k=='values' else k)]=a
                dirs=np.load(verify_file_pin(p['directions']))['directions'];mode_results={}
                for mi,mode in [(0,'OBSERVABLE'),(2,'FULL')]:
                    projected={(s,j):ga['RAW'][i,mi].astype(np.float64)@dirs[j] for i,s in enumerate(base_names) for j in range(20)}
                    records,aggregate=compare_reward_responses(p,truth[role],v[:,mi],projected,base_map)
                    mode_results[mode]=dict(groups=records,aggregate={t:{['residual','support'][k]:a[oldname] for k,oldname in enumerate(['observable_residual','observable_support'])} for t,a in aggregate.items()})
                for key in ['RAW','PD_partial','RAW_direct_partial']:
                    ok,delta=gradient_close(ga[key][:,2],ga[key][:,0]+ga[key][:,1]);checks[prefix+key+'_additive']=ok;differences[prefix+key+'_additive']=delta
                checks[prefix+'values_additive']=bool(np.allclose(v[:,2],v[:,0]+v[:,1],rtol=0,atol=1e-12))
                ok,delta=gradient_close(ga['RAW'][:,2],prior[prefix+'RAW']);checks[prefix+'prior_FULL_gradient']=ok;differences[prefix+'prior_FULL_RAW']=delta
                differences[prefix+'prior_FULL_values']=float(abs(v[:,2]-prior[prefix+'values']).max());checks[prefix+'prior_FULL_values']=differences[prefix+'prior_FULL_values']<=1e-12
                checks[prefix+'actual_PD']=np.array_equal(ga['computed_PD'],data[role]['x'][base_ids,81:101].astype(np.float32));checks[prefix+'quarter_gain']=bool(np.all(ga['controller_diagonal']==.25))
                checks[prefix+'zero_direct_PATH']=not bool(ga['RAW_direct_partial'][:,1].any());checks[prefix+'all_original_groups']=all(len(m['groups'])==len(p['response_groups']) and [g['qualified'] for g in m['groups']]==[g['qualified'] for g in p['response_groups']] for m in mode_results.values())
                checks[prefix+'all_finite']=bool(np.isfinite(v).all() and all(np.isfinite(a).all() for a in ga.values()))
                if name=='RESPONSE_ONLY':checks[prefix+'zero_baseline_path_gradient']=not bool(ga['RAW'][:,1].any())
                attrs=attribution(ga['RAW']);summary=[]
                for k,goal in enumerate(['RESIDUAL','SUPPORT']):
                    rows=[a[k] for a in attrs];ratios=[a['path_to_full_norm'] for a in rows if a['path_to_full_norm'] is not None];angles=[a['full_observable_angle_deg'] for a in rows if a['full_observable_angle_deg'] is not None]
                    summary.append(dict(goal=goal,sources=len(rows),defined_norm_ratios=len(ratios),median_path_to_full_norm=float(np.median(ratios)) if ratios else None,max_path_to_full_norm=float(max(ratios)) if ratios else None,
                        median_full_observable_angle_deg=float(np.median(angles)) if angles else None,max_full_observable_angle_deg=float(max(angles)) if angles else None))
                cohorts[role]['models'][name]=dict(modes=mode_results,base_source_order=base_names,base_row_indices=base_ids,source_attribution=attrs,attribution_summary=summary)
            checks[name+'_unchanged_model']=raw_tree_equal(before,model.state_dict());del arm,model
        for role in prep:
            expected_base=cohorts[role]['models'][ARMS[0]]
            checks[role+'_identical_base_order']=all(m['base_source_order']==expected_base['base_source_order'] and m['base_row_indices']==expected_base['base_row_indices'] for m in cohorts[role]['models'].values())
            for name in ARMS[1:]:
                baseline=arrays[role+'_RESPONSE_ONLY_RAW'][:,0].astype(np.float64);other=arrays[role+'_'+name+'_RAW'][:,0].astype(np.float64)
                denom=float(np.sum(baseline**2));cohorts[role]['models'][name]['between_model_observable_relative_L2']=float(np.sqrt(np.sum((other-baseline)**2)/denom)) if denom else None
        np.savez_compressed(out/'scores_and_gradient_components.npz',**arrays);verify_source_snapshot(receipt,live=True)
        result=dict(schema='pen_frozen_path_gradient_decomposition:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'scores_and_gradient_components.npz'),
            modes=MODES,cohorts=cohorts,checks=checks,passed=sum(checks.values()),total=len(checks),differences=differences,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
            scope='Within-network observable/path additive derivatives,all fixed models/rows/groups. Between-model observable differences describe this fixed training comparison. No new native evidence for observable-only directions,no Actor/multistep/task/ego/acceleration admission.')
        write_json(out/'report.json',result)
        if not all(checks.values()):raise ValueError('Frozen gradient decomposition discrepancy retained')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],attribution={r:{n:m['attribution_summary'] for n,m in c['models'].items()} for r,c in cohorts.items()})),flush=True)
    except Exception as e:
        write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
