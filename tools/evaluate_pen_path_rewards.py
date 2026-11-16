"""All-arm saved native FULL/observable reward response evaluation,no updates."""
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
from supermimic_research.learning.pen_path_fit import data_view,registered_context,restore_arm,fixed_spec
from supermimic_research.learning.pen_path_response import ARMS
from supermimic_research.learning.backend_reward_gradient import batch_context,compare_reward_responses
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.learning.reward_gradient_export import add_reward_arrays
from supermimic_research.mechanics.pen_reward_chain import source_context
from supermimic_research.learning.pen_path_reward import SCORES,scores,gradients
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import perturbed_actor_raw

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();ec=tomllib.loads(cp.read_text())
    expected=dict(schema=1,primary_target='FULL_NATIVE',physical_scalar_amplitude_relative_error=.25,physical_scalar_sign_cosine=.9,candidate_AD_relative_error=1.,candidate_AD_sign_cosine=.9)
    if any(ec.get(k)!=v or type(ec.get(k)) is not type(v) for k,v in expected.items()):raise ValueError('Original scalar screen/AD thresholds required')
    fit_path=(cp.parent/ec['fit']).resolve();fit=read(fit_path);fc=verify_file_pin(fit['config']);cfg=fixed_spec(tomllib.loads(fc.read_text()))
    if file_pin(cp)!=fit['inputs']['evaluation_config']:raise ValueError('Exact preregistered evaluation config required')
    paths={k:(fc.parent/cfg[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']}
    common,old_fit,old,recovery,data,prep,path_data,products=data_view(paths);reg,context=registered_context(cfg,{**paths,'config':fc},inputs,common,old_fit,old,products)
    if fit['schema']!='pen_path_fit_result:1' or fit['inputs']!=inputs or fit['update']!=2000 or fit['training_running'] or fit['optimizer_steps']!=6012 or fit['models']!=ARMS:raise ValueError('Terminal full-budget path models required')
    if set(fit['checkpoint_health'])!={'100','2000'} or fit['work']!=dict(live_attempted_calls=6000,live_completed_steps=6000,QA_attempted_calls=12,QA_completed_steps=12):raise ValueError('Both checkpoint boundaries/exact optimizer work required')
    verify_source_snapshot(read(verify_file_pin(fit['source_receipt'])),live=True)
    for u,pin in fit['checkpoint_health'].items():
        h=read(verify_file_pin(pin))
        if h['passed']!=h['total'] or not all(h['checks'].values()) or h['update']!=int(u):raise ValueError('Both full owned checkpoint health checks required')
    prior_path=(cp.parent/ec['prior_RAW']).resolve();prior=read(prior_path);native_path=(cp.parent/ec['native_reward']).resolve();native=read(native_path)
    if (prior['schema']!='backend_one_step_RAW_reward_gradient:1' or prior['passed']!=prior['total'] or not all(prior['checks'].values())
        or prior['inputs']['common_data']!=inputs['common_data'] or prior['inputs']['native_reward_audit']!=file_pin(native_path)
        or native['passed']!=native['total'] or not all(native['checks'].values()) or native['inputs']['common_data']!=inputs['common_data']
        or native['inputs']['objective_config']!=inputs['objective_config'] or native['inputs']['adapter_config']!=inputs['adapter_config']):raise ValueError('Frozen actual RAW/reward audit required')
    for r in [prior,native]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False);reward=tomllib.loads(paths['adapter_config'].read_text())['reward'];objective=tomllib.loads(paths['objective_config'].read_text())
    actions={};bases={};directions={};truth={};checks={}
    for role,p in prep.items():
        if native['inputs'][role+'_preparation']!=common['preparations'][role] or native['inputs'][role]!=p['probe']:raise ValueError('Exact old RAW/prepared row identity required')
        raw=read(verify_file_pin(p['probe']));exemplar=next(c for c in raw['cases'] if c['available']);base_raw={};bases[role]={}
        for entry in raw['sources']:
            state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(state,old['backend'])
            if not raw_tree_equal(contexts[entry['name']],source_context(state,exemplar)):raise ValueError('Controller source changed')
            base_raw[entry['name']]=state['base_raw']
        dirs=directions[role]=np.load(verify_file_pin(p['directions']))['directions'];rows=[]
        for i,m in enumerate(p['metadata']):
            c=raw['cases'][m['case_index']];profile=m['profile'];direction=np.zeros(20) if profile['direction']<0 else dirs[profile['direction']]
            if c['name']!=m['name'] or c['profile']!=profile:raise ValueError('Prior verified actual-RAW row identity changed')
            rows.append(perturbed_actor_raw(base_raw[m['source_name']],direction,profile['epsilon'],profile['sign']))
            if profile['direction']<0 and profile['repeat']==0:bases[role][m['source_name']]=i
        if not prior['checks'][role+'_all_actual_RAW_actions']:raise ValueError('Prior actual RAW/expected equality missing')
        actions[role]=np.asarray(rows);truth[role]=np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values']
        checks[role+'_masks']=np.array_equal(data[role]['valid'],[m['valid'] for m in p['metadata']])
    out=(cp.parent/ec['output']).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh path reward evaluation required')
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    ck=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);device=torch.device(cfg['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    arrays={role+'_RAW20':v for role,v in actions.items()};cohorts={role:dict(models={}) for role in prep};stage='models'
    try:
        for name in ARMS:
            stage=name;arm=restore_arm(ck['arms'][name],context,device);model=arm['model'];model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for role,p in prep.items():
                metadata=p['metadata'];names=[m['source_name'] for m in metadata];x=torch.tensor(data[role]['x'],device=device,dtype=torch.float32);raw=torch.tensor(actions[role],device=device,dtype=torch.float32)
                with torch.no_grad():values,pred,path=scores(model,x,raw,batch_context(contexts,names,device),reward,objective)
                values=values.cpu().numpy();prediction=pred.cpu().numpy();path=path.cpu().numpy();base_names=list(bases[role]);base_ids=[bases[role][n] for n in base_names]
                grad=gradients(model,x[base_ids],raw[base_ids],batch_context(contexts,base_names,device),reward,objective);ga={k:v.cpu().numpy() for k,v in grad.items()}
                projected={(s,j):ga['RAW'][i].astype(np.float64)@directions[role][j] for i,s in enumerate(base_names) for j in range(20)}
                records,aggregate=compare_reward_responses(p,truth[role],values,projected,bases[role])
                # Preserve the maintained native masks/screens/aggregation;rename
                # score labels only because model scores now include soft path.
                aggregate={target:{SCORES[k]:a[old_name] for k,old_name in enumerate(['observable_residual','observable_support'])} for target,a in aggregate.items()}
                mask=data[role]['valid'];error=((prediction-data[role]['y'])/np.asarray(old['scales']['y_scale'],dtype=np.float32))**2
                metrics={block:float(error[mask,sl].mean()) for block,sl in dict(all90=slice(0,90),object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90)).items()}
                metrics['path_normalized_MSE']=float(np.mean(((path[mask]-path_data[role]['geometry_loss'][mask])/context['path_units']['point_scale'])**2))
                checks[name+'_'+role+'_finite']=bool(np.isfinite(values).all() and np.isfinite(prediction).all() and np.isfinite(path).all() and all(np.isfinite(a).all() for a in ga.values()))
                checks[name+'_'+role+'_quarter_gain']=bool(np.all(ga['controller_diagonal']==.25))
                checks[name+'_'+role+'_PD_base']=np.array_equal(ga['computed_PD'],data[role]['x'][base_ids,81:101].astype(np.float32))
                checks[name+'_'+role+'_all_groups']=len(records)==len(p['response_groups']) and [g['qualified'] for g in records]==[g['qualified'] for g in p['response_groups']]
                prefix=role+'_'+name+'_';add_reward_arrays(arrays,prefix,values,ga);arrays[prefix+'prediction90']=prediction;arrays[prefix+'path_loss']=path
                cohorts[role]['models'][name]=dict(groups=records,aggregate=aggregate,forward=metrics,base_source_order=base_names,base_row_indices=base_ids)
            checks[name+'_unchanged_model']=raw_tree_equal(before,model.state_dict());del model,arm
        np.savez_compressed(out/'scores_gradients_predictions.npz',**arrays);verify_source_snapshot(receipt,live=True)
        report=dict(schema='pen_path_RAW_reward_evaluation:1',config=file_pin(cp),fit=file_pin(fit_path),registration=fit['registration'],inputs=dict(native_reward=file_pin(native_path),prior_RAW=file_pin(prior_path)),source_receipt=file_pin(out/'source_receipt.json'),
            checkpoint=fit['checkpoint'],arrays=file_pin(out/'scores_gradients_predictions.npz'),score_order=SCORES,cohorts=cohorts,checks=checks,passed=sum(checks.values()),total=len(checks),
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,TRAIN_path_unit_fits=0,wall_s=time.perf_counter()-started,
            scope='One-step soft-path corrected proxy compared to original native FULL and observable reward. ALL3 arms/groups/masks retained. Expected RAW uses frozen prior actual-RAW equality and same source/controller/rows. Hard error/failure flags unmodeled,observed allzero;not Actor/task/ego/speedup admission.')
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Path gradient evaluation discrepancy retained')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],forward={r:{n:m['forward'] for n,m in c['models'].items()} for r,c in cohorts.items()})),flush=True)
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args())
