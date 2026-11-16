"""Register and freeze all model directions/prospective scores before physics."""
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
from supermimic_research.learning.pen_path_reward import scores
from supermimic_research.learning.backend_reward_gradient import batch_context
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.learning.pen_native_suggestions import GOALS,EPSILONS,direction_from_raw_gradient,proposal_raw,profiles
from supermimic_research.mechanics.pen_reward_chain import controller,source_context
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.full_rehash_scope import implementation_pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    expected=dict(schema=1,mode='FIXED_ALL_PATH_MODELS_NATIVE_ONE_STEP',models=ARMS,goals=GOALS,PD_eps_rad=EPSILONS,signs=[1,-1],repeats=2,native_budget_max=12000,verifier='SINGLE_RESOLVE_FULL_REHASH')
    if any(cfg.get(k)!=v or type(cfg.get(k)) is not type(v) for k,v in expected.items()):raise ValueError('Fixed all-source/all-model native plan required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['fit','evaluation','model_ledger','backend_manifest','objective_config','protocol','registration','bank_output','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','bank_output','output']};fit=read(paths['fit']);evaluation=read(paths['evaluation']);ledger=read(paths['model_ledger']);review=read(verify_file_pin(ledger['review']))
    if (ledger['fit']!=inputs['fit'] or ledger['evaluation']!=inputs['evaluation'] or review['passed']!=review['total'] or not all(review['checks'].values())
        or evaluation['fit']!=inputs['fit'] or evaluation['passed']!=evaluation['total'] or not all(evaluation['checks'].values())):raise ValueError('Exact reviewed final models/evaluation required')
    fc=verify_file_pin(fit['config']);spec=fixed_spec(tomllib.loads(fc.read_text()));fpaths={k:(fc.parent/spec[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']}
    finputs={k:file_pin(p) for k,p in fpaths.items() if k not in ['registration','output']};common,old_fit,old,recovery,data,prep,path_data,products=data_view(fpaths)
    model_reg,model_context=registered_context(spec,{**fpaths,'config':fc},finputs,common,old_fit,old,products)
    if old['backend']['manifest']!=inputs['backend_manifest'] or fit['inputs']['objective_config']!=inputs['objective_config']:raise ValueError('Same native backend/objective required')
    for r in [fit,evaluation,review]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/probe_pen_native_suggestions.py',cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh model suggestion registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='pen_native_suggestions_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
            implementation_pins=implementation_pins(),declared_sources=36,observed_sources=30,declared_controls=1800,maximum_executed_controls=1500,native_budget_max=12000,optimizer_steps=0,normalizer_fits=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['implementation_pins']!=implementation_pins():raise ValueError('Registered suggestion inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['bank_output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh prospective proposal bank required')
    native=read(verify_file_pin(evaluation['inputs']['native_reward']));contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False);gradient_arrays=dict(np.load(verify_file_pin(evaluation['arrays'])))
    objective=tomllib.loads(paths['objective_config'].read_text());reward=tomllib.loads(fpaths['adapter_config'].read_text())['reward'];records=[];checks={}
    for role,p in prep.items():
        raw=read(verify_file_pin(p['probe']));exemplar=next(c for c in raw['cases'] if c['available'])
        for entry in raw['sources']:
            state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(state,old['backend'])
            if not raw_tree_equal(contexts[entry['name']],source_context(state,exemplar)):raise ValueError('Original controller context differs')
            baserows=[i for i,m in enumerate(p['metadata']) if m['source_name']==entry['name'] and m['profile']['direction']<0 and m['profile']['repeat']==0]
            observed=bool(entry['qualified'] and len(baserows)==1);base_row=baserows[0] if observed else None
            template=None if base_row is None else raw['cases'][p['metadata'][base_row]['case_index']]
            item=dict(name=entry['name'],role=role,native_role=raw['role'],original_source=entry,observed=observed,original_base_row=base_row,
                original_base_physical_valid=bool(observed and p['metadata'][base_row]['valid']),template=template,base_raw=np.asarray(state['base_raw']).tolist(),
                base_x102=None if base_row is None else data[role]['x'][base_row].tolist(),model_directions={},model_base_scores={},profiles=[])
            if observed:
                for name in ARMS:
                    mr=evaluation['cohorts'][role]['models'][name];bi=mr['base_source_order'].index(entry['name'])
                    if mr['base_row_indices'][bi]!=base_row:raise ValueError('Source gradient/base row differs')
                    gradients=gradient_arrays[role+'_'+name+'_RAW'][bi]
                    for k,goal in enumerate(GOALS):
                        u,norm=direction_from_raw_gradient(gradients[k]);item['model_directions'][name+'_'+goal]=dict(unit=None if u is None else u.tolist(),RAW_norm=norm,RAW_gradient=gradients[k].tolist())
            for profile in profiles(ARMS):
                row=dict(profile,available=False,reason='ORIGINAL_SOURCE_OR_GRADIENT_UNAVAILABLE',raw20=None,expected_PD20=None,predicted_scores=None)
                if observed:
                    if profile['model'] is None:
                        row.update(available=True,reason=None,raw20=np.asarray(state['base_raw'],np.float32).tolist())
                    else:
                        d=item['model_directions'][profile['model']+'_'+profile['goal']]
                        if d['unit'] is None:row['reason']='ZERO_RAW_GRADIENT'
                        else:
                            proposal,ok=proposal_raw(state['base_raw'],d['unit'],profile['epsilon'],profile['sign']);row.update(available=ok,raw20=proposal.tolist(),reason=None if ok else 'PROPOSAL_WOULD_CLIP_NO_PROJECTION')
                    if row['available']:
                        PD,_,_=controller(torch.tensor([row['raw20']],dtype=torch.float32),contexts[entry['name']]);row['expected_PD20']=PD[0].tolist()
                item['profiles'].append(row)
            records.append(item)
    checks['all36_declared_sources']=len(records)==36;checks['all30_observed_sources']=sum(r['observed'] for r in records)==30;checks['all1800_declared_controls']=sum(len(r['profiles']) for r in records)==1800
    checks['all_available_PD_matches_declared_radius']=all(np.max(abs(np.asarray(p['expected_PD20'])-np.asarray(r['template']['base_PD20'])-(0 if p['model'] is None else p['sign']*p['epsilon']*np.asarray(r['model_directions'][p['model']+'_'+p['goal']]['unit']))))<=2e-7 for r in records for p in r['profiles'] if p['available'])
    checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);device=torch.device(spec['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ARMS:
        arm=restore_arm(checkpoint['arms'][name],model_context,device);model=arm['model'];model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        selected=[(r,p) for r in records for p in r['profiles'] if p['available'] and p['model'] in [None,name]]
        xx=np.array([r['base_x102'] for r,p in selected],np.float32);xx[:,81:101]=[p['expected_PD20'] for r,p in selected]
        rr=np.array([p['raw20'] for r,p in selected],np.float32);names=[r['name'] for r,p in selected]
        with torch.no_grad():value,_,_=scores(model,torch.tensor(xx,device=device),torch.tensor(rr,device=device),batch_context(contexts,names,device),reward,objective)
        value=value.cpu().numpy()
        for i,(r,p) in enumerate(selected):
            if p['model'] is None:
                if p['repeat']==0:r['model_base_scores'][name]=value[i].tolist()
            else:p['predicted_scores']=value[i].tolist()
        checks[name+'_unchanged_model']=raw_tree_equal(before,model.state_dict());checks[name+'_finite_scores']=bool(np.isfinite(value).all());del arm,model
    for r in records:
        for p in r['profiles']:
            if p['available'] and p['model'] is not None:
                p['predicted_goal_gain']=p['predicted_scores'][p['goal_index']]-r['model_base_scores'][p['model']][p['goal_index']]
                p['predicted_linear_goal_gain']=4*p['sign']*p['epsilon']*r['model_directions'][p['model']+'_'+p['goal']]['RAW_norm']
    if not all(checks.values()):raise ValueError('Fixed suggestion bank discrepancy: '+repr([k for k,v in checks.items() if not v]))
    out.mkdir();receipt=freeze(out);verify_source_snapshot(receipt,live=True)
    report=dict(schema='pen_frozen_native_suggestion_bank:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        sources=records,checks=checks,passed=sum(checks.values()),total=len(checks),declared_controls=1800,available_controls=sum(p['available'] for r in records for p in r['profiles']),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Fixed bank from all3 frozen model gradients/2goals/36 original sources. No native steps,Actor update or clipping projection;all unavailable candidates retained. Fresh same-model base scores for prospective gain comparisons.')
    write(out/'report.json',report);print(json.dumps(dict(bank=file_pin(out/'report.json'),checks=[report['passed'],report['total']],available=report['available_controls'],declared=1800)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
