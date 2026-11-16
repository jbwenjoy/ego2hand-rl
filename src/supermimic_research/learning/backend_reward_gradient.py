"""Read-only one-step observable reward derivatives;full window reward is absent."""
import hashlib
import json
import tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import verify_source_snapshot
from ..runtime.explicit_collision_backend import load_contract
from ..mechanics.pen_reward_chain import controller,observable_proxy
from .backend_response_fit import MODELS,model_spec
from .pen_native_response import repeated_contrast_spread

SCORES=['observable_residual','observable_support']


def frozen_model_view(common_path,fit_path):
    """Reconstruct expected ownership from pinned products,without fitting scales."""
    read=lambda p:json.loads(Path(p).read_text())
    common_path,fit_path=Path(common_path),Path(fit_path);common,fit=read(common_path),read(fit_path)
    cp=verify_file_pin(fit['inputs']['config']);spec=model_spec(tomllib.loads(cp.read_text()))
    backend=load_contract((cp.parent/spec['backend_manifest']).resolve())
    if (common.get('schema')!='backend_pen_response_common_data:1' or common['collision_backend']!=backend
            or common['passed']!=common['total'] or not all(common['checks'].values())
            or common['scale_input_role']!='VALID_NEW_BACKEND_TRAIN_ONLY' or common['validation_rows_used_for_TRAIN']
            or fit.get('schema')!='backend_response_fit_result:1' or fit['update']!=2000 or fit['training_running']
            or fit['optimizer_steps']!=12024 or fit['models']!=MODELS or fit['inputs']['dataset']!=file_pin(common_path)):
        raise ValueError('Exact terminal owned six-arm model and reviewed new dataset required')
    reg=read(verify_file_pin(fit['registration']))
    if reg['inputs']!=fit['inputs']:raise ValueError('Model registration differs')
    for item in [common,fit,reg]:verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
    products={k:dict(np.load(verify_file_pin(v))) for k,v in common['products'].items()}
    units=dict(np.load(verify_file_pin(reg['response_units'])))
    pairs=products['pairs']
    pair_contract=dict(schema='backend_response_pair_bank:1',backend_id=backend['backend_id'],dataset=file_pin(common_path),
        point_bank=common['outputs']['TRAIN'],pair_bank=common['products']['pairs'],
        index_sha256=hashlib.sha256(pairs['TRAIN_group_index'].tobytes()).hexdigest(),rows=len(pairs['epsilon']))
    context=dict(config=spec,backend=backend,dataset=file_pin(common_path),physics=common['physics'],
        scales=products['scales'],head_scales=products['head_scales'],response_scale=units['response_scale'],pair_contract=pair_contract)
    return common,fit,context


def batch_context(contexts,names,device):
    result={}
    for key in ['EMA','reference_joints','encoder_bias','lower','upper','reference_pose','target_world']:
        result[key]=torch.stack([contexts[name][key] for name in names]).to(device)
    result['target_valid']=torch.tensor([contexts[name]['target_valid'] for name in names],device=device,dtype=torch.bool)
    return result


def raw_reward_gradient(model,x,raw,context,reward_cfg,objective_cfg):
    """Actual-PD base value;analytic controller diagonal supplies the chain factor.

    EMA/source state is fixed for this ONE step. No straight-through value offset,
    source replacement,window-geometry approximation or future-controller claim.
    """
    x=x.detach().clone().requires_grad_(True);raw=raw.detach().clone().requires_grad_(True)
    PD,_,_=controller(raw,context)
    diagonal=torch.autograd.grad(PD.sum(),raw,retain_graph=True)[0]
    prediction=model.prediction(x);scores=observable_proxy(x,prediction,raw,context,reward_cfg,objective_cfg)
    chained=[];PD_parts=[];direct_parts=[]
    for score in SCORES:
        gx,gr=torch.autograd.grad(scores[score].sum(),(x,raw),retain_graph=True)
        PD_parts.append(gx[:,81:101]);direct_parts.append(gr)
        chained.append(gx[:,81:101]*diagonal+gr)
    return dict(values=torch.stack([scores[s] for s in SCORES],-1).detach(),
        RAW=torch.stack(chained,1).detach(),PD_partial=torch.stack(PD_parts,1).detach(),
        RAW_direct_partial=torch.stack(direct_parts,1).detach(),controller_diagonal=diagonal.detach(),
        computed_PD=PD.detach())


def scalar_screen(values,metadata,groups):
    """Two native scalar scores,two amplitudes/repeats;no model-dependent filters."""
    paired={}
    for group in groups:
        p,m=group['rows']['1'],group['rows']['-1']
        if len(p)!=2 or len(m)!=2:continue
        qualified=bool(all(metadata[i]['valid'] for i in p+m))
        if qualified!=group['qualified']:raise ValueError('Original physical masks changed')
        fd=(values[p].mean(0)-values[m].mean(0))/(8*group['epsilon'])
        spread=[repeated_contrast_spread(values[p,j:j+1],values[m,j:j+1],np.ones(1)) for j in range(2)]
        paired[group['source_name'],group['direction'],group['epsilon']]=(qualified,fd,spread)
    result={}
    identities=list(dict.fromkeys((g['source_name'],g['direction']) for g in groups))
    for source,j in identities:
        a,b=paired.get((source,j,.002)),paired.get((source,j,.005));flags=[]
        for k in range(2):
            ok=False
            if a is not None and b is not None:
                x,y=a[1][k],b[1][k]
                ok=bool(a[0] and b[0] and y!=0 and abs(x-y)/abs(y)<=.25 and x*y>0
                    and a[2][k]['contrast_exceeds_observed_spread'] and b[2][k]['contrast_exceeds_observed_spread'])
            flags.append(ok)
        result[source,j]=flags
    return result


def compare_reward_responses(preparation,native_components,model_values,raw_gradients,base_rows):
    """Retain declared groups;separate observable/full targets and their screens."""
    metadata=preparation['metadata'];groups=preparation['response_groups']
    native_observable=np.column_stack([native_components[:,4],native_components[:,4]+.2*native_components[:,6]])
    native_full=native_components[:,[7,8]]
    screens={target:scalar_screen(values,metadata,groups) for target,values in [('OBSERVABLE_NATIVE',native_observable),('FULL_NATIVE',native_full)]}
    records=[];cache={}
    for gi,g in enumerate(groups):
        source,j,e=g['source_name'],g['direction'],g['epsilon'];p,m=g['rows']['1'],g['rows']['-1']
        done=len(p)==len(m)==2;qualified=bool(done and all(metadata[i]['valid'] for i in p+m))
        if g['complete']!=done or g['qualified']!=qualified:raise ValueError('All physical group masks must remain')
        base=base_rows.get(source)
        row=dict(group_index=gi,source_name=source,direction=j,epsilon=e,raw_amplitude=4*e,complete=done,qualified=qualified,
            base_row=base,base_physical_valid=base is not None and bool(metadata[base]['valid']),metrics=None)
        if done:
            targets={target:(values[p].mean(0)-values[m].mean(0))/(8*e) for target,values in [('OBSERVABLE_NATIVE',native_observable),('FULL_NATIVE',native_full)]}
            fd=(model_values[p].mean(0)-model_values[m].mean(0))/(8*e)
            ad=raw_gradients.get((source,j));row['metrics']={}
            for target,t in targets.items():
                row['metrics'][target]=[]
                for k in range(2):
                    rel=None if ad is None or t[k]==0 else float(abs(ad[k]-t[k])/abs(t[k]))
                    cosine=None if ad is None or ad[k]*t[k]==0 else float(np.sign(ad[k]*t[k]))
                    candidate=bool(qualified and row['base_physical_valid'] and screens[target][source,j][k]
                        and rel is not None and rel<1 and cosine is not None and cosine>=.9)
                    row['metrics'][target].append(dict(native_FD_per_RAW=float(t[k]),model_FD_per_RAW=float(fd[k]),
                        model_AD_per_RAW=None if ad is None else float(ad[k]),physical_scalar_screen=screens[target][source,j][k],
                        candidate_local_AD=candidate))
            cache[gi]=dict(targets=targets,fd=fd,ad=ad)
        records.append(row)
    aggregate={}
    for target in ['OBSERVABLE_NATIVE','FULL_NATIVE']:
        aggregate[target]={}
        for k,score in enumerate(SCORES):
            aggregate[target][score]={}
            for subset in ['ALL_PHYSICAL_VALID','FROZEN_SCALAR_SCREEN']:
                ids=[i for i,r in enumerate(records) if r['qualified'] and (subset=='ALL_PHYSICAL_VALID' or screens[target][r['source_name'],r['direction']][k])]
                ad_ids=[i for i in ids if cache[i]['ad'] is not None]
                den=sum(float(cache[i]['targets'][target][k]**2) for i in ids)
                ad_den=sum(float(cache[i]['targets'][target][k]**2) for i in ad_ids)
                ae=sum(float((cache[i]['ad'][k]-cache[i]['targets'][target][k])**2) for i in ad_ids)
                fe=sum(float((cache[i]['fd'][k]-cache[i]['targets'][target][k])**2) for i in ids)
                agreement_den=sum(float(cache[i]['fd'][k]**2) for i in ad_ids)
                agreement_err=sum(float((cache[i]['ad'][k]-cache[i]['fd'][k])**2) for i in ad_ids)
                aggregate[target][score][subset]=dict(pairs=len(ids),AD_pairs=len(ad_ids),missing_AD_pairs=len(ids)-len(ad_ids),
                    invalid_native_base_pairs=sum(not records[i]['base_physical_valid'] for i in ad_ids),
                    AD_relative_L2=float(np.sqrt(ae/ad_den)) if ad_den else None,
                    model_FD_relative_L2=float(np.sqrt(fe/den)) if den else None,
                    AD_vs_model_FD_relative_L2=float(np.sqrt(agreement_err/agreement_den)) if agreement_den else None,
                    same_sign_AD_pairs=sum(bool(cache[i]['ad'][k]*cache[i]['targets'][target][k]>0) for i in ad_ids),
                    candidate_local_AD_pairs=sum(records[i]['metrics'][target][k]['candidate_local_AD'] for i in ids),
                    normalized_native_squared_norm=den,AD_subset_native_squared_norm=ad_den)
    return records,aggregate
