"""All declared new-backend validation groups,including empty/masked groups."""
import json
import subprocess
import time
import tomllib
from pathlib import Path
import numpy as np
import torch
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .backend_response_fit import (MODELS,model_spec,load_dataset,restore_backend_arm,read,write)
from .pen_native_response import compare_native_response,vector_comparison

BLOCKS=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))


def comparison(prediction,target):
    result=vector_comparison(prediction,target)
    result['relative_L2_error']=result['absolute_L2_error']/result['target_L2'] if result['target_L2']>0 else None
    return result


def group_responses(report,data,predictions,jvps,base_rows,physical_screens,scale):
    """No fitting:preserve nulls for missing groups and unchanged physical masks."""
    rows=[];native={};predicted={}
    for gi,g in enumerate(report['response_groups']):
        source,j,e=g['source_name'],g['direction'],g['epsilon'];plus=g['rows']['1'];minus=g['rows']['-1']
        complete=len(plus)==len(minus)==2
        if g['complete']!=complete or g['qualified']!=bool(complete and data['valid'][plus+minus].all()):
            raise ValueError('Original physical masks/completeness cannot change')
        screens=physical_screens.get((source,j))
        row=dict(group_index=gi,source_name=source,direction=j,epsilon=e,rows=g['rows'],
            complete=complete,qualified=g['qualified'],base_row=base_rows.get(source),
            base_row_valid=source in base_rows and bool(data['valid'][base_rows[source]]),
            physical_screens={b:bool(screens and screens[b]['physical_screen']) for b in BLOCKS},models=None)
        if complete:
            target=(data['y'][plus].mean(0)-data['y'][minus].mean(0))/(2*e);native[gi]=target;row['models']={}
            for name,prediction in predictions.items():
                fd=(prediction[plus].mean(0)-prediction[minus].mean(0))/(2*e);predicted[name,gi]=fd
                derivative=jvps.get((name,source,j));terms={}
                for b,sl in BLOCKS.items():
                    ad=None if derivative is None else compare_native_response(data['y'][plus,sl],data['y'][minus,sl],derivative[sl],scale[sl],e)[0]
                    if ad is not None:
                        ad['relative_L2_error']=ad['absolute_L2_error']/ad['target_L2'] if ad['target_L2']>0 else None
                        ad['caveat']='Observed two-repeat spread is not a probabilistic bound or proof of differentiability.'
                    terms[b]=dict(AD=ad,model_finite_response=comparison(fd[sl]/scale[sl],target[sl]/scale[sl]),
                        AD_vs_model_finite=None if derivative is None else comparison(derivative[sl]/scale[sl],fd[sl]/scale[sl]),
                        candidate_AD_screen=bool(g['qualified'] and row['base_row_valid'] and row['physical_screens'][b]
                            and ad is not None and ad['relative_L2_error'] is not None and ad['relative_L2_error']<1.
                            and ad['cosine'] is not None and ad['cosine']>=.9))
                row['models'][name]=terms
        rows.append(row)
    aggregate={}
    for name in predictions:
        aggregate[name]={}
        for b,sl in BLOCKS.items():
            aggregate[name][b]={}
            for screen in ['ALL_PHYSICAL_VALID','FROZEN_PHYSICAL_SCREEN']:
                ids=[i for i,r in enumerate(rows) if r['qualified']
                    and (screen=='ALL_PHYSICAL_VALID' or r['physical_screens'][b])]
                ad_ids=[i for i in ids if (name,rows[i]['source_name'],rows[i]['direction']) in jvps]
                den=sum(float(np.sum((native[i][sl]/scale[sl])**2)) for i in ids)
                ad_den=sum(float(np.sum((native[i][sl]/scale[sl])**2)) for i in ad_ids)
                ad_num=sum(float(np.sum(((jvps[name,rows[i]['source_name'],rows[i]['direction']][sl]-native[i][sl])/scale[sl])**2)) for i in ad_ids)
                fd_num=sum(float(np.sum(((predicted[name,i][sl]-native[i][sl])/scale[sl])**2)) for i in ids)
                aggregate[name][b][screen]=dict(pairs=len(ids),AD_pairs=len(ad_ids),finite_response_pairs=len(ids),missing_AD_pairs=len(ids)-len(ad_ids),
                    AD_relative_L2_error=float(np.sqrt(ad_num/ad_den)) if ad_den>0 else None,
                    model_finite_response_relative_L2_error=float(np.sqrt(fd_num/den)) if den>0 else None,
                    zero_response_relative_L2_error=1. if den>0 else None,normalized_native_squared_norm=den,
                    AD_subset_normalized_native_squared_norm=ad_den,
                    candidate_AD_pairs=sum(rows[i]['models'][name][b]['candidate_AD_screen'] for i in ids))
    return rows,aggregate


def evaluate(root,config_path):
    started=time.perf_counter();root=Path(root).resolve();cp=Path(config_path).resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['physical_screen'],cfg['model_AD_relative_error'],cfg['model_AD_cosine'])!=(1,'REGISTERED_RESERVED_VALIDATION_PHYSICAL_SCREEN',1.,.9):
        raise ValueError('Original fixed physical screens and AD thresholds required')
    fp=(cp.parent/cfg['fit_report']).resolve();fit=read(fp);out=(cp.parent/cfg['output']).resolve()
    fit_cp=(cp.parent/cfg['fit_config']).resolve();spec=model_spec(tomllib.loads(fit_cp.read_text()))
    if (fit.get('schema')!='backend_response_fit_result:1' or fit['update']!=2000 or fit['training_running']
            or fit['models']!=MODELS or fit['optimizer_steps']!=12024 or fit['live_optimizer_steps']!=12000
            or fit['clone_only_optimizer_steps']!=24 or fit['validation_evaluation_updates']!=[2000]
            or fit['inputs']['evaluation_config']!=file_pin(cp) or fit['inputs']['config']!=file_pin(fit_cp)):
        raise ValueError('Terminal exact registered new-backend fixed-final trial required')
    pp,prep,data,_,context,_=load_dataset(fit_cp,spec)
    if fit['inputs']['dataset']!=file_pin(pp): raise ValueError('Exact fitted dataset required')
    verify_source_snapshot(read(verify_file_pin(fit['source_receipt'])),live=True)
    for pin in fit['checkpoint_health'].values():
        health=read(verify_file_pin(pin))
        if health['passed']!=health['total'] or not all(health['checks'].values()): raise ValueError('Healthy owned checkpoint boundaries required')
    ck=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    if ck['schema']!='backend_response_fit_checkpoint:1' or list(ck['arms'])!=MODELS or ck['update']!=2000:
        raise ValueError('Complete own final checkpoint required')
    vr=read(verify_file_pin(prep['preparations']['VALIDATE']));raw=read(verify_file_pin(vr['probe']))
    reviewed=read(verify_file_pin(prep['inputs']['data_review']));basis=np.load(verify_file_pin(vr['directions']))['directions']
    screens={(r['source'],r['direction']):r['blocks'] for r in reviewed['cohorts']['VALIDATE']['finite_amplitude_screens']}
    base_rows={}
    for i,m in enumerate(vr['metadata']):
        if m['profile']['direction']<0: base_rows.setdefault(m['source_name'],i)
    keys=[(source,j) for source in base_rows for j in range(20)]
    if out.exists() or not out.is_relative_to(root/'runs'): raise ValueError('Fresh physical evaluation output required')
    out.mkdir();paths=sorted((root/'src').rglob('*.py'))+[cp,root/'tools/evaluate_backend_pen_responses.py']
    receipt=freeze_sources(paths,out/'source_snapshot',root,subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip());write(out/'source_receipt.json',receipt)
    device=torch.device(spec['device']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scale=context['scales']['y_scale'];predictions={};derivatives={};forward={};checks={}
    initial=torch.load(verify_file_pin(fit['initial_NN']),map_location='cpu',weights_only=False)
    for name,state in ck['arms'].items():
        arm=restore_backend_arm(state,context,device);model=arm['model'];model.eval();forward[name]={}
        for role,rows in data.items():
            with torch.no_grad():prediction=model.prediction(torch.tensor(rows['x'],device=device,dtype=torch.float32)).cpu().numpy()
            mask=rows['valid']; forward[name][role]={b:float(np.mean(((prediction[mask,sl]-rows['y'][mask,sl])/scale[sl])**2)) for b,sl in BLOCKS.items()}
            if role=='VALIDATE':predictions[name]=prediction
        for source,j in keys:
            x=torch.tensor(data['VALIDATE']['x'][base_rows[source]:base_rows[source]+1],device=device,dtype=torch.float32)
            tangent=torch.zeros_like(x);tangent[:,81:101]=torch.tensor(basis[j],device=device,dtype=torch.float32)
            _,jvp=torch.autograd.functional.jvp(model.prediction,x,tangent)
            derivatives[name,source,j]=jvp[0].detach().cpu().numpy()
        model.net.load_state_dict(initial,strict=True)
        with torch.no_grad():prediction=model.prediction(torch.tensor(data['VALIDATE']['x'],device=device,dtype=torch.float32)).cpu().numpy()
        mask=data['VALIDATE']['valid'];forward[name]['INITIAL_VALIDATE']={b:float(np.mean(((prediction[mask,sl]-data['VALIDATE']['y'][mask,sl])/scale[sl])**2)) for b,sl in BLOCKS.items()}
        checks[name+'_final_cursor']=state['update']==2000
        del arm,model
    rows,aggregate=group_responses(vr,data['VALIDATE'],predictions,derivatives,base_rows,screens,scale)
    checks.update(all_declared_groups=len(rows)==len(vr['response_groups']),
        original_group_masks=[r['qualified'] for r in rows]==[g['qualified'] for g in vr['response_groups']],
        all_outputs_finite=all(np.isfinite(p).all() for p in predictions.values()),all_JVP_finite=all(np.isfinite(d).all() for d in derivatives.values()))
    np.savez_compressed(out/'VALIDATE_predictions.npz',**predictions)
    np.savez_compressed(out/'VALIDATE_JVP.npz',**{n:np.asarray([derivatives[n,s,j] for s,j in keys]) for n in MODELS})
    verify_source_snapshot(receipt,live=True)
    report=dict(schema='backend_response_physical_evaluation:1',config=file_pin(cp),fit=file_pin(fp),checkpoint=fit['checkpoint'],
        dataset=file_pin(pp),preparation=prep['preparations']['VALIDATE'],data_review=prep['inputs']['data_review'],
        source_receipt=file_pin(out/'source_receipt.json'),predictions=file_pin(out/'VALIDATE_predictions.npz'),
        JVP=file_pin(out/'VALIDATE_JVP.npz'),JVP_source_order=[dict(source_name=s,direction=j) for s,j in keys],
        declared_sources=len(raw['sources']),sources_with_saved_base=len(base_rows),declared_groups=len(rows),
        qualified_groups=sum(r['qualified'] for r in rows),response_pairs=rows,aggregate=aggregate,forward_normalized_MSE=forward,
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,Actor_updates=0,
        loader_normalizer_reconstructions=1,TRAIN_response_unit_computations=1,AD_direction_evaluations=len(derivatives),
        wall_s=time.perf_counter()-started,scope='All new-backend groups/masks retained;empty groups have null metrics. Same new TRAIN point units for all models;physical screens retain separately frozen descriptive units. Viewed controlled development validation,not independent reference/seed/ego/final-test/Actor/full-Jacobian admission.')
    write(out/'report.json',report)
    if not all(checks.values()): raise ValueError('Preserve physical evaluation discrepancy')
    return dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],
                qualified_groups=report['qualified_groups'],AD_direction_evaluations=len(derivatives))
