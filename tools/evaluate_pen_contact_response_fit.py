"""Fixed-final four-arm forward,finite-response and PD-action JVP diagnostics."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_response import model_from_arm
from supermimic_research.learning.pen_native_response import compare_native_response,vector_comparison
from supermimic_research.learning.pen_fullrank_study import MODELS as FULL_MODELS
from supermimic_research.learning.pen_response_units import MODELS as UNIT_MODELS
from supermimic_research.learning.pen_contact_supervision import MODELS as CONTACT_MODELS

BLOCKS=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))
MODELS=['POINT_DIRECT','POINT_IMPULSE','RESPONSE_DIRECT','RESPONSE_IMPULSE']


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());fp=(cp.parent/cfg['fit_report']).resolve();fit=json.loads(fp.read_text());out=a.output.resolve()
    contact=cfg.get('schema',1)==4;unit=cfg.get('schema',1) in [3,4];full=cfg.get('schema',1) in [2,3,4];names=CONTACT_MODELS if contact else UNIT_MODELS if unit else FULL_MODELS if full else MODELS;role='VALIDATE' if full else 'CHECK'
    if out.exists() or not out.is_relative_to(ROOT/'runs') or fit['schema']!=('pen_response_loss_ablation_result:5' if contact else 'pen_response_loss_ablation_result:4' if unit else 'pen_response_loss_ablation_result:3' if full else 'pen_response_loss_ablation_result:2') or fit['update']!=2000 or fit['training_running'] or fit['optimizer_steps']!=(16032 if unit else 12024 if full else 8016) or cfg['physical_screen']!=('REGISTERED_RESERVED_VALIDATION_PHYSICAL_SCREEN' if full else 'FROZEN_PREVIOUS_CHECK_PHYSICAL_SCREEN'):raise ValueError('Fresh fixed-final registered study review')
    prep=json.loads(verify_file_pin(fit['prepared']).read_text());cr=json.loads(verify_file_pin(prep['VALIDATE_preparation' if full else 'CHECK_review']).read_text());probe=json.loads(verify_file_pin(cr['probe']).read_text())
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()};directions=np.load(verify_file_pin(probe['directions']))['directions'];checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);initial=torch.load(verify_file_pin(fit['initial_NN']),map_location='cpu',weights_only=False)
    if full:
        reviewed=json.loads(verify_file_pin(prep['data_review']).read_text())
        if reviewed['passed']!=reviewed['total'] or reviewed['inputs']['VALIDATE']!=prep['VALIDATE_preparation'] or cr['role']!='VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT' or fit['validation_evaluation_updates']!=[2000] or fit['health_probe_role']!='TRAIN':raise ValueError('Exact reserved validation and final-only evaluation contract')
        screen={(r['source'],r['direction']):r['blocks'] for r in reviewed['cohorts']['VALIDATE']['finite_amplitude_screens']}
        original_pairs=[dict(source_name=g['source_name'],direction=g['direction'],epsilon=g['epsilon'],rows=g['rows']['1']+g['rows']['-1'],qualified=g['qualified'],blocks={b:dict(two_amplitude_consistency_screen=screen[(g['source_name'],g['direction'])][b]['physical_screen']) for b in BLOCKS}) for g in cr['response_groups']]
    else:original_pairs=cr['response_pairs']
    if list(checkpoint['arms'])!=names or len(original_pairs)!=(480 if full else 48) or not full and (cr['role']!='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT' or not data['CHECK']['valid'].all()):raise ValueError('Frozen arms and original NEVER_FIT physical masks required')
    for pin in fit['checkpoint_health'].values():
        health=json.loads(verify_file_pin(pin).read_text())
        if health['passed']!=health['total']:raise ValueError('Healthy100/2000 exact reload required')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt);verify_source_snapshot(json.loads(verify_file_pin(fit['source_receipt']).read_text()),live=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
    metadata=cr['metadata'];base_rows={s['name']:next(i for i,r in enumerate(metadata) if r['source_name']==s['name'] and r['profile']['direction']<0) for s in probe['sources'] if s['available']}
    forward={};predictions={};derivatives={};checks={};scale=np.asarray(checkpoint['arms'][names[0]]['scales']['y_scale']);jvp_keys=[(name,d) for name in base_rows for d in range(20 if full else 2)]
    for name,saved in checkpoint['arms'].items():
        checks[name+'_same_frozen_y_scale']=np.array_equal(saved['scales']['y_scale'],scale)
        checks[name+'_final_cursor']=saved['update']==2000
        model=model_from_arm(saved,device);model.eval();forward[name]={}
        for data_role,rows in data.items():
            x=torch.as_tensor(rows['x'],device=device,dtype=torch.float32)
            with torch.no_grad():prediction=model.prediction(x).cpu().numpy()
            mask=rows['valid'];forward[name][data_role]={key:float(np.mean(((prediction[mask,sl]-rows['y'][mask,sl])/scale[sl])**2)) for key,sl in dict(hand40=slice(0,40),warm26=slice(52,78),**BLOCKS).items()}
            if data_role==role:predictions[name]=prediction
        for source,direction in jvp_keys:
            x=torch.tensor(data[role]['x'][base_rows[source]:base_rows[source]+1],device=device,dtype=torch.float32);tangent=torch.zeros_like(x);tangent[:,81:101]=torch.tensor(directions[direction],device=device,dtype=torch.float32)
            _,jvp=torch.autograd.functional.jvp(model.prediction,x,tangent);derivatives[(name,source,direction)]=jvp[0].detach().cpu().numpy()
        model.net.load_state_dict(initial,strict=True)
        with torch.no_grad():prediction=model.prediction(torch.tensor(data[role]['x'],device=device,dtype=torch.float32)).cpu().numpy()
        mask=data[role]['valid'];forward[name]['INITIAL_'+role]={key:float(np.mean(((prediction[mask,sl]-data[role]['y'][mask,sl])/scale[sl])**2)) for key,sl in BLOCKS.items()}
    np.savez_compressed(out/(role+'_predictions.npz'),**predictions)
    np.savez_compressed(out/(role+'_JVP.npz'),**{name:np.array([derivatives[(name,*key)] for key in jvp_keys]) for name in names})
    pairs=[];target_fd={};model_fd={}
    for i,original in enumerate(original_pairs):
        source=original['source_name'];direction=original['direction'];epsilon=original['epsilon'];ids=original['rows'];plus=[j for j in ids if metadata[j]['profile']['sign']==1];minus=[j for j in ids if metadata[j]['profile']['sign']==-1]
        checks[f'pair{i}_repeats']=len(plus)==len(minus)==(2 if full else 3)
        checks[f'pair{i}_source_warm_dt']=all(np.array_equal(data[role]['x'][j,:81],data[role]['x'][base_rows[source],:81]) and data[role]['x'][j,-1]==data[role]['x'][base_rows[source],-1] for j in ids)
        target=(data[role]['y'][plus].mean(0)-data[role]['y'][minus].mean(0))/(2*epsilon);target_fd[i]=target
        row=dict(source_name=source,direction=direction,epsilon=epsilon,qualified=original['qualified'],rows=ids,physical_screens={key:original['blocks'][key]['two_amplitude_consistency_screen'] for key in BLOCKS},models={})
        for name in names:
            pred_fd=(predictions[name][plus].mean(0)-predictions[name][minus].mean(0))/(2*epsilon);model_fd[(name,i)]=pred_fd;row['models'][name]={}
            for key,sl in BLOCKS.items():
                ad,_=compare_native_response(data[role]['y'][plus,sl],data[role]['y'][minus,sl],derivatives[(name,source,direction)][sl],scale[sl],epsilon)
                fd=vector_comparison(pred_fd[sl]/scale[sl],target[sl]/scale[sl]);agreement=vector_comparison(derivatives[(name,source,direction)][sl]/scale[sl],pred_fd[sl]/scale[sl])
                row['models'][name][key]=dict(AD=ad,model_finite_response=fd,AD_vs_model_finite=agreement,candidate_AD_screen=bool(row['physical_screens'][key] and ad['relative_L2_error']<cfg['model_AD_relative_error'] and ad['cosine'] is not None and ad['cosine']>=cfg['model_AD_cosine']))
        pairs.append(row)
    aggregate={}
    for name in names:
        aggregate[name]={}
        for key,sl in BLOCKS.items():
            aggregate[name][key]={}
            for screen in ['ALL_PHYSICAL_VALID','FROZEN_PHYSICAL_SCREEN']:
                ids=[i for i,r in enumerate(pairs) if r['qualified'] and (screen=='ALL_PHYSICAL_VALID' or r['physical_screens'][key])];den=sum(float(np.sum((target_fd[i][sl]/scale[sl])**2)) for i in ids)
                ad_num=sum(float(np.sum(((derivatives[(name,pairs[i]['source_name'],pairs[i]['direction'])][sl]-target_fd[i][sl])/scale[sl])**2)) for i in ids)
                fd_num=sum(float(np.sum(((model_fd[(name,i)][sl]-target_fd[i][sl])/scale[sl])**2)) for i in ids)
                aggregate[name][key][screen]=dict(pairs=len(ids),AD_relative_L2_error=float(np.sqrt(ad_num/max(den,1e-24))),model_finite_response_relative_L2_error=float(np.sqrt(fd_num/max(den,1e-24))),zero_response_relative_L2_error=1.,candidate_AD_pairs=sum(pairs[i]['models'][name][key]['candidate_AD_screen'] for i in ids))
    checks.update(all_pairs_retained=len(pairs)==(480 if full else 48),all_model_outputs_finite=all(np.isfinite(v).all() for v in predictions.values()),all_JVP_finite=all(np.isfinite(v).all() for v in derivatives.values()),physical_mask_unchanged=(sum(r['qualified'] for r in pairs)==475 and int(data[role]['valid'].sum())==1935) if full else all(r['qualified'] for r in pairs))
    if not full:checks={dict(all_pairs_retained='all48_retained',physical_mask_unchanged='all_physical_valid').get(k,k):v for k,v in checks.items()}
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(json.loads(verify_file_pin(fit['source_receipt']).read_text()),live=True)
    result=dict(schema='contact_pen_Fn_response_evaluation:4' if contact else 'contact_pen_response_units_evaluation:3' if unit else 'contact_pen_six_arm_physical_response_evaluation:2' if full else 'contact_pen_four_arm_physical_response_evaluation:1',config=file_pin(cp),protocol=file_pin(protocol),fit=file_pin(fp),checkpoint=fit['checkpoint'],common_data=fit['prepared'],
        source_receipt=file_pin(out/'source_receipt.json'),predictions=file_pin(out/(role+'_predictions.npz')),JVP=file_pin(out/(role+'_JVP.npz')),JVP_source_order=[dict(source_name=s,direction=d) for s,d in jvp_keys],
        forward_normalized_MSE=forward,response_pairs=pairs,aggregate=aggregate,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,Actor_updates=0,
        scope='Fixed final2000,one seed/reference,inspected developmental CHECK NEVER_FIT. Frozen original physical screens are independent of new model scores;new common scales come only from TRAIN. Neither passing arithmetic nor some candidate local directions qualify a full20D Jacobian,task,Actor,ego or amortized speedup.')
    result['VALIDATE_preparation' if full else 'CHECK_review']=prep['VALIDATE_preparation' if full else 'CHECK_review']
    if full:result.update(data_review=prep['data_review'],validation_role=cr['role'],scope='Fixed final2000 six-arm evaluation on reserved development validation after fitting. All480 groups retained,475 physical-valid;frozen independent amplitude/repeat masks unchanged. Same reference/frozen actor seed,not final-test/ego/Actor/full-Jacobian qualification.')
    if unit:result.update(response_scale_preparation=fit['response_scale_preparation'],scope='Fixed final2000 eight-arm response-unit/weight comparison on already-inspected NEVER_FIT development validation. Evaluation uses unchanged POINT y_scale,not derivative training units. All masks retained;not final-test/ego/Actor or full contact-Jacobian qualification.')
    if contact:result['scope']='Fixed final2000 explicit Fn6 response addition against unchanged two-block controls. Inspected NEVER_FIT development data,unchanged POINT evaluation units and physical masks;not Actor/task/ego/full contact-Jacobian qualification.'
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],forward={n:v[role] for n,v in forward.items()},AD_on_frozen_screen={n:{b:z['FROZEN_PHYSICAL_SCREEN'] for b,z in v.items()} for n,v in aggregate.items()})),flush=True)
    if not all(checks.values()):raise ValueError('Retain final evaluation contract discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
