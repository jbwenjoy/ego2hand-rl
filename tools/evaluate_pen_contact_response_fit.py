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

BLOCKS=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))
MODELS=['POINT_DIRECT','POINT_IMPULSE','RESPONSE_DIRECT','RESPONSE_IMPULSE']


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());fp=(cp.parent/cfg['fit_report']).resolve();fit=json.loads(fp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or fit['schema']!='pen_response_loss_ablation_result:2' or fit['update']!=2000 or fit['training_running'] or fit['optimizer_steps']!=8016 or cfg['physical_screen']!='FROZEN_PREVIOUS_CHECK_PHYSICAL_SCREEN':raise ValueError('Fresh fixed-final registered study review')
    prep=json.loads(verify_file_pin(fit['prepared']).read_text());cr=json.loads(verify_file_pin(prep['CHECK_review']).read_text());probe=json.loads(verify_file_pin(cr['probe']).read_text())
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()};directions=np.load(verify_file_pin(probe['directions']))['directions'];checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);initial=torch.load(verify_file_pin(fit['initial_NN']),map_location='cpu',weights_only=False)
    if list(checkpoint['arms'])!=MODELS or cr['role']!='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT' or len(cr['response_pairs'])!=48 or not data['CHECK']['valid'].all():raise ValueError('Frozen four arms and complete NEVER_FIT physical CHECK')
    for pin in fit['checkpoint_health'].values():
        health=json.loads(verify_file_pin(pin).read_text())
        if health['passed']!=health['total']:raise ValueError('Healthy100/2000 exact reload required')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt);verify_source_snapshot(json.loads(verify_file_pin(fit['source_receipt']).read_text()),live=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
    metadata=cr['metadata'];base_rows={s['name']:next(i for i,r in enumerate(metadata) if r['source_name']==s['name'] and r['profile']['direction']<0) for s in probe['sources'] if s['available']}
    forward={};predictions={};derivatives={};checks={};scale=np.asarray(checkpoint['arms'][MODELS[0]]['scales']['y_scale']);jvp_keys=[(name,d) for name in base_rows for d in range(2)]
    for name,saved in checkpoint['arms'].items():
        checks[name+'_same_frozen_y_scale']=np.array_equal(saved['scales']['y_scale'],scale)
        checks[name+'_final_cursor']=saved['update']==2000
        model=model_from_arm(saved,device);model.eval();forward[name]={}
        for role,rows in data.items():
            x=torch.as_tensor(rows['x'],device=device,dtype=torch.float32)
            with torch.no_grad():prediction=model.prediction(x).cpu().numpy()
            mask=rows['valid'];forward[name][role]={key:float(np.mean(((prediction[mask,sl]-rows['y'][mask,sl])/scale[sl])**2)) for key,sl in dict(hand40=slice(0,40),warm26=slice(52,78),**BLOCKS).items()}
            if role=='CHECK':predictions[name]=prediction
        for source,direction in jvp_keys:
            x=torch.tensor(data['CHECK']['x'][base_rows[source]:base_rows[source]+1],device=device,dtype=torch.float32);tangent=torch.zeros_like(x);tangent[:,81:101]=torch.tensor(directions[direction],device=device,dtype=torch.float32)
            _,jvp=torch.autograd.functional.jvp(model.prediction,x,tangent);derivatives[(name,source,direction)]=jvp[0].detach().cpu().numpy()
        model.net.load_state_dict(initial,strict=True)
        with torch.no_grad():prediction=model.prediction(torch.tensor(data['CHECK']['x'],device=device,dtype=torch.float32)).cpu().numpy()
        forward[name]['INITIAL_CHECK']={key:float(np.mean(((prediction[:,sl]-data['CHECK']['y'][:,sl])/scale[sl])**2)) for key,sl in BLOCKS.items()}
    np.savez_compressed(out/'CHECK_predictions.npz',**predictions)
    np.savez_compressed(out/'CHECK_JVP.npz',**{name:np.array([derivatives[(name,*key)] for key in jvp_keys]) for name in MODELS})
    pairs=[];target_fd={};model_fd={}
    for i,original in enumerate(cr['response_pairs']):
        source=original['source_name'];direction=original['direction'];epsilon=original['epsilon'];ids=original['rows'];plus=[j for j in ids if metadata[j]['profile']['sign']==1];minus=[j for j in ids if metadata[j]['profile']['sign']==-1]
        checks[f'pair{i}_repeats']=len(plus)==len(minus)==3
        checks[f'pair{i}_source_warm_dt']=all(np.array_equal(data['CHECK']['x'][j,:81],data['CHECK']['x'][base_rows[source],:81]) and data['CHECK']['x'][j,-1]==data['CHECK']['x'][base_rows[source],-1] for j in ids)
        target=(data['CHECK']['y'][plus].mean(0)-data['CHECK']['y'][minus].mean(0))/(2*epsilon);target_fd[i]=target
        row=dict(source_name=source,direction=direction,epsilon=epsilon,qualified=original['qualified'],rows=ids,physical_screens={key:original['blocks'][key]['two_amplitude_consistency_screen'] for key in BLOCKS},models={})
        for name in MODELS:
            pred_fd=(predictions[name][plus].mean(0)-predictions[name][minus].mean(0))/(2*epsilon);model_fd[(name,i)]=pred_fd;row['models'][name]={}
            for key,sl in BLOCKS.items():
                ad,_=compare_native_response(data['CHECK']['y'][plus,sl],data['CHECK']['y'][minus,sl],derivatives[(name,source,direction)][sl],scale[sl],epsilon)
                fd=vector_comparison(pred_fd[sl]/scale[sl],target[sl]/scale[sl]);agreement=vector_comparison(derivatives[(name,source,direction)][sl]/scale[sl],pred_fd[sl]/scale[sl])
                row['models'][name][key]=dict(AD=ad,model_finite_response=fd,AD_vs_model_finite=agreement,candidate_AD_screen=bool(row['physical_screens'][key] and ad['relative_L2_error']<cfg['model_AD_relative_error'] and ad['cosine'] is not None and ad['cosine']>=cfg['model_AD_cosine']))
        pairs.append(row)
    aggregate={}
    for name in MODELS:
        aggregate[name]={}
        for key,sl in BLOCKS.items():
            aggregate[name][key]={}
            for screen in ['ALL_PHYSICAL_VALID','FROZEN_PHYSICAL_SCREEN']:
                ids=[i for i,r in enumerate(pairs) if r['qualified'] and (screen=='ALL_PHYSICAL_VALID' or r['physical_screens'][key])];den=sum(float(np.sum((target_fd[i][sl]/scale[sl])**2)) for i in ids)
                ad_num=sum(float(np.sum(((derivatives[(name,pairs[i]['source_name'],pairs[i]['direction'])][sl]-target_fd[i][sl])/scale[sl])**2)) for i in ids)
                fd_num=sum(float(np.sum(((model_fd[(name,i)][sl]-target_fd[i][sl])/scale[sl])**2)) for i in ids)
                aggregate[name][key][screen]=dict(pairs=len(ids),AD_relative_L2_error=float(np.sqrt(ad_num/max(den,1e-24))),model_finite_response_relative_L2_error=float(np.sqrt(fd_num/max(den,1e-24))),zero_response_relative_L2_error=1.,candidate_AD_pairs=sum(pairs[i]['models'][name][key]['candidate_AD_screen'] for i in ids))
    checks.update(all48_retained=len(pairs)==48,all_model_outputs_finite=all(np.isfinite(v).all() for v in predictions.values()),all_JVP_finite=all(np.isfinite(v).all() for v in derivatives.values()),all_physical_valid=all(r['qualified'] for r in pairs))
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(json.loads(verify_file_pin(fit['source_receipt']).read_text()),live=True)
    result=dict(schema='contact_pen_four_arm_physical_response_evaluation:1',config=file_pin(cp),protocol=file_pin(protocol),fit=file_pin(fp),checkpoint=fit['checkpoint'],common_data=fit['prepared'],CHECK_review=prep['CHECK_review'],
        source_receipt=file_pin(out/'source_receipt.json'),predictions=file_pin(out/'CHECK_predictions.npz'),JVP=file_pin(out/'CHECK_JVP.npz'),JVP_source_order=[dict(source_name=s,direction=d) for s,d in jvp_keys],
        forward_normalized_MSE=forward,response_pairs=pairs,aggregate=aggregate,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,Actor_updates=0,
        scope='Fixed final2000,one seed/reference,inspected developmental CHECK NEVER_FIT. Frozen original physical screens are independent of new model scores;new common scales come only from TRAIN. Neither passing arithmetic nor some candidate local directions qualify a full20D Jacobian,task,Actor,ego or amortized speedup.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],CHECK_forward={n:v['CHECK'] for n,v in forward.items()},aggregate=aggregate)),flush=True)
    if not all(checks.values()):raise ValueError('Retain final evaluation contract discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
