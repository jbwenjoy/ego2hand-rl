"""Independent saved-array arithmetic,cursors and failed-gradient qualification."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    ev=json.loads(a.evaluation.read_text());full=ev['schema']=='contact_pen_six_arm_physical_response_evaluation:2';diag=json.loads(a.diagnosis.read_text());fit=json.loads(verify_file_pin(ev['fit']).read_text());prep=json.loads(verify_file_pin(fit['prepared']).read_text());cr=json.loads(verify_file_pin(prep['VALIDATE_preparation' if full else 'CHECK_review']).read_text());out=a.output.resolve();role='VALIDATE' if full else 'CHECK'
    if out.exists() or not out.is_relative_to(ROOT/'runs') or diag['fit']!=ev['fit'] or ev['passed']!=ev['total'] or diag['passed']!=diag['total']:raise ValueError('Fresh complete matching evaluation/diagnosis review')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    check=dict(np.load(verify_file_pin(prep['outputs'][role])));pred=dict(np.load(verify_file_pin(ev['predictions'])));ad=dict(np.load(verify_file_pin(ev['JVP'])));train_pred=dict(np.load(verify_file_pin(diag['arrays'])));train_pairs=dict(np.load(verify_file_pin(prep['pairs'])))
    if full:
        independent=json.loads(verify_file_pin(prep['data_review']).read_text());screens={(r['source'],r['direction']):{k:v['physical_screen'] for k,v in r['blocks'].items()} for r in independent['cohorts']['VALIDATE']['finite_amplitude_screens']}
    checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);keys={(r['source_name'],r['direction']):i for i,r in enumerate(ev['JVP_source_order'])};checks={};outcomes={};blocks=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))
    for name,saved in checkpoint['arms'].items():
        scale=np.asarray(saved['scales']['y_scale']);steps=[float(v['step']) for v in saved['optimizer']['state'].values()];checks[name+'_Adam2000']=bool(steps) and all(s==2000 for s in steps);checks[name+'_cursor']=saved['update']==2000
        for key,sl in dict(hand40=slice(0,40),warm26=slice(52,78),**blocks).items():
            mask=check['valid'];value=float(np.mean(((pred[name][mask,sl]-check['y'][mask,sl])/scale[sl])**2));checks[name+'_forward_'+key]=bool(np.isclose(value,ev['forward_normalized_MSE'][name][role][key],atol=1e-12,rtol=1e-12))
        fd_targets=[];ad_rows=[];fd_rows=[]
        for i,row in enumerate(ev['response_pairs']):
            original=cr['response_groups'][i] if full else cr['response_pairs'][i];plus=[j for j in row['rows'] if cr['metadata'][j]['profile']['sign']==1];minus=[j for j in row['rows'] if cr['metadata'][j]['profile']['sign']==-1]
            fd_targets.append((check['y'][plus].mean(0)-check['y'][minus].mean(0))/(2*row['epsilon']));fd_rows.append((pred[name][plus].mean(0)-pred[name][minus].mean(0))/(2*row['epsilon']));ad_rows.append(ad[name][keys[(row['source_name'],row['direction'])]])
            checks[f'pair{i}_physical_mask']=row['physical_screens']==(screens[(row['source_name'],row['direction'])] if full else {k:original['blocks'][k]['two_amplitude_consistency_screen'] for k in blocks})
            if full:checks[f'pair{i}_qualification_unchanged']=row['qualified']==original['qualified']
        target=np.asarray(fd_targets);model_ad=np.asarray(ad_rows);model_fd=np.asarray(fd_rows);outcomes[name]={}
        for key,sl in blocks.items():
            for group in ['ALL_PHYSICAL_VALID','FROZEN_PHYSICAL_SCREEN']:
                ids=[i for i,r in enumerate(ev['response_pairs']) if r['qualified'] and (group=='ALL_PHYSICAL_VALID' or r['physical_screens'][key])];den=np.sum((target[ids,sl]/scale[sl])**2);stored=ev['aggregate'][name][key][group]
                for mode,values,field in [('AD',model_ad,'AD_relative_L2_error'),('FD',model_fd,'model_finite_response_relative_L2_error')]:
                    value=float(np.sqrt(np.sum(((values[ids,sl]-target[ids,sl])/scale[sl])**2)/den));checks[name+'_'+key+'_'+group+'_'+mode]=bool(np.isclose(value,stored[field],atol=1e-10,rtol=1e-10))
            target_train=train_pairs['finite_contrast'][:,sl]/scale[sl];den_train=np.sum(target_train**2)
            for mode,field in [('JVP','AD_relative_L2_error'),('finite','finite_response_relative_L2_error')]:
                value=float(np.sqrt(np.sum(((train_pred[name+'_'+mode][:,sl]-train_pairs['finite_contrast'][:,sl])/scale[sl])**2)/den_train));checks[name+'_'+key+'_TRAIN_'+mode]=bool(np.isclose(value,diag['TRAIN_response'][name][key][field],atol=1e-10,rtol=1e-10))
            s=ev['aggregate'][name][key]['FROZEN_PHYSICAL_SCREEN'];outcomes[name][key]=dict(screen_pairs=s['pairs'],candidate_AD_pairs=s['candidate_AD_pairs'],all_local_screens_pass=s['candidate_AD_pairs']==s['pairs'])
    checks.update(declared_live_updates=fit['live_models']*fit['live_updates_per_model']==(12000 if full else 8000),declared_clone_updates=fit['clone_only_optimizer_steps']==(24 if full else 16),total_optimizer_steps=fit['optimizer_steps']==(12024 if full else 8016),zero_task_native=fit['actual_native_integrations']==0)
    if full:
        for update,curve in fit['curve'].items():
            checks['curve_'+update+'_only_declared_roles']=all(set(rows)==({'TRAIN','VALIDATE'} if update=='2000' else {'TRAIN'}) for rows in curve.values())
        checks['TRAIN_only_health_probe']=fit['health_probe_role']=='TRAIN' and fit['health_probe_rows']==128
        subsets=dict(np.load(verify_file_pin(prep['pair_subsets'])))
        import hashlib
        for name,saved in checkpoint['arms'].items():
            rank=0 if name.startswith('POINT_') else 4 if name.startswith('RANK4_') else 20;ids=np.empty(0,dtype=np.int64) if rank==0 else subsets[f'rank{rank}'];contract=saved['pair_contract']
            checks[name+'_exact_pair_bank_identity']=contract==dict(rank=rank,rows=len(ids),index_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),point_bank=prep['outputs']['TRAIN'],pair_bank=prep['pairs'],subset_bank=prep['pair_subsets'])
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='contact_pen_fullrank_fit_saved_review:2' if full else 'contact_pen_fit_saved_arithmetic_review:1',evaluation=file_pin(a.evaluation),diagnosis=file_pin(a.diagnosis),fit=ev['fit'],source_receipt=file_pin(out/'source_receipt.json'),
        local_screen_outcomes=outcomes,actor_gradient_qualified={name:False for name in outcomes},physical_status='NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION',checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Saved-array arithmetic and optimizer cursors only,not new neural/physics execution. No model passes all original local contact/response screens;none establishes a full20D Jacobian or permits model-based Actor promotion. Original failed data and all four fixed-final outputs retained.')
    if full:result['scope']='Independent saved-array arithmetic,cursors,pair identities,final-only validation and unchanged physical-mask review. Six fixed-final outputs retained;partial object/wrench improvement does not establish full20D contact/Fn6/forward/Actor reliability.'
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],physical_status=result['physical_status'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain saved-arithmetic discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
