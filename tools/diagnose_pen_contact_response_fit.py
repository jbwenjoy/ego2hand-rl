"""Post-hoc TRAIN fitting,excitation rank and objective-scale diagnostics."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_response import model_from_arm


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    fit=json.loads(a.fit.read_text());full=fit['schema']=='pen_response_loss_ablation_result:3';prep=json.loads(verify_file_pin(fit['prepared']).read_text());tr=json.loads(verify_file_pin(prep['TRAIN_preparation']).read_text());cr=json.loads(verify_file_pin(prep['VALIDATE_preparation' if full else 'CHECK_review']).read_text());probe=json.loads(verify_file_pin(cr['probe']).read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or fit['update']!=2000 or fit['schema'] not in ['pen_response_loss_ablation_result:2','pen_response_loss_ablation_result:3']:raise ValueError('Fresh fixed-final post-hoc diagnosis')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);pairs=dict(np.load(verify_file_pin(prep['pairs'])));data=dict(np.load(verify_file_pin(prep['outputs']['TRAIN'])));directions=np.load(verify_file_pin(prep['directions']))['directions'];check_directions=np.load(verify_file_pin(probe['directions']))['directions']
    groups=[g for g in tr['response_groups'] if g['qualified']];metadata=tr['metadata'];base={g['source_name']:next(i for i,r in enumerate(metadata) if r['source_name']==g['source_name'] and r['profile']['direction']<0) for g in groups}
    basis=np.linalg.qr(directions.T,mode='reduced')[0];fractions=np.sum((check_directions@basis)**2,axis=1)/np.sum(check_directions**2,axis=1)
    deltas=np.array([data['x'][r['row_index'],81:101]-data['x'][base[r['source_name']],81:101] for r in metadata if r['valid']]);off=deltas-(deltas@basis)@basis.T
    rank=dict(declared_directions=int(len(directions)),physical_PD_dimension=20,declared_rank=int(np.linalg.matrix_rank(directions)),singular_values=np.linalg.svd(directions,compute_uv=False).tolist(),CHECK_direction_projection_fraction=fractions.tolist(),max_observed_PD_delta_off_span_L2_rad=float(np.linalg.norm(off,axis=1).max()),
        caveat='Local intervention design,not the global action matrix rank or proof of one failure cause. Cross-state action variation may add constraints under structural assumptions. Tiny float32 deviations are not deliberate independent excitation.')
    if full:
        rank['VALIDATE_direction_projection_fraction']=rank.pop('CHECK_direction_projection_fraction')
        rank['VALIDATE_direction_rank4_projection_fraction']=np.sum((check_directions@directions[:4].T)**2,axis=1).tolist()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device='cuda:0';arrays={};aggregate={};losses={};checks={};blocks=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))
    for name,saved in checkpoint['arms'].items():
        model=model_from_arm(saved,device);model.eval();scale=np.asarray(saved['scales']['y_scale']);ad={}
        with torch.no_grad():p=model.prediction(torch.tensor(np.concatenate([pairs['x_plus'],pairs['x_minus']]),device=device,dtype=torch.float32)).cpu().numpy()
        plus,minus=np.split(p,2);fd=(plus-minus)/(2*pairs['epsilon'][:,None])
        for group in groups:
            key=(group['source_name'],group['direction'])
            if key in ad:continue
            x=torch.tensor(data['x'][base[key[0]]:base[key[0]]+1],device=device,dtype=torch.float32);t=torch.zeros_like(x);t[:,81:101]=torch.tensor(directions[key[1]],device=device,dtype=torch.float32)
            _,jvp=torch.autograd.functional.jvp(model.prediction,x,t);ad[key]=jvp[0].detach().cpu().numpy()
        ad_rows=np.array([ad[(g['source_name'],g['direction'])] for g in groups]);arrays[name+'_finite']=fd;arrays[name+'_JVP']=ad_rows;aggregate[name]={}
        for block,sl in blocks.items():
            target=pairs['finite_contrast'][:,sl]/scale[sl];den=float(np.sum(target**2));aggregate[name][block]=dict(pairs=len(groups),AD_relative_L2_error=float(np.sqrt(np.sum(((ad_rows[:,sl]-pairs['finite_contrast'][:,sl])/scale[sl])**2)/max(den,1e-24))),finite_response_relative_L2_error=float(np.sqrt(np.sum(((fd[:,sl]-pairs['finite_contrast'][:,sl])/scale[sl])**2)/max(den,1e-24))),zero_response_relative_error=1.)
        log=np.asarray(checkpoint['logs'][name]);weight=float(saved['contrast_weight']);mean=log[-100:].mean(0);point=mean[1]+saved['config']['auxiliary_weight']*mean[2];weighted=weight*mean[3]
        checks[name+'_loss_accounting']=bool(np.allclose(log[:,0],log[:,1]+saved['config']['auxiliary_weight']*log[:,2]+weight*log[:,3],atol=1e-10,rtol=1e-10))
        losses[name]=dict(first=log[0].tolist(),last100_mean=mean.tolist(),weighted_response_fraction=float(weighted/max(mean[0],1e-24)),point_with_auxiliary=float(point),weighted_response=float(weighted))
    np.savez_compressed(out/'TRAIN_response_predictions.npz',**arrays)
    checks.update(declared_pairs=len(groups)==(949 if full else 188),masked_rows_unchanged=int(data['valid'].sum())==(3866 if full else 808),finite_arrays=all(np.isfinite(v).all() for v in arrays.values()))
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(json.loads(verify_file_pin(fit['source_receipt']).read_text()),live=True)
    result=dict(schema='contact_pen_fullrank_fit_posthoc_diagnosis:2' if full else 'contact_pen_fit_posthoc_diagnosis:1',fit=file_pin(a.fit),common_data=fit['prepared'],source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'TRAIN_response_predictions.npz'),pair_order=groups,
        TRAIN_response=aggregate,local_excitation=rank,loss_diagnostics=losses,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Post-hoc descriptive TRAIN fit/excitation/objective-scale diagnostics;no weight/checkpoint selection,new optimizer steps or causal proof. Reusing inspected CHECK directions for span reporting does not make them a final test. Original physical gates remain unchanged.')
    if full:result['scope']='Post-hoc TRAIN response-fit/rank/loss-value diagnostics after fixed-final six-arm evaluation. Loss fractions are not gradient fractions. Reserved validation is now inspected developmental NEVER_FIT;not final-test or Actor qualification. No NN/native steps or weight selection.'
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],TRAIN_response=aggregate,rank=rank,weighted_response_fraction={k:v['weighted_response_fraction'] for k,v in losses.items()})),flush=True)
    if not all(checks.values()):raise ValueError('Retain diagnosis accounting discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
