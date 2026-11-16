"""Independent saved-arithmetic/ownership review;no simulator or optimizer step."""
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_path_fit import fixed_spec,data_view,registered_context
from supermimic_research.learning.pen_path_response import ARMS,WEIGHTS,PathResponseWM
from supermimic_research.learning.pen_path_reward import SCORES
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_contrast_update import finite_contrast_loss
from supermimic_research.learning.backend_reward_gradient import batch_context
from supermimic_research.learning.checkpoint import _adam_valid,_finite_tree
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.mechanics.pen_reward_chain import observable_proxy
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def close(a,b):return a is None and b is None if a is None or b is None else bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))


def native_screens(values,prep):
    paired={}
    for g in prep['response_groups']:
        p,m=g['rows']['1'],g['rows']['-1']
        if len(p)!=2 or len(m)!=2:continue
        plus,minus=values[p],values[m];fd=(plus.mean(0)-minus.mean(0))/(8*g['epsilon'])
        signal=abs(plus.mean(0)-minus.mean(0));spread=abs(plus-plus.mean(0)).max(0)+abs(minus-minus.mean(0)).max(0)
        paired[g['source_name'],g['direction'],g['epsilon']]=(g['qualified'],fd,signal>spread)
    result={}
    for g in prep['response_groups']:
        key=(g['source_name'],g['direction']);a=paired.get((*key,.002));b=paired.get((*key,.005));flags=[]
        for k in range(2):
            ok=False
            if a is not None and b is not None:
                x,y=a[1][k],b[1][k];ok=bool(a[0] and b[0] and y!=0 and abs(x-y)/abs(y)<=.25 and x*y>0 and a[2][k] and b[2][k])
            flags.append(ok)
        result[key]=flags
    return result


def main():
    torch.set_num_threads(4);fp=ROOT/'runs/pen_path_fit_v1/report.json';ep=ROOT/'runs/pen_path_reward_evaluation_v1/report.json';fit=read(fp);evaluation=read(ep)
    out=ROOT/'runs/pen_path_study_review_v1';archive=ROOT/'provenance/pen_path_study_results_v1'
    if out.exists() or archive.exists() or evaluation['passed']!=evaluation['total'] or not all(evaluation['checks'].values()):raise ValueError('Fresh completed study review required')
    cp=verify_file_pin(fit['config']);cfg=fixed_spec(tomllib.loads(cp.read_text()));paths={k:(cp.parent/cfg[k]).resolve() for k in ['common_data','prior_fit','interval_targets','interval_ledger','objective_config','adapter_config','evaluation_config','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']};common,old_fit,old,recovery,data,prep,path_data,products=data_view(paths)
    reg,context=registered_context(cfg,{**paths,'config':cp},inputs,common,old_fit,old,products);units=context['path_units']
    checks=dict(exact_terminal_work=fit['update']==2000 and not fit['training_running'] and fit['optimizer_steps']==6012 and fit['work']==dict(live_attempted_calls=6000,live_completed_steps=6000,QA_attempted_calls=12,QA_completed_steps=12),
        evaluation29=evaluation['passed']==evaluation['total']==29 and evaluation['fit']==file_pin(fp) and evaluation['checkpoint']==fit['checkpoint'],
        no_native_or_Actor=fit['actual_native_integrations']==fit['Actor_updates']==evaluation['actual_native_integrations']==evaluation['optimizer_steps']==0,
        frozen_unit_counts=reg['TRAIN_path_unit_fits']==2 and reg['production_path_unit_fits']==reg['invalid_TRAIN_isolation_fits']==1 and fit['TRAIN_path_unit_fits']==evaluation['TRAIN_path_unit_fits']==0)
    # A separate TRAIN arithmetic reconstruction is charged as one additional
    # unit computation. No scale is changed or loaded back into a fitted model.
    labels=path_data['TRAIN']['geometry_loss'].astype(np.float64);mask=data['TRAIN']['valid'];contrasts=[];indices=[]
    for gi,g in enumerate(prep['TRAIN']['response_groups']):
        if g['qualified']:
            contrasts.append((labels[g['rows']['1']].mean()-labels[g['rows']['-1']].mean())/(2*g['epsilon']));indices.append(gi)
    ps=max(float(labels[mask].std()),1e-6);rms=float(np.sqrt(np.mean(np.square(contrasts))))
    expected=dict(output_scale=float(labels[mask].mean()/np.log(2)),point_scale=ps,response_RMS=rms,response_floor=.001*ps,response_scale=max(rms,.001*ps))
    checks['TRAIN_only_units_reconstruction']=all(units[k]==v for k,v in expected.items()) and units['TRAIN_group_indices'].tolist()==indices==products['pairs']['TRAIN_group_index'].tolist()
    checkpoint_audits={}
    for u,pin in fit['checkpoint_health'].items():
        h=read(verify_file_pin(pin));saved=torch.load(verify_file_pin(h['checkpoint']),map_location='cpu',weights_only=False);local={}
        if h['passed']!=h['total'] or not all(h['checks'].values()):raise ValueError('GPU owned restoration failed')
        for name,s in saved['arms'].items():
            model=PathResponseWM(context['scales'],context['physics'],units,width=cfg['width'])
            buffers=all(raw_tree_equal(v,s['model'][k]) for k,v in model.named_buffers());model.load_state_dict(s['model'])
            opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(copy.deepcopy(s['optimizer']))
            local[name]=dict(buffers=buffers,Adam_cursor=_adam_valid(opt,model,int(u),torch),finite=_finite_tree([s['model'],s['optimizer'],s['gradient'],saved['logs'][name]],torch),
                context=raw_tree_equal(s['context'],context),weights=s['weights']==WEIGHTS[name],cursor=s['update']==int(u))
            if name=='RESPONSE_ONLY':local[name]['zero_path_parameters_and_moments']=all(not bool(p.any()) and not bool(opt.state[p]['exp_avg'].any()) and not bool(opt.state[p]['exp_avg_sq'].any()) for p in model.path_head.parameters())
        checkpoint_audits[u]=local;checks['checkpoint_'+u+'_CPU_readonly']=saved['update']==int(u) and saved['planned_updates']==2000 and set(local)==set(ARMS) and all(all(v.values()) for v in local.values())
    checks['both_boundaries']=set(checkpoint_audits)=={'100','2000'}
    # Compare the maintained DIRECT objective and gradients at IDENTICAL input
    # weights. New matched baselines are retrained,not old final-state replay.
    initial=torch.load(verify_file_pin(reg['initial_base_NN']),map_location='cpu',weights_only=False)
    terminal=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    final={k[4:]:v for k,v in terminal['arms']['RESPONSE_ONLY']['model'].items() if k.startswith('net.')};baseline_parity={}
    tx=torch.tensor(data['TRAIN']['x'][data['TRAIN']['valid']][:256],dtype=torch.float32);ty=torch.tensor(data['TRAIN']['y'][data['TRAIN']['valid']][:256],dtype=torch.float64)
    pairs=products['pairs'];px=torch.tensor(np.r_[pairs['x_plus'][:32],pairs['x_minus'][:32]],dtype=torch.float32);eps=torch.tensor(pairs['epsilon'][:32],dtype=torch.float64)
    fd=torch.tensor(pairs['finite_contrast'][:32],dtype=torch.float64);rs=torch.tensor(old['response_scale'],dtype=torch.float64)
    for label,weights in [('INITIAL',initial),('FINAL_NEW_BASELINE',final)]:
        reference=PenResponseWM(old['scales'],dict(head_mean=old['scales']['y_mean'],head_scale=old['scales']['y_scale']),old['physics'],width=cfg['width'],coupled=False)
        model=PathResponseWM(context['scales'],context['physics'],units,width=cfg['width']);reference.net.load_state_dict(weights);model.net.load_state_dict(weights)
        rp,_=reference.supervised_loss(tx,ty,ty,auxiliary_weight=.1);np_loss,_=model.point_terms(tx,ty,torch.tensor(labels[mask][:256]),.1)
        ra,rb=reference.prediction(px).chunk(2);na,nb=model.prediction(px).chunk(2)
        rl=rp+finite_contrast_loss(ra,rb,fd,eps,rs);nl=np_loss+finite_contrast_loss(na,nb,fd,eps,rs);rl.backward();nl.backward()
        rg=np.concatenate([p.grad.numpy().astype(np.float64).ravel() for p in reference.net.parameters()]);ng=np.concatenate([p.grad.numpy().astype(np.float64).ravel() for p in model.net.parameters()])
        relative=float(np.linalg.norm(rg-ng)/max(np.linalg.norm(rg),1e-30));difference=abs(float(rl.detach())-float(nl.detach()));baseline_parity[label]=dict(loss_difference=difference,gradient_relative_L2=relative,gradient_max_absolute=float(abs(rg-ng).max()))
        checks['maintained_DIRECT_math_'+label]=difference<=1e-12 and relative<=1e-5
    native=read(verify_file_pin(evaluation['inputs']['native_reward']));contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False)
    reward=tomllib.loads(paths['adapter_config'].read_text())['reward'];objective=tomllib.loads(paths['objective_config'].read_text());arrays=dict(np.load(verify_file_pin(evaluation['arrays'])))
    reviews={};arithmetic=0;score_error={}
    for role,p in prep.items():
        truth=np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values'];targets={'OBSERVABLE_NATIVE':np.column_stack([truth[:,4],truth[:,4]+.2*truth[:,6]]),'FULL_NATIVE':truth[:,[7,8]]}
        screens={key:native_screens(v,p) for key,v in targets.items()};dirs=np.load(verify_file_pin(p['directions']))['directions'];metadata=p['metadata'];role_reviews={}
        for name in ARMS:
            report=evaluation['cohorts'][role]['models'][name];prefix=role+'_'+name+'_';v=arrays[prefix+'values'];prediction=arrays[prefix+'prediction90'];path=arrays[prefix+'path_loss'];base_rows=report['base_row_indices'];base_names=report['base_source_order']
            raw=arrays[role+'_RAW20'];ctx=batch_context(contexts,[m['source_name'] for m in metadata],'cpu')
            proxy=observable_proxy(torch.tensor(data[role]['x'],dtype=torch.float32),torch.from_numpy(prediction),torch.tensor(raw,dtype=torch.float32),ctx,reward,objective)
            rebuilt=np.column_stack([proxy['observable_residual'].numpy(),proxy['observable_support'].numpy()])-objective['geometry_soft_weight']*path[:,None]
            diff=float(abs(v-rebuilt).max());score_error[role+'_'+name]=diff
            checks[role+'_'+name+'_scores']=diff<=5e-6 and np.allclose(arrays[prefix+'base_values'],v[base_rows],rtol=0,atol=5e-6)
            ch=arrays[prefix+'PD_partial']*arrays[prefix+'controller_diagonal'][:,None,:]+arrays[prefix+'RAW_direct_partial']
            checks[role+'_'+name+'_chain']=np.array_equal(ch,arrays[prefix+'RAW']) and np.all(arrays[prefix+'controller_diagonal']==.25) and np.array_equal(arrays[prefix+'computed_PD'],data[role]['x'][base_rows,81:101].astype(np.float32))
            expected_direct=-2*reward['action_weight']*raw[base_rows].astype(np.float64)/20
            checks[role+'_'+name+'_direct_RAW_cost']=bool(abs(arrays[prefix+'RAW_direct_partial']-expected_direct[:,None,:]).max()<=1e-9)
            error=((prediction-data[role]['y'])/np.asarray(old['scales']['y_scale'],np.float32))**2;valid=data[role]['valid']
            forward={block:float(error[valid,sl].mean()) for block,sl in dict(all90=slice(0,90),object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90)).items()}
            forward['path_normalized_MSE']=float(np.mean(((path[valid]-path_data[role]['geometry_loss'][valid])/units['point_scale'])**2))
            checks[role+'_'+name+'_forward']=all(close(value,report['forward'][k]) for k,value in forward.items())
            by_source={s:i for i,s in enumerate(base_names)};base_map={s:i for s,i in zip(base_names,base_rows)};computed={};group_flags=[]
            for gi,g in enumerate(p['response_groups']):
                stored=report['groups'][gi];source,j,e=g['source_name'],g['direction'],g['epsilon'];plus,minus=g['rows']['1'],g['rows']['-1'];done=len(plus)==len(minus)==2
                qualified=bool(done and all(metadata[i]['valid'] for i in plus+minus));base=base_map.get(source);base_valid=base is not None and bool(metadata[base]['valid'])
                ok=stored['group_index']==gi and stored['source_name']==source and stored['direction']==j and stored['epsilon']==e and stored['complete']==done==g['complete'] and stored['qualified']==qualified==g['qualified'] and stored['base_row']==base and stored['base_physical_valid']==base_valid
                if done:
                    model_fd=(v[plus].mean(0)-v[minus].mean(0))/(8*e);ad=None if source not in by_source else arrays[prefix+'RAW'][by_source[source]].astype(np.float64)@dirs[j]
                    native_fd={t:(values[plus].mean(0)-values[minus].mean(0))/(8*e) for t,values in targets.items()}
                    computed[gi]=dict(model_fd=model_fd,ad=ad,targets=native_fd,base_valid=base_valid,source=source,direction=j)
                    for target,t in native_fd.items():
                        for k in range(2):
                            q=stored['metrics'][target][k];rel=None if ad is None or t[k]==0 else abs(ad[k]-t[k])/abs(t[k]);cos=None if ad is None or ad[k]*t[k]==0 else float(np.sign(ad[k]*t[k]))
                            candidate=bool(qualified and base_valid and screens[target][source,j][k] and rel is not None and rel<1 and cos is not None and cos>=.9)
                            ok=ok and close(q['native_FD_per_RAW'],float(t[k])) and close(q['model_FD_per_RAW'],float(model_fd[k])) and close(q['model_AD_per_RAW'],None if ad is None else float(ad[k])) and q['physical_scalar_screen']==screens[target][source,j][k] and q['candidate_local_AD']==candidate;arithmetic+=5
                else:ok=ok and stored['metrics'] is None
                group_flags.append(bool(ok))
            aggregates={};aggregate_flags=[]
            for target in targets:
                aggregates[target]={}
                for k,score in enumerate(SCORES):
                    aggregates[target][score]={}
                    for subset in ['ALL_PHYSICAL_VALID','FROZEN_SCALAR_SCREEN']:
                        ids=[i for i,g in enumerate(p['response_groups']) if g['qualified'] and (subset=='ALL_PHYSICAL_VALID' or screens[target][g['source_name'],g['direction']][k])];ai=[i for i in ids if computed[i]['ad'] is not None]
                        den=sum(float(computed[i]['targets'][target][k]**2) for i in ids);ad_den=sum(float(computed[i]['targets'][target][k]**2) for i in ai)
                        ae=sum(float((computed[i]['ad'][k]-computed[i]['targets'][target][k])**2) for i in ai);fe=sum(float((computed[i]['model_fd'][k]-computed[i]['targets'][target][k])**2) for i in ids)
                        md=sum(float(computed[i]['model_fd'][k]**2) for i in ai);me=sum(float((computed[i]['ad'][k]-computed[i]['model_fd'][k])**2) for i in ai)
                        result=dict(pairs=len(ids),AD_pairs=len(ai),missing_AD_pairs=len(ids)-len(ai),invalid_native_base_pairs=sum(not computed[i]['base_valid'] for i in ai),AD_relative_L2=float(np.sqrt(ae/ad_den)) if ad_den else None,
                            model_FD_relative_L2=float(np.sqrt(fe/den)) if den else None,AD_vs_model_FD_relative_L2=float(np.sqrt(me/md)) if md else None,
                            same_sign_AD_pairs=sum(bool(computed[i]['ad'][k]*computed[i]['targets'][target][k]>0) for i in ai),candidate_local_AD_pairs=sum(report['groups'][i]['metrics'][target][k]['candidate_local_AD'] for i in ids),
                            normalized_native_squared_norm=den,AD_subset_native_squared_norm=ad_den)
                        expected=report['aggregate'][target][score][subset];aggregate_flags.append(all(close(value,expected[key]) for key,value in result.items()));aggregates[target][score][subset]=result;arithmetic+=len(result)
            checks[role+'_'+name+'_all_groups']=len(group_flags)==len(p['response_groups']) and all(group_flags)
            checks[role+'_'+name+'_all_aggregates']=len(aggregate_flags)==8 and all(aggregate_flags)
            role_reviews[name]=dict(forward=forward,aggregate=aggregates,group_flags=group_flags,aggregate_flags=aggregate_flags)
        reviews[role]=role_reviews
    tests=ROOT/'runs/pen_path_response_tests536_v1.log';checks['full536_tests']='Ran 536 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[fit['source_receipt'],evaluation['source_receipt'],reg['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Path study discrepancy retained: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-path-study-results-2026-11-16.md','research-plan-v1.40.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],checkpoint_health={'100':[46,46],'2000':[46,46]},evaluation_checks=[29,29],arithmetic_checks=arithmetic,
        optimizer_steps=6012,live_optimizer_steps=6000,clone_QA_optimizer_steps=12,actual_native_integrations=0,added_review_optimizer_steps=0,
        production_path_unit_fits=1,isolation_path_unit_fits=1,independent_path_unit_recomputations=1,total_TRAIN_path_unit_computations=3,point_normalizer_fits=0,
        fit_and_QA_wall_s=fit['fit_and_QA_wall_s'],total_runner_wall_s=fit['total_runner_wall_s'],evaluation_wall_s=evaluation['wall_s'],CPU_saved_score_max_errors=score_error,maintained_DIRECT_math=baseline_parity,
        scope='Matched three-arm soft interval supervision;all original groups/masks and invalid centers retained. CPU checkpoint view only,not CUDA RNG migration/resume. Hard risk unmodeled,no Actor/task/ego/overall acceleration claim.')
    write(out/'report.json',dict(schema='pen_path_study_independent_review:1',fit=file_pin(fp),evaluation=file_pin(ep),checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,
        checkpoint_CPU_readonly=checkpoint_audits,cohorts=reviews,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_interval_targets_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    roots=[fp,ep,paths['registration'],out/'report.json',out/'source_receipt.json',tests,prior_path]+docs
    todo=[file_pin(p) for p in roots]+receipts;allowed={fp.parent,ep.parent,paths['registration'].parent,out}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='pen_path_study_results_ledger:1',prior_evidence=file_pin(prior_path),fit=file_pin(fp),evaluation=file_pin(ep),registration=file_pin(paths['registration']),review=file_pin(out/'report.json'),summary=file_pin(archive/'summary.json'),
        documents=[file_pin(p) for p in docs],source_receipts=receipts+[file_pin(out/'source_receipt.json')],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),optimizer_steps=6012,actual_native_integrations=0,total_TRAIN_path_unit_computations=3))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),arithmetic_checks=arithmetic,score_errors=score_error)),flush=True)


if __name__=='__main__':main()
