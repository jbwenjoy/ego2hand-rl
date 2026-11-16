"""Original-gate physics/full-reward audit and fixed positive-direction utility."""
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.probe_row_review import review_recorded_row
from supermimic_research.mechanics.pen_reward_chain import native_components,controller
from supermimic_research.learning.pen_path_response import ARMS
from supermimic_research.learning.pen_native_suggestions import GOALS,EPSILONS,summarize_improvement
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();rp=ROOT/'runs/pen_native_suggestions_v2/report.json';r=read(rp);out=ROOT/'runs/pen_native_suggestions_review_v1'
    if out.exists() or r['schema']!='pen_model_suggestion_native_probes:1' or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Fresh completed native probe required')
    bank=read(verify_file_pin(r['bank']));original_reg=read(verify_file_pin(r['registration']));launch=read(verify_file_pin(r['launch_registration']));failure=read(verify_file_pin(r['prior_failure']))
    if launch['inputs']['bank']!=r['bank'] or launch['inputs']['prior_failure']!=r['prior_failure'] or bank['registration']!=r['registration'] or r['default_changed'] or r['source_epoch_used']:raise ValueError('Exact unchanged bank/guard launch required')
    for obj in [r,bank,original_reg,launch,failure]:verify_source_snapshot(read(verify_file_pin(obj['source_receipt'])),live=True)
    contract=load_contract(verify_file_pin(original_reg['inputs']['backend_manifest']));objective=tomllib.loads(verify_file_pin(original_reg['inputs']['objective_config']).read_text())
    evaluation=read(verify_file_pin(bank['inputs']['evaluation']));native=read(verify_file_pin(evaluation['inputs']['native_reward']));reward=tomllib.loads(verify_file_pin(native['inputs']['adapter_config']).read_text())['reward']
    contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False);original_gradients=dict(np.load(verify_file_pin(evaluation['arrays'])))
    models={};physical={};states={};row_audits=[];by_name={};packets=[];bank_checks=[];groups=[];case_map={c['name']:c for c in r['cases']}
    case_indices={c['name']:i for i,c in enumerate(r['cases'])}
    if len(case_map)!=1800:raise ValueError('All original declared case names required')
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    stage='admission'
    try:
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved native probes only')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No regenerated geometry')):
            for source in bank['sources']:
                sp=verify_file_pin(source['original_source']['source'])
                if str(sp) not in states:
                    states[str(sp)]=torch.load(sp,map_location='cpu',weights_only=False);require_backend_probe(states[str(sp)],contract)
                state=states[str(sp)]
                for key,d in source['model_directions'].items():
                    g=np.asarray(d['RAW_gradient'],np.float64);norm=float(np.linalg.norm(g));unit=None if norm==0 else g/norm
                    model_name,goal=key.rsplit('_',1);original_model=evaluation['cohorts'][source['role']]['models'][model_name];bi=original_model['base_source_order'].index(source['name'])
                    original=original_gradients[source['role']+'_'+model_name+'_RAW'][bi,GOALS.index(goal)].astype(np.float64)
                    bank_checks.append(np.array_equal(g,original) and d['RAW_norm']==norm and ((unit is None and d['unit'] is None) or np.array_equal(unit,d['unit'])))
                for p in source['profiles']:
                    name=source['name']+'_'+p['name'];stage=name;c=case_map[name]
                    identity=c['source_name']==source['name'] and c['available']==p['available'] and c['suggestion']==p and c['cohort']==source['role'] and c['role']==source['native_role']
                    if not p['available']:
                        bank_checks.append(identity and not c['qualified'] and c['reason']==p['reason']);continue
                    if p['model'] is None:expected=np.asarray(source['base_raw'],np.float32)
                    else:expected=(np.asarray(source['base_raw'])+4*p['sign']*p['epsilon']*np.asarray(source['model_directions'][p['model']+'_'+p['goal']]['unit'])).astype(np.float32)
                    PD,_,_=controller(torch.tensor([p['raw20']],dtype=torch.float32),contexts[source['name']])
                    identity=identity and np.array_equal(expected,np.asarray(p['raw20'],np.float32)) and np.array_equal(PD[0].numpy(),np.asarray(p['expected_PD20'],np.float32))
                    z=dict(np.load(verify_file_pin(c['trace'])));w=dict(np.load(verify_file_pin(c['window'])));identity=identity and np.array_equal(z['actions20'][0],expected);bank_checks.append(identity)
                    mp=verify_file_pin(c['model'])
                    if str(mp) not in models:models[str(mp)]=mujoco.MjModel.from_binary_path(str(mp));physical[str(mp)]=dict(np.load(verify_file_pin(c['physical_context'])))
                    packet,audit=review_recorded_row(c,z,w,state,models[str(mp)],physical[str(mp)],dict(role=source['native_role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001))
                    values,formula,errors=native_components(c,z,w,contexts[source['name']],reward,objective)
                    score_checks=dict(reconstructed_scores=np.array_equal(values[7:9],np.asarray(c['full_scores'])),executed_source_reward=abs(float(values[-1])-c['source_reward'])<=2e-6)
                    valid=bool(audit['valid'] and all(formula.values()) and all(score_checks.values()) and identity)
                    item=dict(name=name,source_name=source['name'],role=source['role'],profile=p,valid=valid,physics=audit,reward_checks=formula,reward_errors=errors,score_checks=score_checks,native_components=values.tolist())
                    row_audits.append(item);by_name[name]=item;packets.append(dict(**packet,native_components=values,valid=valid,case_index=case_indices[name]))
                    if len(row_audits)%50==0:write_json(out/'progress.json',dict(rows=len(row_audits),total=r['executed_controls'],valid_rows=sum(x['valid'] for x in row_audits),stage=stage,wall_s=time.perf_counter()-started))
        for source in bank['sources']:
            base_names=[source['name']+f'_BASE_R{k}' for k in range(2)]
            for model in ARMS:
                for k,goal in enumerate(GOALS):
                    for epsilon in EPSILONS:
                        selected={sign:[source['name']+'_'+p['name'] for p in source['profiles'] if p['model']==model and p['goal']==goal and p['epsilon']==epsilon and p['sign']==sign] for sign in [1,-1]}
                        names=base_names+selected[1]+selected[-1];complete=all(n in by_name for n in names)
                        g=dict(source_name=source['name'],role=source['role'],model=model,goal=goal,epsilon=epsilon,complete=complete,original_base_physical_valid=source['original_base_physical_valid'],cases=names,result=None)
                        if complete:
                            values=lambda ns:[by_name[n]['native_components'][7+k] for n in ns]
                            physical_ok=all(by_name[n]['valid'] for n in names);d=source['model_directions'][model+'_'+goal]
                            result=summarize_improvement(values(base_names),values(selected[1]),values(selected[-1]),epsilon=epsilon,gradient_norm=d['RAW_norm'],physical_valid=physical_ok,original_base_valid=source['original_base_physical_valid'])
                            # Direct arithmetic cross-check of the helper's unit/
                            # sign/envelope convention,without using its fields.
                            b,p,m=[np.asarray(values(ns)) for ns in [base_names,selected[1],selected[-1]]]
                            if result['positive_gain']!=float(p.mean()-b.mean()) or result['native_FD_per_RAW']!=float((p.mean()-m.mean())/(8*epsilon)):raise ValueError('Native gain/RAW denominator differs')
                            predicted=[case_map[n]['suggestion']['predicted_goal_gain'] for n in selected[1]]
                            component_means=[np.asarray([by_name[n]['native_components'] for n in ns]).mean(0) for ns in [base_names,selected[1],selected[-1]]]
                            g.update(new_all_physics_valid=physical_ok,result=result,predicted_positive_goal_gain=float(np.mean(predicted)),
                                positive_component_gain=(component_means[1]-component_means[0]).tolist(),negative_component_gain=(component_means[2]-component_means[0]).tolist())
                        groups.append(g)
        lookup={(g['source_name'],g['model'],g['goal'],g['epsilon']):g for g in groups}
        for g in groups:
            a=lookup[g['source_name'],g['model'],g['goal'],.002];b=lookup[g['source_name'],g['model'],g['goal'],.005];screen=False
            if a['result'] is not None and b['result'] is not None:
                x,y=a['result'],b['result'];xf,yf=x['native_FD_per_RAW'],y['native_FD_per_RAW']
                screen=bool(x['eligible'] and y['eligible'] and yf!=0 and abs(xf-yf)/abs(yf)<=.25 and xf*yf>0
                    and abs(x['positive_mean']-x['negative_mean'])>x['separation_repeat_envelope'] and abs(y['positive_mean']-y['negative_mean'])>y['separation_repeat_envelope'])
            g['original_two_amplitude_repeat_screen']=screen
        summaries=[]
        for role in ['TRAIN','VALIDATE']:
            for model in ARMS:
                for goal in GOALS:
                    for epsilon in EPSILONS:
                        selected=[g for g in groups if (g['role'],g['model'],g['goal'],g['epsilon'])==(role,model,goal,epsilon)];complete=[g for g in selected if g['complete']];eligible=[g for g in complete if g['result']['eligible']]
                        results=[g['result'] for g in eligible];gain=np.asarray([x['positive_gain'] for x in results]);native_fd=np.asarray([x['native_FD_per_RAW'] for x in results]);ad=np.asarray([x['predicted_AD_per_RAW'] for x in results])
                        den=float(np.sum(native_fd**2));relative=float(np.linalg.norm(ad-native_fd)/np.sqrt(den)) if den else None
                        summaries.append(dict(role=role,model=model,goal=goal,epsilon=epsilon,declared=len(selected),complete=len(complete),missing=len(selected)-len(complete),eligible=len(eligible),
                            new_all_physics_invalid=sum(not g['new_all_physics_valid'] for g in complete),original_center_invalid=sum(not g['original_base_physical_valid'] for g in complete),
                            positive_gain_above_repeat_envelope=sum(x['improvement_exceeds_observed_repeat_envelope'] for x in results),negative_gain_above_repeat_envelope=sum(x['descent_exceeds_observed_repeat_envelope'] for x in results),
                            ascent_beats_reverse_above_repeat_envelope=sum(x['ascent_beats_reverse_exceeds_observed_repeat_envelope'] for x in results),
                            positive_gain_count=int((gain>0).sum()),mean_positive_gain=float(gain.mean()) if len(gain) else None,median_positive_gain=float(np.median(gain)) if len(gain) else None,
                            min_positive_gain=float(gain.min()) if len(gain) else None,max_positive_gain=float(gain.max()) if len(gain) else None,AD_relative_L2=relative,
                            two_amplitude_screen_groups=sum(g['original_two_amplitude_repeat_screen'] for g in selected),
                            mean_positive_component_gains=None if not eligible else dict(zip(native['channels'],np.mean([g['positive_component_gain'] for g in eligible],axis=0).tolist())),
                            scope='Source counts/repeat envelopes,not IID trials,confidence bounds,best-sign selection or multi-step policy success.'))
        packet_keys=['x','y','heads','native_components','valid','case_index'];np.savez_compressed(out/'executed_packets.npz',**{key:np.asarray([p[key] for p in packets]) for key in packet_keys})
        physics_flags=[sum(sum(x['physics']['checks'].values()) for x in row_audits),sum(len(x['physics']['checks']) for x in row_audits)]
        reward_flags=[sum(sum(x['reward_checks'].values()) for x in row_audits),sum(len(x['reward_checks']) for x in row_audits)]
        checks=dict(all1800_cases=len(case_map)==1800,all1500_executed=len(row_audits)==r['executed_controls']==1500,all_original_bank_bytes=all(bank_checks),
            all432_groups=len(groups)==432,all360_complete=sum(g['complete'] for g in groups)==360,
            all24_summaries=len(summaries)==24,exact12000_native=r['actual_native_integrations']==r['attempted_native_integrations']==12000,
            cohort12008=r['total_cohort_native']==r['actual_native_integrations']+failure['reconstructed_actual_native_integrations']==12008,
            zero_optimizer=r['optimizer_steps']==r['attempted_optimizer_steps']==0,
            all_score_reconstructions=all(all(x['score_checks'].values()) for x in row_audits),all_original_reward_formulas=all(all(x['reward_checks'].values()) for x in row_audits),
            invalid_original_center_never_admitted=all(not g['result']['eligible'] for g in groups if g['complete'] and not g['original_base_physical_valid']))
        verify_source_snapshot(receipt,live=True)
        result=dict(schema='pen_native_suggestions_physical_reward_review:1',native=file_pin(rp),bank=r['bank'],prior_failure=r['prior_failure'],source_receipt=file_pin(out/'source_receipt.json'),
            rows=row_audits,groups=groups,summary=summaries,packets=file_pin(out/'executed_packets.npz'),checks=checks,passed=sum(checks.values()),total=len(checks),
            valid_rows=sum(x['valid'] for x in row_audits),total_rows=len(row_audits),invalid_rows=[x['name'] for x in row_audits if not x['valid']],physics_flags=physics_flags,reward_flags=reward_flags,
            new_actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
            scope='All new native packets independently decoded under original physical gates;failures/missing/original invalid centers retained. All model/goals/epsilon outcomes reported. New safe one-step benefit does not qualify multi-step Actor/task/ego or overall acceleration.')
        write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],valid=[result['valid_rows'],result['total_rows']],physics_flags=physics_flags,reward_flags=reward_flags,summary=summaries)),flush=True)
        if not all(checks.values()):raise ValueError('Native suggestion review discrepancy retained')
    except Exception as e:
        write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,reviewed_rows=len(row_audits),new_actual_native_integrations=0,optimizer_steps=0));raise


if __name__=='__main__':main()
