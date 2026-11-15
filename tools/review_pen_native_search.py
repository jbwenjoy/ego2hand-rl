"""Independent actual-control, score, budget and selected native replay audit."""
import argparse
import contextlib
import json
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());cfg=tomllib.loads(verify_file_pin(r['config']).read_text());base=tomllib.loads(verify_file_pin(r['base_config']).read_text())
    if r.get('planning_horizon_controls',1)!=1:raise ValueError('Scheduled forecasts require horizon-aware prefix/score review; this reader is H1 only')
    cp=verify_file_pin(r['config']);gates=tomllib.loads((cp.parent/cfg['gates']).read_text())
    proposal=json.loads(verify_file_pin(r['proposal']).read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    offsets=np.load(verify_file_pin(r['candidates']))['raw_offsets']
    expected={(mode,dt) for mode in ['STATE_ONLY','STATE_PLUS_SUPPORT'] for dt in [.005,.0025]}
    if {(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=4 or a.output.exists():raise ValueError('Exact predeclared4 cases and fresh review')
    for pin in r['sources']+r['task_sources']:verify_file_pin(pin)
    a.output.mkdir()
    try:
        with (a.output/'EFC_console.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):efc=review_saved_warp_contacts(a.report,a.output/'EFC',expected)
    except ValueError:
        if not (a.output/'EFC'/'report.json').exists():raise
        efc=json.loads((a.output/'EFC'/'report.json').read_text())
    results=[];all_decisions=[]
    for c in r['cases']:
        z=dict(np.load(verify_file_pin(c['trace'])));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']));table=np.load(verify_file_pin(c['fixed_model_arrays']))
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        body=c['model_layout']['object_body'];qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'];n=round(.02/c['dt'])
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);actual_w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench']
        native=mujoco.MjData(m);mujoco.mj_setState(m,native,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        mass=float(table['body_mass'][0,body]);gravity=table['opt.gravity'][0];count=len(ref['velocity_world'])-1
        nominal=np.zeros((368,6));nominal[:count,:3]=mass*(np.diff(ref['velocity_world'],axis=0)/.02-gravity)
        nominal[:count,3:]=np.diff(ref['angular_momentum_world'],axis=0)/.02
        global_checks=dict(actual_reference_COM_exact=np.array_equal(ref['COM_world_m'],z['COM_world_m']),
            actual_reference_quat_exact=np.array_equal(ref['quaternion_wxyz'],z['quaternion_wxyz']),nominal_momentum_target=bool(np.max(abs(nominal-ref['wrench_world']))<1e-12),
            nominal_tail_mask=np.array_equal(ref['valid_interval'],np.arange(368)<365),
            first_native_terminal=not np.any(z['terminated_truncated'][:-1]) and (bool(z['terminated_truncated'][-1].any()) or c['actual_intervals']==368),
            original_no_reset_no_aid=c['resets_after_start']==0 and not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            unchanged_model=bool(c['planning_model_unchanged']),all_actual_EFC=all(e['checks'].values()),
            candidate_budget=c['candidates_evaluated']==len(c['planning'])*57 and c['candidate_physics_ticks']==len(c['planning'])*57*n)
        decoded=[]
        for decision in c['planning']:
            k=decision['k'];lo,hi=k*n,(k+1)*n;d=dict(np.load(verify_file_pin(decision['prediction'])));source=np.load(verify_file_pin(decision['source']));index=decision['selected']
            before=d['controller__processed_actions'];raw=(2*selected['delta20']).astype(np.float32)[None]+offsets[k]
            ema=np.float32(.5)*np.clip(np.float32(.5)*raw,-.6,.6)+np.float32(.5)*before
            implied_bias=ref['reference_hand'][k+1]+ema-d['intended_PD']
            q=d['Q'][:,-1];quat=q[:,qa+3:qa+7].astype(float);qw,qx,qy,qz=quat.T
            shaft=np.c_[1-2*(qy*qy+qz*qz),2*(qx*qy+qw*qz),2*(qx*qz-qw*qy)]
            refshaft=Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]]).as_matrix()[:,0]
            angle=np.arccos(np.clip(shaft@refshaft,-1,1));position=((q[:,qa:qa+3]-ref['COM_world_m'][k+1])/.005)**2
            hand=((q[:,qids]-ref['reference_hand'][k+1])/.02)**2;residual=(ema/.02)**2
            state=position.sum(1)+(angle/np.deg2rad(5))**2+.01*hand.mean(1)+.01*residual.mean(1)
            mean=d['W'].mean(1);support=((mean[:,:3]-ref['wrench_world'][k,:3])/.1)**2
            torque=((mean[:,3:]-ref['wrench_world'][k,3:])/.005)**2;support=support.mean(1)+torque.mean(1)
            score=state+(0. if c['name']=='STATE_ONLY' else .1)*bool(ref['valid_interval'][k])*support
            score_error=float(np.max(abs(score-d['score_total'])))
            feasible=d['score_feasible'].astype(bool);violation=d['score_violation'];cost=d['score_total']
            if feasible.any():best=np.flatnonzero(feasible)[np.argmin(cost[feasible])]
            else:
                ids=np.flatnonzero(violation==violation.min());best=ids[np.argmin(cost[ids])]
            warm=native.qacc_warmstart if lo==0 else z['post_qacc'][lo-1]
            checks=dict(raw_table=np.array_equal(raw,d['raw_candidates']),original_EMA=bool(abs(ema-d['EMA_candidates']).max()<1e-7),
                encoder_bias_constant_within_candidates=bool(abs(implied_bias-implied_bias[:1]).max()<2e-7),
                selected_raw_exact=np.array_equal(d['raw_candidates'][index],z['actions20'][k]),
                selected_target_exact=bool(abs(d['PD_targets'][index]-z['post_ctrl'][lo:hi,aids]).max()<2e-7),
                source_state_exact=np.array_equal(source['qpos'][0],z['pre_qpos'][lo]) and np.array_equal(source['qvel'][0],z['pre_qvel'][lo]),
                source_warm_exact=np.array_equal(source['qacc_warmstart'][0],warm),
                source_clock=decision['phase']==k+1 and int(z['pre_action_phase'][k])==k+1,
                live_isolation=bool(decision['live_data_controller_untouched']),fixed57_candidates=d['raw_candidates'].shape==(57,20),
                formula_within_FP32_roundoff=bool(np.all(abs(score-d['score_total'])<=1e-3+1e-5*abs(score))),
                minimum_declared_score_or_violation=int(best)==index,
                no_fallback_hidden=bool(decision['any_feasible'])==bool(feasible.any()) and bool(decision['selected_feasible'])==bool(feasible[index]))
            pred=dict(post_qpos=d['Q'][index].astype(float),post_qvel=d['V'][index].astype(float),post_ctrl=d['U'][index].astype(float),post_GPU_time=d['T'][index].astype(float))
            actual={key:z[key][lo:hi] for key in pred}
            metrics,fidelity=compare_response(pred,actual,d['W'][index].astype(float),actual_w[lo:hi],qids,dofs,qa,da,gates)
            record=dict(mode=c['name'],dt=c['dt'],k=k,checks={key:bool(v) for key,v in checks.items()},score_max_abs_error=score_error,
                fidelity=fidelity,fidelity_metrics=metrics,fidelity_passed=all(fidelity.values()),
                any_feasible=bool(feasible.any()),selected=index,selected_violation=float(violation[index]),
                selected_predicted_geometry=d['pen'][index].tolist(),selected_actual_geometry=[float(z['pre_penetration_m'][lo:hi].max()),float(z['pre_self_penetration_m'][lo:hi].max()),float(z['post_joint_limit_excursion_rad'][lo:hi].max())])
            decoded.append(record);all_decisions.append(record)
        results.append(dict(mode=c['name'],dt=c['dt'],global_checks={key:bool(v) for key,v in global_checks.items()},decisions=decoded,
            actual_duration_s=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified'],task_gates=c['summary']['gates'],
            terminations=c['termination_flags'],fallbacks=sum(not d['any_feasible'] for d in decoded),
            candidate_physics_ticks=c['candidate_physics_ticks'],actual_physics_ticks=c['physics_steps'],
            candidate_cuda_ms=[d['candidate_cuda_ms'] for d in c['planning']],wall_with_artifact_s=[d['total_with_artifact_wall_s'] for d in c['planning']]))
    contract_passed=sum(sum(c['global_checks'].values()) for c in results)+sum(sum(d['checks'].values()) for d in all_decisions)
    contract_total=sum(len(c['global_checks']) for c in results)+sum(len(d['checks']) for d in all_decisions)
    common=[]
    for dt in [.005,.0025]:
        pair=[c for c in r['cases'] if c['dt']==dt];intervals=min(c['actual_intervals'] for c in pair)
        common.append(dict(dt=dt,common_intervals=intervals,common_physical_s=intervals*.02,candidates_per_mode_common_prefix=57*intervals,
            selected_indices_equal=sum(pair[0]['planning'][i]['selected']==pair[1]['planning'][i]['selected'] for i in range(intervals))))
    result=dict(schema='pen_native_search_saved_review:1',run=file_pin(a.report),source=file_pin(__file__),
        comparison_source=file_pin(ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py'),EFC=file_pin(a.output/'EFC'/'report.json'),EFC_counts=[efc['passed'],efc['total']],
        cases=results,common_prefix=common,contract_passed=contract_passed,contract_total=contract_total,
        selected_fidelity_passed=sum(d['fidelity_passed'] for d in all_decisions),selected_fidelity_total=len(all_decisions),
        nominal_model_support_benefit_established=False,neural_updates=0,native_integrations=0,
        scope='Independent saved contracts, mean-reference momentum targets, cost formula(debug FP32 roundoff only), exact argmin/fallback, original raw/EMA/selected PD and EVERY selected native20ms state/force comparison. Implied bias consistency is not a new observed hardware calibration. Actual task gates stay unrelaxed; all failures retained. Per-decision57 budget/common prefix, not equal total survival compute or final learned-policy A1/A2.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[contract_passed,contract_total],EFC=result['EFC_counts'],fidelity=[result['selected_fidelity_passed'],len(all_decisions)],
        cases=[dict(mode=c['mode'],dt=c['dt'],duration=c['actual_duration_s'],fallbacks=c['fallbacks'],terminations=c['terminations'],qualified=c['qualified']) for c in results])))
    if contract_passed!=contract_total or efc['passed']!=efc['total'] or result['selected_fidelity_passed']!=len(all_decisions):raise ValueError('Retain failing audit; no global contract/replay qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
