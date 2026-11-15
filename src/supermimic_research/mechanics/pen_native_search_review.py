"""Independent NumPy native-objective, source and actual replay review."""
import contextlib
import json
import subprocess
import tomllib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());cfg=tomllib.loads(verify_file_pin(r['config']).read_text());base=tomllib.loads(verify_file_pin(r['base_config']).read_text())
    cp=verify_file_pin(r['config']);gates=tomllib.loads((cp.parent/cfg['gates']).read_text())
    proposal=json.loads(verify_file_pin(r['proposal']).read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    offsets=np.load(verify_file_pin(r['candidates']))['raw_offsets']
    modes=({'STATE_SUPPORT':['STATE_ONLY','STATE_PLUS_SUPPORT'],
        'VELOCITY':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_VELOCITY','STATE_PLUS_VELOCITY_SUPPORT'],
        'FEEDBACK':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_FEEDBACK_SUPPORT']}[cfg.get('experiment','FEEDBACK')])
    expected={(mode,dt) for mode in modes for dt in [.005,.0025]}
    if {(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=len(expected) or a.output.exists():raise ValueError('Exact predeclared objective cases and fresh review')
    relocation_path=getattr(a,'source_relocations',None)
    relocations=json.loads(relocation_path.read_text()) if relocation_path else None
    if 'source_receipt' in r:
        verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()))
    for pin in r['sources']+r['task_sources']:verify_source_pin(pin,relocations)
    a.output.mkdir()
    review_sources=freeze_sources([Path(__file__),ROOT/'src/supermimic_research/frozen_sources.py',
        ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py',ROOT/'src/supermimic_research/mechanics/warp_saved_review.py',
        ROOT/'tools/review_pen_native.py'],a.output/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(a.output/'source_receipt.json',review_sources)
    review_source_map={q['observed']['path']:q['snapshot'] for q in review_sources['records']}
    efc_input=a.report
    if relocations is not None:
        # A new explicit review view; the original collected report is never rewritten.
        view={**r,'sources':[file_pin(verify_source_pin(p,relocations)) for p in r['sources']],
            'task_sources':[file_pin(verify_source_pin(p,relocations)) for p in r['task_sources']],
            'source_relocation_view':dict(original_report=file_pin(a.report),relocations=file_pin(relocation_path),
                scope='Only source paths relocated by exact original pin; all case data/physics unchanged.')}
        efc_input=a.output/'explicit_source_view.json';write(efc_input,view)
    try:
        with (a.output/'EFC_console.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):efc=review_saved_warp_contacts(efc_input,a.output/'EFC',expected)
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
            full_angle=(Rotation.from_quat(quat[:,[1,2,3,0]])*Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]]).inv()).magnitude()
            orientation_objective=cfg.get('orientation_objective','SHAFT')
            if orientation_objective not in ['SHAFT','FULL_SO3']:raise ValueError('Unknown saved orientation objective')
            orientation_angle=full_angle if orientation_objective=='FULL_SO3' else angle
            state=position.sum(1)+(orientation_angle/np.deg2rad(5))**2+.01*hand.mean(1)+.01*residual.mean(1)
            position_error=ref['COM_world_m'][k]-source['qpos'][0,qa:qa+3]
            valid=bool(ref['valid_interval'][k])
            velocity_error=ref['velocity_world'][k]-source['qvel'][0,da:da+3] if valid else np.full(3,np.nan)
            correction=mass*(100.*position_error+20.*velocity_error) if c['name']=='STATE_PLUS_FEEDBACK_SUPPORT' and valid else np.zeros(3)
            target=ref['wrench_world'][k].copy();target[:3]+=correction
            mean=d['W'].mean(1);support=((mean[:,:3]-target[:3])/.1)**2
            torque=((mean[:,3:]-target[3:])/.005)**2;support=support.mean(1)+torque.mean(1)
            support_weight=.1 if c['name'] in ['STATE_PLUS_SUPPORT','STATE_PLUS_FEEDBACK_SUPPORT','STATE_PLUS_VELOCITY_SUPPORT'] else 0.
            velocity_weight=1. if c['name'] in ['STATE_PLUS_VELOCITY','STATE_PLUS_VELOCITY_SUPPORT'] else 0.
            endpoint_valid=k+1<len(ref['velocity_world'])
            endpoint_velocity=ref['velocity_world'][k+1] if endpoint_valid else np.zeros(3)
            velocity_cost=(((d['V'][:,-1,da:da+3].astype(float)-endpoint_velocity)/.25)**2).sum(1) if velocity_weight and endpoint_valid else np.zeros(57)
            score=state+support_weight*bool(ref['valid_interval'][k])*support+velocity_weight*velocity_cost
            score_error=float(np.max(abs(score-d['score_total'])))
            feasible=d['score_feasible'].astype(bool);violation=d['score_violation'];cost=d['score_total']
            if feasible.any():best=np.flatnonzero(feasible)[np.argmin(cost[feasible])]
            else:
                ids=np.flatnonzero(violation==violation.min());best=ids[np.argmin(cost[ids])]
            warm=native.qacc_warmstart if lo==0 else z['post_qacc'][lo-1]
            checks=dict(feedback_target_formula=np.allclose(target,d['support_target_world'],atol=1e-6,rtol=1e-5),
                correction_formula=np.allclose(correction,d['support_feedback_force_world'],atol=1e-6,rtol=1e-5),
                source_error_formula=np.allclose(position_error,d['support_source_position_error_world_m'],atol=1e-7,rtol=1e-5) and np.allclose(velocity_error,d['support_source_velocity_error_world_m_s'],atol=1e-7,rtol=1e-5,equal_nan=True),
                raw_table=np.array_equal(raw,d['raw_candidates']),original_EMA=bool(abs(ema-d['EMA_candidates']).max()<1e-7),
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
            if 'endpoint_velocity_world' in d:
                checks.update(endpoint_velocity=np.allclose(endpoint_velocity,d['endpoint_velocity_world'],atol=1e-7,rtol=1e-5),
                    derivative_mask=bool(d['endpoint_velocity_valid'])==endpoint_valid and decision['endpoint_velocity_valid']==endpoint_valid,
                    predeclared_weights=float(d['velocity_weight'])==velocity_weight and float(d['support_weight'])==support_weight,
                    velocity_score=np.allclose(d['score_velocity'],velocity_cost,atol=1e-4,rtol=1e-5))
            if 'orientation_objective' in cfg:
                checks.update(orientation_reference=np.array_equal(ref['quaternion_wxyz'][k+1].astype(np.float32),d['orientation_reference_quaternion_wxyz']),
                    declared_orientation_objective=decision['orientation_objective']==c['orientation_objective']==orientation_objective)
                if orientation_objective=='FULL_SO3':
                    checks['full_SO3_angle']=np.allclose(full_angle,d['score_orientation_error_rad'],atol=2e-6,rtol=1e-6)
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
            selected_indices_all_equal=sum(len({c['planning'][i]['selected'] for c in pair})==1 for i in range(intervals))))
    verify_source_snapshot(review_sources,live=True)
    source_provenance_qualified=r.get('source_import_enumeration_available',True)
    result=dict(schema='pen_native_objective_saved_review:2',run=file_pin(a.report),source=review_source_map[str(Path(__file__).resolve())],source_receipt=file_pin(a.output/'source_receipt.json'),
        source_provenance_qualified=source_provenance_qualified,EFC_input_report=file_pin(efc_input),
        source_relocations=file_pin(relocation_path) if relocation_path else None,comparison_source=review_source_map[str((ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py').resolve())],EFC=file_pin(a.output/'EFC'/'report.json'),EFC_counts=[efc['passed'],efc['total']],
        cases=results,common_prefix=common,contract_passed=contract_passed,contract_total=contract_total,
        selected_fidelity_passed=sum(d['fidelity_passed'] for d in all_decisions),selected_fidelity_total=len(all_decisions),
        nominal_model_support_benefit_established=False,neural_updates=0,native_integrations=0,
        scope='Independent saved contracts, mean-reference momentum targets, source-state translational feedback target and endpoint WORLD velocity targets/masks/cost, cost formula(debug FP32 roundoff only), exact argmin/fallback, original raw/EMA/selected PD and EVERY selected native20ms state/force comparison. Implied bias consistency is not a new observed hardware calibration. Actual task gates stay unrelaxed; all failures retained. Per-decision57 budget/common prefix, not equal total survival compute or final learned-policy A1/A2.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[contract_passed,contract_total],EFC=result['EFC_counts'],fidelity=[result['selected_fidelity_passed'],len(all_decisions)],
        cases=[dict(mode=c['mode'],dt=c['dt'],duration=c['actual_duration_s'],fallbacks=c['fallbacks'],terminations=c['terminations'],qualified=c['qualified']) for c in results])))
    if not source_provenance_qualified or contract_passed!=contract_total or efc['passed']!=efc['total'] or result['selected_fidelity_passed']!=len(all_decisions):raise ValueError('Retain failing audit; no global provenance/contract/replay qualification')
