"""Independent NumPy scheduled-control/score review and actual EFC/prefix checks."""
import argparse
import contextlib
import json
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.learning.pen_stage import write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def main(a):
    r=json.loads(a.report.read_text());cp=verify_file_pin(r['config']);cfg=tomllib.loads(cp.read_text())
    base=tomllib.loads(verify_file_pin(r['base_config']).read_text());gates=tomllib.loads((cp.parent/cfg['gates']).read_text())
    if not r['engine_warmstart_recorded'] or a.output.exists():raise ValueError('Explicit actual engine memory capture and fresh output required')
    Hmax=r['planning_horizon_controls'];offsets=load(r['candidates'])['raw_offsets']
    proposal=json.loads(verify_file_pin(r['proposal']).read_text());selected=load(proposal['artifacts']['selected'])
    for pin in r['sources']+r['task_sources']:verify_file_pin(pin)
    expected={(mode,dt) for mode in ['STATE_ONLY','STATE_PLUS_SUPPORT'] for dt in [.005,.0025]}
    a.output.mkdir();receipt=freeze_sources([Path(__file__).resolve(),ROOT/'src/supermimic_research/mechanics/warp_saved_review.py',
        ROOT/'src/supermimic_research/mechanics/warp_packet_decode.py',ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py'],
        a.output/'source_snapshot',ROOT,'independent-horizon-review')
    write(a.output/'source_receipt.json',receipt)
    try:
        with (a.output/'EFC_console.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):
            efc=review_saved_warp_contacts(a.report,a.output/'EFC',expected)
    except ValueError:
        if not (a.output/'EFC/report.json').exists():raise
        efc=json.loads((a.output/'EFC/report.json').read_text())
    rows=[];cases=[]
    for c in r['cases']:
        z=load(c['trace']);ref=load(c['initialization']['nominal_mean_wrench']);n=round(.02/c['dt'])
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr']
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);W=load(e['wrenches'])['world_COM_wrench']
        native=mujoco.MjData(m);mujoco.mj_setState(m,native,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        first=len(rows);cost=0
        for decision in c['planning']:
            k,index=decision['k'],decision['selected'];H=min(Hmax,368-k);lo,hi=k*n,(k+1)*n
            d=load(decision['prediction']);s=load(decision['source']);memory=d['controller__processed_actions'];PD=[];EMA=[];raws=[];state=[];support=[]
            for j in range(H):
                raw=(2*selected['delta20']).astype(np.float32)[None]+offsets[k+j]
                memory=np.float32(.5)*np.clip(np.float32(.5)*raw,-.6,.6)+np.float32(.5)*memory
                target=np.clip(ref['reference_hand'][k+j+1]+memory-d['encoder_bias20'],d['PD_lower20'],d['PD_upper20'])
                PD.append(target);EMA.append(memory);raws.append(raw)
                q=d['Q'][:,(j+1)*n-1].astype(float);quat=q[:,qa+3:qa+7]
                good=np.isfinite(quat).all(1)&(np.linalg.norm(quat,axis=1)>0);angle=np.full(57,np.nan)
                Rtarget=Rotation.from_quat(ref['quaternion_wxyz'][k+j+1][[1,2,3,0]])
                angle[good]=(Rotation.from_quat(quat[good][:,[1,2,3,0]])*Rtarget.inv()).magnitude()
                state.append(np.sum(((q[:,qa:qa+3]-ref['COM_world_m'][k+j+1])/cfg['COM_scale_m'])**2,1)+
                    (angle/np.deg2rad(cfg['shaft_scale_deg']))**2+cfg['hand_weight']*np.mean(((q[:,qids]-ref['reference_hand'][k+j+1])/cfg['hand_scale_rad'])**2,1)+
                    cfg['residual_weight']*np.mean((memory/cfg['residual_scale_rad'])**2,1))
                mean=d['W'][:,j*n:(j+1)*n].astype(float).mean(1);desired=ref['wrench_world'][k+j]
                support.append(np.mean(((mean[:,:3]-desired[:3])/cfg['force_scale_N'])**2,1)+np.mean(((mean[:,3:]-desired[3:])/cfg['torque_scale_Nm'])**2,1))
            PD,EMA,raws=[np.stack(v,1) for v in [PD,EMA,raws]]
            state,support=np.array(state),np.array(support);weight=cfg['support_weight'] if c['name']=='STATE_PLUS_SUPPORT' else 0.
            total=np.mean(state+weight*ref['valid_interval'][k:k+H,None]*support,0)
            finite=np.isfinite(total)&np.isfinite(d['pen']).all(1)&np.isfinite(d['other']).all(1)
            for key in ['Q','V','U','W']:finite &= np.isfinite(d[key]).all(axis=(1,2))
            capacity=~d['bad'].astype(bool).any(1);bounds=np.array([base[key] for key in ['max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad']])
            geometric=np.sum(np.maximum(0,d['pen']-bounds)/bounds,1)
            nonhand=np.maximum(0,d['other'].max(1)-base['max_object_nonhand_normal_force_N'])/base['max_object_nonhand_normal_force_N']
            feasible=finite&capacity&(geometric==0)&(nonhand==0)
            violation=np.nan_to_num(geometric+nonhand+(~capacity)*1e6+(~finite)*1e12,nan=1e12,posinf=1e12)
            clean=np.nan_to_num(d['score_total'],nan=1e30,posinf=1e30,neginf=1e30)
            ids=np.flatnonzero(feasible) if feasible.any() else np.flatnonzero(violation==violation.min());best=ids[np.argmin(clean[ids])]
            warm=native.qacc_warmstart if lo==0 else z['post_qacc_warmstart'][lo-1]
            score_finite=np.isfinite(total)&np.isfinite(d['score_total'])
            score_ok=(np.array_equal(np.isfinite(total),np.isfinite(d['score_total'])) and
                      np.all(abs(total[score_finite]-d['score_total'][score_finite])<=1e-3+1e-5*abs(total[score_finite])))
            checks=dict(horizon_clock=int(d['prediction_controls'])==H and decision['phase']==k+1 and int(z['pre_action_phase'][k])==k+1,
                prediction_shape=d['Q'].shape[1]==H*n,raw_first=np.array_equal(raws[:,0],d['raw_candidates']),
                future_raw=np.array_equal(raws,d['raw_sequence']) if H>1 else True,
                EMA=bool(np.max(abs(EMA-(d['EMA_schedule'] if H>1 else d['EMA_candidates'][:,None])))<1e-7),
                PD=bool(np.max(abs(PD-(d['PD_schedule'] if H>1 else d['PD_targets'][:,None])))<2e-7),
                graph_PD_exact=np.array_equal(d['U'][:,:,aids],np.repeat(PD,n,axis=1)),
                applied_first_raw=np.array_equal(z['actions20'][k],raws[index,0]),
                applied_first_PD=np.array_equal(z['post_ctrl'][lo:hi][:,aids],np.broadcast_to(PD[index,0],(n,20))),
                source_qv=np.array_equal(s['qpos'][0],z['pre_qpos'][lo]) and np.array_equal(s['qvel'][0],z['pre_qvel'][lo]),
                source_actual_warm=np.array_equal(s['qacc_warmstart'][0],warm),
                source_time=float(s['time'][0])==(float(native.time) if lo==0 else float(z['post_GPU_time'][lo-1])),
                live_isolation=decision['live_data_controller_untouched'],independent_mean_cost=bool(score_ok),
                full_forecast_safety=np.array_equal(feasible,d['score_feasible']),
                selected_minimum=int(best)==index,budget=decision['native_candidate_ticks']==57*H*n)
            pred={key:d[name][index,:n] for key,name in [('post_qpos','Q'),('post_qvel','V'),('post_ctrl','U'),('post_GPU_time','T')]}
            metrics,fidelity=compare_response(pred,{key:z[key][lo:hi] for key in pred},d['W'][index,:n],W[lo:hi],qids,dofs,qa,da,gates)
            score_error=float(np.max(abs(total[score_finite]-d['score_total'][score_finite]))) if score_finite.any() else None
            rows.append(dict(mode=c['name'],dt=c['dt'],k=k,H=H,checks={k:bool(v) for k,v in checks.items()},
                fidelity=fidelity,fidelity_metrics=metrics,score_max_abs_error=score_error))
            cost+=57*H*n
        global_checks=dict(no_aid_reset=c['resets_after_start']==0 and not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            actual_native_budget=len(z['post_qpos'])==c['physics_steps']==c['actual_intervals']*n,
            forecast_budget=cost==c['candidate_physics_ticks'],fixed_model=c['planning_model_unchanged'],
            first_terminal=not np.any(z['terminated_truncated'][:-1]) and (bool(z['terminated_truncated'][-1].any()) or c['actual_intervals']==368),
            reference_COM=np.array_equal(ref['COM_world_m'],z['COM_world_m']),reference_quat=np.array_equal(ref['quaternion_wxyz'],z['quaternion_wxyz']))
        cases.append(dict(mode=c['name'],dt=c['dt'],H=Hmax,global_checks={k:bool(v) for k,v in global_checks.items()},
            qualified=c['summary']['qualified'],task=c['summary'],duration=c['summary']['metrics']['actual_duration_s'],
            forecast_integrations=cost,actual_integrations=c['physics_steps'],decisions=[first,len(rows)],
            fallbacks=sum(not d['any_feasible'] for d in c['planning'])))
    checks=[c['global_checks'] for c in cases]+[r['checks'] for r in rows];verify_source_snapshot(receipt,live=True)
    result=dict(schema='native_pen_scheduled_horizon_review:1',run=file_pin(a.report),source_receipt=file_pin(a.output/'source_receipt.json'),
        cases=cases,decisions=rows,contract_passed=sum(sum(v.values()) for v in checks),contract_total=sum(len(v) for v in checks),
        EFC=file_pin(a.output/'EFC/report.json'),EFC_counts=[efc['passed'],efc['total']],
        selected_prefix_counts=[sum(all(r['fidelity'].values()) for r in rows),len(rows)],optimizer_updates=0,native_integrations=0,
        scope='Independent scheduled EMA/reference/PD/mean score and actual first20ms prefix. Actual EFC and task gates separate. Longer forecasts already unqualified by matched-source pilot; no control benefit, expert or ego claim.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[result['contract_passed'],result['contract_total']],
        EFC=result['EFC_counts'],prefix=result['selected_prefix_counts'],cases=[{k:c[k] for k in ['mode','dt','H','duration','qualified','fallbacks','forecast_integrations','actual_integrations']} for c in cases])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
