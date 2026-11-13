"""Shared privileged native-controller execution/persistence, no copied planner."""
import json,time
import numpy as np
from .compound_transition import CompoundTransitionReader
from .native_oracle import NativeBranchOracle
from .native_tracking import TrackingResidual,NativeTrackingPlanner
from .transition_store import save_transitions
from ..artifacts import file_pin


def execute_native_case(binding,reference,config,task,dt,folder,summarize,*,oracle_budget,on_progress=None):
    folder.mkdir();reader=CompoundTransitionReader(binding,reference['initial_qpos27'],reference['feedforward_PD_target20'][0],dt_s=dt)
    oracle=NativeBranchOracle(reader,max_steps=oracle_budget);objective=TrackingResidual(reference,reference['grasp_q20_reference'][0],reader.model.jnt_range[:20],config,task);planner=NativeTrackingPlanner(oracle,objective,config)
    records=[];decisions=[];replays=[];prediction_checks=[];started=time.perf_counter()
    while not reader.terminated and reader.k<100:
        snapshot=reader.snapshot();before=planner.state_dict();action,decision=planner.choose(snapshot);after=planner.state_dict()
        if snapshot['k'] in config['decision_replay_k']:
            planner.restore(before);second,repeat=planner.choose(snapshot);equal=np.array_equal(action,second) and np.array_equal(decision['proposal_actions'],repeat['proposal_actions']) and np.array_equal(decision['proposal_costs'],repeat['proposal_costs']);replays.append(dict(k=snapshot['k'],passed=equal));planner.restore(after)
            if not equal:raise ValueError('Exact decision restore failed')
        t=reader.step(action);records.append(t);decisions.append(decision)
        if t['complete_H']:
            equal=np.array_equal(t['next_native_state'],decision['selected_first_native_state']);prediction_checks.append(dict(k=t['k'],passed=equal))
            if not equal:raise ValueError('Forecast/actual numerical state mismatch')
        if on_progress is not None and reader.k%10==0:on_progress(dict(task=task,dt=dt,k=reader.k,physical_s=reader.tick*dt,oracle_steps=oracle.steps,wall_s=time.perf_counter()-started))
    reader.verify_physics();execution=save_transitions(folder/'execution',records,dt,[dict(task=task,controller='NATIVE_MPC',dt=dt)]*len(records));summary=summarize(records,reference,task,dt,int(reader.d.warning.number.sum()))
    np.savez_compressed(folder/'decisions.npz',actions=np.array([t['decision']['action20'] for t in records]),selected_cost=np.array([d['selected_cost'] for d in decisions]),incumbent_cost=np.array([d['incumbent_cost'] for d in decisions]),
        proposal_actions=np.concatenate([d['proposal_actions'] for d in decisions]),proposal_costs=np.concatenate([d['proposal_costs'] for d in decisions]),proposal_ranges=np.cumsum([0]+[len(d['proposal_costs']) for d in decisions]),
        proposal_residuals_flat=np.concatenate([d['proposal_residuals'].ravel() for d in decisions]),proposal_residual_shapes=np.array([d['proposal_residuals'].shape for d in decisions]),proposal_residual_offsets=np.cumsum([0]+[d['proposal_residuals'].size for d in decisions]),
        selected_first_native_state=np.array([d['selected_first_native_state'] for d in decisions]),decision_wall_s=np.array([d['wall_s'] for d in decisions]))
    result=dict(task=task,controller='NATIVE_MPC',dt=dt,horizon=config['horizon'],execution=execution,summary=summary,oracle_steps=oracle.steps,oracle_calls=oracle.calls,decision_replays=replays,prediction_checks=prediction_checks,
        median_decision_wall_s=float(np.median([d['wall_s'] for d in decisions])),case_wall_s=time.perf_counter()-started,decisions=file_pin(folder/'decisions.npz'))
    (folder/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');return result
