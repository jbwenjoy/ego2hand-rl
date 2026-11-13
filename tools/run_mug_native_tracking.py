"""Execute a fixed reference/hold native feedback baseline with full raw traces."""
from pathlib import Path
import argparse,json,sys,time,tomllib
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import CompoundTransitionReader
from supermimic_research.mechanics.native_oracle import NativeBranchOracle
from supermimic_research.mechanics.native_tracking import TrackingResidual,NativeTrackingPlanner
from supermimic_research.mechanics.transition_store import save_transitions
from supermimic_research.runtime.native_evaluation import GATES


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def summarize(records,reference,task,dt,warning_count):
    trace={k:np.concatenate([t['trace'][k] for t in records]) for k in records[0]['trace']};count=len(trace['post_qpos']);times=(np.arange(count)+1)*dt;tail=times>=1.5;pre_tail=np.arange(count)*dt>=1.5
    center0=reference['COM_world_m'][0];R0=reference['T_world_from_BODY'][0,:3,:3];quat0=np.r_[Rotation.from_matrix(R0).as_quat()[3],Rotation.from_matrix(R0).as_quat()[:3]]
    R=Rotation.from_quat(np.c_[trace['post_quat'][:,1:],trace['post_quat'][:,:1]]).as_matrix()
    if task=='HOLD':target=np.tile(center0,(count,1));targetR=np.tile(R0,(count,1,1));targetv=np.zeros((count,3));targetw=np.zeros((count,3))
    else:
        clock=np.arange(101)*.02;target=np.column_stack([np.interp(times,clock,reference['COM_world_m'][:,i]) for i in range(3)]);targetR=Slerp(clock,Rotation.from_matrix(reference['T_world_from_BODY'][:,:3,:3]))(np.minimum(times,2.)).as_matrix()
        targetv=np.column_stack([np.interp(times,clock,reference['COM_velocity_world_m_s'][:,i]) for i in range(3)]);targetw=np.column_stack([np.interp(times,clock,reference['angular_velocity_world_rad_s'][:,i]) for i in range(3)])
    position=np.linalg.norm(trace['post_COM']-target,axis=1);angle=Rotation.from_matrix(R@np.swapaxes(targetR,-1,-2)).magnitude()
    def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
    metrics=dict(actual_duration_s=count*dt,tracking_COM_RMS_m=rms(position),tracking_COM_max_m=float(position.max()),tracking_angle_RMS_rad=rms(angle),tracking_angle_max_rad=float(angle.max()),
        final_tracking_COM_m=float(position[-1]),final_tracking_angle_rad=float(angle[-1]),tracking_velocity_RMS_m_s=rms(trace['post_COM_velocity']-targetv)*(3**.5),tracking_omega_RMS_rad_s=rms(trace['post_omega_world']-targetw)*(3**.5),final_COM_drift_m=float(np.linalg.norm(trace['post_COM'][-1]-center0)),
        final_angle_rad=float(2*np.arccos(np.clip(abs(trace['post_quat'][-1]@quat0),0,1))),
        tail_linear_velocity_RMS_m_s=rms(trace['post_COM_velocity'][tail])*(3**.5) if tail.any() else None,tail_angular_velocity_RMS_rad_s=rms(trace['post_omega_world'][tail])*(3**.5) if tail.any() else None,
        tail_triple_pad_force_coverage=float(np.mean(np.all(trace['pre_group_normal_N'][pre_tail]>.1,axis=1))) if pre_tail.any() else None,
        max_native_penetration_m=float(trace['pre_penetration_m'].max()),max_self_penetration_m=float(trace['pre_self_penetration_m'].max()),max_phone_nonpad_normal_force_N=float(trace['pre_other_normal_N'].max()),
        max_table_normal_force_N=float(trace['pre_table_normal_N'].max()),max_joint_limit_excursion_rad=float(trace['post_limit_excursion_rad'].max()),warning_count=int(warning_count))
    physical={k:bool(metrics[k] is not None and (metrics[k]>=GATES[k] if k=='tail_triple_pad_force_coverage' else metrics[k]<=GATES[k])) for k in ['max_native_penetration_m','max_self_penetration_m','max_phone_nonpad_normal_force_N','max_table_normal_force_N','max_joint_limit_excursion_rad','warning_count','tail_triple_pad_force_coverage']}
    duration=len(records)==100 and not records[-1]['terminated'] and records[-1]['complete_H']
    original_hold={k:bool(metrics[k] is not None and metrics[k]<=GATES[k]) for k in ['final_COM_drift_m','final_angle_rad','tail_linear_velocity_RMS_m_s','tail_angular_velocity_RMS_rad_s']}
    tracking=dict(duration_complete=bool(duration),COM_RMS_5mm=metrics['tracking_COM_RMS_m']<=.005,COM_max_10mm=metrics['tracking_COM_max_m']<=.01,angle_RMS_5deg=metrics['tracking_angle_RMS_rad']<=np.pi/36,angle_max_10deg=metrics['tracking_angle_max_rad']<=np.pi/18,velocity_RMS_5cm_s=metrics['tracking_velocity_RMS_m_s']<=.05,omega_RMS_half_rad_s=metrics['tracking_omega_RMS_rad_s']<=.5)
    return dict(metrics=metrics,physical_gates=physical,tracking_gates=tracking,tracking_qualified=all(physical.values()) and all(tracking.values()),
        original_static_hold_qualified=duration and all(physical.values()) and all(original_hold.values()),original_static_hold_gates=original_hold,task_qualified=(duration and all(physical.values()) and all(original_hold.values())) if task=='HOLD' else (all(physical.values()) and all(tracking.values())),
        termination=records[-1]['termination'],scope='Full per-tick execution metrics; new reference tracking uses linear COM/SLERP interpolation on declared50Hz goals. Original static hold gates retained separately, not substituted for moving-reference tracking.')


def main(args):
    config_path=args.config.resolve();config=tomllib.loads(config_path.read_text());out=args.output.resolve();paths={k:(config_path.parent/config[k]).resolve() for k in ['oracle_parity','reference','binding']};parity=json.loads(paths['oracle_parity'].read_text());binding=json.loads(paths['binding'].read_text());reference_report=json.loads(paths['reference'].read_text());ref=dict(np.load(verify_file_pin(reference_report['output'])))
    if out.exists() or not out.is_relative_to(ROOT/'runs') or (parity['passed'],parity['total'])!=(18,18):raise ValueError('Fresh unified output and native branch parity required')
    if config['active_joints']!=list(range(12)) or config['horizon']!=3:raise ValueError('Fixed controller and action subset')
    for pin in parity['sources']:verify_file_pin(pin)
    sources=[file_pin(ROOT/p) for p in ['tools/run_mug_native_tracking.py','src/supermimic_research/mechanics/native_tracking.py','src/supermimic_research/mechanics/native_oracle.py',
        'src/supermimic_research/mechanics/compound_transition.py','src/supermimic_research/mechanics/residual_control.py','src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/transition_store.py']]
    out.mkdir();write(out/'intent.json',dict(schema='mug_reference_native_feedback:1',sources=sources,config=file_pin(config_path),inputs={k:file_pin(v) for k,v in paths.items()},model=binding['model'],
        task_scope='One fixed nominal initial mug/hand. HOLD diagnostic plus actual anchored object REFERENCE. Fixed original2s/initial50mm45deg aborts; reference range9.57mm/.0605rad is within abort envelope.',
        optimization='One finite-difference Gauss-Newton step at each actual50Hz observation with four native line-search trials. Future reference allowed; privileged native full state used only in oracle baseline.',
        no_neural_updates=True,no_world_model_gradients=True,no_object_force_aid=True,branch_sampled_contacts_not_execution_gate=True,max_total_oracle_steps=config['max_total_oracle_steps']))
    cases=[];oracle_steps=0;active_steps=0;start=time.perf_counter()
    for task in config['tasks']:
        for controller in config['controllers']:
            for dt in config['dt']:
                name=f'{task}_{controller}_{dt}';folder=out/name;folder.mkdir();reader=CompoundTransitionReader(binding,ref['initial_qpos27'],ref['feedforward_PD_target20'][0],dt_s=dt)
                oracle=NativeBranchOracle(reader,max_steps=config['max_total_oracle_steps']-oracle_steps);objective=TrackingResidual(ref,ref['grasp_q20_reference'][0],reader.model.jnt_range[:20],config,task);planner=NativeTrackingPlanner(oracle,objective,config)
                records=[];decisions=[];replays=[];prediction_checks=[]
                while not reader.terminated and reader.k<100:
                    snapshot=reader.snapshot();before=planner.state_dict()
                    if controller=='NATIVE_MPC':
                        action,decision=planner.choose(snapshot);after=planner.state_dict()
                        if snapshot['k'] in config['decision_replay_k']:
                            planner.restore(before);second,repeat=planner.choose(snapshot);equal=np.array_equal(action,second) and np.array_equal(decision['proposal_actions'],repeat['proposal_actions']) and np.array_equal(decision['proposal_costs'],repeat['proposal_costs']);replays.append(dict(k=snapshot['k'],passed=equal));planner.restore(after)
                            if not equal:raise ValueError('Restored planner decision changed')
                    else:action=np.zeros(20);decision=None
                    t=reader.step(action);records.append(t);active_steps+=t['executed_native_steps']
                    if decision is not None:
                        if t['complete_H']:
                            equal=np.array_equal(t['next_native_state'],decision['selected_first_native_state']);prediction_checks.append(dict(k=t['k'],passed=equal))
                            if not equal:raise ValueError('Selected prediction differs from actual first H')
                        decisions.append(decision)
                        if reader.k%10==0:print(json.dumps(dict(case=name,k=reader.k,physical_s=reader.tick*dt,oracle_steps=oracle.steps,wall_s=time.perf_counter()-start)),flush=True)
                reader.verify_physics();saved=save_transitions(folder/'execution',records,dt,[dict(task=task,controller=controller,dt=dt)]*len(records));summary=summarize(records,ref,task,dt,int(reader.d.warning.number.sum()))
                if decisions:
                    np.savez_compressed(folder/'decisions.npz',actions=np.array([r['action20'] for r in saved_arrays(records)]),selected_cost=np.array([d['selected_cost'] for d in decisions]),incumbent_cost=np.array([d['incumbent_cost'] for d in decisions]),
                        proposal_actions=np.concatenate([d['proposal_actions'] for d in decisions]),proposal_costs=np.concatenate([d['proposal_costs'] for d in decisions]),proposal_ranges=np.cumsum([0]+[len(d['proposal_costs']) for d in decisions]),
                        proposal_residuals_flat=np.concatenate([d['proposal_residuals'].ravel() for d in decisions]),proposal_residual_shapes=np.array([d['proposal_residuals'].shape for d in decisions]),proposal_residual_offsets=np.cumsum([0]+[d['proposal_residuals'].size for d in decisions]),
                        selected_first_native_state=np.array([d['selected_first_native_state'] for d in decisions]),decision_wall_s=np.array([d['wall_s'] for d in decisions]))
                oracle_steps+=oracle.steps;result=dict(task=task,controller=controller,dt=dt,execution=saved,summary=summary,oracle_steps=oracle.steps,oracle_calls=oracle.calls,decision_replays=replays,prediction_checks=prediction_checks,
                    median_decision_wall_s=float(np.median([d['wall_s'] for d in decisions])) if decisions else None,decisions=file_pin(folder/'decisions.npz') if decisions else None)
                write(folder/'report.json',result);cases.append(result);write(out/'progress.json',dict(completed_cases=len(cases),oracle_steps=oracle_steps,active_steps=active_steps,last=result,wall_s=time.perf_counter()-start));print(json.dumps(dict(case=name,actual_duration_s=summary['metrics']['actual_duration_s'],termination=summary['termination'],qualified=summary['task_qualified'],oracle_steps=oracle.steps)),flush=True)
    for pin in sources+[binding['model']]:verify_file_pin(pin)
    report=dict(schema='mug_reference_native_feedback:1',sources=sources,config=file_pin(config_path),inputs={k:file_pin(v) for k,v in paths.items()},cases=cases,total_oracle_steps=oracle_steps,total_active_steps=active_steps,wall_s=time.perf_counter()-start,neural_updates=0,
        scope='Fixed privileged CPU native feedback baseline, one initialization and anchored object reference; no trained policy, reliable WM gradient, real-time guarantee, source contact truth/RGB or full-goal success. Oracle computation cost fully charged.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),oracle_steps=oracle_steps,active_steps=active_steps,qualified=sum(c['summary']['task_qualified'] for c in cases))))


def saved_arrays(records):return [t['decision'] for t in records]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
