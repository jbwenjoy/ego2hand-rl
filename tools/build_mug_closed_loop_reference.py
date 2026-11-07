"""Prepare an explicitly new anchored50Hz object-reference input contract."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.ego.trajectory_clock import resample_body_poses


def main(args):
    import mujoco
    reference,parent,out=map(lambda p:p.resolve(),[args.reference,args.parent,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((parent/'report.json').read_text());selected=prior['results'][0]
    if not selected['native_eligible']:raise ValueError('Predeclared native6D qualified support pose, not best dynamic case selection')
    source_pin=file_pin(reference/'reference61.npz');source=np.load(verify_file_pin(source_pin));state=np.load(verify_file_pin(selected['candidate']))
    case=next(c for c in selected['native_cases'] if c['control']=='inverse_PD_strict_LP' and c['dt_s']==.0003125);trace=np.load(verify_file_pin(case['trace']))
    q,u=state['qpos27'],trace['ctrl20'];model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    anchor=np.eye(4);anchor[:3,:3]=data.xmat[bind['mug_body_id']].reshape(3,3);anchor[:3,3]=data.xpos[bind['mug_body_id']]
    original=source['T_robot_world_from_object_BODY'];alignment=anchor@np.linalg.inv(original[0]);anchored=alignment[None]@original
    sampled=resample_body_poses(source['timestamps_ns'],anchored,source['object_COM_in_BODY_m']);meta=sampled.pop('metadata')
    if len(sampled['timestamps_ns'])!=101 or meta['excluded_tail_ns']!=4434 or np.max(abs(sampled['T_world_from_BODY'][0]-anchor))>1e-12:raise ValueError('Expected exact clock and new start pose')
    relative_old=np.linalg.inv(original[0])[None]@original;relative_new=np.linalg.inv(anchor)[None]@anchored
    relative_error=float(abs(relative_old-relative_new).max())
    out.mkdir(parents=True);np.savez_compressed(out/'reference101.npz',**sampled,source_frame_left=source['frame_ids'][sampled['source_left_index']],
        source_frame_ids=source['frame_ids'],source_timestamps_ns=source['timestamps_ns'],source_anchored_BODY_poses61=anchored,source_original_BODY_poses61=original,
        anchor_world_from_BODY=anchor,alignment_new_world_from_old_world=alignment,COM_BODY_m=source['object_COM_in_BODY_m'],
        initial_qpos27=q,initial_qvel26=np.zeros(26),grasp_q20_reference=np.tile(q[:20],(101,1)),feedforward_PD_target20=np.tile(u,(101,1)),initial_EMA_residual20_rad=np.zeros(20))
    contract=dict(schema='mug_closed_loop_input_contract_DRAFT:1',learning_enabled=False,new_task_not_phone_resume=True,
        object_reference='Anchored source-relative object motion; original fixed-palm mapping and new support initial pose remain explicit',
        hand_reference='Constant mechanically chosen grasp posture, not retargeted human joint trajectory',feedforward='Constant legal strict-LP PD target; distinct from observed grasp q',
        control_period_ns=20_000_000,control_intervals=100,proposed_residual_limit_rad=.05,proposed_EMA_memory='20 physical joint residual radians; initial zero; update factor0.5',
        executable_controller_and_native_observation_adapter_validated=False,contact_ground_truth='UNKNOWN; no zero force/contact supervision generated',
        dataset_role='INPUT_CONTRACT_ONLY_NOT_TRAIN',frozen_checkpoint_compatible=False,source_gravity_measured=False,simulator_gravity_changed=False,
        next_required='Explicit bounded control/EMA implementation, state/COM/frame/units adapter, actual response interventions and role-separated new data; analytic static hold is not a required success prerequisite for learning.')
    (out/'controller_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
    report=dict(schema='mug_anchored_closed_loop_reference:1',sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/ego/trajectory_clock.py')],
        inputs=[file_pin(reference/'binding.json'),source_pin,file_pin(parent/'report.json'),selected['candidate'],case['trace']],model=bind['model'],
        output=file_pin(out/'reference101.npz'),controller_contract=file_pin(out/'controller_contract.json'),time=meta,samples=101,control_intervals=100,source_frames=61,
        relative_object_pose_preservation_error=relative_error,initial_pose_anchor_error=float(abs(sampled['T_world_from_BODY'][0]-anchor).max()),
        initial_source_mapping_translation_change_m=alignment[:3,3].tolist(),alignment_rotation_trace=float(np.trace(alignment[:3,:3])),
        root_reference_displacement_m=float(np.linalg.norm(sampled['T_world_from_BODY'][-1,:3,3]-sampled['T_world_from_BODY'][0,:3,3])),
        raw_PD_target_legal=bool(np.all(u>=model.actuator_ctrlrange[:,0]) and np.all(u<=model.actuator_ctrlrange[:,1])),
        native_observation_and_controller_integration_complete=False,full_source_trajectory_executed=False,independent_RGB_reconstruction=False,integration_steps=0,neural_updates=0,
        scope='Exact source-clock input preparation only. A declared rigid start alignment preserves object-relative pose changes, not world gravity/source wrist alignment or human joint retargeting. No dynamic trajectory execution, training, observed contacts or ego/generalization benefit.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8');verify_file_pin(bind['model']);print(json.dumps(dict(report=file_pin(out/'report.json'),samples=101,tail_ns=meta['excluded_tail_ns'],relative_error=relative_error)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
