"""Explicit pre-start joint/EMA setup and actual-GPU frozen-contact queries."""
import copy
import numpy as np
from ..artifacts import file_pin
from ..mechanics.warp_contact_labels import read_warp_contact_labels
from ..mechanics.indexed_contact_support import indexed_static_support
from ..mechanics.support import static_feasibility
from ..mechanics.wrench_separation import separating_witness


def initialize_and_query_pen(env,folder,source_qpos,projected_qpos,delta20,dynamic_target_BODY):
    import torch,mujoco,mujoco_warp as mjwarp
    from .controlled_evaluation import write
    model=env.sim.mj_model;action=env.action_manager.get_term('joint_pos');robot=env.scene['robot'];command=env.command_manager.get_term('motion')
    qids=np.array([model.jnt_qposadr[model.joint(f'robot/right_finger{i}_joint{j}').id] for i in range(1,6) for j in range(1,5)])
    dofs=np.array([model.jnt_dofadr[model.joint(f'robot/right_finger{i}_joint{j}').id] for i in range(1,6) for j in range(1,5)])
    aids=action._position_ctrl_ids.cpu().numpy()
    if action.cfg.alpha!=.5 or action.cfg.scale!=.5 or action.cfg.mask_prob!=0:raise ValueError('Original unmasked residual .5-scale/.5-EMA action contract')
    if np.any(abs(delta20)>.1+1e-12) or np.any(abs(2*delta20)>1):raise ValueError('Bounded predeclared residual')
    initial=env.sim.data.qpos[0].cpu().numpy().copy();velocity=env.sim.data.qvel.clone();warm=env.sim.data.qacc_warmstart.clone()
    if not np.array_equal(initial,np.array(source_qpos,dtype=np.float32).astype(np.float64)):raise ValueError('Original seeded state before correction changed')
    expected=np.array(projected_qpos,dtype=np.float32)
    if not np.array_equal(np.delete(expected,qids),np.delete(initial.astype(np.float32),qids)):raise ValueError('Only robot joint positions may change')
    robot.write_joint_state_to_sim(torch.as_tensor(expected[qids],device=env.device)[None],robot.data.joint_vel[:,action._target_ids].clone(),joint_ids=action._target_ids)
    delta=torch.as_tensor(delta20,dtype=torch.float32,device=env.device)[None]
    action._processed_actions[:]=delta;action._raw_actions[:]=2*delta
    action._targets[:]=action.limit_target(command._frames['joint_pos'][0][None]+delta);action._targets_valid[:]=True
    action.apply_actions();env.scene.write_data_to_sim()
    # Static support probe is BEFORE trajectory start; original velocities and
    # warmstart are restored before the one official execution. No episode reset.
    env.sim.data.qvel.zero_();env.sim.data.qacc_warmstart.zero_();env.sim.forward()
    host_model=copy.copy(model)
    if np.shares_memory(host_model.body_mass,model.body_mass) or np.shares_memory(host_model.geom_size,model.geom_size):raise ValueError('Detached query model required')
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','geom_size','geom_friction','geom_solref','geom_solimp',
                'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','jnt_range','dof_damping','dof_armature']:
        value=getattr(env.sim.wp_model,key,None)
        if value is not None:getattr(host_model,key)[:]=value.numpy()[0]
    data=mujoco.MjData(host_model);mujoco.mj_setConst(host_model,data);mjwarp.get_data_into(data,host_model,env.sim.wp_data)
    labels=read_warp_contact_labels(env.sim.wp_model,env.sim.wp_data)
    if labels['overflow'] or not np.all(model.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)):raise ValueError('No contact overflow/adhesion actuation')
    # Host-only field initialization fixes the demonstrated bridge omission.
    # Actual GPU constraints, contacts, model and forces are never overwritten.
    data.contact.adhesion[:]=0.
    for i,force in enumerate(labels['local_force6']):
        f=np.zeros(6);mujoco.mj_contactForce(host_model,data,i,f)
        if np.max(abs(f-force))>1e-10:raise ValueError('Corrected host contact decode must match actual GPU EFC values')
    body=host_model.body('object/cube').id;parts=[g for g in range(host_model.ngeom) if host_model.geom_bodyid[g]==body and (host_model.geom_contype[g] or host_model.geom_conaffinity[g])]
    allowed=[host_model.geom(f'robot/right_finger{i}_{s}').id for i in [1,2,3] for s in ['link4_col','tip_sensor_col']]
    q=data.qpos[qids];jids=host_model.actuator_trnid[aids,0];lo=np.maximum(q-.05,np.maximum(host_model.jnt_range[jids,0],host_model.actuator_ctrlrange[aids,0]));hi=np.minimum(q+.05,np.minimum(host_model.jnt_range[jids,1],host_model.actuator_ctrlrange[aids,1]))
    support=indexed_static_support(host_model,data,body,parts,allowed,dofs,aids,lo,hi)
    pure_static=static_feasibility(support['wrench_BODY'],support['target_BODY'],radius=.125)
    dynamic=static_feasibility(support['wrench_BODY'],dynamic_target_BODY,radius=.125)
    static_certificate=None if pure_static['success'] else separating_witness(support['wrench_BODY'],support['target_BODY'],radius_m=.125)
    dynamic_certificate=None if dynamic['success'] else separating_witness(support['wrench_BODY'],dynamic_target_BODY,radius_m=.125)
    static_state=np.empty(mujoco.mj_stateSize(host_model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(host_model,data,static_state,mujoco.mjtState.mjSTATE_INTEGRATION)
    path=folder/'initial_GPU_support.npz'
    np.savez_compressed(path,wrench_BODY=support['wrench_BODY'],target_static_BODY=support['target_BODY'],target_dynamic_BODY=dynamic_target_BODY,
        loads20=support['loads20'],torque_lower=support['torque_lower'],torque_upper=support['torque_upper'],torque_offset=support['offset'],
        GPU_geom=labels['geom'],GPU_distance=labels['dist'],GPU_frame=labels['frame'],GPU_position=labels['pos'],GPU_friction=labels['friction'],GPU_dim=labels['dim'],
        GPU_efc_addresses=labels['efc_address'],GPU_efc_force=labels['efc_force'],GPU_local_force6=labels['local_force6'],
        static_native_integration_state=static_state,static_qpos=data.qpos.copy(),static_qvel=data.qvel.copy(),source_initial_qpos=initial,
        projected_qpos=expected,controller_delta20=delta20,
        static_ray_coefficients_N=np.array([]) if support['lp']['witness'] is None else support['lp']['witness']['ray_coefficients_N'],
        static_control_witness20=np.array([]) if support['control'] is None else support['control'],
        dynamic_ray_coefficients_N=np.array([]) if dynamic['witness'] is None else dynamic['witness']['ray_coefficients_N'])
    receipt=dict(schema='projected_pen_GPU_initialization:1',source=file_pin(__file__),snapshot=file_pin(path),
        joint_float32_rounding_max=float(abs(expected-np.asarray(projected_qpos)).max()),max_joint_delta_rad=float(abs(expected[qids]-initial[qids]).max()),
        static_support_with_PD_bounds=support['lp']['success'],static_wrench_cone_feasible=pure_static['success'],dynamic_wrench_cone_feasible=dynamic['success'],
        static_separating_certificate=static_certificate,dynamic_separating_certificate=dynamic_certificate,
        actual_active_support_contacts=support['contacts'],full_wrench_rank=support['full_wrench_rank'],
        observation_scope='Actual GPU contact/constraint snapshot at zero static velocities. Host-only adhesion field cleared; no GPU force or pose overwritten for support query.',
        neural_updates=0,native_integration_steps=0)
    env.sim.data.qvel.copy_(velocity);env.sim.data.qacc_warmstart.copy_(warm);env.sim.forward()
    if not torch.equal(env.sim.data.qvel,velocity) or not np.array_equal(env.sim.data.qpos[0].cpu().numpy(),expected):raise ValueError('Original velocities and projected state restored before start')
    receipt['source_initial_velocities_preserved']=True
    receipt['scope']='One explicit pre-start joint/EMA correction; static probe followed by original source velocities, no mid-trajectory reset, object/wrist pose fixed.'
    write(folder/'initial_GPU_support.json',receipt)
    return receipt
