"""Recorded GPU cube/pen evaluation with one reset and first-terminal stopping."""
from pathlib import Path
import json
import random
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ..artifacts import file_pin
from .controlled_assets import bind_controlled_assets
from ..mechanics.compound_holding import object_state
from ..mechanics.warp_contact_labels import read_warp_contact_labels


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def summarize_trace(z, reference, dt, complete, config):
    t = np.arange(1, len(z['post_qpos']) + 1) * dt
    clock = reference['time_s']
    target = np.stack([np.interp(t, clock, reference['COM_world_m'][:, j]) for j in range(3)], axis=1)
    targetR = Slerp(clock, Rotation.from_quat(reference['quaternion_wxyz'][:, [1, 2, 3, 0]]))(np.minimum(t, clock[-1]))
    actualR = Rotation.from_quat(z['post_quaternion_wxyz'][:, [1, 2, 3, 0]])
    distance = np.linalg.norm(z['post_COM'] - target, axis=1)
    angle = (actualR * targetR.inv()).magnitude()
    m = dict(actual_duration_s=float(t[-1]), COM_RMS_m=float(np.sqrt(np.mean(distance**2))),
        COM_max_m=float(distance.max()), angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),
        angle_max_rad=float(angle.max()), max_penetration_m=float(z['pre_penetration_m'].max()),
        max_self_penetration_m=float(z['pre_self_penetration_m'].max()),
        max_joint_limit_excursion_rad=float(z['post_joint_limit_excursion_rad'].max()),
        max_object_nonhand_normal_force_N=float(z['pre_object_nonhand_normal_N'].max()),
        hand_contact_coverage_over_0_1N=float(np.mean(z['pre_hand_normal_N'] > .1)),
        object_speed_RMS_m_s=float(np.sqrt(np.mean(np.sum(z['post_COM_velocity']**2, axis=1)))),
        object_omega_RMS_rad_s=float(np.sqrt(np.mean(np.sum(z['post_omega_world']**2, axis=1)))))
    gates = {key: m[key] <= config[key] for key in [
        'COM_RMS_m', 'COM_max_m', 'angle_RMS_rad', 'angle_max_rad', 'max_penetration_m',
        'max_self_penetration_m', 'max_joint_limit_excursion_rad', 'max_object_nonhand_normal_force_N']}
    gates.update(duration_complete=bool(complete), no_applied_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
        finite_dynamics=all(np.isfinite(v).all() for v in z.values()),
        no_contact_constraint_overflow=not np.any(z['overflow']))
    return dict(metrics=m, gates={k: bool(v) for k, v in gates.items()}, qualified=bool(all(gates.values())),
        warning_count=None, warning_scope='Warp has no equivalent copied native warning counter; finite/capacity checks are explicit, not native warning=0.')


def evaluate_case(task, controller, saved, config, config_dir, registry, folder, *, initializer=None, constant_action=None):
    import torch
    import mujoco
    import mujoco_warp as mjwarp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg, SceneEntityCfg
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder.mkdir()
    cfg = make_env_cfg(1, 50, play=True, task=task,
        recipe_path=str((config_dir / config['pen_recipe']).resolve()) if task == 'hand2-pen-spin' else None,
        clips_root=str((config_dir / config['clips_root']).resolve()) if task == 'hand2-pen-spin' else None)
    cfg.seed = seed; cfg.auto_reset = False; cfg.scene.num_envs = 1
    if 'physics_dt' in config:
        cfg.sim.mujoco.timestep=config['physics_dt'];cfg.decimation=round(.02/config['physics_dt'])
        if abs(cfg.decimation*config['physics_dt']-.02)>1e-15:raise ValueError('Exact declared 50Hz clock')
    assets = bind_controlled_assets(cfg, task, registry)
    if task == 'hand2-pen-spin':
        from dex_supermimic.tasks.pen_spin import mdp
        motion = cfg.commands['motion']
        motion.first_frame_hold_prob = 0.; motion.debug_vis = False
        cfg.terminations['object_position_error'].params['threshold'] = .05
        cfg.terminations['object_orientation_error'] = TerminationTermCfg(
            func=mdp.object_shaft_direction_error_exceeded,
            params=dict(robot_cfg=SceneEntityCfg('robot', body_names=('right_palm_link',)),
                        object_cfg=SceneEntityCfg('object'), command_name='motion', threshold_deg=60.))
    else:
        cfg.commands['reorient_command'].min_goal_interval = config['cube_duration_s'] + 1
        cfg.commands['reorient_command'].debug_vis = False
    if any(event.mode not in ['reset', 'startup'] for event in cfg.events.values()):
        raise ValueError('Unexpected step/interval event in no-aid configuration')
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    original_step = env.sim.step
    started = time.perf_counter()
    try:
        observation = env.reset()[0]['policy']  # The only reset in this case.
        initialization_receipt=None
        if initializer is not None:
            if task!='hand2-pen-spin' or constant_action is None:raise ValueError('Declared constant-residual pen diagnostic only')
            initialization_receipt=initializer(env)
            env.observation_manager.reset()
            observation=env.observation_manager.compute(update_history=True)['policy']
        m = env.sim.mj_model; host = mujoco.MjData(m)
        mjwarp.get_data_into(host, m, env.sim.wp_data)
        if m.neq or np.any(m.body_gravcomp) or np.any(host.qfrc_applied) or np.any(host.xfrc_applied):
            raise ValueError('No applied force/equality/gravity compensation')
        if not np.all(m.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)):
            raise ValueError('Only joint actuators; GPU cone labels exclude adhesion actuation')
        body = m.body('object/cube').id
        joint = m.body_jntadr[body]; qa = int(m.jnt_qposadr[joint]); da = int(m.jnt_dofadr[joint])
        if int(m.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(m.body('robot/right_palm_link').jntnum.item()):
            raise ValueError('Free object and fixed wrist required')
        robot_joints = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
        qids = m.jnt_qposadr[robot_joints]; joint_ranges = m.jnt_range[robot_joints]
        dt = float(m.opt.timestep)
        com_body = env.sim.wp_model.body_ipos.numpy()[0, body].copy()
        initial_qpos, initial_qvel = host.qpos.copy(), host.qvel.copy()
        signature = mujoco.mjtState.mjSTATE_INTEGRATION
        initial_native = np.empty(mujoco.mj_stateSize(m, signature))
        mujoco.mj_getState(m, host, initial_native, signature)
        mujoco.mj_saveModel(m, str(folder/'host_topology.mjb'))
        physical = {}
        for key in ['body_mass','body_inertia','body_ipos','body_iquat','geom_size','geom_friction',
                    'geom_solref','geom_solimp','actuator_gainprm','actuator_biasprm',
                    'actuator_forcerange','actuator_ctrlrange','jnt_range','dof_damping','dof_armature']:
            value = getattr(env.sim.wp_model, key, None)
            if value is not None: physical[key] = value.numpy()
        np.savez_compressed(folder/'GPU_physical_context.npz', **physical)
        mocap0 = np.r_[host.mocap_pos.ravel(), host.mocap_quat.ravel()]
        if task == 'hand2-pen-spin':
            command = env.command_manager.get_term('motion')
            if command.group.num_trajectories != 1 or int(command.time_steps[0]) != 1:
                raise ValueError('Single clip and explicit reset frame0 -> command1 clock required')
            from mjlab.utils.lab_api.math import combine_frame_transforms
            pose = command._frames['object_pose']
            wrist = command._frames['wrist_pose_w']
            pos, quat = combine_frame_transforms(wrist[:, :3], wrist[:, 3:], pose[:, :3], pose[:, 3:])
            p, q = pos.cpu().numpy(), quat.cpu().numpy()
            R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
            reference = dict(time_s=np.arange(len(p))*.02, COM_world_m=p + R@com_body,
                             quaternion_wxyz=q)
            intervals = len(p) - 1
        else:
            command = env.command_manager.get_term('reorient_command')
            center = object_state(m, host.qpos, host.qvel, qa, da, com_body)[0]
            intervals = round(config['cube_duration_s']/.02)
            reference = dict(time_s=np.array([0., config['cube_duration_s']]),
                COM_world_m=np.tile(center, (2, 1)), quaternion_wxyz=np.tile(command.goal_quat_w[0].cpu().numpy(), (2, 1)))
        actor=None
        if constant_action is None:
            actor = Actor(saved['modules']['actor']['net.0.weight'].shape[1]).cuda().eval()
            actor.load_state_dict(saved['modules']['actor'], strict=True)
            if observation.shape[1] != actor.net[0].weight.shape[1]:
                raise ValueError('Saved Actor observation width mismatch')
        elif np.asarray(constant_action).shape!=(20,) or not np.isfinite(constant_action).all() or np.any(abs(constant_action)>1):
            raise ValueError('Explicit bounded constant action20')
        trace = {k: [] for k in ['pre_qpos', 'pre_qvel', 'post_qpos', 'post_qvel', 'post_ctrl',
            'post_qfrc_applied', 'post_xfrc_applied', 'post_COM', 'post_quaternion_wxyz',
            'post_COM_velocity', 'post_omega_world', 'pre_object_COM', 'pre_hand_normal_N',
            'pre_object_nonhand_normal_N', 'pre_penetration_m', 'pre_self_penetration_m',
            'post_joint_limit_excursion_rad', 'overflow', 'post_actuator_force', 'post_qfrc_actuator', 'post_GPU_time',
            'post_qfrc_constraint','post_qfrc_bias','post_qfrc_passive','post_qacc']}
        raw = []; action_rows = []; phase = []; terminations = []; ticks = 0
        observations = []; terminal_rows = []
        efc_rows=[];efc_ranges=[0];contact_addresses=[];legacy_force=[];legacy_adhesion=[]
        def record_step():
            nonlocal ticks
            preq = env.sim.data.qpos[0].cpu().numpy().copy()
            prev = env.sim.data.qvel[0].cpu().numpy().copy()
            original_step()
            mjwarp.get_data_into(host, m, env.sim.wp_data)
            labels=read_warp_contact_labels(env.sim.wp_model,env.sim.wp_data)
            if len(labels['dim'])!=host.ncon or not np.array_equal(labels['geom'],host.contact.geom):
                raise ValueError('GPU/host contact geometry order mismatch')
            efc_rows.append(labels['efc_force']);efc_ranges.append(efc_ranges[-1]+len(labels['efc_force']))
            contact_addresses.extend(labels['efc_address'])
            if not np.array_equal(np.r_[host.mocap_pos.ravel(), host.mocap_quat.ravel()], mocap0):
                raise ValueError('Wrist mocap moved after reset')
            if np.any(host.qfrc_applied) or np.any(host.xfrc_applied):
                raise ValueError('Applied force aid after reset')
            center, quat, velocity, omega = object_state(m, host.qpos, host.qvel, qa, da, com_body)
            hand_N = other_N = penetration = self_pen = 0.
            for ci in range(host.ncon):
                c = host.contact[ci]; g1, g2 = map(int, c.geom)
                old_force=np.zeros(6);mujoco.mj_contactForce(m,host,ci,old_force)
                legacy_force.append(old_force);legacy_adhesion.append(float(c.adhesion))
                force=labels['local_force6'][ci]
                raw.append(np.r_[ticks, g1, g2, c.dim, labels['efc_address'][ci,0], c.dist, c.pos, c.frame, force])
                penetration = max(penetration, -c.dist)
                bodies = m.geom_bodyid[[g1, g2]]
                if (bodies == body).sum() == 1:
                    other = g2 if bodies[0] == body else g1
                    name = m.geom(other).name or ''
                    if name.startswith('robot/'):
                        hand_N += max(force[0], 0.)
                    else:
                        other_N += max(force[0], 0.)
                elif all((m.geom(g).name or '').startswith('robot/') for g in [g1, g2]):
                    self_pen = max(self_pen, -c.dist)
            excursion = max(0., float((joint_ranges[:, 0] - host.qpos[qids]).max()),
                             float((host.qpos[qids] - joint_ranges[:, 1]).max()))
            wd = env.sim.wp_data
            overflow = int(wd.nacon.numpy()[0]) > wd.naconmax or int(wd.nefc.numpy()[0]) > wd.njmax or labels['overflow']
            values = [preq, prev, host.qpos.copy(), host.qvel.copy(), host.ctrl.copy(), host.qfrc_applied.copy(),
                host.xfrc_applied.copy(), center, quat, velocity, omega, host.xipos[body].copy(), hand_N, other_N,
                penetration, self_pen, excursion, overflow, host.actuator_force.copy(), host.qfrc_actuator.copy(), float(host.time),
                host.qfrc_constraint.copy(),host.qfrc_bias.copy(),host.qfrc_passive.copy(),host.qacc.copy()]
            for key, value in zip(trace, values): trace[key].append(value)
            ticks += 1
        env.sim.step = record_step
        completed = False
        with torch.no_grad():
            for k in range(intervals):
                before_phase = int(command.time_steps[0]) if task == 'hand2-pen-spin' else k+1
                if task == 'hand2-pen-spin' and before_phase != k+1:
                    raise ValueError('Unexpected reference clock resampling')
                action = torch.as_tensor(constant_action,dtype=torch.float32,device='cuda:0')[None] if constant_action is not None else torch.zeros((1, 20), device='cuda:0') if controller == 'ZERO' else actor(observation)
                observations.append(observation[0].cpu().numpy().copy())
                action_rows.append(action[0].cpu().numpy())
                obs, reward, terminated, truncated, _ = env.step(action)
                observation = obs['policy']
                phase.append(before_phase)
                flags = {name: bool(env.termination_manager.get_term(name)[0]) for name in cfg.terminations}
                terminations.append(flags)
                terminal_rows.append([bool(terminated[0]), bool(truncated[0])])
                if task == 'hand2-reorient' and not np.array_equal(command.goal_quat_w[0].cpu().numpy(), reference['quaternion_wxyz'][0]):
                    raise ValueError('Cube target switched')
                if bool(terminated[0] or truncated[0]):
                    completed = k+1 == intervals and not bool(terminated[0])
                    break
                if k+1 == intervals: completed = True
                if (k+1)%50 == 0: print(json.dumps(dict(task=task, controller=controller, k=k+1, physical_s=(k+1)*.02)), flush=True)
        z = {key: np.array(value) for key, value in trace.items()}
        np.savez_compressed(folder/'trace.npz', **z, raw_contact_tick_plus23=np.array(raw).reshape(-1, 24),
                            actions20=np.array(action_rows), pre_action_phase=np.array(phase),
                            initial_qpos=initial_qpos, initial_qvel=initial_qvel, initial_native_state=initial_native,
                            pre_action_observations=np.array(observations), terminated_truncated=np.array(terminal_rows),
                            termination_flags=np.array([list(row.values()) for row in terminations]),
                            GPU_efc_force_flat=np.concatenate(efc_rows),GPU_efc_ranges=np.array(efc_ranges),
                            GPU_contact_efc_addresses=np.array(contact_addresses),
                            legacy_host_local_force6=np.array(legacy_force).reshape(-1,6),
                            legacy_host_adhesion=np.array(legacy_adhesion), **reference)
        summary = summarize_trace(z, reference, dt, completed, config)
        result = dict(task=task, controller=controller, assets=assets, dt=dt, control_dt=.02,
            initial_reset_count=1, resets_after_start=0, planned_intervals=intervals,
            actual_intervals=len(action_rows), physics_steps=ticks, summary=summary,
            termination_flags=terminations[-1], phase_first=phase[0], phase_last=phase[-1],
            termination_names=list(terminations[0]),
            initialization=initialization_receipt,
            contact_label_schema='GPU_constraint_indexed_no_adhesion:1',
            trace=file_pin(folder/'trace.npz'), wall_s=time.perf_counter()-started,
            host_topology=file_pin(folder/'host_topology.mjb'), GPU_physical_context=file_pin(folder/'GPU_physical_context.npz'),
            host_model_scope='Compiled host topology plus separate actual GPU physical tables; host binary alone is not a GPU replay checkpoint.',
            model_layout=dict(nq=m.nq, nv=m.nv, nu=m.nu, object_qposadr=qa, object_dofadr=da,
                              object_body=body, COM_BODY=com_body.tolist()),
            scope=('Actual original GPU plant, declared constant action; one reset and explicit pre-start initialization, stop first native terminal, no Actor inference or fitting.' if constant_action is not None else
                   'Actual original GPU plant; one reset, stop first native terminal, recorded per-substep GPU contacts. Saved smoke Actor, not a trained expert or ego generalization.'))
        write(folder/'report.json', result)
        return result
    finally:
        env.sim.step = original_step
        env.close()
