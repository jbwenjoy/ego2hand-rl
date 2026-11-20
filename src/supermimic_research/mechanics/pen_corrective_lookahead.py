"""CPU shooting screen using the candidate's own simulated hand contact impulses.

Reference samples are desired end-of-control COM positions and body orientations.
The fixed gains are kp=100/s² and kd=20/s, for both translation and rotation.
All velocities are in world coordinates. When omitted, desired velocities use
backward differences of the reference (forward difference at index 0, or zero
for a singleton). The first source reference is extrapolated back one control.
No measured future motion is used.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


TICKS = 8
KP = 100.
KD = 20.
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _rotation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1, axis=-1))


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return value


def _reference_velocity(provided, increments, horizon, control_dt, name):
    if provided is not None:
        return _array(provided, (horizon, 3), name)
    if horizon == 1:
        return np.zeros((1, 3))
    return np.concatenate([increments[:1], increments]) / control_dt


def _body_state(data, qa, da, com_local):
    attitude = _rotation(data.qpos[qa + 3:qa + 7])
    offset = attitude.apply(com_local)
    omega = attitude.apply(data.qvel[da + 3:da + 6])
    com = data.qpos[qa:qa + 3] + offset
    velocity = data.qvel[da:da + 3] + np.cross(omega, offset)
    return com, attitude, velocity, omega


def _contacts(model, data, body, object_geom, hand_geom, local):
    wrench = np.zeros(6)
    hand_pen = self_pen = object_pen = nonhand_normal = 0.
    for index, contact in enumerate(data.contact):
        g1, g2 = contact.geom
        if g1 < 0 or g2 < 0:
            raise ValueError('Pen shooting requires rigid geom contacts')
        depth = max(0., -contact.dist)
        if hand_geom[g1] and hand_geom[g2]:
            self_pen = max(self_pen, depth)
        if object_geom[g1] == object_geom[g2]:
            continue
        object_pen = max(object_pen, depth)
        other = g1 if object_geom[g2] else g2
        if hand_geom[other]:
            hand_pen = max(hand_pen, depth)
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, index, local)
        if not hand_geom[other]:
            nonhand_normal += max(0., local[0])
            continue
        # Native wrench acts on geom2. Both contact and xipos are from the solve.
        basis = contact.frame.reshape(3, 3).T
        sign = 1 if object_geom[g2] else -1
        force = sign * (basis @ local[:3])
        wrench[:3] += force
        wrench[3:] += sign * (basis @ local[3:]) + np.cross(contact.pos - data.xipos[body], force)
    return wrench, (object_pen, hand_pen, self_pen, nonhand_normal)


def score_candidates(model, integration_state, pd_sequences, ref_pos, ref_quat_wxyz,
                     *, body, qa, da, length_m=.10, dt=.0025, ref_v=None, ref_omega=None):
    """Return one dict per candidate; each starts at the same mjSTATE_INTEGRATION.

    Supply one integration state per call to compare different initial preloads.
    The caller's model/state are never advanced. Hand geoms have a robot/ name or
    belong to a robot/ body. Each control holds 20 PD targets for eight native
    steps; the model's own control limits apply. External force arrays must be zero.

    Requested force is m[kp*dx + kd*dv - gravity]. Requested torque about the
    COM is I_world[kp*Log(R_des R.T) + kd*domega] + omega × I_world*omega.
    Both are fixed at interval start. The measured force/torque is the arithmetic
    mean of eight native hand wrenches, each about the *current* COM. The reported
    wrenches are expressed in interval-start body axes. Given these two wrenches,
    z = ||dF/(mg)||² + ||dT/(mg*length_m)||² and support_cost = mean[z/(1+z)].
    pose_cost = mean[(COM distance/.01m)² + (SO(3) geodesic/10deg)²].

    valid_controls counts the prefix before the first interval exceeding 2 mm
    object penetration, .2 mm self penetration, .001 rad joint excursion or
    .1 N nonhand normal load, or encountering a numerical/solver warning. This
    is only a CPU ranking screen; strict GPU/EFC qualification is separate.
    """
    targets = np.asarray(pd_sequences, dtype=np.float64)
    if targets.ndim != 3 or min(targets.shape[:2]) < 1 or targets.shape[2] != 20 or model.nu != 20:
        raise ValueError('Expected [K,H,20] controls for a 20-actuator pen model')
    horizon = targets.shape[1]
    _array(targets, targets.shape, 'pd_sequences')
    ref_pos = _array(ref_pos, (horizon, 3), 'ref_pos')
    ref_q = _array(ref_quat_wxyz, (horizon, 4), 'ref_quat_wxyz')
    if not np.allclose(np.linalg.norm(ref_q, axis=1), 1., atol=1e-4):
        raise ValueError('Expected unit reference quaternions in wxyz order')
    if not np.isfinite([dt, length_m]).all() or min(dt, length_m) <= 0 or not np.isclose(model.opt.timestep, dt):
        raise ValueError('Positive length and native dt matching model.opt.timestep required')
    if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICIT,
                                    mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError('Contact impulse screen requires a single-stage native integrator')
    body = model.body(body).id if isinstance(body, str) else int(body)
    if not 0 < body < model.nbody or model.body_jntnum[body] != 1:
        raise ValueError('Expected a rigid free joint body')
    joint = model.body_jntadr[body]
    if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE or
            model.jnt_qposadr[joint] != qa or model.jnt_dofadr[joint] != da):
        raise ValueError('qa and da must address the selected body free joint')
    source = _array(integration_state, (mujoco.mj_stateSize(model, STATE),), 'integration_state')
    mass = model.body_mass[body]
    gravity = model.opt.gravity
    weight = mass * np.linalg.norm(gravity)
    if weight <= 0:
        raise ValueError('Support scoring requires positive mass and gravity')
    inertial_rotation = _rotation(model.body_iquat[body]).as_matrix()
    body_inertia = (inertial_rotation * model.body_inertia[body]) @ inertial_rotation.T
    object_geom = model.geom_bodyid == body
    hand_geom = np.array([model.geom(i).name.startswith('robot/') or
                          model.body(int(model.geom_bodyid[i])).name.startswith('robot/')
                          for i in range(model.ngeom)])
    limited = model.jnt_limited.astype(bool) & (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    hand_qadr = model.jnt_qposadr[limited]
    lo, hi = model.jnt_range[limited].T
    ref_rot = _rotation(ref_q)
    control_dt = TICKS * dt
    rotations = ((ref_rot[1:] * ref_rot[:-1].inv()).as_rotvec() if horizon > 1 else np.empty((0, 3)))
    vel = _reference_velocity(ref_v, np.diff(ref_pos, axis=0), horizon, control_dt, 'ref_v')
    omega = _reference_velocity(ref_omega, rotations, horizon, control_dt, 'ref_omega')
    start_pos = np.vstack([ref_pos[0] - control_dt * vel[0], ref_pos[:-1]])
    start_rot = [Rotation.from_rotvec(-control_dt * omega[0]) * ref_rot[0], *ref_rot[:-1]]
    local = np.empty(6)
    results = []
    for sequence in targets:
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, source, STATE)
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError('Expected zero externally applied forces in the integration state')
        intervals = []
        valid_controls = native_steps = 0
        for t, target in enumerate(sequence):
            pos, rot, v, w = _body_state(data, qa, da, model.body_ipos[body])
            R = rot.as_matrix()
            I = R @ body_inertia @ R.T
            requested_force = mass * (KP * (start_pos[t] - pos) + KD * (vel[t] - v) - gravity)
            angle_error = (start_rot[t] * rot.inv()).as_rotvec()
            requested_torque = I @ (KP * angle_error + KD * (omega[t] - w)) + np.cross(w, I @ w)
            requested = np.r_[R.T @ requested_force, R.T @ requested_torque]
            measured_sum = np.zeros(6)
            maxima = np.zeros(4)
            max_joint = 0.
            finite = True
            data.ctrl[:] = target
            for _ in range(TICKS):
                mujoco.mj_step(model, data)
                native_steps += 1
                finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and
                              np.isfinite(data.qacc).all() and not np.any(data.warning.number))
                if not finite:
                    break
                measured_tick, geometry = _contacts(model, data, body, object_geom, hand_geom, local)
                measured_sum += measured_tick
                np.maximum(maxima, geometry, out=maxima)
                if hand_qadr.size:
                    joints = data.qpos[hand_qadr]
                    max_joint = max(max_joint, (lo - joints).max(), (joints - hi).max())
            if finite:
                pos, rot, _, _ = _body_state(data, qa, da, model.body_ipos[body])
                distance = np.linalg.norm(pos - ref_pos[t])
                relative = ref_rot[t].inv() * rot
                angle = relative.magnitude()
                shaft = relative.apply([1., 0., 0.])
                shaft_deg = np.rad2deg(np.arctan2(np.hypot(shaft[1], shaft[2]), shaft[0]))
                measured = (measured_sum.reshape(2, 3) @ R).ravel() / TICKS
                scaled = (measured - requested) / np.repeat([weight, weight * length_m], 3)
                z = scaled @ scaled
                pose_cost = (distance / .01) ** 2 + (angle / np.deg2rad(10)) ** 2
                support_cost = z / (1 + z)
                finite = bool(np.isfinite([pose_cost, support_cost, *maxima, max_joint]).all())
            feasible = bool(finite and maxima[0] <= .002 and maxima[2] <= .0002 and
                            max_joint <= .001 and maxima[3] <= .1)
            intervals.append(dict(finite=finite, geometry_ok=feasible,
                                  max_object_penetration_m=float(maxima[0]), max_hand_penetration_m=float(maxima[1]),
                                  max_self_penetration_m=float(maxima[2]), max_joint_excursion_rad=float(max_joint),
                                  max_nonhand_normal_N=float(maxima[3]), requested_wrench_body=requested.tolist(),
                                  measured_wrench_body=measured.tolist() if finite else None,
                                  mm=float(distance * 1000) if finite else None,
                                  deg=float(np.rad2deg(angle)) if finite else None,
                                  shaft_deg=float(shaft_deg) if finite else None,
                                  pose_cost=float(pose_cost) if finite else None,
                                  support_cost=float(support_cost) if finite else None))
            if valid_controls == t and feasible:
                valid_controls += 1
            if not finite:
                break
        end = intervals[-1]
        results.append(dict(
            pose_cost=float(np.mean([i['pose_cost'] for i in intervals])) if end['finite'] else None,
            support_cost=float(np.mean([i['support_cost'] for i in intervals])) if end['finite'] else None,
            valid_controls=valid_controls, native_steps=native_steps, finite=end['finite'],
            endpoint_pos=pos.tolist() if end['finite'] else None,
            endpoint_quat_wxyz=np.roll(rot.as_quat(), 1).tolist() if end['finite'] else None,
            endpoint_mm=end['mm'], endpoint_deg=end['deg'], intervals=intervals))
    return results
