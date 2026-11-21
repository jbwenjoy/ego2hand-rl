"""Source-world kinematic retarget and physically driven, floating wrist carton replay."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from supermimic_research.assets.hand2_scene import fixed_hand
from supermimic_research.ego.hand_frames import palm_frame


TIPS = [4, 8, 12, 16, 20]
BASES = [5, 9, 13, 17]
MIDS = [2, 6, 10, 14, 18]


def quat_from_rotation(matrix):
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True)


def create_scene(source_hand, initial_object, mesh, mass, friction, normal, table_height, dt):
    spec, _, _, registration = fixed_hand()
    local_palm = registration['T_robot_root_from_robot_palm']
    wrist = np.array([palm_frame(h[0], h[BASES]) @ np.linalg.inv(local_palm) for h in source_hand])
    palm = spec.body('right_palm_link')
    palm.pos = wrist[0, :3, 3]
    palm.quat = quat_from_rotation(wrist[0, :3, :3])
    palm.add_freejoint(name='wrist_free')
    obj = spec.worldbody.add_body(name='carton', pos=initial_object[:3, 3],
                                  quat=quat_from_rotation(initial_object[:3, :3]))
    obj.add_freejoint(name='carton_free')
    spec.add_mesh(name='carton_hull', uservert=mesh.vertices.ravel(), userface=mesh.faces.ravel())
    obj.add_geom(name='carton_collision', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='carton_hull',
                 mass=mass, friction=[friction,.005,.0001], rgba=[.08,.43,.78,1.])
    spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0,0,table_height], zaxis=normal, size=[.72,.72,.02],
                            friction=[.6,.005,.0001], rgba=[.66,.64,.51,1.])
    # Explicit object pairs keep friction identical when robot geoms carry a higher value.
    hand_geoms = [g.name for g in spec.geoms if g.name not in ('table','carton_collision')
                  and (g.contype or g.conaffinity)]
    for other in hand_geoms + ['table']:
        mu = .6 if other == 'table' else friction
        spec.add_pair(geomname1='carton_collision', geomname2=other, condim=3,
                      friction=[mu,mu,.005,.0001,.0001], solref=[.006,1])
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 80
    spec.option.tolerance = 1e-8
    model = spec.compile()
    assert (model.nq,model.nv,model.nu,model.neq,model.nmocap) == (34,32,20,0,0)
    assert not model.body_gravcomp.any()
    return model, wrist


def solve_kinematics(model, wrists, source_hand, tip_targets):
    data = mujoco.MjData(model)
    hinge_joints = model.actuator_trnid[:,0]
    qidx = model.jnt_qposadr[hinge_joints]
    didx = model.jnt_dofadr[hinge_joints]
    lo = np.maximum(model.jnt_range[hinge_joints,0],model.actuator_ctrlrange[:,0])+1e-7
    hi = np.minimum(model.jnt_range[hinge_joints,1],model.actuator_ctrlrange[:,1])-1e-7
    sites = [model.site(f'right_finger{i}_tip').id for i in range(1,6)]
    mids = [model.body(f'right_finger{i}_link3').id for i in range(1,6)]
    jp = np.empty((3,model.nv)); jr = np.empty_like(jp)
    q = np.clip(np.zeros(20),lo,hi)
    poses, tips, steps = [], [], []
    for i, wrist in enumerate(wrists):
        data.qpos[:3] = wrist[:3,3]
        data.qpos[3:7] = quat_from_rotation(wrist[:3,:3])
        lastq = None
        def fk(x):
            nonlocal lastq
            if lastq is None or not np.array_equal(x,lastq):
                data.qpos[qidx] = x
                mujoco.mj_kinematics(model,data)
                mujoco.mj_comPos(model,data)
                lastq = x.copy()
        def residual(x):
            fk(x)
            return np.r_[((data.site_xpos[sites]-tip_targets[i])*100).ravel(),
                         ((data.xpos[mids]-source_hand[i,MIDS])*22).ravel(),.10*(x-q),.015*x]
        def jacobian(x):
            fk(x)
            rows=[]
            for site in sites:
                mujoco.mj_jacSite(model,data,jp,jr,site)
                rows.append(jp[:,didx]*100)
            for body in mids:
                mujoco.mj_jacBody(model,data,jp,jr,body)
                rows.append(jp[:,didx]*22)
            return np.vstack([*rows,.10*np.eye(20),.015*np.eye(20)])
        result = least_squares(residual,q,jac=jacobian,bounds=(lo,hi),max_nfev=45,
                               ftol=1e-6,xtol=1e-6,gtol=1e-6)
        fk(result.x)
        q = result.x.copy()
        poses.append(q); tips.append(data.site_xpos[sites].copy()); steps.append(result.nfev)
    return np.stack(poses),np.stack(tips),np.array(steps)


def limit_norm(v, bound):
    return v * min(1.,bound/max(1e-15,np.linalg.norm(v)))


def body_pose(data,body):
    t=np.eye(4)
    t[:3,:3]=data.xmat[body].reshape(3,3)
    t[:3,3]=data.xpos[body]
    return t


def replay(model, wrists, fingers, times, limits):
    # The six external generalized commands actuate the wrist only. The separate
    # free object's generalized and Cartesian applied forces must remain identically zero.
    data=mujoco.MjData(model)
    palm=model.body('right_palm_link').id
    carton=model.body('carton').id
    gid=model.geom('carton_collision').id
    table=model.geom('table').id
    obj_dof=int(model.jnt_dofadr[model.joint('carton_free').id])
    qidx=model.jnt_qposadr[model.actuator_trnid[:,0]]
    sites=[model.site(f'right_finger{i}_tip').id for i in range(1,6)]
    handmass=model.body_subtreemass[palm]
    data.qpos[qidx]=fingers[0]
    data.ctrl[:]=fingers[0]
    mujoco.mj_forward(model,data)
    frames={}
    contact_force=np.zeros(6)
    def save(i, maxforce, maxtorque, peakpen, maxcount, hand_impulse, table_impulse, peakself, joint_excursion):
        # After mj_step, the derived geom/contact arrays precede the final qpos integration.
        mujoco.mj_forward(model,data)
        gf=0.; hf=0.; count=0
        for k,contact in enumerate(data.contact):
            g1,g2=int(contact.geom1),int(contact.geom2)
            if gid not in (g1,g2): continue
            mujoco.mj_contactForce(model,data,k,contact_force)
            if table in (g1,g2): gf+=contact_force[0]
            else: hf+=contact_force[0];count+=1
        fields=dict(frame_index=i,time=data.time,sim_qpos=data.qpos.copy(),sim_qvel=data.qvel.copy(),
                    finger_ctrl=data.ctrl.copy(),world_from_carton_sim=body_pose(data,carton),
                    sim_tips=data.site_xpos[sites].copy(),sim_wrist=body_pose(data,palm),
                    hand_carton_contact_count=count,table_carton_normal_force=gf,
                    hand_carton_normal_force=hf,wrist_force_max_N=maxforce,wrist_torque_max_Nm=maxtorque,
                    interval_max_carton_penetration_m=peakpen,interval_max_hand_contacts=maxcount,
                    interval_hand_carton_normal_impulse_Ns=hand_impulse,
                    interval_table_carton_normal_impulse_Ns=table_impulse,
                    interval_max_hand_self_penetration_m=peakself,interval_joint_excursion_rad=joint_excursion)
        for key,value in fields.items(): frames.setdefault(key,[]).append(value)
    save(0,0,0,0,0,0,0,0,0)
    step_controls, step_wrenches = [], []
    frame_ends = [0]
    dt=model.opt.timestep
    rot_all=Rotation.from_matrix(wrists[:,:3,:3])
    for i in range(1,len(wrists)):
        duration=times[i]-times[i-1]
        steps=int(round(duration/dt))
        if abs(dt*steps-duration)>1e-8:
            raise ValueError('Source frame period must be multiple of physics timestep')
        startR=rot_all[i-1]
        rotvec=(rot_all[i]*startR.inv()).as_rotvec()
        vref=limit_norm((wrists[i,:3,3]-wrists[i-1,:3,3])/duration,limits['vmax'])
        wref=limit_norm(rotvec/duration,limits['wmax'])
        maxf=maxt=penetration=maxcontacts=hand_impulse=table_impulse=peakself=excursion=0.
        for step in range(steps):
            phase=step/steps
            pos=(1-phase)*wrists[i-1,:3,3]+phase*wrists[i,:3,3]
            rot=Rotation.from_rotvec(phase*rotvec)*startR
            now=Rotation.from_quat(data.qpos[3:7],scalar_first=True)
            # Freejoint translational velocities are world axes, angular velocities are body axes.
            vgoal=limit_norm(vref+limits['pos_gain']*(pos-data.qpos[:3]),limits['vmax'])
            wgoal=limit_norm(wref+limits['rot_gain']*(rot*now.inv()).as_rotvec(),limits['wmax'])
            ff=-handmass*model.opt.gravity
            ff_tau=np.cross(data.subtree_com[palm]-data.qpos[:3],ff)
            force=limit_norm(limits['lin_damping']*(vgoal-data.qvel[:3])+ff,limits['max_force'])
            torque=limit_norm(limits['ang_damping']*(wgoal-now.apply(data.qvel[3:6]))+ff_tau,limits['max_torque'])
            data.qfrc_applied[:]=0
            data.qfrc_applied[:3]=force
            data.qfrc_applied[3:6]=now.inv().apply(torque)
            data.ctrl[:]=(1-phase)*fingers[i-1]+phase*fingers[i]
            step_controls.append(data.ctrl.copy())
            step_wrenches.append(data.qfrc_applied[:6].copy())
            assert not data.qfrc_applied[obj_dof:obj_dof+6].any() and not data.xfrc_applied.any()
            mujoco.mj_step(model,data)
            maxf=max(maxf,float(np.linalg.norm(force)));maxt=max(maxt,float(np.linalg.norm(torque)))
            count=0
            for k,c in enumerate(data.contact):
                pair=(int(c.geom1),int(c.geom2))
                if gid in pair:
                    penetration=max(penetration,-c.dist)
                    mujoco.mj_contactForce(model,data,k,contact_force)
                    if table in pair: table_impulse+=contact_force[0]*dt
                    else:
                        hand_impulse+=contact_force[0]*dt
                        count+=1
                elif table not in pair: peakself=max(peakself,-c.dist)
            maxcontacts=max(maxcontacts,count)
            angles=data.qpos[qidx]
            exc=np.maximum(model.jnt_range[model.actuator_trnid[:,0],0]-angles,
                           angles-model.jnt_range[model.actuator_trnid[:,0],1])
            excursion=max(excursion,float(exc.max()))
        save(i,maxf,maxt,penetration,maxcontacts,hand_impulse,table_impulse,peakself,excursion)
        frame_ends.append(len(step_controls))
    frames.update(substep_finger_ctrl=step_controls, substep_wrist_generalized_wrench=step_wrenches,
                  frame_end_substep=frame_ends)
    return {key:np.asarray(values) for key,values in frames.items()}


def audit_commands(model, trace):
    data = mujoco.MjData(model)
    data.qpos[:] = trace['sim_qpos'][0]
    data.qvel[:] = trace['sim_qvel'][0]
    data.ctrl[:] = trace['finger_ctrl'][0]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, data)
    error = 0.
    for frame, (start, end) in enumerate(zip(trace['frame_end_substep'][:-1], trace['frame_end_substep'][1:]), 1):
        for step in range(start, end):
            data.ctrl[:] = trace['substep_finger_ctrl'][step]
            data.qfrc_applied[:6] = trace['substep_wrist_generalized_wrench'][step]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        error = max(error, float(np.max(np.abs(data.qpos-trace['sim_qpos'][frame]))),
                    float(np.max(np.abs(data.qvel-trace['sim_qvel'][frame]))))
    if error > 1e-9:
        raise ValueError(f'Independent command replay diverged by {error}')
    return error
