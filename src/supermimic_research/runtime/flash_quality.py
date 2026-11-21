"""Physical substep recorder and independent command replay for FlashCHORD evaluation."""
import numpy as np
import newton
from scipy.spatial.transform import Rotation

from flash_chord.runtime.sim import build_force_contacts
from supermimic_research.runtime.flash_solver import build_solver


COMMANDS = ("joint_target_pos","joint_target_vel","joint_f")
STATES = ("joint_q","joint_qd","body_q","body_qd")


class SolverRecorder:
    def __init__(self, env):
        self.env = env
        self.solver = env.solver
        self.original_step = self.solver.step
        self.initial_state = {k:getattr(env.state_0,k).numpy().copy() for k in STATES}
        self.commands = {k:[] for k in COMMANDS}
        self.samples = []
        self.contact_rows = []
        self.contact_offsets = [0]
        self.endpoint_states = [self.initial_state.copy()]
        self.env.solver.step = self.step
        self.hand = env.scene.layout.hand("right")
        self.obj = env.scene.objects[0]
        self.obj_id = self.obj.bodies[0].body_id
        self.object_ids = np.asarray(list(self.obj.shapes.ids()))
        self.hand_ids = np.array([i for link in self.hand.link_geometry for i in link.shape_ids])
        self.support_ids = np.array(list(env.scene.support.shapes.ids()))
        self.lo = env.model.joint_limit_lower.numpy()[list(self.hand.finger_dof_ids)]
        self.hi = env.model.joint_limit_upper.numpy()[list(self.hand.finger_dof_ids)]
        self.violations = []

    def step(self,s0,s1,control,contacts,dt):
        if np.any(s0.body_f.numpy()[self.obj_id]) or np.any(control.joint_f.numpy()[list(self.obj.root.free_dof_ids)]):
            raise ValueError("A direct external carton force invalidates unaided contact evaluation")
        for key, values in self.commands.items():
            values.append(getattr(control,key).numpy().copy())
        self.original_step(s0,s1,control,contacts,dt)
        self.solver.update_contacts(contacts,s1)
        n=int(contacts.rigid_contact_count.numpy()[0])
        first=contacts.rigid_contact_shape0.numpy()[:n]
        second=contacts.rigid_contact_shape1.numpy()[:n]
        raw_normals=contacts.rigid_contact_normal.numpy()[:n]
        forces=contacts.force.numpy()[:n,:3]
        normal_load=np.maximum(-np.einsum("ij,ij->i",forces,raw_normals),0.)
        # Newton maps each mjw contact to the same shape-slot index, preserving the original
        # solver midpoint distance. Read that signed distance before another integration.
        dist=self.solver.mjw_data.contact.dist.numpy()[:n]
        hand1=np.isin(first,self.hand_ids);hand2=np.isin(second,self.hand_ids)
        obj1=np.isin(first,self.object_ids);obj2=np.isin(second,self.object_ids)
        table1=np.isin(first,self.support_ids);table2=np.isin(second,self.support_ids)
        hc=(hand1&obj2)|(hand2&obj1)
        tc=(table1&obj2)|(table2&obj1)
        sc=hand1&hand2
        relevant=hc|tc|sc
        self.contact_rows.append(np.column_stack((first[relevant],second[relevant],dist[relevant],
                                 raw_normals[relevant],forces[relevant],normal_load[relevant])))
        self.contact_offsets.append(self.contact_offsets[-1]+int(relevant.sum()))
        q=s1.joint_q.numpy()[list(self.hand.finger_q_ids)]
        excursion=max(0.,float(np.max(np.maximum(self.lo-q,q-self.hi))))
        cp=max(0.,float(-np.min(dist[hc|tc],initial=0.)))
        sp=max(0.,float(-np.min(dist[sc],initial=0.)))
        self.samples.append([normal_load[hc].sum(),normal_load[tc].sum(),int(hc.sum()),cp,sp,excursion])
        if (n>=contacts.rigid_contact_max or not np.isfinite(self.samples[-1]).all()):
            raise ValueError(f"Invalid or saturated contact extraction {n}/{contacts.rigid_contact_max}")
        # These six scalars use the actual native load and geometry across every substep.
        # Unloaded manifold points still count for the terminal release criterion.

    def end_frame(self):
        self.endpoint_states.append({k:getattr(self.env.state_0,k).numpy().copy() for k in STATES})

    def close(self):
        self.solver.step = self.original_step

    def materialize(self):
        out={f"command_{k}":np.stack(v) for k,v in self.commands.items()}
        out.update({k:np.stack([s[k] for s in self.endpoint_states]) for k in STATES})
        out["substep_contacts"]=np.array(self.samples,dtype=float)
        out["contact_offsets"]=np.asarray(self.contact_offsets)
        out["contact_rows"]=np.concatenate(self.contact_rows)
        out["contact_fields"]=np.array(["shape0","shape1","solver_distance_m","nx","ny","nz",
                                       "force_on_shape0_x_N","force_on_shape0_y_N","force_on_shape0_z_N","normal_load_N"])
        out["hand_shape_ids"]=self.hand_ids
        out["object_shape_ids"]=self.object_ids
        out["support_shape_ids"]=self.support_ids
        return out

    def independent_replay(self,trace):
        env=self.env;model=env.model;cfg=env.config.sim
        solver=build_solver(model,cfg)
        contacts=build_force_contacts(model,solver,cfg,device=env.device)
        s0,s1=model.state(),model.state()
        for k in STATES: getattr(s0,k).assign(self.initial_state[k])
        control=model.control()
        err=0.
        sub=cfg.substeps
        for i in range(len(trace["command_joint_f"])):
            s0.clear_forces()
            for name in COMMANDS:
                getattr(control,name).assign(trace["command_"+name][i])
            solver.step(s0,s1,control,contacts,env.sim_dt)
            if env.joint_actuator_limits is not None:
                env.joint_actuator_limits.enforce_velocity(s1)
                newton.eval_fk(model,s1.joint_q,s1.joint_qd,s1)
            s0,s1=s1,s0
            if (i+1)%sub==0:
                f=(i+1)//sub
                for k in STATES:
                    err=max(err,float(np.max(np.abs(getattr(s0,k).numpy()-trace[k][f]))))
        return err


def world_tips(body_poses, hand):
    points=[]
    for frame in hand.fingertip_frames:
        body=body_poses[:,frame.body_id]
        offsets=np.tile(np.asarray(frame.body_to_frame_pos,dtype=float),(len(body),1))
        points.append(body[:,:3]+Rotation.from_quat(body[:,3:]).apply(offsets))
    return np.stack(points,axis=1)


def repeat_control_trajectory(fresh_env, original, original_stop, source_frame_count):
    """Re-execute identical low-level commands from a new explicit source reset."""
    env=fresh_env()
    if env.config.auto_reset or np.any(env.object_control_scale.numpy()) or int(env.timestep.numpy()[0])!=0:
        raise ValueError("Physical repeat requires an unassisted, non-resetting frame-zero environment")
    recorder=SolverRecorder(env)
    for k in STATES:
        if not np.array_equal(recorder.initial_state[k],original[k][0]):
            raise ValueError(f"Fresh scene did not reconstruct identical initial {k}")
    sub=env.config.sim.substeps
    intended=len(original["command_joint_f"])//sub
    stop=dict(failed=bool(original_stop["failed"] or intended!=source_frame_count-1),
              truncated=False,reason="end of supplied commands",termination_causes=[])
    for i in range(intended):
        env.timestep.assign(np.array([i],dtype=np.int32))
        env.episode_step.assign(np.array([i],dtype=np.int32))
        env.command.gather(env.timestep)
        for s in range(i*sub,(i+1)*sub):
            s0,s1=env.state_0,env.state_1
            s0.clear_forces()
            for k in COMMANDS:
                getattr(env.control,k).assign(original["command_"+k][s])
            env.solver.step(s0,s1,env.control,env.contacts,env.sim_dt)
            if env.joint_actuator_limits is not None:
                env.joint_actuator_limits.enforce_velocity(s1)
                newton.eval_fk(env.model,s1.joint_q,s1.joint_qd,s1)
            env.state_0,env.state_1=s1,s0
        recorder.end_frame()
        env.termination.evaluate(env.state_0.body_q,env.timestep)
        failed=bool(env.termination.terminated.numpy()[0]); truncated=bool(env.termination.truncated.numpy()[0])
        if failed or truncated:
            stop=dict(failed=failed or i+1!=source_frame_count-1,truncated=truncated,reason="native episode ended",
                      termination_causes=[name for name,arr in zip(env.termination.cause_names,env.termination.cause_arrays)
                                          if arr.numpy()[0]])
            break
    recorder.close()
    trace=recorder.materialize()
    for k in COMMANDS:
        a=trace["command_"+k]
        if not np.array_equal(a,original["command_"+k][:len(a)]):
            raise ValueError(f"Repeat changed stored actuator channel {k}")
    return env,recorder,trace,stop


def pairwise_trajectory_differences(a,b,object_id):
    n=min(len(a["body_q"]),len(b["body_q"]))
    p=a["body_q"][:n,object_id];q=b["body_q"][:n,object_id]
    pos=np.linalg.norm(p[:,:3]-q[:,:3],axis=1)*1000
    rot=np.rad2deg((Rotation.from_quat(p[:,3:]).inv()*Rotation.from_quat(q[:,3:])).magnitude())
    return dict(common_frames=n,position_rms_mm=float(np.sqrt(np.mean(pos**2))),
                position_max_mm=float(pos.max()),rotation_rms_deg=float(np.sqrt(np.mean(rot**2))),
                rotation_max_deg=float(rot.max()),q_max=float(np.max(abs(a["joint_q"][:n]-b["joint_q"][:n]))),
                qd_max=float(np.max(abs(a["joint_qd"][:n]-b["joint_qd"][:n]))))
