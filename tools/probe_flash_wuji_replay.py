"""Short reproducibility and solver-force diagnostic for a saved Wuji actor command trace."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import mujoco
import newton
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.data import load_reference
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config

from supermimic_research.runtime.flash_quality import COMMANDS,STATES,SolverRecorder


def rebuilt(parquet,checkpoint,condition=False):
    cfg=checkpoint_config(read_checkpoint_metadata(checkpoint))
    term=deepcopy(cfg['env']['termination'])
    apply_evaluation_composition(cfg,world_count=1,reset_mode='explicit')
    cfg['env']['termination']=term
    if condition:
        cfg['env']['sim']['_target_']='supermimic_research.runtime.flash_solver.WujiSimConfig'
    cfg=OmegaConf.create(cfg)
    ec=instantiate_typed(cfg.env,RLEnvConfig)
    setup=setup_scene(parquet=parquet,control_fps=30.,motion_speed=1.,source_frame_playback=True,
                      embodiment=instantiate_typed(cfg.embodiment,Embodiment),
                      collision=instantiate_typed(cfg.collision,CollisionPolicy),world_count=1,
                      include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
                      object_scale_min=cfg.scene.object_scale_min,object_scale_max=cfg.scene.object_scale_max,
                      contact_friction=cfg.scene.contact_friction,object_free_joint_damping=cfg.scene.object_free_joint_damping)
    env=RLEnv(setup.scene,setup.reference,ec)
    env.reset(frame_id=0)
    return env


def experiment(kind,reference,checkpoint,source_commands,control_frames,override_tolerance,condition):
    env=rebuilt(reference,checkpoint,condition);m=env.solver.mj_model;model=env.model;solver=env.solver
    # Identify the carton by the published binding and its unique free-joint child.
    data=solver.mjw_data; mgbody=np.asarray(m.geom_bodyid)
    if override_tolerance is not None:
        solver.mjw_model.opt.tolerance.fill_(override_tolerance)
    native_free=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    body_token=reference.parents[1].name.split('_')[-2]
    object_j=next((j for j in native_free if body_token in str(m.body(int(m.jnt_bodyid[j])).name)),
                  native_free[0] if len(native_free)==1 else None)
    if object_j is None:
        raise ValueError(f"Cannot locate carton free joint: {[m.body(int(m.jnt_bodyid[j])).name for j in native_free]}")
    bi=int(m.jnt_bodyid[object_j]);dof=int(m.jnt_dofadr[object_j]);body=env.scene.objects[0].bodies[0].body_id
    transforms=[];vel=[];inputs={k:[] for k in COMMANDS};sensors=[];initial={}
    for k in ('qacc_warmstart','qacc','time','qpos','qvel'):
        arr=getattr(data,k,None)
        if arr is not None: initial[k]=arr.numpy().copy()
    for k in STATES: initial['input_'+k]=getattr(env.state_0,k).numpy().copy()
    initial['meaninertia']=solver.mjw_model.stat.meaninertia.numpy().copy()
    initial['tolerance']=solver.mjw_model.opt.tolerance.numpy().copy()
    initial['ls_tolerance']=solver.mjw_model.opt.ls_tolerance.numpy().copy()
    initial['ls_iterations']=np.array([solver.mjw_model.opt.ls_iterations])
    initial['max_iterations']=np.array([solver.mjw_model.opt.iterations])
    scratch=mujoco.MjData(m); mujoco.mj_forward(m,scratch)
    M0=np.empty((m.nv,m.nv)); mujoco.mj_fullM(m,M0,scratch.qM)
    support_dof=np.asarray(env.scene.support.free_dof_ids,dtype=int)
    active=np.ones(m.nv,dtype=bool);active[support_dof]=False
    initial['mass_matrix_diagonal0']=M0.diagonal().copy()
    initial['original_dynamic_meaninertia']=np.array([np.diag(M0)[active].mean()])
    initial['dynamic_sum_divided_by_all_dofs']=np.array([np.diag(M0)[active].sum()/m.nv])
    orig=solver.step
    def capture(s0,s1,c,contacts,dt):
        before=data.qvel.numpy()[0,dof:dof+3].copy()
        for k in COMMANDS:inputs[k].append(getattr(c,k).numpy().copy())
        fbody=s0.body_f.numpy()
        orig(s0,s1,c,contacts,dt)
        solver.update_contacts(contacts,s1)
        count=int(contacts.rigid_contact_count.numpy()[0])
        normal=contacts.rigid_contact_normal.numpy()[:count]
        force=contacts.force.numpy()[:count,:3]
        norm=-np.einsum('ij,ij->i',normal,force)
        s0id=contacts.rigid_contact_shape0.numpy()[:count];s1id=contacts.rigid_contact_shape1.numpy()[:count]
        objsg=set(env.scene.objects[0].shapes.ids())
        obj0=np.isin(s0id,list(objsg));obj1=np.isin(s1id,list(objsg))
        net=force[obj0].sum(axis=0)-force[obj1].sum(axis=0)
        adr=data.contact.efc_address.numpy()[:count]
        dims=data.contact.dim.numpy()[:count]
        ef=data.efc.force.numpy()[0]
        summed=[]
        for index,(base,dim) in enumerate(zip(adr[:,0],dims)):
            if base<0 or not (obj0[index] or obj1[index]): continue
            nrow=1 if dim==1 else 2*(int(dim)-1)
            summed.append(float(np.sum(ef[base:base+nrow])))
        after=data.qvel.numpy()[0,dof:dof+3]
        qc=data.qfrc_constraint.numpy()[0,dof:dof+3]
        other=[float(getattr(data,name).numpy()[0,dof+2]) for name in
               ("qfrc_smooth","qfrc_applied","qfrc_passive","qfrc_bias","qfrc_actuator")]
        qM=data.qM.numpy()[0]
        if qM.shape==(m.nv,m.nv):
            M=qM
        else:
            M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,M,qM.reshape(-1).astype(float))
        ma=(M@data.qacc.numpy()[0])[dof:dof+3]
        sensors.append(np.r_[count,float(norm[obj0|obj1].sum()),sum(summed),net,qc,(after-before)/dt,
                             np.max(abs(fbody)),m.body_mass[bi],data.qacc.numpy()[0,dof:dof+3],other,
                             m.dof_armature[dof:dof+3],data.solver_niter.numpy()[0],ma])
        transforms.append(s1.joint_q.numpy().copy());vel.append(s1.joint_qd.numpy().copy())
    if kind=='actor_actions':
        solver.step=capture
        for i in range(control_frames):
            env.step(wp.array(source_commands['actor_action'][i],dtype=wp.float32,device=env.device))
        solver.step=orig
    else:
        for i in range(control_frames*env.config.sim.substeps):
            s0,s1=env.state_0,env.state_1
            s0.clear_forces()
            for k in COMMANDS:getattr(env.control,k).assign(source_commands['command_'+k][i])
            capture(s0,s1,env.control,env.contacts,env.sim_dt)
            if env.joint_actuator_limits is not None:
                env.joint_actuator_limits.enforce_velocity(s1)
                newton.eval_fk(model,s1.joint_q,s1.joint_qd,s1)
            env.state_0,env.state_1=s1,s0
    return dict(q=np.stack(transforms),v=np.stack(vel),sensor=np.stack(sensors),
                **{f'initial_{k}':v for k,v in initial.items()},**{f'command_{k}':np.stack(v) for k,v in inputs.items()})


def diff(a,b):
    output={}
    for k in ('q','v','command_joint_target_pos','command_joint_target_vel','command_joint_f'):
        delta=np.max(abs(a[k]-b[k]),axis=1)
        output[k]=dict(first_bit_difference=int(np.flatnonzero(delta>0)[0]) if (delta>0).any() else None,
                       first_over_1e_6=int(np.flatnonzero(delta>1e-6)[0]) if (delta>1e-6).any() else None,
                       max=float(delta.max()),first20=delta[:20].tolist())
    output['initial']={k:float(np.max(abs(a[k]-b[k]))) for k in a if k.startswith('initial_')}
    return output


def run(a):
    a.out.mkdir(exist_ok=False,parents=True)
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device(a.device)
    with np.load(a.trace,allow_pickle=False) as arch:
        replay={k:arch[k] for k in ['actor_action',*(f'command_{field}' for field in COMMANDS)]}
    first=experiment('actor_actions',a.parquet,a.checkpoint,replay,a.frames,a.tolerance,a.condition)
    second=experiment('solver_commands',a.parquet,a.checkpoint,first,a.frames,a.tolerance,a.condition)
    third=experiment('solver_commands',a.parquet,a.checkpoint,first,a.frames,a.tolerance,a.condition)
    for name,arr in [('env',first),('solver_a',second),('solver_b',third)]:np.savez_compressed(a.out/f'{name}.npz',**arr)
    s=first['sensor'];contact=s[:,0]>0
    report=dict(frames=a.frames,substeps=len(s),env_vs_replay=diff(first,second),replay_vs_replay=diff(second,third),
                sensor_fields=['count','sum_normal','summed_efc_pyramids','net_contact_x','net_contact_y','net_contact_z',
                               'qfrc_constraint_x','qfrc_constraint_y','qfrc_constraint_z','fd_dv_x','fd_dv_y','fd_dv_z','external_force_max','mj_mass',
                               'mj_qacc_x','mj_qacc_y','mj_qacc_z','qfrc_smooth_z','qfrc_applied_z','qfrc_passive_z',
                               'qfrc_bias_z','qfrc_actuator_z','armature_x','armature_y','armature_z','solver_niter','M_times_a_x','M_times_a_y','M_times_a_z'],
                contact_proof=dict(mass=float(s[0,13]),max_native_minus_efc_normal=float(np.max(abs(s[:,1]-s[:,2]))),
                                   max_force_projection_vs_constraint_z=float(np.max(abs(s[:,5]-s[:,8]))),
                                   mean_reported_vertical_load_N=float(s[:,5].mean()),mean_constraint_vertical_N=float(s[:,8].mean()),
                                   mean_fd_dv_z=float(s[:,11].mean()),mean_qacc_z=float(s[:,16].mean()),external_force_peak=float(s[:,12].max()),
                                   mean_smooth_z=float(s[:,17].mean()),mean_effective_vertical_force=float((s[:,8]+s[:,17]).mean())))
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('DIAGNOSED',report['env_vs_replay']['q']['first_over_1e_6'],report['replay_vs_replay']['q']['first_over_1e_6'],report['contact_proof'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('parquet','checkpoint','trace','out','warp-cache'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--frames',type=int,default=6)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--tolerance',type=float,default=None)
    p.add_argument('--condition',action='store_true')
    run(p.parse_args())
