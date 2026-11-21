"""Inspect Wuji collision geometry and effective joint drives after the Newton solver compiles them."""
import argparse
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.runtime.replay import ReplayRunner, ReplayConfig
from flash_chord.runtime.sim import SimConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from supermimic_research.runtime.flash_wuji import WujiHand2,WujiActionConfig,_limit_finger_commands


def joint_id(model,name):
    ids=[j for j in range(model.njnt) if model.joint(j).name==name or model.joint(j).name.endswith("_"+name)]
    if len(ids)!=1:raise ValueError((name,ids))
    return ids[0]


def is_candidate(model,g,h):
    b1,b2=map(int,model.geom_bodyid[[g,h]])
    sig=(min(b1,b2)<<16)+max(b1,b2)
    if sig in model.exclude_signature: return False
    # MuJoCo's parent filter uses the welded parent of the welded body.
    a,b=map(int,model.body_weldid[[b1,b2]])
    if a==b:return False
    ap=model.body_weldid[model.body_parentid[a]];bp=model.body_weldid[model.body_parentid[b]]
    if a and b and not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT) and (a==bp or b==ap):return False
    return bool(model.geom_contype[g]&model.geom_conaffinity[h] or model.geom_contype[h]&model.geom_conaffinity[g])


def projection(model,data,g,dirs):
    pos,rot=data.geom_xpos[g],data.geom_xmat[g].reshape(3,3)
    if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE:
        return dirs@pos + model.geom_size[g,0] + np.abs(dirs@rot[:,2])*model.geom_size[g,1]
    mesh_id=model.geom_dataid[g];start=model.mesh_vertadr[mesh_id];n=model.mesh_vertnum[mesh_id]
    world=model.mesh_vert[start:start+n]@rot.T+pos
    return (world@dirs.T).max(axis=0)


def main(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device("cuda:0")
    wu=WujiHand2(a.asset);meta=wu.meta
    scene=setup_scene(parquet=a.parquet,control_fps=30,motion_speed=1,source_frame_playback=True,embodiment=wu,
            collision=CollisionPolicy(ground=False,robot_scope="all",hand_self_collision=True,robot_object_collision=True,robot_support_collision=True),
            world_count=1,include_support=True,decompose_objects=False,contact_friction=.7)
    runner=ReplayRunner(scene.scene,scene.reference,ReplayConfig(sim=SimConfig(fps=30,substeps=40,njmax=512,nconmax=256),
                        action=WujiActionConfig(),voc_scale=0),device="cuda:0")
    runner.reset(0)
    solver=runner.solver.mj_model
    native=mujoco.MjModel.from_xml_path(str(a.asset/"wuji_right.xml"))
    body_masses={};geom_ids={}
    for name,g in meta["geometry"].items():
        ids=[i for i in range(solver.ngeom) if re.search(re.escape(name)+r"_\d+$",solver.geom(i).name)]
        if len(ids)!=1: raise ValueError(f"Cannot match {name}: {ids}")
        orig=native.geom(name).id
        if solver.geom_type[ids[0]]!=native.geom_type[orig]: raise ValueError(name)
        geom_ids[name]=(orig,ids[0])
    for name in meta["native_body_masses"]:
        match=[i for i in range(solver.nbody) if solver.body(i).name.endswith(name)]
        if len(match)!=1:raise ValueError((name,match))
        body_masses[name]=float(abs(solver.body_mass[match[0]]-native.body_mass[native.body(name).id]))
    joint_err={}
    for name,c in meta["joints"].items():
        j=joint_id(solver,name);v=solver.jnt_dofadr[j]
        actuators=np.flatnonzero(solver.actuator_trnid[:,0]==j)
        assert len(actuators)==2
        pos=actuators[np.argmax(-solver.actuator_biasprm[actuators,1])]
        vel=actuators[np.argmax(-solver.actuator_biasprm[actuators,2])]
        joint_err[name]=dict(range=float(np.max(abs(solver.jnt_range[j]-c["range"]))),
              kp=abs(float(solver.actuator_gainprm[pos,0])-c["kp"]),kd=abs(float(solver.actuator_gainprm[vel,0])-c["kd"]),
              friction=abs(float(solver.dof_frictionloss[v])-c["friction"]),armature=abs(float(solver.dof_armature[v])-c["armature"]),
              damping=abs(float(solver.dof_damping[v])-c["damping"]),
              effort=float(np.max(abs(solver.jnt_actfrcrange[j]-[-c["effort"],c["effort"]]))))
        assert solver.jnt_actfrclimited[j]
    differences=[]
    for i,(name1,(g1,n1)) in enumerate(geom_ids.items()):
        for name2,(g2,n2) in list(geom_ids.items())[i+1:]:
            x=is_candidate(native,g1,g2);y=is_candidate(solver,n1,n2)
            if x!=y:differences.append([name1,name2,x,y])
    diff_geom={}
    unit=np.random.default_rng(7).normal(size=(96,3));unit/=np.linalg.norm(unit,axis=1)[:,None]
    dirs=np.vstack([np.eye(3),-np.eye(3),unit])
    native_d=mujoco.MjData(native);sol_d=mujoco.MjData(solver)
    frames=[0,scene.reference.num_frames//2,scene.reference.num_frames-1]
    names=list(meta['joints'])
    refnames=scene.reference.robot_joint_names()
    for frame in frames:
        for model,data in ((solver,sol_d),(native,native_d)):
            for k,name in enumerate(('right_wrist_x','right_wrist_y','right_wrist_z')):
                data.qpos[model.jnt_qposadr[joint_id(model,name)]]=scene.reference.robot_root_pos_w()[frame,k]
            k=model.jnt_qposadr[joint_id(model,'right_wrist_rotvec')]
            data.qpos[k:k+4]=scene.reference.robot_root_quat_w()[frame]
            for name in names:
                data.qpos[model.jnt_qposadr[joint_id(model,name)]]=scene.reference.robot_joint_pos()[frame,refnames.index(name)]
            mujoco.mj_forward(model,data)
        for name,(g,h) in geom_ids.items():
            diff=float(np.max(abs(projection(native,native_d,g,dirs)-projection(solver,sol_d,h,dirs))))
            diff_geom[name]=max(diff_geom.get(name,0),diff)
    action=runner.action
    ids=scene.scene.layout.hand("right").finger_dof_ids
    lower=np.array([meta['joints'][n]['ctrl_range'][0] for n in names]);upper=np.array([meta['joints'][n]['ctrl_range'][1] for n in names])
    # Check extreme residuals through the normal per-step processing path, and additionally
    # force both out-of-range sides through the actual GPU clamp to prove its boundaries.
    max_excess=0.; clamp_err=0.
    for sign in (-1.,1.):
        inp=wp.full(shape=runner.zero_action.shape,value=sign*1000,dtype=wp.float32,device='cuda:0')
        for frame in frames:
            runner.timestep.assign(np.array([frame],np.int32))
            action.process(inp,runner.timestep,runner.state_0); action.prepare_control(runner.control)
            target=action.joint_target.numpy()[list(ids)]
            max_excess=max(max_excess,float(np.max(target-upper)),float(np.max(lower-target)))
        temp=wp.full(shape=action.joint_target.shape,value=sign*1000,dtype=wp.float32,device='cuda:0')
        wp.launch(_limit_finger_commands,dim=(1,20),inputs=[temp,action.finger_indices,action.model.joint_limit_lower,
                    action.model.joint_limit_upper,action.model.joint_dof_count],device='cuda:0')
        expected=lower if sign<0 else upper
        clamp_err=max(clamp_err,float(np.max(abs(temp.numpy()[list(ids)]-expected))))
    result=dict(native_geometries=len(geom_ids),sampled_directions=len(dirs),frames=frames,
                geom_support_max_difference_m=diff_geom,collision_pair_disagreements=differences,
                max_body_mass_deviation_kg=max(body_masses.values()),finger_joint_errors=joint_err,
                max_policy_target_excess_rad=max_excess,oversized_input_clamp_limit_error_rad=clamp_err,
                source_table_object_mu=.6,flashchord_global_mu=.7,native_only_geom_support_is_not_rollout_parity=True)
    print('AUDIT',max(diff_geom.values()),len(differences),max(body_masses.values()),max_excess,clamp_err,flush=True)
    (out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    assert max_excess<2e-6 and clamp_err<2e-6


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("asset","parquet","out","warp-cache"):p.add_argument("--"+key,type=Path,required=True)
    main(p.parse_args())
