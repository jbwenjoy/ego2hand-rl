"""Verify table support, free fall and true generalized equation residual under FlashCHORD."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from probe_flash_wuji_replay import rebuilt


def check(parquet,checkpoint,condition,height,duration):
    start=time.perf_counter()
    env=rebuilt(parquet,checkpoint,condition)
    solver=env.solver
    m=solver.mj_model;d=solver.mjw_data
    ob=env.scene.objects[0];bid=ob.bodies[0].body_id
    sims=[b for b in range(m.nbody) if f'{parquet.parents[1].name.split("_")[-2]}_' in m.body(b).name and m.body_jntnum[b]>0]
    if len(sims)!=1: raise ValueError(f'Cannot locate single object in solver: {sims}')
    body=sims[0];dof=int(m.jnt_dofadr[m.body_jntadr[body]])
    mass=float(m.body_mass[body]);weight=mass*9.81;dt=env.sim_dt
    readset=env.reference if hasattr(env,'reference') else None
    obj=env.state_0.body_q.numpy()[bid]
    # Active collision mesh support, in the actual Newton scene and original top plane z=0.
    bottoms=[]
    for shape in ob.shapes.ids():
        if not int(env.model.shape_flags.numpy()[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES):continue
        geom=env.model.shape_source[shape];verts=np.array(geom.vertices)*env.model.shape_scale.numpy()[shape]
        xf=env.model.shape_transform.numpy()[shape]
        local=Rotation.from_quat(xf[3:]).apply(verts)+xf[:3]
        world=Rotation.from_quat(obj[3:]).apply(local)+obj[:3]
        bottoms.append(float(world[:,2].min()))
    q=env.state_0.joint_q.numpy();q[list(ob.root.free_q_ids)[2]]+=height-min(bottoms)
    env.state_0.joint_q.assign(q);env.state_0.joint_qd.zero_()
    newton.eval_fk(env.model,env.state_0.joint_q,env.state_0.joint_qd,env.state_0)
    # Move the hand away in world x in both the commanded and actual coordinates.
    # This is a passive object validation fixture only. The robot exerts no object wrench.
    s0,s1=env.state_0,env.state_1
    q=s0.joint_q.numpy();h=env.scene.layout.hand("right");q[h.wrist_pos_q_ids[0]]+=.8
    s0.joint_q.assign(q);newton.eval_fk(env.model,s0.joint_q,s0.joint_qd,s0)
    targets=np.array(env.scene.robot_reference.joint_target[0])
    env.control.joint_target_pos.zero_();buf=env.control.joint_target_pos.numpy()
    buf[:len(targets)]=targets;buf[h.wrist_pos_dof_ids[0]]+=.8
    env.control.joint_target_pos.assign(buf)
    initial=float(s0.body_q.numpy()[bid,2])
    rows=[];dense=np.empty((m.nv,m.nv))
    table_ids=list(env.scene.support.shapes.ids());objids=list(ob.shapes.ids());hids=[k for hg in h.link_geometry for k in hg.shape_ids]
    for i in range(int(round(duration/dt))):
        s0.clear_forces();env.control.joint_f.zero_()
        # Only rotational hand holding is omitted. No controller acts on a free Cartesian object.
        solver.step(s0,s1,env.control,env.contacts,dt)
        solver.update_contacts(env.contacts,s1)
        ct=env.contacts;n=int(ct.rigid_contact_count.numpy()[0])
        x,y=ct.rigid_contact_shape0.numpy()[:n],ct.rigid_contact_shape1.numpy()[:n]
        pair=(np.isin(x,objids)&np.isin(y,table_ids))|(np.isin(y,objids)&np.isin(x,table_ids))
        hand=(np.isin(x,objids)&np.isin(y,hids))|(np.isin(y,objids)&np.isin(x,hids))
        if hand.any():raise RuntimeError("Passive experiment invalidated by hand contact")
        f=ct.force.numpy()[:n,:3];norm=ct.rigid_contact_normal.numpy()[:n]
        load=float(np.maximum(-np.einsum('ij,ij->i',f[pair],norm[pair]),0).sum())
        accel=d.qacc.numpy()[0];mat=d.qM.numpy()[0]
        if mat.shape==(m.nv,m.nv):dense=mat
        else:mujoco.mj_fullM(m,dense,mat.ravel().astype(float))
        ma=(dense@accel)[dof:dof+3]
        qsum=(d.qfrc_smooth.numpy()+d.qfrc_constraint.numpy())[0,dof:dof+3]
        rows.append([dt*(i+1),s1.body_q.numpy()[bid,2]-initial,load,float(np.linalg.norm(ma-qsum)),d.solver_niter.numpy()[0]])
        s0,s1=s1,s0
    v=np.array(rows);late=v[:,0]>duration-.25
    active=np.flatnonzero(v[:,2]>.01*weight)
    expected=np.sqrt(2*height/9.81)
    touch=float(v[active[0],0]) if active.size else None
    flight=(v[:,0]<expected-.005)
    drift=float(np.max(np.abs(v[flight,1]+.5*9.81*v[flight,0]**2))*1000) if flight.any() else None
    settled=float(v[late,2].mean()/weight)
    residual=float(np.sqrt(np.mean(v[:,3]**2)))
    gates=dict(finite=bool(np.isfinite(v).all()),static_load=abs(settled-1)<=.05,
               generalized_force_residual=residual<=.03*weight,
               freefall=(height==0 or touch is not None and abs(touch-expected)<=.006 and drift<1.))
    return dict(artifact_role="passive geometry fixture",scheme="movable_dof_mean_v1" if condition else "upstream default",
                conditioned=condition,drop_m=height,sim_dt=dt,mass_kg=mass,stationary_load_bodyweights=settled,
                full_equation_xyz_rms_N=residual,peak_equation_xyz_N=float(v[:,3].max()),
                time_to_contact_s=touch,ballistic_expected_s=expected,freefall_peak_error_mm=drift,
                meaninertia=float(solver.mjw_model.stat.meaninertia.numpy()[0]),
                solver_tolerance=float(solver.mjw_model.opt.tolerance.numpy()[0]),
                median_iterations=float(np.median(v[:,4])),wall_s=time.perf_counter()-start,
                gates=gates,passed=all(gates.values())),v


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','parquet','out','warp-cache'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--duration',type=float,default=.8)
    p.add_argument('--corrected-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device(a.device)
    rows=[];arrays={}
    for condition in ([True] if a.corrected_only else [False,True]):
        for drop in [0.,.05]:
            row,v=check(a.parquet,a.checkpoint,condition,drop,a.duration);rows.append(row)
            arrays[f'condition{int(condition)}_drop{int(1000*drop)}mm']=v
            print(row,flush=True)
    np.savez_compressed(a.out/'states.npz',**arrays)
    (a.out/'report.json').write_text(json.dumps(dict(device=a.device,fields=['t_s','dz_m','table_normal_load_N','force_residual_xyz_norm_N','iterations'],rows=rows),indent=2,allow_nan=False))
