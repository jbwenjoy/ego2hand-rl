"""CPU diagnostic of actual FlashCHORD hand/carton contact on measured native grasp snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_contacts(args):
    import mujoco
    native=mujoco.MjModel.from_binary_path(str(args.mjb))
    d=mujoco.MjData(native)
    with np.load(args.trace) as tr:
        results={}
        for f in args.frames:
            d.qpos[:]=tr["sim_qpos"][f];d.qvel[:]=0
            d.ctrl[:]=tr["target_finger_q"][f]
            mujoco.mj_forward(native,d)
            results[f]={"pairs":[(native.geom(c.geom1).name,native.geom(c.geom2).name,float(c.dist)) for c in d.contact
                         if "carton_collision" in (native.geom(c.geom1).name,native.geom(c.geom2).name) and not (native.geom(c.geom1).name=="table" or native.geom(c.geom2).name=="table")]}
    return results


def pair_mask(solver,scene):
    m=solver.mj_model
    role=solver.mjc_geom_to_newton_shape.numpy()[0]
    obj=set(scene.objects[0].shapes.ids())
    hand={i for link in scene.layout.hand("right").link_geometry for i in link.shape_ids}
    fingers={int(s):i for i,s in enumerate(role) if s in hand}
    objs=[i for i,s in enumerate(role) if s in obj and (m.geom_contype[i] or m.geom_conaffinity[i])]
    exclusions=set(map(int,m.exclude_signature)); pairs=set(zip(m.pair_geom1.tolist(),m.pair_geom2.tolist()))
    tests={}
    for sid,gi in fingers.items():
        name=scene.model.shape_label[sid].split("/")[-1]
        rows=[]
        for gj in objs:
            i,j=sorted([gi,gj]);b1,b2=sorted([int(m.geom_bodyid[gi]),int(m.geom_bodyid[gj])])
            explicit=(i,j) in pairs
            excluded=((b1<<16)+b2) in exclusions
            mask=bool((m.geom_contype[i]&m.geom_conaffinity[j]) or (m.geom_contype[j]&m.geom_conaffinity[i]))
            rows.append(dict(object_geom=gj,mask=mask,body_exclusion=excluded,explicit_pair=explicit,
                             eligible=bool(explicit or mask and not excluded)))
        tests[name]=rows
    return tests


def test_snapshot(args,f,shift,native):
    import newton
    from probe_flash_wuji_replay import rebuilt
    env=rebuilt(args.parquet,args.checkpoint,True)
    assert getattr(env.solver,"wuji_conditioning",None)
    h=env.scene.layout.hand("right");g=env.scene.objects[0];oid=g.bodies[0].body_id
    wm=native["sim_wrist"][f];cm=native["world_from_carton_sim"][f]
    q=env.state_0.joint_q.numpy()
    q[list(h.wrist_pos_q_ids)]=wm[:3,3]+shift
    q[h.wrist_orient_q_id:h.wrist_orient_q_id+4]=Rotation.from_matrix(wm[:3,:3]).as_quat()
    q[list(h.finger_q_ids)]=native["sim_qpos"][f,7:27]
    q[list(g.root.free_q_ids)]=np.r_[cm[:3,3],Rotation.from_matrix(cm[:3,:3]).as_quat()]
    s0,s1=env.state_0,env.state_1
    s0.joint_q.assign(q);s0.joint_qd.zero_();newton.eval_fk(env.model,s0.joint_q,s0.joint_qd,s0)
    reference=env.scene.robot_reference.joint_target[f].copy()
    reference[list(h.wrist_pos_dof_ids)]=wm[:3,3]+shift
    reference[list(h.wrist_orient_dof_ids)]=Rotation.from_matrix(wm[:3,:3]).as_rotvec()
    reference[list(h.finger_dof_ids)]=native["target_finger_q"][f]
    driver=env.control.joint_target_pos.numpy().copy()
    driver[:len(reference)]=reference
    env.control.joint_target_pos.assign(driver)
    env.control.joint_target_vel.zero_()
    env.control.joint_f.zero_()
    hand={i for link in h.link_geometry for i in link.shape_ids}
    carton=set(g.shapes.ids())
    rows=[];max_err=0.
    for sub in range(args.substeps):
        s0.clear_forces()
        env.action.wrist.apply_wrist_effort(s0,env.control,env.control.joint_target_pos)
        if np.any(s0.body_f.numpy()[oid]) or np.any(env.control.joint_f.numpy()[list(g.root.free_dof_ids)]):
            raise ValueError("probe must never push the object")
        env.solver.step(s0,s1,env.control,env.contacts,env.sim_dt)
        env.solver.update_contacts(env.contacts,s1)
        c=env.contacts;n=int(c.rigid_contact_count.numpy()[0]);sA=c.rigid_contact_shape0.numpy()[:n];sB=c.rigid_contact_shape1.numpy()[:n]
        load=c.force.numpy()[:n,:3];norm=c.rigid_contact_normal.numpy()[:n];dist=env.solver.mjw_data.contact.dist.numpy()[:n]
        records=[]
        for k,(a,b) in enumerate(zip(sA,sB)):
            if not ((a in hand and b in carton) or (b in hand and a in carton)): continue
            sid=a if a in hand else b
            xyz=norm[k] if a in hand else -norm[k]
            records.append(dict(link=env.model.shape_label[sid].split("/")[-1],
                   normal_force_on_object_N=float(max(0,-np.dot(load[k],norm[k]))),
                   normal_toward_object_world=xyz.tolist(),separation_mm=float(1000*dist[k])))
        max_err=max(max_err,float(np.max(abs(s1.body_q.numpy()[oid,:3]-cm[:3,3]))))
        rows.append(records)
        s0,s1=s1,s0
    hands=[[r for r in rs if r["normal_force_on_object_N"]>.01] for rs in rows]
    def opposed(rs):
        a=[r for r in rs if "finger1_" in r["link"]]
        b=[r for r in rs if any(t in r["link"] for t in ("finger2_","finger3_","finger4_","finger5_"))]
        return any(np.dot(x["normal_toward_object_world"],y["normal_toward_object_world"])<-.5 for x in a for y in b)
    return dict(frame=f,shift_world_m=np.asarray(shift).tolist(),loaded_steps=sum(bool(r) for r in hands),
                opposed_thumb_steps=sum(opposed(r) for r in hands),maximum_single_normal_N=max([0]+[r["normal_force_on_object_N"] for rs in rows for r in rs]),
                loaded_geometries=sorted({r["link"] for rs in hands for r in rs}),first_step=rows[0],records=rows,
                maximum_object_excursion_axis_m=max_err,solver_supply="corrected official MuJoCo-Warp",
                physical_fidelity_success_claim=False),pair_mask(env.solver,env.scene)


if __name__=="__main__":
    # Compile/inspect the original MJB on the same MuJoCo build that created it.
    # The Newton runtime can use the resulting small JSON across OS/MuJoCo versions.
    if sys.argv[1:2]==["native"]:
        native_args=argparse.ArgumentParser(description="Native MJB snapshot contact readout")
        for n in ("mjb","trace","out"):
            native_args.add_argument("--"+n,type=Path,required=True)
        native_args.add_argument("--frames",type=int,nargs="+",default=[30,38,45,65])
        request=native_args.parse_args(sys.argv[2:])
        contacts=native_contacts(request)
        with request.out.open("x",encoding="utf8") as writer:json.dump(contacts,writer,indent=2)
        raise SystemExit(0)
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("parquet","checkpoint","mjb","trace","out","warp-cache"): p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--frames",type=int,nargs="+",default=[30,38,45,65]);p.add_argument("--substeps",type=int,default=12)
    p.add_argument("--native-record",type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import warp as wp
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device("cpu")
    with np.load(a.trace,allow_pickle=False) as src:
        native={k:src[k] for k in ('sim_wrist','world_from_carton_sim','sim_qpos','target_finger_q')}
    results=[];settings=None
    for frame,shift in [*[(f,[0.,0.,0.]) for f in a.frames],(a.frames[-1],[.30,0.,0.])]:
        v,settings=test_snapshot(a,frame,shift,native);results.append(v)
    out=dict(scope="Controlled matched physical pose probe on development sequence only, zero initial velocities, finger commands from saved source, bounded wrist, object never aided.",
             checkpoint_sha256=digest(a.checkpoint),parquet_sha256=digest(a.parquet),native_trace_sha256=digest(a.trace),
             native_mjb_sha256=digest(a.mjb),
             source_native_zero_velocity_snapshot_contacts=json.loads(a.native_record.read_bytes()) if a.native_record else native_contacts(a),
             native_record_sha256=digest(a.native_record) if a.native_record else None,pair_filter_rows=settings,
             substeps_per_probe=a.substeps,results=results)
    (a.out/"report.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print([(v["frame"],v["shift_world_m"],v["loaded_steps"],v["opposed_thumb_steps"],v["maximum_single_normal_N"]) for v in results])
