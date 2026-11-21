"""Replay the original S1 stable-placement export and measure hand geometry against its native table."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def support_distances(model, pose, table, hand, meshes):
    """Exact lowest point along the compiled table normal, independent of active contacts."""
    normal = pose.geom_xmat[table].reshape(3,3)[:,2]
    center_gaps = (pose.geom_xpos[hand]-pose.geom_xpos[table])@normal
    local_n = pose.geom_xmat[hand].reshape(-1,3,3).transpose(0,2,1)@normal
    for i,g in enumerate(hand):
        if g in meshes:
            center_gaps[i] += (meshes[g]@local_n[i]).min()
        else:
            if model.geom_type[g] != mj.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError(f"Unsupported physical hand geom: {model.geom(g).name}")
            radius,half_length = model.geom_size[g,:2]
            center_gaps[i] -= radius+half_length*abs(local_n[i,2])
    return center_gaps


def main():
    source = ROOT/"runs/hocap_dev_release_stability_v1/export"
    receipt = json.loads((source/"report.json").read_bytes())
    archive = source/"Wuji-HOCap-carton-replay.zip"
    checksum = digest(archive.read_bytes())
    if checksum != receipt["archive"]["sha256"]:
        raise ValueError("Export differs from its verified release")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as z:
        filename = Path(tmp)/"scene.mjb"
        scene_bytes = z.read("scene.mjb")
        filename.write_bytes(scene_bytes)
        m = mj.MjModel.from_binary_path(str(filename))
        controls = z.read("controls.npz")
        with np.load(io.BytesIO(controls),allow_pickle=False) as f:
            clip = dict(f)
        evaluation = json.loads(z.read("development_evaluation.json"))
    table,obj = m.geom("table").id,m.geom("carton_collision").id
    pairs = {frozenset((int(x),int(y))) for x,y in zip(m.pair_geom1,m.pair_geom2)}
    hand = sorted(g for pair in pairs if obj in pair for g in pair if g not in (obj,table))
    if m.geom_type[table] != mj.mjtGeom.mjGEOM_PLANE or not hand:
        raise ValueError("Expected a native table plane and explicitly named hand-carton pairs")
    meshes = {}
    for g in hand:
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mesh=m.geom_dataid[g]; start=m.mesh_vertadr[mesh]
            meshes[g]=m.mesh_vert[start:start+m.mesh_vertnum[mesh]].astype(float)
    body=m.body("carton").id
    if m.nmocap or m.neq or m.nu!=20 or m.jnt_dofadr[m.joint("wrist_free").id]!=0 or m.jnt_dofadr[m.joint("carton_free").id]<6:
        raise ValueError("Model differs from the hand-only command replay layout")
    d,pose=mj.MjData(m),mj.MjData(m)
    d.qpos[:],d.qvel[:],d.ctrl[:]=clip["initial_qpos"],clip["initial_qvel"],clip["initial_ctrl"]
    mj.mj_forward(m,d);mj.mj_forward(m,d)
    plane=dict(point_m=d.geom_xpos[table].tolist(),world_normal=d.geom_xmat[table].reshape(3,3)[:,2].tolist(),
               contype=int(m.geom_contype[table]),conaffinity=int(m.geom_conaffinity[table]))
    group=[]
    for g in hand:
        a,b=sorted((int(m.geom_bodyid[g]),int(m.geom_bodyid[table])))
        group.append(dict(id=g,name=m.geom(g).name,contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),
                          explicit_hand_table_pair=frozenset((table,g)) in pairs,
                          compatible_masks=bool((m.geom_contype[g]&m.geom_conaffinity[table]) or
                                               (m.geom_contype[table]&m.geom_conaffinity[g])),
                          excluded_body_pair=bool(((a<<16)+b) in m.exclude_signature)))
    boundaries=clip["frame_end_substep"]; N=int(boundaries[-1]); dt=float(m.opt.timestep)
    if len(clip["wrist_wrench"])!=N or len(clip["finger_target"])!=N:raise ValueError("Wrong command length")
    signed=np.empty((N+1,len(hand)));native=np.full(N,np.nan); loads=np.zeros(N); counts=np.zeros(N,int)
    force=np.zeros(6); checked_errors=np.zeros(3)
    for frame,end in enumerate(boundaries):
        if frame:
            for i in range(boundaries[frame-1],end):
                pose.qpos[:]=d.qpos
                mj.mj_kinematics(m,pose)
                signed[i]=support_distances(m,pose,table,hand,meshes)
                d.ctrl[:]=clip["finger_target"][i]
                d.qfrc_applied[:6]=clip["wrist_wrench"][i]
                mj.mj_step(m,d)
                for c_idx,c in enumerate(d.contact):
                    if table in c.geom and (c.geom1 in hand or c.geom2 in hand):
                        native[i]=np.fmin(native[i],c.dist)
                        counts[i]+=1
                        mj.mj_contactForce(m,d,c_idx,force)
                        loads[i]+=force[0]
            mj.mj_forward(m,d)
        body_pose=np.eye(4)
        body_pose[:3,3]=d.xpos[body]
        body_pose[:3,:3]=d.xmat[body].reshape(3,3)
        errors=[abs(d.qpos-clip["expected_qpos"][frame]).max(),abs(d.qvel-clip["expected_qvel"][frame]).max(),
                abs(body_pose-clip["expected_carton"][frame]).max()]
        checked_errors=np.maximum(checked_errors,errors)
    if max(checked_errors)>1e-9:raise ValueError(f"Trajectory no longer reproduces: {checked_errors}")
    pose.qpos[:]=d.qpos;mj.mj_kinematics(m,pose)
    signed[-1]=support_distances(m,pose,table,hand,meshes)
    per_state=signed.min(axis=1); has_contact=np.isfinite(native)
    worst_delta=float(abs(native[has_contact]-per_state[:-1][has_contact]).max(initial=0))
    names=[m.geom(g).name for g in hand]
    def event(t):
        g=int(signed[t].argmin()); frame=int(np.searchsorted(boundaries,t,side="right")-1)
        return dict(signed_gap_mm=float(1000*signed[t,g]),geom=names[g],state_tick=int(t),
                    time_s=float(t*dt),source_interval_start=frame,substep=int(t-boundaries[frame]))
    intervals=np.searchsorted(boundaries,np.arange(N),side="right")-1
    def block(mask):
        ix=np.flatnonzero(mask)
        return dict(worst=event(int(ix[per_state[ix].argmin()])),steps=len(ix),
                    geometric_overlap_steps=int((per_state[ix]<0).sum()),
                    steps_deeper_than_2mm=int((per_state[ix]<-.002).sum()),
                    steps_deeper_than_10mm=int((per_state[ix]<-.01).sum()),
                    solver_contact_steps=int(has_contact[ix].sum()),peak_native_normal_load_N=float(loads[ix].max()),
                    native_normal_impulse_Ns=float(loads[ix].sum()*dt))
    def spans(cutoff):
        mask=per_state[:-1]<-cutoff
        changes=np.diff(np.r_[False,mask,False].astype(int))
        return [dict(start_tick=int(a),end_tick_exclusive=int(b),start_time_s=a*dt,end_time_s=b*dt)
                for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
    phase_rows={name:block((intervals>=value["frames"][0])&(intervals<=value["frames"][1]))
                for name,value in evaluation["phases"].items()}
    out=ROOT/"runs/hocap_s1_offline_hand_table_audit_v1"
    out.mkdir(exist_ok=True,parents=True)
    trace=out/"hand_table.npz"
    np.savez_compressed(trace,geom_names=names,signed_plane_distances_m=signed,native_min_distance_m=native,
                        actual_contact_count=counts,actual_normal_force_N=loads,frame_end_substep=boundaries)
    result=dict(archive=receipt["archive"],scene_mjb_sha256=digest(scene_bytes),controls_sha256=digest(controls),
                tool_sha256=digest(Path(__file__).read_bytes()),mujoco=mj.__version__,trace_sha256=digest(trace.read_bytes()),
                scope="Original stable-release S1 development commands, native sloped plane; signed negative means overlap. "
                "Geometry at all 5,601 sampled states; native solved contacts are from the 5,600 integrated substeps. "
                "2 mm and 10 mm counts describe magnitude only and introduce no hand-table qualification threshold.",
                nominal_source_gates=evaluation["gates"],states_verified_at_endpoints=len(boundaries),
                max_endpoint_reference_errors=dict(zip(("qpos","qvel","carton_matrix"),checked_errors)),
                steps=N,timestep_s=dt,plane=plane,disableflags=int(m.opt.disableflags),
                global_contacts_enabled=not bool(m.opt.disableflags & int(mj.mjtDisableBit.mjDSBL_CONTACT)),
                carton_contact_hand_geometry=group,first=event(0),worst=event(int(per_state.argmin())),last=event(N),
                initial_integrated_native_distance_mm=float(native[0]*1000),initial_integrated_native_hand_table_contacts=int(counts[0]),
                initial_integrated_hand_table_normal_load_N=float(loads[0]),max_native_vs_geometry_difference_mm=1000*worst_delta,
                all_substeps=block(np.ones(N,bool)),by_phase=phase_rows,spans_deeper_than_2mm=spans(.002),
                spans_deeper_than_10mm=spans(.01))
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(out/"report.json")


if __name__=="__main__":
    main()
