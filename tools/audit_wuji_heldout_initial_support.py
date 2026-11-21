"""CPU-only post hoc static geometry and archived cold-actor contact audit; never advances the simulator."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import yaml
from audit_hocap_carton_support import backproject
from probe_flash_wuji_replay import rebuilt
from run_wuji_matched_tranche import local_path
from supermimic_research.runtime.flash_quality import world_tips


def read(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def checked(pin):
    path=local_path(pin["path"])
    if hash_file(path)!=pin["sha256"]:
        raise ValueError(f"Changed artifact: {path}")
    return path


def tea_anchor_plane(world, poses, depth):
    anchor=world[[0,1,2,-3,-2,-1]]
    def bottom(slopes):
        heights=anchor[:,:,2]-anchor[:,:,:2]@slopes
        ix=heights.argmin(axis=1)
        return anchor[np.arange(6),ix],ix
    points,indices=bottom([0.,0.])
    start,end=points[:3].mean(axis=0),points[3:].mean(axis=0)
    delta=end-start
    slopes=delta[:2]*delta[2]/(delta[:2]@delta[:2])
    active,ids=bottom(slopes)
    np.testing.assert_array_equal(indices,ids)
    intercept=float(np.mean(active[:,2]-active[:,:2]@slopes))
    candidate=np.r_[slopes,intercept]
    refnorm=np.r_[-slopes,1];refnorm/=np.linalg.norm(refnorm)
    xy=np.array([start[:2],end[:2]])
    stations=np.c_[xy,np.ones(2)]
    compare=[]
    for f in depth["plane_fits"]:
        if not f["used"]:continue
        coeff=np.asarray(f["coef_z_xy1"])
        norm=np.r_[-coeff[:2],1];norm/=np.linalg.norm(norm)
        sensor_points,si=bottom(coeff[:2])
        gaps=(sensor_points[:,2]-sensor_points[:,:2]@coeff[:2]-coeff[2])*1000*norm[2]
        intrinsics,extrinsics=(yaml.safe_load(checked(k).read_bytes()) for k in f["calibration"])
        matrices=extrinsics["extrinsics"]
        def matrix(key):return np.vstack((np.array(matrices[key]).reshape(3,4),[0,0,0,1]))
        cloud=backproject(checked(f["depth"]),intrinsics["color"],np.linalg.solve(matrix("tag_1"),matrix(f["camera"])))
        design=np.c_[cloud[:,:2],np.ones(len(cloud))]
        inlier=abs(cloud[:,2]-design@coeff)<.006
        if int(inlier.sum())!=f["inliers"]:raise ValueError("Depth inlier reconstruction differs from frozen fit")
        delta_z=(cloud[:,2]-design@candidate)*1000
        local=[]
        for center in xy:
            rad=np.linalg.norm(cloud[:,:2]-center,axis=1)
            take=inlier&(rad<=.03)
            local.append(dict(nearest_fit_inlier_xy_mm=float(rad[inlier].min()*1000),inliers_in_30mm=int(take.sum()),
                              median_cloud_minus_candidate_mm=float(np.median(delta_z[take])) if take.any() else None))
        compare.append(dict(camera=f["camera"],depth_frame=f["frame"],normal=norm.tolist(),
                  tilt_deg=float(np.rad2deg(np.arccos(norm[2]))),uphill_azimuth_deg=float(np.rad2deg(np.arctan2(coeff[1],coeff[0]))),
                  angle_to_candidate_deg=float(np.rad2deg(np.arccos(np.clip(refnorm@norm,-1,1)))),
                  all_active_vertex_ids_match=bool(np.array_equal(indices,si)),anchor_signed_gaps_mm=gaps.tolist(),
                  end_minus_start_mean_gap_mm=float(gaps[3:].mean()-gaps[:3].mean()),
                  start_end_candidate_minus_depth_height_mm=(stations@(candidate-coeff)*1000).tolist(),
                  sensor_inliers_vertical_p95_mm=f["residual_p95_mm_inliers"],
                  candidate_on_same_inliers_abs_vertical_p95_mm=float(np.quantile(abs(delta_z[inlier]),.95)),
                  local_start_end=local))
    center_delta=poses[-3:,:2,3].mean(axis=0)-poses[:3,:2,3].mean(axis=0)
    return dict(object_center_end_minus_start_xy_mm=(center_delta*1000).tolist(),
        support_vertex_ids_zero_slope=indices.tolist(),support_start_xyz_mm=(start*1000).tolist(),support_end_xyz_mm=(end*1000).tolist(),
        support_delta_xyz_mm=(delta*1000).tolist(),anchor_only_plane_coef_z_xy1=candidate.tolist(),normal=refnorm.tolist(),
        tilt_deg=float(np.rad2deg(np.arctan(np.linalg.norm(slopes)))),uphill_azimuth_deg=float(np.rad2deg(np.arctan2(slopes[1],slopes[0]))),
        anchor_gaps_mm=((active[:,2]-active[:,:2]@slopes-intercept)*1000*refnorm[2]).tolist(),depth_comparison=compare)


def support_geometry(source, stage, depth):
    hull=checked(source["geometry"]["hull"])
    verts=np.loadtxt([line[2:] for line in hull.read_text().splitlines() if line.startswith("v ")])
    with np.load(checked(source["source_poses"])) as ar:
        poses=ar["world_from_carton"]
        frames=ar["frames"]
    world=verts@poses[:,:3,:3].transpose(0,2,1)+poses[:,None,:3,3]
    limits=world[:,:,2].min(axis=1)
    p0=world[0]
    anchor=np.r_[limits[:3],limits[-3:]]*1000
    fits=[]
    for f in depth["plane_fits"]:
        if not f["used"]:continue
        a,b,c=f["coef_z_xy1"]
        residual=(p0[:,2]-p0[:,0]*a-p0[:,1]*b-c)/np.linalg.norm([a,b,1])
        fits.append(dict(camera=f["camera"],depth_frame=f["frame"],coef_z_xy1=[a,b,c],
                    residual_p95_mm=f["residual_p95_mm_inliers"],first_hull_plane_gap_mm=float(residual.min()*1000)))
    return poses,dict(first_frame=int(frames[0]),last_frame=int(frames[-1]),source_hull_sha256=hash_file(hull),
                tea_anchor_plane=tea_anchor_plane(world,poses,depth) if source["object_id"]=="G15_2" else None,
                minimum_vertex_world_m=p0[p0[:,2].argmin()].tolist(),flat_plane_z0_gap_mm=float(limits[0]*1000),
                first_three_flat_gap_mm=(limits[:3]*1000).tolist(),last_three_flat_gap_mm=(limits[-3:]*1000).tolist(),
                horizontal_plane_anchor_minimax_mismatch_mm=float(np.ptp(anchor)/2),
                horizontal_plane_anchor_minimax_height_mm=float((anchor.min()+anchor.max())/2),
                first_pose_aligned_last_three_gaps_mm=(anchor[-3:]-anchor[0]).tolist(),
                stage_table_evidence=stage["table_evidence"],depth_first_pose_vs_each_fit=fits)


def static(env,object_id,pose,bottom):
    solver=env.solver
    m=solver.mj_model
    d=mujoco.MjData(m)
    solver._update_mjc_data(d,env.model,env.state_0)
    ob,=[i for i in range(m.nbody) if m.body(i).name==f"{object_id}_{object_id}"]
    table=m.body("support").id
    zidx=int(m.jnt_qposadr[m.body_jntadr[table]])+2
    q=d.qpos.copy()
    geoms={k:[i for i in range(m.ngeom) if m.geom_bodyid[i]==body and (m.geom_contype[i] or m.geom_conaffinity[i])]
           for k,body in (("box",ob),("support",table))}
    go,=geoms["box"]; gt,=geoms["support"]
    geometry=dict(box_geom=go,support_geom=gt,box_margin_m=float(m.geom_margin[go]),support_margin_m=float(m.geom_margin[gt]),
                  box_gap_m=float(m.geom_gap[go]),support_gap_m=float(m.geom_gap[gt]),convex_mesh_vertices=int(m.mesh_vertnum[m.geom_dataid[go]]))
    cases={}
    for label,shift in (("frozen",0.),("source_bottom_support_translation",bottom)):
        d.qpos[:]=q
        d.qpos[zidx]+=shift
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.xpos[ob],pose[:3,3],atol=5e-7,rtol=0)
        np.testing.assert_allclose(d.xmat[ob].reshape(3,3),pose[:3,:3],atol=5e-7,rtol=0)
        if label=="frozen":
            unchanged=d.xpos.copy()
            orientations=d.xmat.copy()
        else:
            ids=np.arange(m.nbody)!=table
            np.testing.assert_array_equal(d.xpos[ids],unchanged[ids])
            np.testing.assert_array_equal(d.xmat[ids],orientations[ids])
        distances={"box_support":[],"hand_support":[],"hand_box":[]}
        for c in d.contact:
            pair={int(c.geom1),int(c.geom2)}
            if gt in pair:group="box_support" if go in pair else "hand_support"
            elif go in pair:group="hand_box"
            else:continue
            distances[group].append(float(c.dist)*1000)
        top=float(d.geom_xpos[gt,2]+m.geom_size[gt,2])
        if abs(top-shift)>1e-6:raise ValueError("Unexpected support top or orientation")
        mesh=int(m.geom_dataid[go]);start=int(m.mesh_vertadr[mesh]);size=int(m.mesh_vertnum[mesh])
        vm=m.mesh_vert[start:start+size]@d.geom_xmat[go].reshape(3,3).T+d.geom_xpos[go]
        cases[label]=dict(support_z_translation_mm=shift*1000,top_world_z_mm=top*1000,
                  native_cpu_box_support_signed_distance_mm=float(mujoco.mj_geomDistance(m,d,go,gt,.03,None))*1000,
                  signed_contact_distances_mm=distances,source_hull_gap_mm=(bottom-top)*1000,
                  compiled_geometry_min_vertex_gap_mm=float(vm[:,2].min()-top)*1000)
    return dict(compiled_model=geometry,cases=cases,integrations=0,controls_applied=0,all_nonsupport_body_transforms_identical=True)


def trace_result(path, expected, env, thresholds):
    report=read(path)
    actor=next((a for a in report["realizations"] if a["id"]=="actor"),None)
    if actor is None:return dict(status=report["status"],applied_intervals=report["numerical_failure"]["applied_intervals"])
    folder=path.parent/"raw"
    raw=read(folder/"report.json")
    trace=folder/"trace.npz"
    if hash_file(trace)!=actor["trace_sha256"]:raise ValueError("Actor trace digest mismatch")
    with np.load(trace) as t:
        ids=t["source_frames"];n=len(ids)-1;sub=raw["substeps_per_interval"]
        phases={k:v["source_frames"] for k,v in raw["phases"].items() if v}
        source=t["raw_object_pose"];np.testing.assert_array_equal(source,expected)
        ticks=np.repeat(np.arange(n*sub),np.diff(t["contact_offsets"]))
        row=t["contact_rows"]; a,b=row[:,0],row[:,1]
        hand=t["hand_shape_ids"]; obj=t["object_shape_ids"]; table=t["support_shape_ids"]
        def pair(A,B):return (np.isin(a,A)&np.isin(b,B))|(np.isin(a,B)&np.isin(b,A))
        masks=dict(hand_box=pair(hand,obj),box_support=pair(table,obj),hand_self=pair(hand,hand))
        if not np.all(sum(mask.astype(int) for mask in masks.values())==1):raise ValueError("Unclassified contact row")
        depth={}
        for k,mask in masks.items():
            v=np.zeros(n*sub)
            np.maximum.at(v,ticks[mask],-1000*row[mask,2])
            depth[k]=v
        samp=t["substep_contacts"]
        np.testing.assert_allclose(np.maximum(depth["hand_box"],depth["box_support"]),1000*samp[:,3],atol=1e-8)
        np.testing.assert_allclose(depth["hand_self"],1000*samp[:,4],atol=1e-8)
        def point(tick):
            i,substep=divmod(tick,sub)
            f=int(ids[i])
            phase,=(p for p,(lo,hi) in phases.items() if lo<=f<=hi)
            return dict(frame_from=f,frame_to=int(ids[i+1]),substep=int(substep),phase=phase)
        def peak(v,start):
            if start>=len(v):return None
            k=int(np.argmax(v[start:])+start)
            return dict(value=float(v[k]),**point(k))
        depth["joint_excursion_deg"]=np.rad2deg(samp[:,5])
        limits=dict(hand_box=thresholds["maximum_carton_penetration_mm"],box_support=thresholds["maximum_carton_penetration_mm"],
                    hand_self=thresholds["maximum_self_penetration_mm"],joint_excursion_deg=thresholds["maximum_joint_excursion_deg"])
        detail={}
        frame0=np.repeat(ids[:-1],sub)
        for k,v in depth.items():
            breach=v>limits[k]
            phase_values={}
            for p,(lo,hi) in phases.items():
                selection=np.flatnonzero((frame0>=lo)&(frame0<=hi)&(np.arange(len(frame0))>=sub))
                if len(selection):
                    peak_tick=int(selection[v[selection].argmax()])
                    phase_values[p]=dict(count_over=int(breach[selection].sum()),peak=float(v[peak_tick]),**point(peak_tick))
            detail[k]=dict(peak=peak(v,0),excluding_first_tick_peak=peak(v,1),excluding_first_interval_peak=peak(v,sub),
                           over_limit_after_initial_interval=int(breach[sub:].sum()),phases_after_first_interval=phase_values)
        body=t["body_q"]; ob=env.scene.objects[0].bodies[0].body_id
        np.testing.assert_allclose(body[0,ob,:3],source[0,:3,3],atol=5e-7,rtol=0)
        pos=1000*np.linalg.norm(body[:,ob,:3]-source[:len(ids),:3,3],axis=1)
        ang=np.rad2deg((Rotation.from_quat(body[:,ob,3:]).inv()*Rotation.from_matrix(source[:len(ids),:3,:3])).magnitude())
        tips=1000*np.linalg.norm(world_tips(body,env.scene.layout.hand("right"))-t["raw_human_tips"][:len(ids)],axis=2)
        phase_errors={}
        for label,(lo,hi) in phases.items():
            pick=(ids>=lo)&(ids<=hi)
            if pick.any():
                phase_errors[label]=dict(frames=[lo,hi],position_rms_mm=float(np.mean(pos[pick]**2)**.5),
                    rotation_rms_deg=float(np.mean(ang[pick]**2)**.5),tip_rms_mm=float(np.mean(tips[pick]**2)**.5))
        return dict(status=report["status"],applied_intervals=n,trace_sha256=actor["trace_sha256"],
                    initial=dict(contact_rows=row[ticks==0].tolist(),shape_ids=dict(hand=hand.tolist(),box=obj.tolist(),support=table.tolist()),
                                 samples=samp[0].tolist()),contact_peaks_and_counts=detail,phase_errors=phase_errors,
                    retained_metrics=raw["metrics"],failed_heldout_gates=[k for k,x in actor["heldout_gates"].items() if not x],
                    actor_terminal=actor["terminal"])


def main():
    out=REPO/"runs/wuji_heldout_initial_support_diagnosis_v1"
    root=REPO/"runs/wuji_heldout_formal_v1"
    close=read(root/"complete.json");primary=read(root/"cost_and_outcomes.json")
    if (hash_file(root/"registered_stages.json")!=close["registered_stages_sha"] or
            hash_file(root/"final_selection.json")!=close["selection_sha"]):
        raise ValueError("Frozen primary inputs changed")
    stages=read(root/"registered_stages.json")["subjects"]; plan=read(root/"final_selection.json")
    quarter=plan["steps"]
    wp.config.kernel_cache_dir=str(Path(plan["upstream"]).parent/"cache/warp")
    wp.init();wp.set_device("cpu")
    if wp.get_cuda_devices():raise ValueError("Diagnostic requires a CPU-only process")
    cfg=read(REPO/"configs/hocap_carton_evaluation.json")
    results={}
    for s,item in stages.items():
        pq=checked(item["parquet"]); source=read(checked(item["source"])); contract=read(checked(item["contract"]))
        evidence=read(root/"source_jobs"/f"{s}_source_evidence"/"attempt_001"/"evidence.json")
        depth=checked(evidence["source_depth_fit_report"])
        poses,geom=support_geometry(source,contract,read(depth))
        env=rebuilt(pq,checked(plan["passive_config_checkpoint"]),True)
        snap=static(env,source["object_id"],poses[0],geom["flat_plane_z0_gap_mm"]/1000)
        checkpoints={}
        for step in quarter:
            name=f"{s}_cold_single_eval_{step}"
            attempts=sorted((root/"jobs"/name).glob("attempt_*.json"))
            scored=[job for p in attempts if (job:=read(p))["success"]]
            if not scored:
                stop=primary["methods"]["cold_single"]["subjects"][s].get("adaptation_stop")
                if not stop or step<stop["step"] or (step==stop["step"] and stop["actor_usable"]):
                    raise ValueError(f"Unexplained missing cold evaluation: {s} {step}")
                checkpoints[step]=dict(status="no_physical_rollout_after_"+stop["classification"],applied_intervals=None)
                continue
            job,=scored
            checkpoints[step]=dict(receipt=job["receipt"],**trace_result(checked(job["receipt"]),poses,env,cfg))
        results[s]=dict(source=source["sequence"],object=source["object_id"],source_geometry=geom,
                        source_pin=item["source"],depth_sha256=hash_file(depth),static_cpu=snap,archived_cold_actor=checkpoints)
    result=dict(scope="Post hoc diagnostic. Source poses, hand poses, frozen metrics and original rollouts preserved. Positive penetration = overlap; negative signed distance = overlap.",
                enrolled_steps=quarter,source_stages_sha256=hash_file(root/"registered_stages.json"),
                diagnostic_sha256=hash_file(Path(__file__)),subjects=results)
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(out/"audit.json")


if __name__=="__main__":
    main()
