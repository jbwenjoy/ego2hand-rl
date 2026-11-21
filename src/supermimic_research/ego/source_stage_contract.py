"""Build prospective held-out stage evidence from human data, then score a physical replay."""
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.assets.proximity import SurfaceQuery


def load_frozen_stages(path, *, contract_sha256, raw_source_report_sha256, frames, truth):
    """Validate the pre-registered contract and original source before reading metrics."""
    path=Path(path)
    contents=path.read_bytes()
    if hashlib.sha256(contents).hexdigest()!=contract_sha256:
        raise ValueError("Held-out stage contract changed after registration")
    meta=json.loads(contents)
    if meta["raw_source_report_sha256"]!=raw_source_report_sha256:
        raise ValueError("Held-out stage contract refers to another source window")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=meta["logic"]["sha256"]:
        raise ValueError("Stage scoring implementation changed after source freeze")
    if meta["intervals_file"]!="source_intervals.npz":
        raise ValueError("Unexpected intervals file")
    data_path=path.parent/meta["intervals_file"]
    if hashlib.sha256(data_path.read_bytes()).hexdigest()!=meta["intervals_sha256"]:
        raise ValueError("Stage interval masks changed after freeze")
    with np.load(data_path,allow_pickle=False) as raw:
        arrays=dict(raw)
    np.testing.assert_array_equal(arrays["source_frame_ids"],frames)
    np.testing.assert_array_equal(arrays["source_poses"],truth)
    return arrays,meta


def _angles(rotations):
    return np.rad2deg(Rotation.from_matrix(rotations).magnitude())


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def _points(poses, local):
    return local @ poses[:, :3, :3].transpose(0,2,1) + poses[:,None,:3,3]


def _stable_anchor(poses, config):
    translations = np.linalg.norm(poses[:,:3,3]-poses[0,:3,3], axis=1)*1000
    rotations = _angles(poses[0,:3,:3].T@poses[:,:3,:3])
    return (translations.max()<=config["anchor_max_displacement_mm"]
            and rotations.max()<=config["anchor_max_rotation_deg"])


def _clearance(poses, vertices, config, fits):
    """Conservative minimum signed full-scan height in each source frame, metres."""
    world = _points(poses,vertices)
    n=config["source_anchor_frames"]
    camera_bounds=[]
    used=[]
    for row in fits:
        if not row.get("used",True) or row.get("points",0)<1000 or row.get("inliers",0)<800:
            continue
        a,b,c=row["coef_z_xy1"]
        scale=np.sqrt(1+a*a+b*b)
        gap=((world[:,:,2]-a*world[:,:,0]-b*world[:,:,1]-c)/scale).min(1)
        anchor=max(abs(np.median(gap[:n])),abs(np.median(gap[-n:])))
        if anchor*1000>config["depth_max_anchor_discrepancy_mm"]:
            continue
        uncertainty=max(config["depth_residual_floor_mm"],row["residual_p95_mm_inliers"])/1000+anchor
        if "inlier_xy_bounds_m" in row:
            lower,upper=np.asarray(row["inlier_xy_bounds_m"])
            xy=world[:,:,:2]
            outside=np.maximum(np.maximum(lower-xy,xy-upper),0.)
            distance=np.linalg.norm(outside,axis=-1).max(axis=1)
            uncertainty=uncertainty+distance*np.tan(np.deg2rad(config["anchor_fallback_max_tilt_deg"]))
        camera_bounds.append(gap-uncertainty)
        used.append(str(row["camera"]))
    if len(set(used))>=config["depth_min_distinct_cameras"]:
        return np.min(camera_bounds,axis=0),dict(method="independent_depth_plane_envelope",cameras=sorted(set(used)))
    # Orient the full scan in every frame. For any planar tabletop tilted up to
    # theta, the height increase across horizontal displacement D is <= D tan(theta).
    # Use the entire XY bounding diagonal as a conservative D, with additional
    # vertical registration uncertainty. This fallback requires visible supported
    # endpoints and the source dataset's table-world convention.
    floor=world[:,:,2].min(1)
    xy=world[:,:,:2].reshape(-1,2)
    diameter=float(np.linalg.norm(np.ptp(xy,axis=0)))
    tilt=np.deg2rad(config["anchor_fallback_max_tilt_deg"])
    vertical=config["anchor_fallback_vertical_uncertainty_mm"]/1000
    threshold=max(np.median(floor[:n]),np.median(floor[-n:]))+vertical+diameter*np.tan(tilt)
    return (floor-threshold)*np.cos(tilt),dict(method="source_supported_endpoint_envelope",
            xy_diameter_m=diameter,max_table_tilt_deg=config["anchor_fallback_max_tilt_deg"],
            vertical_uncertainty_mm=config["anchor_fallback_vertical_uncertainty_mm"])


def derive_stages(poses, human_tips, source_mesh, config, depth_fits=(), *, table_seen_at_both_ends):
    """Use source-only poses and scan; never accept replay, actions, rewards or predicted contacts."""
    poses=np.asarray(poses,dtype=np.float64)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or len(poses)<2*config["source_anchor_frames"]+1:
        raise ValueError("Invalid source sequence")
    if not np.isfinite(poses).all() or human_tips.shape!=(len(poses),5,3) or not np.isfinite(human_tips).all():
        raise ValueError("Invalid hand/object reference")
    n=len(poses)
    corners=np.array(list(product(*zip(*source_mesh.bounds,strict=True))))
    witness=_points(poses,corners)
    tips=(human_tips-poses[:,None,:3,3]) @ poses[:,:3,:3]
    query=SurfaceQuery(source_mesh.vertices,source_mesh.faces)
    tip_mm=1000*query.nearest(tips.reshape(-1,3))[1].reshape(n,5).min(1)
    lower,plane=_clearance(poses,source_mesh.convex_hull.vertices,config,depth_fits)
    above=lower*1000>config["extra_clearance_margin_mm"]
    air=above[:-1]&above[1:]
    min_run=config["minimum_consecutive_airborne_intervals"]
    edges=np.diff(np.r_[False,air,False].astype(int))
    for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1),strict=True):
        if hi-lo<min_run:air[lo:hi]=False
    k=config["motion_difference_frames_each_side"]
    endpoints=np.arange(1,n)
    low=np.maximum(0,endpoints-k)
    high=np.minimum(n-1,endpoints+k)
    motion=witness[high]-witness[low]
    speed=np.sqrt(np.mean(np.sum(motion*motion,axis=2),axis=1))*1000
    engaged=(speed>=config["minimum_witness_motion_mm"])&(tip_mm[1:]<=config["source_contact_distance_mm"])
    static_disp=np.linalg.norm(witness-witness[0],axis=2)*1000
    pos=np.linalg.norm(poses[:,:3,3]-poses[0,:3,3],axis=1)*1000
    angle=_angles(poses[0,:3,:3].T@poses[:,:3,:3])
    length=config["source_anchor_frames"]
    # Final N physical intervals depend on N+1 source frames.
    released=bool(np.all(tip_mm[-(config["release_suffix_intervals"]+1):]>=config["detached_distance_mm"]))
    eligibility=dict(
        initial_and_final_table_support_visible=bool(table_seen_at_both_ends),
        initial_stable=bool(_stable_anchor(poses[:length],config)),
        final_stable=bool(_stable_anchor(poses[-length:],config)),
        released_at_end=released,
        informative_motion=bool(static_disp.max()>=config["minimum_informative_motion_mm"]),
        source_has_engaged_motion=bool(engaged.sum()>=config["minimum_engaged_moving_intervals"]),
    )
    arrays=dict(source_poses=poses,corners=corners,source_witness=witness,source_tip_distance_mm=tip_mm,
                source_clearance_lower_m=lower,airborne_intervals=air,engaged_moving_intervals=engaged,
                witness_window_lo=low,witness_window_hi=high)
    meta=dict(eligibility=eligibility,scorable=all(eligibility.values()),table_evidence=plane,
              airborne_applicable=bool(air.any()),airborne_intervals=int(air.sum()),engaged_intervals=int(engaged.sum()),
              informative_translation=bool(pos.max()>=config["translation_informative_mm"]),
              informative_rotation=bool(angle.max()>=config["rotation_informative_deg"]),
              static_position_rms_mm=_rms(pos),static_rotation_rms_deg=_rms(angle),static_witness_rms_mm=_rms(static_disp))
    return arrays,meta


def score_frozen_source_stages(arrays, meta, config, simulated_poses, hand_load_bw, table_load_bw, max_hand_contacts):
    """Scores supplement existing absolute/full-run gates. Load units are normal load / bodyweight."""
    poses=np.asarray(simulated_poses)
    truth=arrays["source_poses"]
    hand,table,contacts=map(np.asarray,(hand_load_bw,table_load_bw,max_hand_contacts))
    if poses.shape[1:]!=(4,4) or not 1<=len(poses)<=len(truth):
        raise ValueError("Invalid physical rollout length")
    if any(a.shape!=(len(poses)-1,) or not np.isfinite(a).all() for a in (hand,table,contacts)) or not np.isfinite(poses).all():
        raise ValueError("Physical interval evidence missing")
    full=len(poses)==len(truth)
    applicable=meta["airborne_applicable"]
    gates={"scorable_source_stages":meta["scorable"],"beats_fixed_pose":False,
           "contact_during_demonstrated_motion":False,"release":False}
    if applicable:gates["hand_support_in_air"]=False
    metrics={"airborne_applicability":"applicable" if applicable else "N/A: no robust source airborne interval",
             "complete_source_window":full,"source_stage_eligibility":meta["eligibility"]}
    if not full or not meta["scorable"]:return gates,metrics
    witness=_points(poses,arrays["corners"])
    target=arrays["source_witness"]
    engaged=arrays["engaged_moving_intervals"]
    air=arrays["airborne_intervals"]
    lo,hi=arrays["witness_window_lo"][engaged],arrays["witness_window_hi"][engaged]
    reference_delta=target[hi]-target[lo]
    # Dot-product progress is signed; moving the box the wrong way cannot score.
    progress=float(np.sum((witness[hi]-witness[lo])*reference_delta)/np.sum(reference_delta**2))
    contact=(hand>=config["minimum_normal_load_bodyweights"])&(contacts>0)
    support=float(np.mean(contact[engaged]))
    pw=_rms(1000*np.linalg.norm(witness-target,axis=2))
    p=_rms(1000*np.linalg.norm(poses[:,:3,3]-truth[:,:3,3],axis=1))
    a=_rms(_angles(poses[:,:3,:3].transpose(0,2,1)@truth[:,:3,:3]))
    gates["beats_fixed_pose"]=bool(pw<meta["static_witness_rms_mm"]
        and (not meta["informative_translation"] or p<meta["static_position_rms_mm"])
        and (not meta["informative_rotation"] or a<meta["static_rotation_rms_deg"]))
    gates["contact_during_demonstrated_motion"]=bool(support>=config["minimum_supported_engaged_fraction"]
        and progress>=config["minimum_contact_conditioned_directional_progress"])
    if applicable:
        free=(table<config["maximum_airborne_table_load_bodyweights"])&contact
        fraction=float(np.mean(free[air]))
        gates["hand_support_in_air"]=fraction>=config["minimum_airborne_supported_fraction"]
        metrics["airborne_supported_fraction"]=fraction
    n=config["release_suffix_intervals"]
    gates["release"]=bool(np.all(table[-n:]>=config["minimum_release_table_load_bodyweights"])
                          and np.all(contacts[-n:]==0) and _stable_anchor(poses[-(n+1):],config))
    metrics.update(witness_rms_mm=pw,engaged_load_coverage=support,directional_witness_progress=progress)
    return gates,metrics
