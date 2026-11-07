"""Frozen pilot task criteria. These are simulation task labels, not hardware limits."""
import hashlib
import json
import numpy as np
from .geometry import quat_matrix


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _unit(v):
    v=np.asarray(v,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        raise ValueError("expected finite nonzero 3D direction")
    return v/np.linalg.norm(v)


def progress_from_axes(axes, normal_w, tangent_w, *, min_projection=.25):
    """Unwrap a directed shaft's azimuth around a declared external axis.

    This measures turn progress, not pen roll. Singular projections remain
    invalid; do not interpolate across a lost orientation observation.
    """
    axes=np.asarray(axes,dtype=float)
    if axes.ndim!=2 or axes.shape[1]!=3 or not np.isfinite(axes).all():
        raise ValueError("expected finite [T,3] shaft axes")
    n=_unit(normal_w); e1=_unit(tangent_w)
    if abs(n@e1)>1e-5: raise ValueError("progress frame must be orthogonal")
    e2=np.cross(n,e1)
    projected=axes-np.outer(axes@n,n)
    norms=np.linalg.norm(projected,axis=-1)
    valid=norms>=min_projection
    phase=np.arctan2(projected@e2,projected@e1)
    out=np.full(len(axes),np.nan)
    # Each observed segment is unwrapped independently. Any gap disqualifies
    # full-clip turn-count acceptance, although observed portions remain useful.
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
    for begin,end in edges.reshape(-1,2): out[begin:end]=np.unwrap(phase[begin:end])
    if len(out) and valid[0]: out-=out[0]
    return np.rad2deg(out),valid


def fit_reference_progress_frame(object_pose_w):
    """Fit solely from the given target, before any formal policy comparison."""
    axes=quat_matrix(np.asarray(object_pose_w)[:,3:])[:,:,0]
    _,singular,vt=np.linalg.svd(axes-axes.mean(0),full_matrices=False)
    if len(singular)!=3 or singular[1]<.05:
        raise ValueError("target does not identify a sufficiently excited single-axis rotation plane")
    ratio=float(singular[2]/singular[1])
    if ratio>.30:
        raise ValueError("target is insufficiently planar for single-axis progress; use a different declared task")
    normal=vt[-1]
    if normal[np.argmax(abs(normal))]<0: normal=-normal
    tangent=_unit(axes[0]-(axes[0]@normal)*normal)
    progress,valid=progress_from_axes(axes,normal,tangent)
    if not valid.all(): raise ValueError("target progress has singular shaft projection")
    if abs(progress[-1])<45:
        raise ValueError("target has less than 45 degrees net turn; do not call it a turn-progress experiment")
    if progress[-1]<0:
        normal=-normal
        progress,valid=progress_from_axes(axes,normal,tangent)
    return {"mode":"azimuth","normal_w":normal.tolist(),"tangent_w":tangent.tolist(),"plane_singular_values":singular.tolist(),
            "plane_residual_ratio":ratio,"target_net_progress_deg":float(progress[-1]),
            "fit_source":"reference shaft axes only; not policy rollout","min_projection":.25}


def shaft_increment_vectors(axes):
    """Minimal observed shaft rotations; axial roll is deliberately excluded."""
    axes=np.asarray(axes,dtype=float)
    if axes.ndim!=2 or axes.shape[1]!=3 or not np.isfinite(axes).all(): raise ValueError("invalid shaft axes")
    norm=np.linalg.norm(axes,axis=-1)
    if not np.allclose(norm,1,atol=1e-5): raise ValueError("shaft axes must be unit vectors")
    cross=np.cross(axes[:-1],axes[1:]);sine=np.linalg.norm(cross,axis=-1)
    cosine=np.clip((axes[:-1]*axes[1:]).sum(-1),-1,1)
    if np.any((sine<1e-8)&(cosine<0)): raise ValueError("antipodal samples have ambiguous shaft progress")
    angle=np.arctan2(sine,cosine)
    return cross*(angle/np.maximum(sine,1e-12))[:,None]


def projected_shaft_progress(axes, normal_w):
    increments=shaft_increment_vectors(axes)
    return np.rad2deg(np.r_[0,np.cumsum(increments@_unit(normal_w))]),np.ones(len(axes),dtype=bool)


def fit_reference_projected_rotation(object_pose_w):
    """A fixed, reference-only dominant axis for nonplanar shaft trajectories.

    Progress is an integral of observable minimal shaft rotation projected on
    this axis, not azimuth or a count of physical axial-roll revolutions.
    """
    axes=quat_matrix(np.asarray(object_pose_w)[:,3:])[:,:,0]
    increments=shaft_increment_vectors(axes)
    covariance=increments.T@increments
    values,vectors=np.linalg.eigh(covariance)
    if values.sum()<1e-9: raise ValueError("reference has no measurable shaft rotation")
    dominance=float(values[-1]/values.sum())
    if dominance<.60: raise ValueError("reference has no sufficiently dominant external shaft rotation axis")
    normal=vectors[:,-1]
    if increments.sum(0)@normal<0:normal=-normal
    progress,_=projected_shaft_progress(axes,normal)
    extent=float(np.rad2deg(np.arccos(np.clip(axes@axes[0],-1,1))).max())
    path=float(np.rad2deg(np.linalg.norm(increments,axis=-1).sum()))
    if progress[-1]<45 or extent<60 or path<90:
        raise ValueError("reference lacks the preregistered 45deg projected / 60deg excursion / 90deg path motion")
    return {"mode":"projected_shaft_rotation","normal_w":normal.tolist(),"dominant_rotation_energy_fraction":dominance,
            "rotation_increment_eigenvalues":values.tolist(),"target_net_progress_deg":float(progress[-1]),
            "target_shaft_excursion_deg":extent,"target_shaft_path_deg":path,
            "fit_source":"reference minimal shaft increments only; no policy data",
            "meaning":"integrated minimal shaft rotation projected on a fixed reference axis; excludes unobservable axial roll and is not winding count"}


def make_pilot_protocol(reference, reference_npz_sha256, *, progress_mode="azimuth"):
    if progress_mode not in {"azimuth","projected_shaft_rotation"}:raise ValueError("unknown progress mode")
    fit=fit_reference_progress_frame if progress_mode=="azimuth" else fit_reference_projected_rotation
    frame=fit(reference.arrays["object_pose_w"])
    return {"schema":"supermimic_chord_pilot_acceptance:1","status":"frozen_pilot_simulation_criteria_not_hardware_safety_limits",
            "reference_npz_sha256":reference_npz_sha256,"fixed_before_formal_A1_A2":True,
            "sample_semantics":"control samples at 50Hz, not a continuous-time contact certificate",
            "position_max_m":.02,"shaft_angle_max_deg":20.,"penetration_max_m":.003,
            "progress_frame":frame,"progress_abs_error_max_deg":30.,"progress_endpoint_error_max_deg":20.,
            "angular_step_max_deg":90.,"drop_center_below_wrist_m":.15,"allow_object_environment_contacts":False,
            "actuator_load_ratio_max":1.01,"load_charge_max":.8,"charge_decay_time_s":2.,
            "target_speed_abs_max_rad_s":15.,"joint_limit_margin_rad":.001,
            "source":"preliminary 250mm pen pilot task definition; freeze before formal comparisons; changes require a new protocol"}


def evaluate_pilot_task(reference,rollout,rollout_meta,base_report):
    protocol=reference.meta.get("task_acceptance")
    if protocol is None:
        return {"task_success":None,"task_success_unavailable_reason":"No frozen task protocol; progress/drop/load acceptance unavailable."}
    if protocol.get("schema")!="supermimic_chord_pilot_acceptance:1": raise ValueError("unsupported task protocol")
    if reference.meta.get("task_acceptance_sha256")!=protocol_digest(protocol): raise ValueError("task protocol hash mismatch")
    if rollout_meta.get("reference_bundle_sha256")!=protocol["reference_npz_sha256"]:
        raise ValueError("rollout reference bundle does not match the frozen task protocol")
    required={"actuator_torque_Nm","actuator_force_limit_Nm","object_ang_vel_w_rad_s","object_environment_contact_count"}
    missing=required-set(rollout)
    if missing:
        return {"task_success":None,"task_success_unavailable_reason":f"Missing task evidence: {sorted(missing)}","task_protocol_sha256":protocol_digest(protocol)}
    n=len(rollout["time_s"])
    for key,shape in (("actuator_torque_Nm",(n,20)),("actuator_force_limit_Nm",(20,)),
                      ("object_ang_vel_w_rad_s",(n,3)),("object_environment_contact_count",(n,))):
        v=np.asarray(rollout[key])
        if v.shape!=shape or not np.isfinite(v).all(): raise ValueError(f"invalid task evidence {key}")
    limit=rollout["actuator_force_limit_Nm"]
    if np.any(limit<=0) or np.any(rollout["object_environment_contact_count"]<0): raise ValueError("invalid load/contact evidence")
    frame=protocol["progress_frame"]
    ref_axis=quat_matrix(reference.arrays["object_pose_w"][:n,3:])[:,:,0]
    axis=quat_matrix(rollout["object_pose_w"][:,3:])[:,:,0]
    if frame.get("mode","azimuth")=="projected_shaft_rotation":
        ref_progress,ref_valid=projected_shaft_progress(ref_axis,frame["normal_w"])
        try:
            actual_progress,valid=projected_shaft_progress(axis,frame["normal_w"])
        except ValueError:
            # Ambiguous rapid observed motion is a failed progress observation,
            # not a reason to erase the otherwise valid physical rollout.
            actual_progress=np.full(n,np.nan);valid=np.zeros(n,dtype=bool)
    else:
        ref_progress,ref_valid=progress_from_axes(ref_axis,frame["normal_w"],frame["tangent_w"],min_projection=frame["min_projection"])
        actual_progress,valid=progress_from_axes(axis,frame["normal_w"],frame["tangent_w"],min_projection=frame["min_projection"])
    good=valid&ref_valid
    error=np.abs(actual_progress-ref_progress)
    dt=np.diff(rollout["time_s"])
    angular_bound=np.linalg.norm(rollout["object_ang_vel_w_rad_s"],axis=-1)
    # Bound sampled angular advance with both interval endpoints to catch
    # obvious aliasing. No claim about unobserved substep extrema is made.
    step_deg=np.rad2deg(np.maximum(angular_bound[1:],angular_bound[:-1])*dt)
    alias_ok=bool(np.all(step_deg<=protocol["angular_step_max_deg"]))
    progress_ok=bool(good.all() and alias_ok and np.all(error<=protocol["progress_abs_error_max_deg"])
                     and error[-1]<=protocol["progress_endpoint_error_max_deg"])
    drop=(rollout["object_pose_w"][:,2] < rollout["wrist_pose_w"][:,2]-protocol["drop_center_below_wrist_m"])
    environment_contact=rollout["object_environment_contact_count"]>0
    drop_ok=bool(not np.any(drop) and (protocol["allow_object_environment_contacts"] or not np.any(environment_contact)))
    ratio=np.abs(rollout["actuator_torque_Nm"])/limit
    charge=np.zeros(20);max_charge=0.
    for k,delta_t in enumerate(dt,1):
        charge=np.clip(charge+delta_t*(np.minimum(ratio[k],1)-charge/protocol["charge_decay_time_s"]),0,1)
        max_charge=max(max_charge,float(charge.max()))
    load_ok=bool(ratio.max()<=protocol["actuator_load_ratio_max"] and max_charge<=protocol["load_charge_max"])
    speed=np.diff(rollout["joint_target_rad"],axis=0)/dt[:,None] if n>1 else np.zeros((0,20))
    smooth_ok=bool(np.all(np.abs(speed)<=protocol["target_speed_abs_max_rad_s"]))
    pos=np.linalg.norm(rollout["object_pose_w"][:,:3]-reference.arrays["object_pose_w"][:n,:3],axis=-1)
    shaft_angle=np.rad2deg(np.arccos(np.clip((axis*ref_axis).sum(-1),-1,1)))
    limits_ok=all(np.all((rollout[k]>=rollout["joint_lower_rad"]-protocol["joint_limit_margin_rad"])
                        &(rollout[k]<=rollout["joint_upper_rad"]+protocol["joint_limit_margin_rad"])) for k in ("joint_pos_rad","joint_target_rad"))
    tracking_ok=bool(np.all(pos<=protocol["position_max_m"]) and np.all(shaft_angle<=protocol["shaft_angle_max_deg"]))
    penetration_ok=bool(np.max(rollout["penetration_m"])<=protocol["penetration_max_m"])
    transport_ok=bool(rollout_meta.get("resets_during_rollout")==0 and rollout_meta.get("object_state_writes_after_initialization")==0
                      and rollout_meta.get("rollout_censored") is False)
    success=bool(base_report["complete_clip"] and base_report["unassisted"] and base_report["fixed_wrist_ok"]
                 and tracking_ok and penetration_ok and limits_ok and progress_ok and drop_ok and load_ok and smooth_ok and transport_ok)
    finite_error=error[good]
    return {"task_success":success,"task_success_unavailable_reason":None,"task_protocol_sha256":protocol_digest(protocol),
            "task_protocol_status":protocol["status"],"progress_ok":progress_ok,"progress_observed_fraction":float(good.mean()),
            "progress_metric_mode":frame.get("mode","azimuth"),
            "rotation_progress_error_deg":{"mean":float(finite_error.mean()) if finite_error.size else None,
                                           "max":float(finite_error.max()) if finite_error.size else None,
                                           "endpoint":float(error[-1]) if good[-1] else None},
            "reference_progress_deg":float(ref_progress[-1]) if ref_valid[-1] else None,
            "actual_progress_deg":float(actual_progress[-1]) if valid[-1] else None,
            "progress_alias_bound_ok":alias_ok,"drop_ok":drop_ok,"drop_detected":bool(drop.any()),
            "object_environment_contact_frames":int(environment_contact.sum()),"load_ok":load_ok,"load_ratio_max":float(ratio.max()),
            "load_charge_max":max_charge,"target_rate_ok":smooth_ok,"target_speed_abs_max_rad_s":float(np.abs(speed).max()) if speed.size else 0.,
            "task_tracking_ok":tracking_ok,"task_penetration_ok":penetration_ok,"task_joint_limits_ok":bool(limits_ok),
            "task_logging_contract_ok":transport_ok}
