"""Decoded simulator contact wrenches/momentum residuals. Saved arrays only.

No MuJoCo, Torch, FK, model load, forward or integration. This is an author label
export/consistency audit, not measured real forces or holding qualification.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
CODEC = PILOT/"phone_transition_dataset_v1/transition_codec_v1.py"
CODEC_SHA = "1bf33a7fd27ae51d38a71c1f7e095dbe3047c4828ee6e65c199117bf536f4cac"
SCHEMA = "phone_saved_object_wrench_label_protocol:1"
PHONE_GEOM = 71  # Pinned Reader asserts this exact compiled ID.


def require(ok, message):
    if not ok:raise ValueError(message)


def pin(path):
    p = Path(path).resolve(strict=True); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def checked(rec):
    p = Path(rec["path"]); a = pin(p)
    require(p.is_absolute() and a["sha256"] == rec["sha256"] and a["bytes"] == rec["bytes"], "Saved input/source identity changed")
    return p.resolve(strict=True)


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:json.dump(value, f, indent=2, allow_nan=False)


def codec_module():
    require(pin(CODEC)["sha256"] == CODEC_SHA, "Frozen saved-state codec")
    spec = importlib.util.spec_from_file_location("wrench_frozen_phone_codec", CODEC)
    c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)
    return c


def rotation_wxyz(q):
    """Raw unit-quaternion sandwich, no normalization or projection."""
    q = np.asarray(q, dtype=np.float64)
    require(q.shape[-1] == 4 and bool(np.isfinite(q).all()) and bool(np.all(abs(np.sum(q*q, -1)-1) <= 1e-12)), "Saved unit quaternion domain")
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack((w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z), -1).reshape(*q.shape[:-1],3,3)


def raw_states(q, v, ipos, codec):
    R = rotation_wxyz(q[:,23:27])
    return codec.state.to_com_world_state(q[:,:20], v[:,:20], q[:,20:23], R,
        v[:,20:23], v[:,23:26], ipos), R


def contact_wrenches(contact_rows24, pre_tick, COM_pre):
    """All phone-boundary contacts, no positive-Fn/constraint/address filtering.

    frame rows map world vectors to local contact axes. mj_contactForce is the
    local wrench on geom2: sign + if phone geom2, - if phone geom1. Direct local
    moments are included, and contact lever is about this tick's actual COM.
    """
    c = np.asarray(contact_rows24, dtype=np.float64)
    require(c.ndim == 2 and c.shape[1] == 24 and bool(np.isfinite(c).all()), "Raw decoded contact24 finite ABI")
    n = len(pre_tick); total = np.zeros((n,6)); phone = (c[:,1] == PHONE_GEOM) ^ (c[:,2] == PHONE_GEOM)
    internal = (c[:,1] == PHONE_GEOM) & (c[:,2] == PHONE_GEOM)
    r = c[phone]
    if not len(r):return total, {"generated_contacts": len(c), "phone_boundary_contacts": 0, "phone_internal_pairs": int(internal.sum()), "max_contact_frame_defect": 0.}
    indices = r[:,0].astype(np.int64)-int(pre_tick[0])
    require(np.array_equal(r[:,0], r[:,0].astype(np.int64)) and bool(np.all((indices >= 0)&(indices < n)))
        and np.array_equal(r[:,0], pre_tick[indices]), "Original integer PRE contact tick, not clock/h rounding")
    frame = r[:,9:18].reshape(-1,3,3); sign = np.where(r[:,2] == PHONE_GEOM, 1., -1.)
    F = sign[:,None]*np.einsum("nji,nj->ni",frame,r[:,18:21])
    direct = sign[:,None]*np.einsum("nji,nj->ni",frame,r[:,21:24])
    tau = np.cross(r[:,6:9]-COM_pre[indices],F)+direct
    np.add.at(total[:,:3],indices,F); np.add.at(total[:,3:],indices,tau)
    defect = np.max(abs(frame@frame.transpose(0,2,1)-np.eye(3)))
    return total, {"generated_contacts":len(c),"phone_boundary_contacts":len(r),"phone_internal_pairs":int(internal.sum()),
        "max_contact_frame_defect":float(defect),"other_geom_ids":sorted(set(np.where(r[:,1] == PHONE_GEOM,r[:,2],r[:,1]).astype(int).tolist()))}


def rotate_wrench_axes(world, R_body_to_world):
    return np.column_stack((np.einsum("nji,nj->ni",R_body_to_world,world[:,:3]),
                            np.einsum("nji,nj->ni",R_body_to_world,world[:,3:])))


def angular_momentum(R_body, omega_world, I_body):
    return np.einsum("...ij,...j->...i", R_body@I_body@np.swapaxes(R_body,-1,-2), omega_world)


def fixture():
    Rz = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    c = np.zeros((1,24)); c[0,[0,1,2,3]] = [0,15,71,3]; c[0,6:9] = [0,1,0]
    c[0,9:18] = np.eye(3).ravel(); c[0,18:24] = [2,0,0,0,0,.5]
    world,_ = contact_wrenches(c,np.array([0]),np.zeros((1,3)))
    require(np.array_equal(world,np.array([[2.,0,0,0,0,-1.5]])), "Point force/direct moment lever sign")
    body = rotate_wrench_axes(world,Rz[None]); require(np.array_equal(body,np.array([[0.,-2,0,0,0,-1.5]])), "Rotated body axes")
    reverse = c.copy();reverse[0,1:3] = [71,15];reverse[0,9:18] = np.diag([-1.,-1,1]).ravel();reverse[0,21:24] = [0,0,-.5]
    flipped,_ = contact_wrenches(reverse,np.array([0]),np.zeros((1,3)))
    require(np.array_equal(world,flipped),"Reversed geom-order physical wrench invariance")
    fixed_F = np.array([[1.,0,0,0,0,0],[1.,0,0,0,0,0]])
    instant = rotate_wrench_axes(fixed_F,np.array([np.eye(3),Rz])).mean(0)
    require(np.array_equal(instant[:3], [.5,-.5,0]) and np.array_equal(fixed_F.mean(0)[:3],[1,0,0]), "Instantaneous/body-fixed axes differ")
    fixed_origin = fixed_F.copy();fixed_origin[:,3:] += np.cross(np.array([[0.,0,0],[0,1,0]]),fixed_F[:,:3])
    require(np.array_equal(fixed_origin.mean(0)[3:],[0,0,-.5]),"Explicit fixed-origin lever")
    Rx = np.array([[1.,0,0],[0,0,-1],[0,1,0]]); Ry = np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    Ib = Rx@np.diag([2.,3,5])@Rx.T
    require(np.array_equal(angular_momentum(Ry,np.array([1.,2,3]),Ib),[3.,10,6]),"Inertial vs body vs world axes")
    return {"schema":"phone_wrench_analytic_fixture:1","passed":True,"cases":3,
        "case_scope":["rotated contact/lever/directmoment/geomorder","instantaneous vs source axes and origin","noncommuting body/inertial/world tensor"],
        "model_native_NN_physics_calls":0}


def preflight(path):
    p=read(path);require(p["schema"]==SCHEMA and p["actual_new_model_native_NN_physics"] is False,"Saved-only protocol")
    for r in [*p["inputs"].values(),*p["sources"].values()]:checked(r)
    require(checked(p["sources"]["exporter"]).samefile(__file__),"Pinned current exporter")
    acq=read(checked(p["inputs"]["acquisition_report"]));audit=read(checked(p["inputs"]["dataset_audit"]))
    require(acq["schema"]=="phone_episode_acquisition_result:1" and acq["status"]=="COMPLETE_GROUPED_NATIVE_ACQUISITION"
        and acq["sample_identities"]==p["inputs"]["sample_identities"] and acq["episodes"]==p["inputs"]["episodes"],"Exact raw collection")
    require(audit["dataset_report"]==p["inputs"]["acquisition_report"] and audit["engineering_qualified"] is True
        and all(v is True for v in audit["checks"].values()),"Existing saved dataset review")
    out=Path(p["output"]);require(out.is_absolute() and out.parent.resolve()==HERE.resolve() and not out.exists(),"Fresh label output")
    return p,acq,out


def statistics(rows, mask, arrays):
    rr=np.flatnonzero(mask);result={"rows":len(rr),"duration_sum_s":float(arrays["duration_s"][rr].sum())}
    for label,key in (("linear_impulse_residual_Ns","linear_residual_world_Ns"),("angular_impulse_residual_Nms","angular_residual_world_Nms"),
                      ("linear_force_equivalent_residual_N","linear_residual_force_equivalent_N"),("angular_torque_equivalent_residual_Nm","angular_residual_torque_equivalent_Nm")):
        v=arrays[key][rr]
        result[label]={"vector_RMS":float(np.sqrt(np.mean(np.sum(v*v,-1)))) if len(rr) else None,
            "max_vector_norm":float(np.linalg.norm(v,axis=-1).max()) if len(rr) else None,
            "per_axis_signed_mean":np.mean(v,axis=0).tolist() if len(rr) else None}
    return result


def run(path):
    p,acq,out=preflight(path);out.mkdir();write(out/"protocol.json",p)
    write(out/"analytic_fixture.json",fixture())
    started=time.perf_counter();codec=codec_module();phase="parameters"
    inputs=p["inputs"];sources=[read(checked(inputs[k])) for k in ("source_exact","source_side")]
    config=[s["native_probe"]["model_configuration"] for s in sources]
    fields=["phone_mass_kg","phone_body_inertia3","phone_body_iquat_wxyz","phone_dof_armature6","phone_massive_descendants"]
    require(all(config[0][k]==config[1][k] for k in fields),"Same saved phone physical parameters")
    mass=config[0]["phone_mass_kg"];require(mass==.2,"Actual native mass")
    IQ=rotation_wxyz(np.array(config[0]["phone_body_iquat_wxyz"]));Ibody=IQ@np.diag(config[0]["phone_body_inertia3"])@IQ.T
    gravity=np.asarray(sources[0]["candidate"]["static"]["support_operands"]["gravity_world"],dtype=float)
    require(np.array_equal(gravity,sources[1]["candidate"]["static"]["support_operands"]["gravity_world"]),"Same saved native world gravity")
    audits=list(acq["model_audits"].values());ipos=np.asarray(audits[0]["body_ipos_body_m"],dtype=float)
    require(all(np.array_equal(ipos,a["body_ipos_body_m"]) for a in audits),"Same saved COM offset")
    params=dict(mass_kg=mass,I_body_3x3=Ibody.tolist(),principal_inertia=config[0]["phone_body_inertia3"],
        body_iquat_wxyz=config[0]["phone_body_iquat_wxyz"],body_ipos_body_m=ipos.tolist(),gravity_world_m_s2=gravity.tolist(),
        sourcepins={k:inputs[k] for k in ("source_exact","source_side","acquisition_report")})
    write(out/"physical_parameters.json",params)
    ids=read(checked(inputs["sample_identities"]));episodes=read(checked(inputs["episodes"]))
    require(len(ids)==960 and len(episodes)==96,"All original rows and episodes")
    rows=[];ticks=[];tick_parts=[];means=[];checks=[];trace_pins={};npz_reads=0
    phase="saved_trace_processing"
    try:
        for episode in episodes:
            ep=episode["episode_id"]
            with np.load(checked(episode["initial"]),allow_pickle=False) as z:
                lastq=z["qpos27"].copy();lastv=z["qvel26"].copy()
            npz_reads+=1
            selected=sorted([s for s in ids if s["episode_id"]==ep],key=lambda s:s["interval_index"])
            for s in selected:
                rid=s["sample_id"];h=float(s["dt_s"])
                trace_pins[str(rid)]=s["trace"]
                with np.load(checked(s["trace"]),allow_pickle=False) as z:
                    a={k:z[k].copy() for k in ["pre_tick","post_tick","post_qpos27","post_qvel26","post_state58",
                        "contact_rows24","pre_contact_wrench_about_COM_world6","source_state58"]}
                npz_reads+=1;n=len(a["pre_tick"])
                require(n==s["executed_substeps"] and n>0 and np.array_equal(a["post_tick"],a["pre_tick"]+1)
                    and np.array_equal(a["pre_tick"],np.arange(a["pre_tick"][0],a["pre_tick"][0]+n)),"Integer tick/prefix identity")
                qpre=np.vstack((lastq,a["post_qpos27"][:-1]));vpre=np.vstack((lastv,a["post_qvel26"][:-1]))
                Spre,Rpre=raw_states(qpre,vpre,ipos,codec)
                Spost,Rpost=raw_states(a["post_qpos27"],a["post_qvel26"],ipos,codec)
                source_err=float(np.max(abs(Spre[0]-a["source_state58"])))
                post_err=float(np.max(abs(Spost-a["post_state58"])))
                W,info=contact_wrenches(a["contact_rows24"],a["pre_tick"],Spre[:,40:43])
                Wbody=rotate_wrench_axes(W,Rpre)
                R0=Rpre[0];Wsource=rotate_wrench_axes(W,np.broadcast_to(R0,(n,3,3)))
                Wsource_origin_W=W.copy();Wsource_origin_W[:,3:]+=np.cross(Spre[:,40:43]-Spre[0,40:43],W[:,:3])
                Wsource_origin=rotate_wrench_axes(Wsource_origin_W,np.broadcast_to(R0,(n,3,3)))
                duration=n*h;J=W.sum(0)*h
                dP=mass*(Spost[-1,52:55]-Spre[0,52:55]);gravityJ=mass*gravity*duration
                L0=angular_momentum(Rpre[0],Spre[0,55:58],Ibody);L1=angular_momentum(Rpost[-1],Spost[-1,55:58],Ibody)
                lin=dP-gravityJ-J[:3];ang=L1-L0-J[3:]
                source_capture_diff=float(np.max(abs(W-a["pre_contact_wrench_about_COM_world6"])))
                rows.append({k:s[k] for k in ["sample_id","family_id","episode_id","interval_index","split","source_branch","dt_s","complete_H20ms","training_eligible","termination","exclusion_reasons"]})
                means.append(dict(row_id=rid,duration_s=duration,world_mean_wrench6=W.mean(0),world_contact_impulse6=J,
                    instantaneous_BODY_mean_wrench6=Wbody.mean(0),instantaneous_BODY_integral6=Wbody.sum(0)*h,
                    source_BODY_axes_moving_COM_mean_wrench6=Wsource.mean(0),source_BODY_axes_moving_COM_impulse6=Wsource.sum(0)*h,
                    source_BODY_fixed_origin_mean_wrench6=Wsource_origin.mean(0),source_BODY_fixed_origin_impulse6=Wsource_origin.sum(0)*h,
                    linear_residual_world_Ns=lin,angular_residual_world_Nms=ang,
                    linear_residual_force_equivalent_N=lin/duration,angular_residual_torque_equivalent_Nm=ang/duration,
                    delta_linear_momentum_world_Ns=dP,gravity_impulse_world_Ns=gravityJ,
                    angular_momentum_start_world_Nms=L0,angular_momentum_end_world_Nms=L1,
                    source_COM_world_m=Spre[0,40:43],target_prefix_COM_world_m=Spost[-1,40:43],source_Rbody_world9=R0.ravel()))
                checks.append(dict(row_id=rid,source_state_rebuild_max_abs=source_err,post_state_rebuild_max_abs=post_err,
                    decoded_reader_wrench_max_abs_difference=source_capture_diff,**info))
                tick_parts.append(dict(row_id=np.full(n,rid,dtype=np.int64),pre_tick=a["pre_tick"],post_tick=a["post_tick"],
                    native_h_s=np.full(n,h),COM_pre_world_m=Spre[:,40:43],Rbody_pre_world9=Rpre.reshape(n,9),
                    world_wrench6=W,instantaneous_BODY_wrench6=Wbody,source_BODY_axes_moving_COM_wrench6=Wsource,
                    source_BODY_fixed_origin_wrench6=Wsource_origin,reader_saved_world_wrench6=a["pre_contact_wrench_about_COM_world6"]))
                lastq=a["post_qpos27"][-1].copy();lastv=a["post_qvel26"][-1].copy()
        require(len(rows)==960 and sum(r["complete_H20ms"] for r in rows)==912,"Retained912complete+48partial, no source filtering")
        order=np.argsort([v["row_id"] for v in means]);means=[means[i] for i in order];rows=[rows[i] for i in order];checks=[checks[i] for i in order]
        arrays={k:np.asarray([v[k] for v in means]) for k in means[0]}
        arrays.update(complete_H_bool=np.array([r["complete_H20ms"] for r in rows]),training_eligible_bool=np.array([r["training_eligible"] for r in rows]),
            family_id=np.array([r["family_id"] for r in rows],dtype=np.int64),episode_id=np.array([r["episode_id"] for r in rows],dtype=np.int64))
        np.savez_compressed(out/"all960_interval_wrench_labels_and_residuals.npz",**arrays)
        np.savez_compressed(out/"all_PRE_tick_wrenches.npz",**{k:np.concatenate([v[k] for v in tick_parts]) for k in tick_parts[0]})
        write(out/"all960_row_identities.json",rows);write(out/"all960_reconstruction_checks.json",checks)
        write(out/"all960_raw_trace_pins.json",trace_pins)
        stats=[]
        for complete in (True,False):
            base=np.array([r["complete_H20ms"]==complete for r in rows])
            label="complete_H" if complete else "actual_partial_prefix"
            stats.append(dict(cohort="ALL",duration_class=label,**statistics(rows,base,arrays)))
            for split in ("train","validation","test"):
                stats.append(dict(cohort=split,duration_class=label,**statistics(rows,base&np.array([r["split"]==split for r in rows]),arrays)))
            for dt in sorted(set(r["dt_s"] for r in rows)):
                stats.append(dict(cohort=f"dt_{dt:g}",duration_class=label,**statistics(rows,base&np.array([r["dt_s"]==dt for r in rows]),arrays)))
            for branch in sorted(set(r["source_branch"] for r in rows)):
                stats.append(dict(cohort=branch,duration_class=label,**statistics(rows,base&np.array([r["source_branch"]==branch for r in rows]),arrays)))
        write(out/"momentum_residual_statistics.json",stats)
        with (out/"momentum_residual_statistics.csv").open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader()
            for r in stats:writer.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()})
        outputs={f.name:pin(f) for f in out.iterdir() if f.is_file()}
        report=dict(schema="phone_saved_object_wrench_labels:1",status="EXPORTED_SIMULATOR_DECODED_WRENCHES_AND_OBSERVED_MOMENTUM_DEFECTS",
            protocol=pin(path),outputs=outputs,rows=960,complete_H_rows=912,actual_partial_prefix_rows=48,
            native_PRE_ticks=sum(len(v["pre_tick"]) for v in tick_parts),NPZ_reads=npz_reads,
            phone_boundary_contact_rows=sum(v["phone_boundary_contacts"] for v in checks),
            max_reader_wrench_reconstruction_abs=max(v["decoded_reader_wrench_max_abs_difference"] for v in checks),
            max_POST_state_reconstruction_abs=max(v["post_state_rebuild_max_abs"] for v in checks),
            complete_ALL=stats[0],elapsed_s=time.perf_counter()-started,
            model_native_FK_NN_AD_physics_new_training_calls=0,real_world_force_GT=False,holding_qualified=False,
            scopes=["All decoded phone-boundary contacts including nonpad/table, no Fn threshold or manual pyramid/hostaddress decoding.",
                "Torque is about current physical COM. Instantaneous BODY integral is not equal to frozen-source-axes integral when orientation changes.",
                "Source BODY axes at moving COM differs from a truly source-fixed-origin wrench; both are exported explicitly.",
                "Only the WORLD moving-COM torque is used in DeltaL balance; body-frame endpoint subtraction needs transport terms, while a fixed-origin balance needs orbital momentum/gravity lever terms.",
                "World momentum equations use actualprefix_duration=n*h, not accumulated clock difference or fabricated H targets.",
                "Residual is observed simulator/discretization/omitted-term defect, not forced zero and not a learned correction or physical law certificate.",
                "Reader asserts no external applied forces/callbacks/gravcomp; no independent per-step passive/applied-force arrays were exported. Missing passive/drift/integrator terms remain limitations.",
                "m/I/COM/g belong to assumed fixed native model, not measured human/phone parameters or real ego force ground truth."])
        write(out/"report.json",report)
        print(json.dumps({"report":pin(out/"report.json"),"rows":960,"ticks":report["native_PRE_ticks"],"complete_ALL":stats[0]}),flush=True)
    except BaseException as e:
        write(out/"failure.json",dict(phase=phase,error=repr(e),completed_rows=len(rows),NPZ_reads=npz_reads,automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--run",action="store_true");a=p.parse_args()
    if a.run:run(a.protocol.resolve(strict=True))
    else:
        _,_,out=preflight(a.protocol.resolve(strict=True));print(json.dumps(dict(status="METADATA_ONLY_NO_ARRAYS_OR_MODELS",output=str(out))))

if __name__=="__main__":main()
