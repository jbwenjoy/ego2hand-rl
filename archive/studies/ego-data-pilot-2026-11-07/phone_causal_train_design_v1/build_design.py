"""Generate a TRAIN-only same-state intervention plan and fresh-final metadata.

Reads frozen TRAIN/cache/XML metadata. No native/NN/AD/FK/physics/training. No
acquisition is launched. Baseline graph observations never become strict zeros.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import causal_contract as C

H=Path(__file__).resolve().parent
PILOT=H.parent
SCHEMA="phone_causal_training_design_protocol:1"


def require(ok,msg):
    if not ok:raise ValueError(msg)


def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def checked(r):
    p=Path(r["path"]);a=pin(p)
    require(p.is_absolute() and a["bytes"]==r["bytes"] and a["sha256"]==r["sha256"],"Pinned source/data changed")
    return p.resolve(strict=True)


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p,v):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(v,f,indent=2,allow_nan=False)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def geometry_metadata(xml_path,binding):
    """Topology from unchanged hand source; contact eligibility from actual explicit pairs.

    XML physics/options are NOT substituted for the later compiled binding.
    Compiled geom IDs are not guessed from XML enumeration order.
    """
    root=ET.parse(xml_path).getroot();parent={c:p for p in root.iter() for c in p}
    joints={j.get("name"):j for j in root.findall(".//body/joint")}
    actuators=list(root.find("actuator"));names=[a.get("joint") for a in actuators]
    require(len(names)==20 and len(set(names))==20 and all(n in joints for n in names),"Source20 direct-actuator joint topology")
    geoms={g.get("name"):g for g in root.findall(".//geom") if g.get("name")}
    pairs=binding["explicit_pairs"]
    names_phone=[]
    for p in pairs:
        require("phone24_collision_convex" in (p["geom1"],p["geom2"]),"Actual explicit phone-boundary pairs")
        names_phone.append(p["geom2"] if p["geom1"]=="phone24_collision_convex" else p["geom1"])
    require(len(names_phone)==22 and len(set(names_phone))==22,"Complete22 explicit phone pairs")
    rows=[]
    for axis,joint in enumerate(names):
        body=parent[joints[joint]];desc=set(body.iter())
        descendant=[name for name in names_phone if name in geoms and geoms[name] in desc]
        rows.append(dict(actuator_index=axis,joint_name=joint,body_name=body.get("name"),
            directly_phone_enabled_descendant_geoms=descendant,possible_direct_contact_path=bool(descendant),
            strict_zero_path_certified=False))
    return dict(schema="phone_fixed_model_contact_topology_metadata:1",actual_explicit_phone_pair_names=names_phone,
        actuator_paths=rows,compiled_geom_ids_inferred_from_XML=False,
        static_XML_physics_not_reused=True,pair_rule="Explicit pairs matter even when geom contype/conaffinity are0",
        strict_scope="Potential paths only. Active/absent baseline contacts do not certify intervention or whole-H disconnection.")


def side(u,joint,sign,limits,eps):
    margin=(limits[joint,1]-u[joint]) if sign>0 else (u[joint]-limits[joint,0])
    step=min(eps,max(0.,.5*float(margin)))
    endpoint=u.copy();endpoint[joint]=u[joint]+sign*step
    du=float(endpoint[joint]-u[joint]);du32=float(np.float32(endpoint[joint])-np.float32(u[joint]))
    other=np.arange(20)!=joint
    require(endpoint[other].tobytes()==u[other].tobytes(),"Other19 coordinates unchanged")
    legal=bool(np.all(endpoint>=limits[:,0]) and np.all(endpoint<=limits[:,1]))
    available=bool(sign*du>=1e-6 and legal)
    return dict(joint_index=joint,sign=sign,requested_step_rad=eps,actual_delta_u_F64_rad=du,
        represented_delta_u_FP32_rad=du32,model_FP32_endpoint_collapsed=du32==0,
        step_rule="min(.01,half actual remaining legal margin); require actual_absdu>=1e-6; no clipping or step-up rescue",
        status="PLANNED_NATIVE_SIDE" if available else "UNAVAILABLE_AT_BOUND_COLLAPSE_OR_ACTUAL_STEP_BELOW_1e-6",
        available=available,endpoint_u20_F64=endpoint.tolist(),endpoint_u_raw_sha256=array_sha(endpoint))


def uniform(tag,index):
    raw=hashlib.sha256((tag+":"+str(index)).encode()).digest()
    return ((int.from_bytes(raw[:8],"big")>>11)+.5)/(2**53)


def new_final_families(plan,design):
    """Literal new seeds/initials/actions; not acquired and not used for fitting.

    Same two known source poses/model, so not independent ego scenes. Distinct
    controlled initializations/actions can support later fresh-family validation.
    """
    qlimits=np.asarray(design["actual_limits"]["joint_range20x2_F64"])
    ulim=np.asarray(design["actual_limits"]["actuator_ctrlrange20x2_F64"])
    oldq={f["initial_qpos27_raw_F64_sha256"] for f in plan["families"]}
    oldu={f["held_ctrl10x20_raw_F64_sha256"] for f in plan["families"]}
    bases={}
    for f in plan["families"]:bases.setdefault(f["source_branch"],f)
    result=[];rejected=[]
    for bi,(branch,b) in enumerate(sorted(bases.items())):
        source=read(checked(b["source_record"]))["native_probe"]["raw"]
        q0=np.asarray(source["qpos27"],dtype=np.float64);u0=np.asarray(b["base_ctrl20_F64"],dtype=np.float64)
        for profile in range(3):
            for rep in range(2):
                fid=1000+len(result);tag=f"phone_causal_fresh_final_v1:{branch}:{profile}:{rep}"
                q=None
                for attempt in range(64):
                    candidate=q0.copy();candidate[:8]+=.01*(np.array([uniform(tag+":q:"+str(attempt),j) for j in range(8)])-.5)
                    if np.all(candidate[:20]>=qlimits[:,0]) and np.all(candidate[:20]<=qlimits[:,1]) and array_sha(candidate) not in oldq:
                        q=candidate;break
                    rejected.append(dict(family_id=fid,attempt=attempt,qpos27_F64=candidate.tolist(),reason="hard_joint_range_or_old_initial_hash"))
                require(q is not None,"Fresh-final initial rejection budget exhausted; no replacement scene")
                delta=np.zeros((10,20))
                for k in range(10):
                    for j in range(20):
                        phase=2*np.pi*uniform(tag+":phase",j);amplitude=.05 if j<8 else .025
                        v=np.sin(2*np.pi*(k+.5)/10+phase)
                        if profile==1 and j<8:v=np.sin(2*np.pi*(k+.5)/10)*(-1 if j>=4 else 1)
                        if profile==2 and j<8:v=-abs(v)
                        delta[k,j]=amplitude*v
                alpha=np.ones(20)
                for j in range(20):
                    ratios=[(ulim[j,1]-u0[j])/v if v>0 else (ulim[j,0]-u0[j])/v for v in delta[:,j] if v!=0]
                    alpha[j]=min([1.]+ratios)
                    require(alpha[j]>=0,"Legal original base control")
                    if alpha[j]<1:alpha[j]*=(1-1e-12)
                controls=u0+delta*alpha
                require(np.all(controls>=ulim[:,0]) and np.all(controls<=ulim[:,1]),"Whole-curve legal scaling, no clipping")
                require(array_sha(controls) not in oldu,"Fresh final control curve must differ from original48")
                oldq.add(array_sha(q));oldu.add(array_sha(controls))
                result.append(dict(family_id=fid,source_branch=branch,source_record=b["source_record"],
                    split="NEW_FINAL_VALIDATION_SEALED_BEFORE_ACQUISITION",profile=profile,replicate=rep,
                    seed_namespace=tag,initialization_seed=3100000+fid,control_seed=4200000+fid,
                    initial_qpos27_F64=q.tolist(),initial_qvel26_F64=[0.]*26,base_ctrl20_F64=u0.tolist(),
                    requested_delta10x20_F64=delta.tolist(),whole_curve_alpha20_F64=alpha.tolist(),
                    held_ctrl10x20_F64=controls.tolist(),native_dt_s=[.0003125,.00015625],
                    initial_raw_sha256=array_sha(q),control_raw_sha256=array_sha(controls),
                    original48_family_relabelled=False,acquired=False,used_for_fit_or_model_selection=False))
    return dict(schema="phone_new_fresh_final_family_plan:1",families=result,rejected_initial_proposals=rejected,
        new_families=12,max_native_steps=23040,max_H20ms_rows=240,
        scope="New controlled initials/actions, same source geometry/model. Not acquired, not fit; never called independent ego/scene generalization.")


def preflight(path):
    p=read(path);require(p["schema"]==SCHEMA and p["execute_native_or_NN_training"] is False,"Metadata-only design")
    for r in [*p["inputs"].values(),*p["sources"].values()]:checked(r)
    require(checked(p["sources"]["generator"]).samefile(__file__),"Current generator")
    out=Path(p["output"]);require(out.is_absolute() and out.parent.resolve()==H.resolve() and not out.exists(),"Fresh design output")
    return p,out


def run(path):
    p,out=preflight(path);out.mkdir();save(out/"protocol.json",p)
    acq=read(checked(p["inputs"]["acquisition_report"]));audit=read(checked(p["inputs"]["dataset_audit"]))
    require(audit["engineering_qualified"] is True and audit["dataset_report"]==p["inputs"]["acquisition_report"],"Frozen saved acquisition audit")
    plan=read(checked(p["inputs"]["family_plan"]));design=read(checked(p["inputs"]["acquisition_design"]))
    require(acq["family_plan"]==p["inputs"]["family_plan"] and acq["partition_qualified"] is True,"Original48-family split")
    binding=read(checked(p["inputs"]["model_binding"]))
    require(binding["robot_gains_limits_geometry_wrist_unchanged"] is True
        and all(binding["model"]==a["model_pin"] for a in acq["model_audits"].values()),"Same compiled model and unchanged hand geometry/tree")
    topology=geometry_metadata(checked(p["inputs"]["hand_scene_XML"]),binding);save(out/"geometry_topology_contract.json",topology)
    selected={f["family_id"]:("CAUSAL_FIT_KNOWN_TRAIN_STATE" if f["replicate"]==0 else "HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE")
        for f in plan["families"] if f["split"]=="train" and f["replicate"] in (0,5)}
    require(len(selected)==12,"Six fit/six action-check families, all originallyTRAIN")
    with np.load(checked(acq["outputs"]["train"]),allow_pickle=False) as z:
        tr={k:z[k].copy() for k in ("row_id_i64","family_id_i64","episode_id_i64","interval_index_i64","native_dt_s_F64","n_substeps_i64","training_eligible_bool","x58_F64","u20_F64")}
    old80=np.column_stack((tr["x58_F64"],tr["u20_F64"],tr["native_dt_s_F64"],tr["n_substeps_i64"]))
    old64={row.tobytes() for row in old80[tr["training_eligible_bool"]]}
    old32={row.astype(np.float32).tobytes() for row in old80[tr["training_eligible_bool"]]}
    identities=read(checked(acq["sample_identities"]));idmap={s["sample_id"]:s for s in identities}
    episodes=read(checked(acq["episodes"]));epmap={e["episode_id"]:e for e in episodes}
    families={f["family_id"]:f for f in plan["families"]};ulim=np.asarray(design["actual_limits"]["actuator_ctrlrange20x2_F64"])
    pick=[i for i in range(len(tr["row_id_i64"])) if int(tr["family_id_i64"][i]) in selected and int(tr["interval_index_i64"][i]) in (0,2,5)]
    pick.sort(key=lambda i:(int(tr["family_id_i64"][i]),float(tr["native_dt_s_F64"][i]),int(tr["interval_index_i64"][i])))
    require(len(pick)==72,"All predeclared source states; no outcome replacement")
    contexts=[];vectors=[];aftervectors=[];states=[];controls=[];empirical=[];baselines=[];sides=[];initial_cache={}
    for i in pick:
        rid=int(tr["row_id_i64"][i]);fid=int(tr["family_id_i64"][i]);ep=int(tr["episode_id_i64"][i]);j=int(tr["interval_index_i64"][i]);s=idmap[rid]
        require(s["split"]=="train" and s["family_id"]==fid and bool(tr["training_eligible_bool"][i]),"Only complete finite originalTRAIN sources")
        with np.load(checked(s["trace"]),allow_pickle=False) as z:
            before=z["source_integration_state"].copy();after=z["source_after_action_integration_state"].copy()
            x=z["source_state58"].copy();u=z["actual_control20"].copy();contact=z["contact_rows24"].copy();pretick=z["pre_tick"].copy()
        require(before.dtype==after.dtype==np.float64 and before.shape==after.shape==(s["source_receipt"]["size"],)
            and array_sha(before)==s["source_receipt"]["sha256"] and array_sha(after)==s["source_after_action_receipt"]["sha256"],"Full before/after-action integration raw receipts")
        require(x.tobytes()==tr["x58_F64"][i].tobytes() and u.tobytes()==tr["u20_F64"][i].tobytes()
            and u.tobytes()==np.asarray(families[fid]["held_ctrl10x20_F64"],dtype=np.float64)[j].tobytes(),"Source state and actual/planned u bytes")
        require(np.all(u>=ulim[:,0]) and np.all(u<=ulim[:,1]),"True actual actuator limits")
        if ep not in initial_cache:
            with np.load(checked(epmap[ep]["initial"]),allow_pickle=False) as z:
                initial_cache[ep]={k:z[k].copy() for k in ("qpos27","state58")}
        phone=(contact[:,1]==71)^(contact[:,2]==71);other=np.where(contact[:,1]==71,contact[:,2],contact[:,1])
        cidx=len(contexts);h=float(tr["native_dt_s_F64"][i]);n=int(tr["n_substeps_i64"][i])
        obs=dict(baseline_generated_phone_contact_rows=int(phone.sum()),baseline_phone_contact_at_any_PRE=bool(phone.any()),
            baseline_designated_pad_contact_rows=int((phone&np.isin(other,[15,28])).sum()),
            baseline_no_generated_phone_contact_over_H=not bool(phone.any()),
            strict_counterfactual_zero_certified=False,scope="Whole baseline H only, not swept/intervention/coupling-path proof")
        ctx=dict(context_id=cidx,source_sample_id=rid,original_family_id=fid,new_response_family_id=f"known_train_response_{fid:03d}",
            partition=selected[fid],original_partition="train",source_episode_id=ep,interval_index=j,source_branch=s["source_branch"],
            native_h_s=h,n_substeps=n,H_s=.02,source_trace=s["trace"],before_action_receipt=s["source_receipt"],
            after_action_receipt=s["source_after_action_receipt"],restore_array_index=cidx,baseline_u20_F64=u.tolist(),
            original_episode_initial=epmap[ep]["initial"],baseline_observation=obs,
            initial_COM_world_m=initial_cache[ep]["state58"][40:43].tolist(),
            initial_phone_quaternion_wxyz=initial_cache[ep]["qpos27"][23:27].tolist(),
            hard_stops=design["safety"],strict_zero_mask_source="NO_V1_FULL_H_CERTIFICATE")
        contexts.append(ctx);vectors.append(before);aftervectors.append(after);states.append(x);controls.append(u);empirical.append(obs)
        baselines.append(dict(kind="BASELINE",context_id=cidx,new_response_family_id=ctx["new_response_family_id"],partition=selected[fid],
            endpoint_u20_F64=u.tolist(),endpoint_u_raw_sha256=array_sha(u),status="PLANNED",source_before_action_array_index=cidx))
        for joint in range(20):
            for sign in (-1,1):
                sd=side(u,joint,sign,ulim,.01)
                raw80=np.r_[x,sd["endpoint_u20_F64"],h,n].astype(np.float64)
                alias64=raw80.tobytes() in old64;alias32=raw80.astype(np.float32).tobytes() in old32
                sd.update(endpoint_already_in_original_TRAIN_raw80_F64=alias64,
                    endpoint_already_in_original_TRAIN_represented80_FP32=alias32,
                    new_action_response_check_eligible=bool(sd["available"] and not alias64 and not alias32))
                sides.append(dict(kind="FINITE_SIDE",context_id=cidx,new_response_family_id=ctx["new_response_family_id"],partition=selected[fid],
                    source_before_action_array_index=cidx,**sd))
    branches=baselines+sides
    for i,b in enumerate(branches):b["branch_id"]=i
    require(all(b["kind"]=="BASELINE" for b in branches[:72]),"All baselines before every side")
    # Same-state receipts, native h/model and whole family are preserved. Any
    # duplicate source crossing fit/check blocks future acquisition admission.
    alias={}
    for c in contexts:
        key=(c["before_action_receipt"]["signature"],c["before_action_receipt"]["sha256"],c["native_h_s"])
        alias.setdefault(key,[]).append(c)
    cross=[dict(context_ids=[c["context_id"] for c in cc],partitions=sorted(set(c["partition"] for c in cc)))
        for cc in alias.values() if len(set(c["partition"] for c in cc))>1]
    require(not cross,"Cross fit/action-check full-state alias: block, never silently reassign")
    strict=C.strict_object_zero_mask(72)
    np.savez_compressed(out/"source_before_action_restore_arrays.npz",source_integration_state_F64=np.asarray(vectors),
        source_after_action_integration_state_F64=np.asarray(aftervectors),x58_F64=np.asarray(states),baseline_u20_F64=np.asarray(controls))
    np.savez_compressed(out/"causal_masks_and_descriptors.npz",strict_object_action_zero72x20=strict,
        empirical_baseline_no_phone_contact72=np.array([e["baseline_no_generated_phone_contact_over_H"] for e in empirical]),
        potential_direct_contact_path20=np.array([a["possible_direct_contact_path"] for a in topology["actuator_paths"]]),
        native_side_available72x20x2=np.array([s["available"] for s in sides]).reshape(72,20,2),
        model_FP32_endpoint_collapsed72x20x2=np.array([s["model_FP32_endpoint_collapsed"] for s in sides]).reshape(72,20,2))
    save(out/"contexts72.json",contexts);save(out/"all_baselines_then_sides_plan.json",branches)
    fresh=new_final_families(plan,design);save(out/"new_final_family_plan.json",fresh)
    exposure=dict(original48_family_roles_unchanged=True,original_TRAIN_used_in_WM_fit=True,
        known_state_response_fit_family_ids=sorted(fid for fid,v in selected.items() if v.startswith("CAUSAL_FIT")),
        known_state_heldout_action_check_family_ids=sorted(fid for fid,v in selected.items() if v.startswith("HELDOUT")),
        known_state_action_check_is_unseen_state_or_ego=False,original_VAL476_diagnostic_branches_allowed_in_TRAIN=False,
        original_TEST6_exposed_in_first_model_evaluation=True,original_TEST_relabelled_unseen=False,
        fresh_final_family_ids=[f["family_id"] for f in fresh["families"]],fresh_final_not_acquired_or_fit=True)
    save(out/"exposure_and_partition_policy.json",exposure)
    feature=dict(schema=C.SCHEMA,fields=sorted(C.FEATURE_FIELDS),phone_pair_names=topology["actual_explicit_phone_pair_names"],
        designated_pad_geom_ids=[15,28],third_fingertip_still_other=True,
        source_PRE_input_fields=sorted(C.FEATURE_FIELDS),
        target_supervision_only_fields=["whole_H_contacts","whole_H_path_certificate","baseline_no_phone_contact_over_H","native_endpoint_delta52","decoded_contact_impulses","counterfactual_response"],
        future_contact_certificate_is_not_deployment_input=True,
        coordinate_contract="Geometry from current/predicted q20 + full body R/COM state, root recovered with actual ipos; normals/points in BODY axes. No future/native decoded force inputs.",
        strict_zero_policy="No certificates issued in v1. Current collision emptiness or whole-baseline empty contacts are not proof for altered actions/fullH.",
        provider_required="Differentiable hand FK + compiled collision geometry query on current imagined state; source snapshot geom positions are not future features.",
        unavailable_provider_at_this_stage=True,pad_functional_face_semantics="UNKNOWN unless separate hardware/CAD evidence; whole sensor mesh is not a certified pad patch.",
        future_targets="New native matched baseline/side state58, common world delta52, sourceBODY movingCOM impulse6 per all/pad/other contacts; targets stay outside forward inputs.")
    save(out/"geometry_contact_input_contract.json",feature)
    steps=sum(c["n_substeps"]*41 for c in contexts)
    report=dict(schema="phone_causal_train_metadata_design:1",status="GENERATED_VERIFIABLE_TRAIN_INTERVENTION_AND_NEW_FINAL_PLANS_NOT_ACQUIRED",
        protocol=pin(path),sources=p["sources"],inputs=p["inputs"],contexts=72,all20_axes=True,baseline_branches=72,side_records=2880,
        available_native_sides=sum(s["available"] for s in sides),unavailable_sides=sum(not s["available"] for s in sides),
        side_endpoints_alias_original_TRAIN_F64=sum(s["endpoint_already_in_original_TRAIN_raw80_F64"] for s in sides),
        side_endpoints_alias_original_TRAIN_FP32=sum(s["endpoint_already_in_original_TRAIN_represented80_FP32"] for s in sides),
        model_FP32_collapsed_sides=sum(s["model_FP32_endpoint_collapsed"] for s in sides),maximum_native_steps_stage1=steps,
        fresh_final_families=12,fresh_final_max_native_steps=23040,strict_zero_mask_true_count=int(strict.sum()),
        empirical_baseline_no_phone_contact_contexts=sum(e["baseline_no_generated_phone_contact_over_H"] for e in empirical),
        all20_structural_direct_paths_possible=all(a["possible_direct_contact_path"] for a in topology["actuator_paths"]),
        exposure_policy=exposure,outputs={f.name:pin(f) for f in out.iterdir() if f.is_file()},
        execution_counts=dict(TRAIN_NPZ_reads=1,selected_TRAIN_trace_NPZ_reads=72,TRAIN_initial_NPZ_reads=len(initial_cache),
            VAL_TEST_NPZ_reads=0,new_models_FK_native_steps_NN_AD_training=0),
        limitations=["Design generation only. All new baseline/side/fresh outcomes and failures remain unobserved; no execution is authorized here.",
            "Known-state check holds out new action responses only; original states were used in5000-update fit. Fresh12 families require later acquisition/evaluation without model selection on their results.",
            "Every axis can potentially contact phone through an eligible descendant; HH/dynamic coupling also exists. No strict object zero masks are justified by current metadata.",
            "Future failed/partial branches and unavailable sides must remain in dataset with masks/reasons, never replaced or selected on outcomes."])
    save(out/"report.json",report);print(json.dumps(dict(report=pin(out/"report.json"),contexts=72,branches=len(branches),max_steps=steps,strict_zero_count=0,actual_native_NN_training=0)))


def main():
    a=argparse.ArgumentParser();a.add_argument("--protocol",type=Path,required=True);a.add_argument("--build-metadata",action="store_true");p=a.parse_args()
    if p.build_metadata:run(p.protocol.resolve(strict=True))
    else:
        _,out=preflight(p.protocol.resolve(strict=True));print(json.dumps(dict(status="SOURCE_METADATA_PREFLIGHT",output=str(out))))

if __name__=="__main__":main()
