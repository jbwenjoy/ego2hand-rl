"""Bounded CPU path: strict3field reader -> bone-only IK -> native BODY ABI.

All10source[0,151) are geometry audits, including heldout sources. No learned
statistics/weights, cachedsourceq/link/wrist, contactquery, physics or GPU.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEX = Path("D:/workspace/dex-supermimic-research")
sys.path.insert(0, str(ROOT / "src"))
from shared_reference import open_catalog, load_observation, geometry_arrays
from native_reference_abi import export_clip

MODEL = Path("D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb")
KIN = Path("D:/workspace/supermimic-observation-retarget-study/scripts/retarget_palmroot_cpu.py")
EXPORTER = ROOT / "src/native_reference_abi.py"
CONSTANTS = DEX / "src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py"
NATIVE_READER = DEX / "src/dex_supermimic/tasks/pen_spin/mdp/commands.py"
SOFT_LIMIT_S = 300.


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def deadline():
    if time.monotonic() - STARTED > SOFT_LIMIT_S:
        raise TimeoutError("WholeCPU soft300s reached; retainpartial/no retry/no sourceqfallback")


def pinned_inputs():
    pins = {
        "model": {"path": str(MODEL), "sha256": "24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd"},
        "legacy_kinematics": {"path": str(KIN), "sha256": "60a2496633415e65f1b851e44992f5f703be61981c05d9ae6948dc5804eda9e3"},
        "root_ABI_exporter": {"path": str(EXPORTER), "sha256": "8cefd4f1e4ec6939b8201c26670a691c913ab0b5e9e9f87347904e4889fd6e2d"},
        "native_names": {"path": str(CONSTANTS), "sha256": "2a1a37a0c97f3bcab631006ac6e0dc47a1a63a92c7717ac4c6c712314ba35986"},
        "native_reader": {"path": str(NATIVE_READER), "sha256": "e1adb7c183aa28dde49f6dc6e7795ed450567e6c54347bd7f1e0543922fcbcf0"},
        "G1_reader": {"path": str(ROOT / "src/shared_reference/reader.py"), "sha256": "8b601a98ac00884625fc55898238e74d5332ed7f165954359e3e6872e1e191bf"},
        "audit_catalog": {"path": str(ROOT / "catalogs/audit_all10.json"), "sha256": "ed6dfebd418150ef0296456bc7e0ebd45a394190325ad7d61d05db06eee0bb4c"},
        "G1_manifest": {"path": str(ROOT / "outputs/g1_reader_v1/manifest.json"), "sha256": "58e13461dd73fb2d3b07d42807687651a60cfcf17bf6b978b9cbc0846bb9e618"},
    }
    for name, path in (("batch_source", Path(__file__)), ("G1_package", ROOT / "src/shared_reference/__init__.py"),
                       ("native_recipe", DEX / "src/dex_supermimic/tasks/pen_spin/tooling/recipe.py"),
                       ("native_clip_store", DEX / "src/dex_supermimic/tasks/pen_spin/tooling/clip_store.py")):
        pins[name] = {"path": str(path), "sha256": sha(path)}
    for name, pin in pins.items():
        if sha(pin["path"]) != pin["sha256"]: raise ValueError("Source/input drift: " + name)
    return pins


def optimize_clip(obs, kin, directory):
    started = time.monotonic(); directory.mkdir(parents=True, exist_ok=False)
    arrays = geometry_arrays(obs)
    t, human, object_pose = (arrays[k] for k in ("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
    robot = kin.RobotKinematics(MODEL)  # Reset nominal/temporal prior for EACHclip.
    assert kin.TEMPORAL_WEIGHT == .03 and kin.NOMINAL_WEIGHT == .001 and kin.MAX_NFEV == 80
    human_frames = kin.hand_frames(human)
    rotations = np.einsum("ij,tkj->tik", robot.palm_frame, human_frames)
    desired = np.einsum("tij,tkj->tki", rotations, kin.directions(human))
    translation = robot.neutral[0] - np.einsum("tij,tj->ti", rotations, human[:, 0])
    mapped_pos = np.einsum("tij,tj->ti", rotations, object_pose[:, :3]) + translation
    object_R = Rotation.from_quat(object_pose[:, [4, 5, 6, 3]]).as_matrix()
    mapped_quat = Rotation.from_matrix(rotations @ object_R).as_quat()[:, [3, 0, 1, 2]]
    mapped = np.c_[mapped_pos, mapped_quat]
    qs, fks, targets, angles, optimizers = [], [], [], [], []
    residuals, jacobians, gradients, active_masks = [], [], [], []
    previous = robot.nominal.copy(); identity = np.eye(20); failure = None
    write(directory / "observation_provenance.json", obs.provenance_dict())
    try:
        for frame, goal in enumerate(desired):
            deadline()
            def residual(q):
                deadline(); feature, _ = robot.bone_features(q)
                return np.r_[(feature-goal).ravel(), .03*(q-previous), .001*(q-robot.nominal)]
            def jacobian(q):
                deadline(); return np.vstack((robot.bone_features(q)[1], .03*identity, .001*identity))
            result = least_squares(residual, previous.copy(), jac=jacobian, bounds=(robot.lower, robot.upper),
                                   max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = result.x.copy(); points, _ = robot.forward(q); feature, _ = robot.bone_features(q)
            if not np.isfinite(q).all(): raise FloatingPointError("Nonfinite optimizedq; nofallback")
            angle = np.rad2deg(np.arccos(np.clip(np.sum(feature*goal, axis=-1), -1, 1)))
            target = robot.neutral.copy()
            for finger in range(5):
                first = 1+4*finger
                for bone in range(3):
                    j = 3*finger+bone
                    target[first+bone+1] = target[first+bone] + robot.lengths[j]*goal[j]
            qs.append(q); fks.append(points.copy()); targets.append(target); angles.append(angle)
            residuals.append(result.fun.copy()); jacobians.append(np.asarray(result.jac).copy())
            gradients.append(result.grad.copy()); active_masks.append(result.active_mask.copy())
            optimizers.append({"frame": frame, "success": bool(result.success), "status": int(result.status),
                               "message": str(result.message), "nfev": int(result.nfev), "njev": int(result.njev),
                               "cost": float(result.cost), "optimality": float(result.optimality), "full_arrays": "optimizer_full_arrays.npz"})
            previous = q  # Unconvergedfinite result retained, no qualityselection.
    except Exception as error:
        failure = repr(error)
    count = len(qs)
    qs = np.asarray(qs).reshape(count, 20); fks = np.asarray(fks).reshape(count, 21, 3)
    targets = np.asarray(targets).reshape(count, 21, 3); angles = np.asarray(angles).reshape(count, 15)
    geometric = directory / "geometric_retarget.npz"
    np.savez_compressed(geometric, time_s=t[:count], joint_pos_rad=qs, fk_hand_points_w=fks,
                        morphology_normalized_target_points_w=targets, bone_angle_error_deg=angles,
                        source_to_fixed_wrist_R=rotations[:count], source_to_fixed_wrist_translation_m=translation[:count],
                        mapped_object_pose_w=mapped[:count], joint_lower_rad=robot.lower, joint_upper_rad=robot.upper,
                        desired_unit_bone_directions=desired[:count], nominal_q_initial=robot.nominal)
    np.savez_compressed(directory / "optimizer_full_arrays.npz", residual=np.asarray(residuals).reshape(count, 85),
                        jacobian=np.asarray(jacobians).reshape(count, 85, 20), gradient=np.asarray(gradients).reshape(count, 20),
                        active_mask=np.asarray(active_masks).reshape(count, 20), q=qs)
    write(directory / "optimizer_all_frames.json", {"rows": optimizers, "unfinished_frame_index": count if count<len(t) else None, "failure": failure})
    excess = np.maximum(np.maximum(robot.lower-qs, qs-robot.upper), 0)
    report = {"status": "COMPLETED_GEOMETRY_ONLY" if count==len(t) and failure is None else "FAILED_PARTIAL_GEOMETRY",
              "source_id": obs.provenance["source_id"], "selected_source_frames": [0,151], "frames_requested": len(t), "frames_completed": count,
              "failure": failure, "geometric_npz_sha256": sha(geometric), "all_optimizer_records_sha256": sha(directory/"optimizer_all_frames.json"),
              "all_optimizer_arrays_sha256": sha(directory/"optimizer_full_arrays.npz"), "optimizer_success_count": sum(x["success"] for x in optimizers),
              "optimizer_unconverged_frames": [x["frame"] for x in optimizers if not x["success"]],
              "hard_joint_excess_max_rad": float(excess.max()) if count else None,
              "bone_angle_mean_deg": float(angles.mean()) if count else None, "bone_angle_max_deg": float(angles.max()) if count else None,
              "normalized_target_tip_error_mean_m": float(np.linalg.norm(fks[:, [4,8,12,16,20]]-targets[:, [4,8,12,16,20]], axis=-1).mean()) if count else None,
              "q_goal_FD_abs_max_rad_s": float(np.abs(np.diff(qs,axis=0)/np.diff(t[:count])[:,None]).max()) if count>1 else None,
              "q_goal_FD_scope": "Geometric commandgoal derivative, not measuredvelocity or actualexecutedtarget rate",
              "nominal_prior_reset_each_clip": True, "unknown_confidence_preserved": obs.provenance_dict()["confidence"],
              "task_redesign": "Per-frame landmark palm R/t to fixednominal robot palm; no cachedwrist. Object/gravity relation changes and remains unqualified.",
              "macro_FK_scope": "palm/link1/link3/link4/tip; native BODY21 must be separatelyrecomputed",
              "source_robot_q_link_wrist_read": False, "physical_reference_qualified": False, "training_authorized": False,
              "physics_steps": 0, "wall_seconds_geometry_body": time.monotonic()-started}
    write(directory / "report.json", report)
    return report, qs, mapped[:count], fks, geometric


def audit_native(out, exported):
    deadline(); started=time.monotonic()
    import torch
    from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader, object_velocity_from_pose
    selection=resolve_recipe(out/"native_recipe.json", clips_root=out/"clips", verify_clip_hashes=True)
    assert len(selection.clip_sets)==1
    clipset=selection.clip_sets[0]; loaded=SingleHandTrackingReader(clipset,device="cpu").read()
    expected_order=["observation_derived_geometry/"+x["clip_id"] for x in exported]
    assert [clip.clip_id for clip in clipset.clips]==expected_order
    lengths=loaded["trajectory_lengths"].numpy(); expected_lengths=[151]*len(exported)
    assert lengths.tolist()==expected_lengths
    starts=np.r_[0,np.cumsum(lengths)[:-1]];ends=np.cumsum(lengths)
    checks=[]
    for row,start,end in zip(exported,starts,ends):
        deadline()
        with np.load(row["native_path"],allow_pickle=False) as z:
            native={k:z[k].copy() for k in z.files}
        assert set(native)=={"reference_qpos","wrist_pose_w","object_pose_w","link_pos_w"}
        q=native["reference_qpos"].astype(np.float32);v=np.zeros_like(q)
        v[:-1]=(q[1:]-q[:-1])/np.float32(.02);v[-1]=v[-2]
        wrist=native["wrist_pose_w"];pose=native["object_pose_w"]
        Rw=Rotation.from_quat(wrist[:,[4,5,6,3]]).as_matrix();Ro=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
        expect_p=np.einsum("tji,tj->ti",Rw,pose[:,:3]-wrist[:,:3])
        expect_R=np.einsum("tji,tjk->tik",Rw,Ro)
        expect_links=np.einsum("tji,tkj->tki",Rw,native["link_pos_w"]-wrist[:,None,:3])
        got_pose=loaded["object_pose"][start:end].numpy();got_R=Rotation.from_quat(got_pose[:,[4,5,6,3]]).as_matrix()
        errors={"q_FP32":float(np.abs(loaded["joint_pos"][start:end].numpy()-q).max()),
                "goal_qd_FP32":float(np.abs(loaded["joint_vel"][start:end].numpy()-v).max()),
                "object_position_local_m":float(np.abs(got_pose[:,:3]-expect_p).max()),
                "object_R_local":float(np.abs(got_R-expect_R).max()),
                "native_BODY_local_m":float(np.abs(loaded["link_pos"][start:end].numpy()-expect_links).max())}
        assert errors["q_FP32"]==errors["goal_qd_FP32"]==0 and max(errors[k] for k in errors if k not in ("q_FP32","goal_qd_FP32"))<1e-6
        # Invoke the real posevelocity helper pertrajectory, including its edge
        # rule. There is no simulationenv and no crossclip object differentiation.
        world=torch.tensor(pose,dtype=torch.float32,device="cpu")
        lin,ang=object_velocity_from_pose(world,frame_dt=.02)
        lin_start,ang_start=object_velocity_from_pose(world[:2],frame_dt=.02)
        lin_end,ang_end=object_velocity_from_pose(world[-2:],frame_dt=.02)
        assert torch.equal(lin[0],lin_start[0]) and torch.equal(ang[0],ang_start[0])
        assert torch.equal(lin[-1],lin_end[-1]) and torch.equal(ang[-1],ang_end[-1])
        boundary={"joint_last_velocity_equals_own_last_difference":bool(np.array_equal(loaded["joint_vel"][end-1].numpy(),v[-1])),
                  "object_first_last_velocity_from_own_clip_pairs":True,"object_goal_velocity_is_designed_FD_not_measured":True}
        if end<int(ends[-1]):
            erroneous=(loaded["joint_pos"][end]-loaded["joint_pos"][end-1])/.02
            boundary["if_crossclip_differenced_discrepancy_max_rad_s"]=float((erroneous-loaded["joint_vel"][end-1]).abs().max())
        with np.load(row["geometric_path"],allow_pickle=False) as geom:
            macro_difference=float(np.linalg.norm(native["link_pos_w"]-geom["fk_hand_points_w"],axis=-1).max())
        assert macro_difference>1e-4
        checks.append({"source_id":row["source_id"],"trajectory_start":int(start),"trajectory_end_exclusive":int(end),"errors":errors,
                       "boundary":boundary,"native_vs_macro_21_max_distance_m":macro_difference})
    assert not torch.cuda.is_initialized() and all(not x.is_cuda for x in loaded.values() if torch.is_tensor(x))
    return {"status":"PASS_REAL_NATIVE_READER_CPU", "loaded_clip_order":expected_order,"trajectory_lengths":lengths.tolist(),
            "trajectory_starts":starts.tolist(),"trajectory_ends":ends.tolist(),"per_clip":checks,"CUDA_initialized":False,
            "all_tensors_CPU":True,"normalizer_or_modelweights_created_or_updated":False,"reader_wall_seconds":time.monotonic()-started,
            "reader_source_sha256":sha(NATIVE_READER),"physical_reference_qualified":False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=ROOT/"outputs/geometry_native_batch_v1")
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args(); pins=pinned_inputs()
    recipe={"schema":"observation_geometry_native_batch:1","status":"FROZEN_CPU_RECIPE_NOT_TRAINING_OR_PHYSICAL_AUTHORITY",
            "source_pins":pins,"source_frame_range_half_open":[0,151],"catalog_use":"geometry_preparation","clips":10,
            "least_squares":{"temporal_weight":.03,"nominal_weight":.001,"max_nfev":80,"ftol":1e-8,"xtol":1e-8,"gtol":1e-8,"initial_each_clip":"robotnominal"},
            "soft_total_CPU_wall_seconds":300,"no_quality_selection_or_sourceqfallback":True,"unknown_confidence_preserved":True,
            "all10_including_holdout_scope":"Geometry auditonly, no learnednormalizer/modelstatistics update. Futuretraincatalog still excludes009/010.",
            "training_authorized":False,"physical_reference_qualified":False,"GPU_authorized":False}
    if not args.run:print(json.dumps(recipe,indent=2));return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/"frozen_recipe.json",recipe);(out/"batch_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"stage":"FROZEN_BEFORE_OPTIMIZATION","recipe_sha256":sha(out/"frozen_recipe.json"),"softCPUlimit_s":300}),flush=True)
    original={};forbidden={}
    for name in ("mj_step","mj_step1","mj_step2","mj_forward","mj_inverse","mj_contactForce","mj_geomDistance","mj_collision","mj_fwdPosition","mj_fwdVelocity"):
        original[name]=getattr(mujoco,name)
        def deny(*a,_name=name,**kw):
            forbidden[_name]=forbidden.get(_name,0)+1;raise AssertionError("Forbidden physics/force/collision API: "+_name)
        setattr(mujoco,name,deny)
    rows=[];exported=[];native_audit=None;global_error=None
    try:
        spec=importlib.util.spec_from_file_location("pinned_palmroot_geometry_for_batch",KIN)
        kin=importlib.util.module_from_spec(spec);spec.loader.exec_module(kin)
        cat=open_catalog(pins["audit_catalog"]["path"],expected_sha256=pins["audit_catalog"]["sha256"])
        assert len(cat.entries)==10
        group=out/"clips/observation_derived_geometry"
        for sid in cat.source_ids:
            if time.monotonic()-STARTED>300:
                rows.append({"source_id":sid,"status":"NOT_RUN_SOFT_DEADLINE","training_authorized":False});continue
            try:
                for pin in pins.values():
                    if sha(pin["path"])!=pin["sha256"]:raise ValueError("Frozen source drift")
                obs=load_observation(cat,sid,use="geometry_preparation",frame_range=(0,151))
                local=sid.rsplit("/",1)[1];directory=out/"geometry"/local
                result,q,pose,macro,geometric=optimize_clip(obs,kin,directory)
                result["geometry_directory"]=str(directory)
                if result["status"]=="COMPLETED_GEOMETRY_ONLY":
                    deadline();clip_id=local+"_f0000_n0151"
                    native,sidecar,info=export_clip(group,clip_id,obs.time_s,q,pose,model_path=MODEL,constants_path=CONSTANTS,
                        provenance={"is_real_ego":False,"source_robot_labels_used":False,
                                    "observation_source":{"path":obs.provenance["export_file"],"sha256":obs.provenance["export_sha256"]},
                                    "geometric_derivative":{"path":str(geometric),"sha256":sha(geometric)},
                                    "strict_reader_provenance":obs.provenance_dict(),"confidence":obs.provenance_dict()["confidence"],
                                    "lineage":obs.provenance["lineage"],"split_group":obs.provenance["split_group"],
                                    "physical_reference_qualified":False,"training_authorized":False})
                    result.update(native_npz=str(native),native_npz_sha256=sha(native),native_provenance_sha256=sha(sidecar),
                                  native_BODY_FK_recomputed=True,macro_FK_renamed=False)
                    exported.append({"source_id":sid,"clip_id":clip_id,"native_path":str(native),"geometric_path":str(geometric)})
                rows.append(result)
            except Exception as error:
                rows.append({"source_id":sid,"status":"FAILED_CPU_GEOMETRY_OR_ABI","error":repr(error),"physical_reference_qualified":False})
            print(json.dumps({"source_id":sid,"status":rows[-1]["status"],"frames":rows[-1].get("frames_completed"),
                              "optimizer_success":rows[-1].get("optimizer_success_count"),"native_exported":bool(rows[-1].get("native_npz")),
                              "CPU_elapsed_s":time.monotonic()-STARTED},allow_nan=False),flush=True)
            with (out/"progress.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(rows[-1],allow_nan=False)+"\n")
        if exported:
            group.mkdir(parents=True,exist_ok=True)
            write(group/"index.json",{"schema":"wuji_pen_spin_clips:1","group_id":"observation_derived_geometry", "group_name":"All10 observation-derived bone-only geometry audits, not physicalreferences", "frame_dt":.02,"clip_count":len(exported),"clips":[{"clip_id":r["clip_id"],"file":Path(r["native_path"]).name,"sha256":sha(r["native_path"])} for r in exported]})
            write(out/"native_recipe.json",{"schema":"wuji_pen_spin_recipe:1","clips":["observation_derived_geometry/"+r["clip_id"] for r in exported]})
            native_audit=audit_native(out,exported);write(out/"native_reader_audit.json",native_audit)
    except Exception as error:
        global_error=repr(error)
    finally:
        for name,value in original.items():setattr(mujoco,name,value)
        status="COMPLETE_GEOMETRY_NATIVE_ABI_CPU_ONLY" if len(exported)==10 and native_audit and global_error is None else "PARTIAL_OR_FAILED_CPU_BATCH"
        summary={"status":status,"source_recipe_sha256":sha(out/"frozen_recipe.json"),"source_sha256":sha(__file__),"rows":rows,
                 "native_export_count":len(exported),"native_reader_audit":native_audit,"global_error":global_error,
                 "forbidden_calls":forbidden,"CPU_wall_seconds_including_imports":time.monotonic()-STARTED,
                 "wall_scope":"OneCPUprocess softdeadline includesimports; perclip/reader timers nest and are notadded",
                 "all_unconverged_or_failures_retained":True,"source_q_link_wrist_fallback":False,
                 "all10_scope":"Geometry auditonly, not a traincatalog or model/normalizer update; external009/010 remainheldout for futuretraining",
                 "physical_reference_qualified":False,"training_authorized":False,"physics_steps":0,"GPU_used":False}
        write(out/"report.json",summary)
        print(json.dumps({k:summary[k] for k in ("status","native_export_count","CPU_wall_seconds_including_imports","global_error","forbidden_calls","training_authorized","physical_reference_qualified")},indent=2),flush=True)
    raise SystemExit(status!="COMPLETE_GEOMETRY_NATIVE_ABI_CPU_ONLY")


if __name__=="__main__":main()
