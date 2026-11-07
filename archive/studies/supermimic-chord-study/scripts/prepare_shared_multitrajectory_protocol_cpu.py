"""Prepare DRAFT shared-trajectory protocol/inventory; never run physics/training.

Only observation arrays are copied from the10source clips. Robot-label arrays
are inventoried by their NPY headers, never used to construct observations.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEX = Path("D:/workspace/dex-supermimic-research")
NATIVE = DEX / "src/dex_supermimic"
CLIPS = NATIVE / "tasks/pen_spin/data/rawdata/clips/01_single_axis"
LINEAGE = "upstream_01_single_axis_original_parent_unknown_ALL10"
OBS_KEYS = {"time_s", "hand_landmarks_source_w", "object_pose_source_w"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, d):
    p = Path(p)
    if p.exists(): raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0): shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0): shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else: raise ValueError("Unsupported NPY header version")
            assert not dtype.hasobject
            result[member.removesuffix(".npy")] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(order)}
    return result


def validate_observations(arrays):
    if set(arrays) != OBS_KEYS: raise ValueError("Observationallowlist rejects source robotq/wrist/link/force/ID labels")
    t, hand, obj = (arrays[k] for k in ("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
    if t.ndim != 1 or len(t) < 2 or hand.shape != (len(t), 21, 3) or obj.shape != (len(t), 7): raise ValueError("Observationlayout")
    if not all(np.isfinite(a).all() for a in arrays.values()) or not np.all(np.diff(t) > 0): raise ValueError("Finite/timecontract")
    if np.abs(np.linalg.norm(obj[:, 3:], axis=1) - 1).max() > 1e-6: raise ValueError("Objectquaternion")


def span(path, class_name, method):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in parent.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method)
    return {"path": str(path), "sha256": sha(path), "class": class_name, "method": method, "line": node.lineno, "end_line": node.end_lineno}


def actual_native_boundary_check(pen_path):
    """Execute exactsource methods in isolation, with synthetic trajectoryIDs."""
    import torch
    from types import SimpleNamespace
    cls = next(n for n in ast.parse(pen_path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "PenMDP")
    ns = {"torch": torch}
    for method in ("reference", "command_step"):
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(pen_path), "exec"), ns)
    ref = torch.tensor([11., 12., 21., 22., 23., 31.])
    fake = SimpleNamespace(motion=SimpleNamespace(_frames={"joint_pos": ref[:, None].expand(-1, 20), "object_pose": ref[:, None].expand(-1, 7), "link_pos": ref[:, None, None].expand(-1, 21, 3)}))
    start = torch.tensor([[0.], [2.], [5.]])
    end = torch.tensor([[2.], [5.], [6.]])
    cases = [(-2., [11., 21., 31.]), (0., [11., 21., 31.]), (1., [12., 22., 31.]), (100., [12., 23., 31.])]
    for time, expected in cases:
        memory = {"start": start, "end": end, "time": torch.full((3, 1), time)}
        q, obj, link = ns["reference"](fake, memory)
        assert torch.equal(q[:, 0], torch.tensor(expected)) and torch.equal(obj[:, 0], q[:, 0]) and torch.equal(link[:, 0, 0], q[:, 0])
        nxt = ns["command_step"](fake, None, memory, None)
        assert torch.equal(nxt["time"], memory["time"] + 1) and torch.equal(nxt["start"], start) and torch.equal(nxt["end"], end)
    assert not torch.cuda.is_initialized()
    return {"status": "PASS_exact_native_methods_CPU_synthetic_only", "trajectory_lengths": [2, 3, 1], "cases": len(cases), "boundary_clamp_and_globaloffsets": True, "command_increments_localtime_only": True, "CUDA_initialized": False, "does_not_validate_new_multitrajectory_training": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/shared_multitrajectory_inventory_v1")
    ap.add_argument("--protocol", type=Path, default=ROOT / "protocols/shared-multitrajectory-controlled-v1.json")
    ap.add_argument("--doc", type=Path, default=ROOT / "docs/shared-multitrajectory-protocol-v1.md")
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    index_path = CLIPS / "index.json"; index = read(index_path)
    previous_path = ROOT / "outputs/source-inventory.json"; previous = read(previous_path)
    previous = {c["clip_id"].split("/")[-1]: c for c in previous["clips"]}
    assert index["clip_count"] == len(index["clips"]) == 10 and index["frame_dt"] == .02
    rows = []; offset = 0
    for entry in index["clips"]:
        path = CLIPS / entry["file"]; digest = sha(path)
        assert digest == entry["sha256"] == previous[entry["clip_id"]]["sha256"]
        schema = headers(path)
        with np.load(path, allow_pickle=False) as z:
            # Deliberate three-field data boundary: no q/wrist/link labels.
            hand, obj = z["mediapipe_landmarks_w"].copy(), z["object_pose_w"].copy()
        n = len(hand); t = np.arange(n, dtype=np.float64) * index["frame_dt"]
        arrays = dict(time_s=t, hand_landmarks_source_w=hand, object_pose_source_w=obj)
        validate_observations(arrays)
        assert n == previous[entry["clip_id"]]["frames"] and n >= 151
        export = out / "observations_only" / (entry["clip_id"] + ".npz"); export.parent.mkdir(exist_ok=True)
        np.savez_compressed(export, **arrays)
        with np.load(export, allow_pickle=False) as check: validate_observations({k: check[k] for k in check.files})
        role = "train_pool" if entry["clip_id"] <= "clip_008" else "validation_within_lineage" if entry["clip_id"] == "clip_009" else "test_within_lineage"
        rows.append({"source_id": "01_single_axis/" + entry["clip_id"], "source_file": str(path), "source_sha256": digest,
                     "source_catalog_sha256": sha(index_path), "source_arrays_header_only": schema,
                     "source_frame_range_half_open": [0, n], "frames": n, "frame_dt_s": .02, "duration_s": float(t[-1]),
                     "global_source_frame_range_half_open": [offset, offset+n],
                     "draft_controlled_window_half_open": [0, 151], "draft_window_duration_s": 3., "window_physical_qualified": False,
                     "observation_export": {"path": str(export), "sha256": sha(export), "fields": sorted(OBS_KEYS), "original_robot_labels_exported": False},
                     "lineage": LINEAGE, "split_group": LINEAGE, "trajectory_role": role,
                     "role_for_seen_reference_cost_curve": "training_and_seen_evaluation",
                     "original_video_parent_id": None, "original_ego_demonstrations_verified": 0,
                     "source_contact_or_force_labels_available": False, "source_metric_camera_frame_qualified": False,
                     "source_wrist_max_angle_deg_prior_inventory": previous[entry["clip_id"]]["source_wrist_angle_from_start_max_deg"],
                     "source_wrist_near_fixed_prior_inventory": previous[entry["clip_id"]]["source_wrist_near_fixed_5mm_5deg"],
                     "observation_derived_robot_reference": None, "physics_qualified_init_registry": None})
        offset += n
    assert len({r["lineage"] for r in rows}) == len({r["split_group"] for r in rows}) == 1
    pen = NATIVE / "rl/dex_supermimic/pen.py"; runner = NATIVE / "rl/dex_supermimic/runner.py"
    buffer = NATIVE / "rl/dex_supermimic/buffer.py"; command = NATIVE / "tasks/pen_spin/mdp/commands.py"
    trainer = Path("D:/workspace/supermimic-c2-training-study/scripts/train_controlled_pen.py")
    contact = ROOT / "src/chord_study/contact_objective.py"
    bindings = {
        "native_reference": span(pen, "PenMDP", "reference"), "native_reference_observation": span(pen, "PenMDP", "observation"),
        "native_control": span(pen, "PenMDP", "control_target"), "native_memory": span(pen, "PenSimReader", "memory"),
        "native_collect": span(runner, "DexSuperMimicRunner", "collect"), "native_sample_starts": span(runner, "DexSuperMimicRunner", "sample_starts"),
        "native_WM_loss": span(runner, "DexSuperMimicRunner", "fit_world_model"), "native_imagination": span(runner, "DexSuperMimicRunner", "policy_update"),
        "native_learner": span(runner, "DexSuperMimicRunner", "learn"), "native_model_transfer_guard": span(runner, "DexSuperMimicRunner", "load_model_only"),
        "native_replay_sample": span(buffer, "ReplayBuffer", "sample"),
        "native_pertrajectory_wrist": span(command, "SingleHandTrackingCommand", "_configure_fixed_wrist_reference"),
        "single_bundle_objective": span(contact, "GeometricContactObjective", "forward"),
        "current_singleclip_entry": {"path": str(trainer), "sha256": sha(trainer), "exact_guard": "if len(selected) != 1:", "line": next(i for i, l in enumerate(trainer.read_text(encoding="utf-8").splitlines(), 1) if "if len(selected) != 1:" in l)},
    }
    boundary = actual_native_boundary_check(pen)
    # Label-injection negatives on the actual exported schema, not only wording.
    extra_rejected = []
    for key in ("reference_qpos", "wrist_pose_source_w", "link_pos_w", "contact_force", "trajectory_id"):
        try: validate_observations(dict(arrays, **{key: np.zeros((len(t), 1))}))
        except ValueError: extra_rejected.append(key)
        else: raise AssertionError("Forbidden observationlabel accepted: " + key)
    inventory = {"schema": "controlled_shared_multitrajectory_inventory:1", "status": "CPU_INVENTORY_COMPLETE_NOT_PHYSICAL_QUALIFICATION",
                 "source_kind": "upstream_kinematic_controlled_cached", "source_catalog": {"path": str(index_path), "sha256": sha(index_path)},
                 "previous_inventory": {"path": str(previous_path), "sha256": sha(previous_path)}, "source_count": 10,
                 "total_frames": offset, "lineage_group_count": 1, "verified_independent_ego_demonstrations": 0,
                 "rows": rows, "export_code_sha256": sha(__file__), "GPU_used": False, "new_physics_steps": 0}
    write(out / "inventory.json", inventory)
    order = ["clip_007", "clip_001", "clip_002", "clip_003", "clip_004", "clip_005", "clip_006", "clip_008", "clip_009", "clip_010"]
    quotas = {str(n): {"real_transitions": 648192*n, "WM_optimizer_steps": 4000*n, "failure_head_optimizer_steps": 4000*n,
                       "actor_optimizer_steps": 1800*n, "critic_optimizer_steps": 7200*n,
                       "shared_max_iterations": 2000*n, "shared_wm_only_iters": 200*n, "shared_warmup_steps": 64*n,
                       "independent_runs": n, "independent_iterations_each": 2000, "independent_warmup_each": 64,
                       "status": "DRAFT_NOT_EXECUTED"} for n in (1, 5, 8, 10)}
    protocol = {
        "schema": "shared_multitrajectory_controlled_protocol:1", "status": "DRAFT_CPU_BOUND_NOT_EXECUTION_AUTHORIZATION",
        "GPU_authorized": False, "training_executed": False, "teacher_dataset_generated": False,
        "inventory": {"path": str(out / "inventory.json"), "sha256": sha(out / "inventory.json")},
        "source_bindings": bindings,
        "existing_work_reused": {"conceptual_plan": "configs/controlled-study.json multitrajectory + docs/research-plan-v0.3.md section4", "data_inventory": "outputs/source-inventory.json", "existing_executable_shared_training_protocol_found": False, "scope": "This artifact adds source-bound operational contracts, not another conceptual-only plan"},
        "lineage_policy": {"all10_lineage": LINEAGE, "all10_split_group": LINEAGE, "independent_video_split_claim": False,
                           "heldout_claim": "controlled trajectory holdout within the same conservatively grouped lineage only",
                           "historical_exposure": "All10clips inspected by kinematic and initialization screens;001 engineering and007training/eval seen;009staticcontrol seen. This draft is posthoc to those exposures, not virgin data or blind ego generalization."},
        "data_tracks": {
            "seen_reference_cost_curve": {"N": [1, 5, 10], "prefix_order": order, "train_eval_overlap_intentional": True, "claim": "Costs/quality on already conditioned references only, no trajectory holdout"},
            "controlled_within_lineage_holdout": {"training_pool": order[:8], "N": [1, 5, 8], "validation": ["clip_009"], "test": ["clip_010"], "prefix_order": order[:8], "test_not_loaded_into_training": True, "no_independent_video_generalization": True},
            "N20": {"status": "NOT_AVAILABLE", "reason": "Only10source clips exist; slicing, augmentation or embodiments must not be counted as20independent trajectories"}},
        "input_and_label_boundary": {
            "demonstrator_input_allowlist": sorted(OBS_KEYS), "forbidden_demonstrator_inputs": ["cached reference_qpos", "cached link_pos_w", "cached wrist_pose_w", "solverforces", "teacheractualq", "sourcevelocity labels", "trajectory onehot/filename embedding"],
            "reference_goals": {"desired_q20": "Observation-only IK output, derivedrobot command goal, not sourceproprio or measuredteacherq", "desired_link_positions63": "Knownrobot FK of derivedgoalq, not cached source link labels", "desired_object_axis_points6": "Controlled observation objectpose mapped into fixedpalm task; camera/scale remainunqualified", "future_reference": "Only declared exogenous precomputed command schedule; never futuretrue physical states/contacts or policyoutputs"},
            "actor155": {"goal89": "derivedq20+objectaxis6+derivedFKlinks63", "current66": "robotq20+robotqd20+currentobjectaxis6+previousaction20", "source": "Current controlled rollout supplies these robot/object observations; realimage estimation and sensor errors are not qualified", "robot_proprio_at_execution_allowed": True, "actor_sees_teacher_state_or_forces": False},
            "WM_physics": {"state58": "actualrobotq20/qd20+objectp3/R9/v3/w3 in palmframe", "input_u20": "Actualexecuted FP32C2absolute target", "context": "Frozenactual physics46 + pinnedexogenous174 layout, no trajectoryID/reference needed for physicaltransition", "labels": "True simulator x1 plus native15selfcollision labels; these are robot/simulator supervision, not no-proprio video labels", "loss": "Unchanged native state loss+selfcollision BCE .1; no targetcontact/support loss added"},
            "teacher_optional": {"currently_available": False, "if_added": "Separate dataset must store actualq/qd/objecttwist/executedu/controlmemory and provenance, not equate actualq with PDtarget; cost and retries charged to all applicable conditions"}},
        "sharing_conditions": {"I0": "IndependentWM+actor pertrajectory", "S1": "SharedWM then separate actor adaptation, deferred until explicit safe transfer loader exists", "S2": "SharedWM+single reference-conditioned actor", "primary_pair": ["I0", "S2"], "model_free_secondary": "Native generic PPO path exists, but matched controlled3arm/controller/prior pipeline remains a separate implementation/gate, not automatically comparable"},
        "objective_arms": {"A1L": "unchanged native objective", "P_ONLY": "native + dt*(-.1*sameP)", "A2": "native + dt*(knownphase_support-.1*sameP)", "placement": "Imagination policyreward only; dt=.02 once; native realreward/WMloss unchanged; targets independent/fixed perreference", "scope": "Soft geometric prior ablation, not a qualified force/contact dynamics predictor"},
        "reference_and_replay_contract": {"native_already_supported": "Concatenated frames/trajectoryboundaries, memory(start,end,time), pertrajectory velocity/rebasing; CPU exactsource boundary contract attached", "ID_policy": "Stableint64trajectory/sourceID only in provenance/audit/sampler, not actor/WMinputs. Referencecontent itself identifies commands and may be memorized", "memory": "All EMA/action/load/camera/frozentarget state stays perworld/replay; localtime resets within selectedtrajectory; integeroffsets exactly representable (<2^24 for existingfloat32memory)", "replay": "Everytransition records actualu/residual/memory plus catalogID/globalframe/sourceframe; everywindow masked at done/reset/trajectorychange", "sampling": "Freeze uniformpertrajectory initialstrata and realtransition quotas; disable adaptive outcome-based trajectorysampling for primary comparison; log pertrajectory transitions/updates", "A2_bundles": "Concatenate or index immutablepertrajectory targetbundles in exactly the samecatalog order as references, bind eachSHA/start/end; knownunknown masks remainseparate; no currentspolicy targetrefresh", "no_live_future_state": True},
        "split_and_statistics": {"trajectory_holdout": "External validation/test env+catalog never contribute replay, actor/WMnormalizers, samplerstatistics, modelselectiongradients or contacttargets for training", "native_env_holdout_not_trajectory_holdout": True, "training_seed_DRAFT": [7, 8, 9], "engineering_seed": 0, "process_repeats_per_checkpoint_pertrajectory": 3, "checkpoint_primary_endpoint": "All3fixed freshprocess repeats pass everyfrozen taskgate", "hierarchy": "processes nested in checkpoint×trajectory, trajectories repeated within trainingseed andsame lineage; methodseed N=3, not frames/processes/clips independentdemo N", "failed_init_or_missing_run": "UNQUALIFIED/PENDING/NOT_RUN, not taskFAIL or droppedfromdenominator; preserve enrolledrefs and qualificationcoverage"},
        "budget_DRAFT": {"status": "NOT_EXECUTED", "per_N_aggregate_equal_across_objective_and_sharing_conditions": quotas,
                         "derivation": "128*(64N+2000N/.4)=648192N;2WM/failsteps periteration; actor after200Nwarmup;4criticsteps peractor. IndependentN×2000runs andshared(2000N,wm_only200N,warmup64N) matchaggregatecounts, subject to measuredledger checks/newentry implementation.",
                         "batch_and_horizon": {"num_envs": 128, "WM_batch": 256, "policy_batch": 128, "H": 8, "WM_batches": 2},
                         "replay_total_capacity": {"aggregate_transition_slots": 128000, "shared_buffer_steps": 1000, "independent_buffer_rows_each": "1000/N for declaredN=1/5/8/10", "scope": "Draft equalaggregate retainedslots, not equalpeakmemory or equal samplehistory distributions; must record actualbytes/strata"},
                         "secondary_equal_wall": {"method_total_wall_s": "1200*N DRAFT", "starts": "ActualOScreation of first attributable stage, includes setup/pretraining/adaptation and allmethodprocesses/retries", "selection": "Latestwholecheckpoint strictly complete before methodtotaldeadline; never qualitybest. Independentincomplete models explicitlyNOT_RUN/PENDING", "main_fixed_budget_not_replaced": True},
                         "shared_pretraining": "No freeall10pretraining beforeN1. Primary fromscratch per(N,seed,arm,sharingcondition); transfertrack records parent/dataexposure and charges full pretrainingcost.",
                         "teacher_budget": "Primary proposes0externalteachertransitions; ifrequired changeversionbeforeexecution andfreeze identical teacherdata/teacherbudget acrossarms. Retarget/initsearch/failedteacher attempts stillcharged.",
                         "do_not_claim": "No sharedspeedup/breakeven until matchedquality+inclusivecost measured"},
        "cost_ledger_contract": {"stages": ["datafetch", "observationexport", "reconstruction/calibration", "IK/contactintent", "initphysicalqualification", "teachertraining", "teacherrollouts", "coldcompile/warmcache", "sharedWMpretraining", "pernewtrajectoryadaptation", "realcollection", "WMfit", "actorcriticupdate", "validation", "fixedfinalevaluation", "engineeringfailure/retry"], "fields": ["parentcheckpoint/dataSHA", "stageID", "actualPID+OScreation", "UTCstart/end", "wallCPU/GPUmemory", "realtransitions", "alloptimizersteps", "teachertransitions", "invalid/notrunreason", "cachetemperature"], "accounting": "Report uniqueexperimentcost and inclusive percondition cost separately; nesteddriver/child/evaluator wall scopes not summed. Shared commondata cost never silentlyomitted or counted as free. Everyattempt retained."},
        "engineering_and_stops": {"CPU_now": "Inventory/allowlist/exactnative referenceboundary only", "DRAFT_smoke": "Seed0,N2,128env,H8,10iter with wm_only2 soactor/prior is exercised; qualificationfirst mandatory. Does not countasformaldata or selectquality.", "formal": "NoGPUuntil newentry/loader/sampler/bundles/parity/checkpointauditor+allreference init/physicaltaskprotocol pass andRootauthorizes", "caps": "Measuredsmoke determinesengineeringwall/VRAM caps beforeformal freeze; no borrowed1200/3600guarantee", "stop": "Engineeringnonfinite/sourcehash/capacity/deadline -> failstop/noautomaticretry. Qualityfailureskeepfulltrajectory/latch, do not earlyselect or stopqueue; referenceend naturalcompletion", "training_resume": "Not supported by current103adapter; new shared/transfer format+explicit contracts required, no changingversion bits to bypass loader"},
        "physical_launch_prerequisites": {"currently_qualified_observation_derived_initializations": 0, "observation_rest_baseline": {"path": "D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/evaluation.json", "status": "qualifiedmeasurement_but_restFAIL", "does_not_disprove_motion": True}, "v1_contact_hypotheses": "H23/H123 objectwrenchinfeasible; correctedmapping/newcandidatework not yet a qualified multi-trajectory init registry", "requirements": ["Eachreference camera/scale/frame/handedness limitations explicit", "Observationderivedq/object/freeinitialqvel plus separatelyderivedtargetu verified; noobjectassistance/qprojection", "Actualbackend shortphysicalqualification and fulltargettask protocol perreference beforetraining", "No sourceegoindependence or force/crossbodyqualification claim"], "CPU_inventory_allowed_while_pending": True},
        "implementation_gaps": [
            {"id": "G1", "source": "current_singleclip_entry", "change": "New isolatedmulti catalog freeze/observationderivedreference+perclip provenance, remove singleclipguard only in newentry; currentfrozenentry unchanged", "status": "NOT_IMPLEMENTED"},
            {"id": "G2", "source": "native_memory", "change": "Qualify each selectedtrajectory t0 q/qd/object/EMA/u pairing; currentwrapper references motion.joint_pos and is not proven wrong, but multiinitregistry andevidence missing", "status": "UNQUALIFIED"},
            {"id": "G3", "source": "single_bundle_objective", "change": "Multireference contacttarget registry/prefixmapping/hash+phaseknownmasks; singlebundle cannot be reused bylocaltime for allrefs", "status": "NOT_IMPLEMENTED"},
            {"id": "G4", "source": "native_replay_sample", "change": "Trajectoryquota/provenance and externalholdout split; envholdout alone is not trajectoryholdout. Actor norm/sample_starts currently use alltrainingenvs including WMheldoutenvs", "status": "NOT_IMPLEMENTED"},
            {"id": "G5", "source": "native_model_transfer_guard", "change": "Distinctshared checkpoint schema/explicitcatalogchange transfer with statelayout/physics/controller/norm contracts; oldexactreferencesignature andformat103fromscratch rules must not be silentlybypassed", "status": "NOT_IMPLEMENTED"},
            {"id": "G6", "source": "native_learner", "change": "Aggregatebudgets/initialwarmup/pertrajectoryadaptation/teacher accounting plus fairnessaudit; sharedmaxiters alone would changeactorbudget if warmupnot scaled", "status": "NOT_IMPLEMENTED"},
            {"id": "G7", "source": "native_reference_observation", "change": "Use onlyobservationderivedq/FKgoals for intendedno-demoproprio route; cachedreferenceq/link fields confineddiagnosticupperbound, never silentlyprimary", "status": "NOT_IMPLEMENTED"},
            {"id": "G8", "source": "native_WM_loss", "change": "Retain same58state/u/native15selfcollisionloss/normalization; referenceconditioningbelongs actor/reward. Auditphysicalgradient/calibration pertrajectory; no force-label invention", "status": "CONTRACT_ONLY"}],
        "CPU_evidence": {"exact_native_reference_boundary": boundary, "forbidden_export_fields_rejected": extra_rejected, "observation_exports": 10, "labels_exported": False, "source_hashes_verified": 10, "all10_same_lineage_splitgroup": True},
    }
    write(args.protocol, protocol)
    checks = {"status": "PASS_CPU_ARTIFACTS_ONLY", "source_sha256": sha(__file__), "protocol_sha256": sha(args.protocol), "inventory_sha256": sha(out / "inventory.json"), "source_count": len(rows), "all_source_hashes_match_catalog": True, "all_observation_exports_exact_allowlist": True, "forbidden_fields_rejected": extra_rejected, "lineage_groups": 1, "independent_ego_demonstrations": 0, "native_reference_check": boundary, "training_or_GPU_authorized": False, "new_physics_steps": 0}
    write(out / "checks.json", checks)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    if args.doc.exists(): raise FileExistsError(args.doc)
    table = "\n".join(f"| {r['source_id']} | {r['frames']} | {r['duration_s']:.2f} | {r['source_sha256'][:12]} | {r['trajectory_role']} |" for r in rows)
    text = f"""# 共享多轨迹受控协议 v1（DRAFT，未执行）

本交付是 CPU 数据清单、仅观测字段导出、源码绑定与实现差距；没有启动 GPU、训练或新物理步进。机器协议：[shared-multitrajectory-controlled-v1.json]({args.protocol.resolve().as_posix()})；清单：[inventory.json]({(out/'inventory.json').as_posix()})；核验：[checks.json]({(out/'checks.json').as_posix()})。

已有 `configs/controlled-study.json` 与 `docs/research-plan-v0.3.md` 提出1/5/10/20条共享成本曲线，`outputs/source-inventory.json` 已核10条源。本次复用它们，并补上真实源帧、SHA、输入导出、训练接口与预算约束。当前只有10条，N20不能由切片、扰动或换本体凑数。

## 数据与 lineage

全部10条保守归入同一 `{LINEAGE}` lineage/split_group；原视频父ID缺失，独立ego演示N=0。所有源已被几何/初始化筛查；001工程链路、007训练评估、009静态对照已有历史暴露。以下划分是现在制定的受控同lineage方案，不是假称历史盲测。每条完整源范围见清单；默认待资格窗口是[0,151)，3秒，尚不允许物理启动。旧007[200,351)已见pilot单独保留，不自动替代新观测派生参考。

| 源clip | 帧数 | 时长秒 | SHA前缀 | 同lineage留出轨道用途 |
|---|---:|---:|---|---|
{table}

成本轨道N=1/5/10有意在训练过的参考上评估，只回答已条件化参考的成本与质量。另设同lineage轨迹留出轨道：001–008训练、009验证、010测试，训练N=1/5/8；外部留出轨迹不得进入训练env/replay/normalizer。两个轨道不混池，不能称独立视频或真实ego泛化。

## 输入、参考和仿真标签

仅观测导出实际包含 `time_s/hand_landmarks_source_w/object_pose_source_w`，不包含cached robotq、link、wrist或力。机器人标签仅由NPY header登记维度，不参与观测构造。原坐标、尺度、相机和MediaPipe名称均不构成真实ego资格。

原生actor155维分为89维参考命令（q20、物体轴点6、link63）和66维当前机器人/物体/上一动作。无示范者本体输入路线中，q/link必须是观测IK与已知机器人FK的派生目标，不能偷偷使用cached `reference_qpos/link_pos_w`。执行机器人自身q/qd可用，与示范者没有proprio不是同一个条件。物体当前姿态目前由受控仿真提供；真实图像估计仍待资格。参考未来值仅作为预先声明的外生命令，不可读取未来真实物理状态、接触或当前策略的成功轨迹作为目标。

WM保留58维机器人/对象状态和实际执行u20；context46/外生扰动174布局明确。x1和15维手指自碰是机器人仿真监督标签，不是视频观测或手物力标签。物理WM不需要clip one-hot；actor已通过参考内容条件化。整数trajectory/sourceID只用于索引、分层采样和审计，不加入网络输入。即使无ID，参考内容与phase仍可能被记忆，因此只在明确留出轨道报告受控同lineage迁移。

## 已有接口与必须实现的差距

原生MotionCommand已拼接多轨迹并保存start/end，逐轨迹重定位固定腕并计算速度；PenMDP.reference使用start+localtime且在end−1截断，当前memory已保存这些值。本次执行了真实源码方法的CPU合成边界检查（长度2/3/1，负time、首帧、末帧和溢出time），不把它当多轨迹训练通过。

研究entry仍在 `{trainer.as_posix()}:{bindings['current_singleclip_entry']['line']}` 强制单clip。新实现需隔离新entry并冻结整个catalog/派生参考/初态注册表；保留旧研究源。当前reset目标来自所选motion.joint_pos，并非已证明错位，但多轨迹q/qd/object/EMA/u一致性尚未实际验证。A2目前只有单reference bundle；必须按相同拼接offset构造独立固定多bundle，不能让每条都用localtime查同一目标。unknown/no-contact继续取消吸引，P始终存在，外部dt只乘一次。

ReplayBuffer的holdout只分env，不分轨迹；actor.normalizer更新所有训练env，sample_starts亦包括WM heldout env。这可用于训练环境中的WM诊断，不能充当actor或轨迹留出。外部009/010须单独catalog与env，完全排除训练统计/梯度/采样。新增trajectory元数据和quota必须在replay window跨reset或trajectory变化处掩码，不能跨片拼接监督。

原load_model_only仍严格绑定reference_signature，C2 format103 adapter又明确from-scratch-only；共享预训练迁移不能改format位或禁签名来绕过。必须定义新schema与显式允许的catalog变化，核物理context/控制器/状态布局/normalizer/数据暴露/父checkpoint。旧格式与检查器不被本DRAFT授权复用。原WM loss、真实reward与15维自碰监督不改变；A1L/P_ONLY/A2差异只在原约定imagination reward。

## 公平预算与成本（全部DRAFT）

主比较I0独立WM+actor与S2共享WM+参考条件actor；S1共享WM后逐轨迹actor适配需独立transfer合同后再做。每N、每种子、每目标臂、每共享条件匹配总真实transition和optimizer次数：648192N / WM4000N / fail4000N / actor1800N / critic7200N。提议独立N次各2000iter、warmup64、wm_only200；共享2000Niter、warmup64N、wm_only200N，因此不能只增maxiter而忘记额外actor预算。算法计数基于当前runner learn的128*(64+2000/.4)；仍须新entry实际全批审计，绝不称已训练。

共享batch/H固定，replay提议总128000transition slots（共享1000rows；独立各1000/N rows），峰值内存与历史采样分布另报。工程seed0、N2/10iter/wm_only2只验actor/prior路径，不能充正式结果；正式种子7/8/9仅提议，当前没有执行授权。共享预训练不得在N1之前免费看过全部10条；每新增参考适配、教师、失败尝试、编译、重定向、初态筛选、评估均分列收费。不把历史cache/teacher当免费，不假定共享提速。

次口径提议方法级总墙钟1200N秒，包含预训练/适配/各独立进程启动，从实际OS创建时刻计；只取截止前完整的最近checkpoint，不能按质量挑。未启动/未完成模型标NOT_RUN/PENDING，不能记taskFAIL或删除。driver/child/evaluator时间嵌套，不相加；不同失败尝试成本保留。主口径固定预算不由次口径替换。

## 物理门槛、层级与启动条件

每条先冻结完整原task acceptance字段、input/context/control/参考与接触bundle SHA；3秒151状态、实际u/EMA/t0/capacity/noreset等工程资格和task质量分开。原2cm/20度/3mm/actual-q1mrad、progress/load/rate等字段必须由每参考完整协议定义，旋转progress轴不能把旧clip007轴直接复制给所有轨迹。静态轨迹另标任务，不挑更易任务冒称转笔。物理质量失败锁存且完整继续，只有工程异常/非有限/硬时限停止；无质量停队或自动重试。

每个checkpoint×trajectory固定3个fresh-process重复，以3次全过为该checkpoint/trajectory主终点。方法训练seed N=3，process与轨迹嵌套，帧不作样本N；同lineage轨迹不作独立ego示范N。跨轨迹macro结果与每条完整失败/初始化覆盖率一起报告，不仅显示胜者。教师若未来生成，actualq与targetu分开、控制memory/真实对象初速度保留，成本计入。

目前观测派生rest首帧虽测量工程合格但保持失败，H23/H123 v1还不具静力wrench支持；新映射/候选工作未形成合格multi init registry。本协议允许先完成CPU inventory；不会因此启动物理或大规模训练。新框架、controller real/imag parity、所有目标bundle、checkpoint loader/auditor、逐参考初态与任务门槛实际资格及Root明确授权，都是训练前置条件。
"""
    args.doc.write_text(text, encoding="utf-8")
    print(json.dumps({"status": checks["status"], "clips": len(rows), "frames": offset, "protocol": str(args.protocol), "doc": str(args.doc), "observation_exports": len(rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
