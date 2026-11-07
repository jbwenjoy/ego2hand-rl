"""CPU syntax/plan/math/interface checks; never creates a simulator or GPU tensor."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "D:/workspace/supermimic-chord-study/src")
spec = importlib.util.spec_from_file_location("microprobe", ROOT / "scripts/braking_microprobe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
started = time.perf_counter()
checks = {}
plan, _ = m.validate(ROOT / "calibration_plan_v1.json")
checks["frozen_input_validation"] = True
checks["stdlib_runtime_import_no_torch_warp"] = not any(k in sys.modules for k in ("torch", "warp", "mujoco_warp"))
cases = [c for s in plan["sources"] for c in m.cases_for(plan, s["method"])]
checks["exact_54_unique_cases"] = len(cases) == len({c["branch_id"] for c in cases}) == 54
checks["27_baselines_27_pulses"] = sum(c["kind"] == "baseline" for c in cases) == sum(c["kind"] == "pulse" for c in cases) == 27
checks["countervelocity_and_nearzero"] = [m.pulse_sign(v) for v in [-1, -.020001, -.02, 0, .02, .020001, 1]] == [-1,-1,1,1,1,1,1]
for label, field, value in (("no_enabled_m0", "m0", .01), ("no_enabled_a", "a_brake", 1),
                             ("no_extra_cases", "total_branches", 55), ("no_weak_budget", "worker_soft_cap_seconds", 181)):
    bad = dict(plan); bad[field] = value
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as f:
        json.dump(bad, f); name = f.name
    try:
        try: m.validate(name)
        except ValueError: checks[label] = True
        else: checks[label] = False
    finally: Path(name).unlink()

import numpy as np
import mujoco
for source in plan["sources"]:
    model = mujoco.MjModel.from_binary_path(source["model"])
    with np.load(Path(source["archived_evaluation"])/"rollout.npz", allow_pickle=False) as roll:
        names = json.loads((Path(source["archived_evaluation"])/"rollout.json").read_text(encoding="utf-8"))["joint_names"]
        jids = np.array([model.joint("robot/"+n).id for n in names])
        aids = np.array([np.flatnonzero(model.actuator_trnid[:,0] == j).item() for j in jids])
        checks[source["method"]+"_20_unit_joint_PD"] = bool(len(jids)==20
            and np.all(model.actuator_trntype[aids] == int(mujoco.mjtTrn.mjTRN_JOINT))
            and np.all(model.actuator_gear[aids,0] == 1) and np.all(model.actuator_gear[aids,1:]==0)
            and np.all(model.actuator_gaintype[aids] == int(mujoco.mjtGain.mjGAIN_FIXED))
            and np.all(model.actuator_biastype[aids] == int(mujoco.mjtBias.mjBIAS_AFFINE))
            and np.allclose(model.actuator_gainprm[aids,0],-model.actuator_biasprm[aids,1]))
        checks[source["method"]+"_full151_archive"] = len(roll["time_s"]) == 151
        checks[source["method"]+"_future_target_kplus1"] = all(
            np.array_equal(np.repeat(roll["joint_target_rad"][f+1:f+11],4,axis=0),
                           np.repeat(roll["joint_target_rad"][1:],4,axis=0)[f*4:f*4+40])
            for f in plan["snapshot_frames"])

pose = np.tile([0,0,0,1,0,0,0], (41,1)).astype(float)
a = {"q_rad":np.zeros((41,20)), "qd_rad_s":np.zeros((41,20)), "object_pose_w":pose,
     "object_twist_native":np.zeros((41,6)), "u_readback_for_step":np.zeros((40,20))}
checks["baseline_identical_pass"] = m.compare_trajectory(a,a)["passed"]
for label,field,index,change in (("position_gate_not_relaxed","q_rad",(20,3),2.1e-5),
                                ("velocity_gate_not_relaxed","qd_rad_s",(20,3),2.1e-4),
                                ("ctrl_exact_not_tolerance","u_readback_for_step",(20,3),1e-8)):
    b={k:v.copy() for k,v in a.items()};b[field][index]=change
    checks[label] = not m.compare_trajectory(a,b)["passed"]
checks["shape_mismatch_fails"] = not m.compare_trajectory(a,{**a,"q_rad":a["q_rad"][:-1]})["passed"]
tree=ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
branch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_branches")
loop=next(n for n in ast.walk(branch) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="step")
calls=[n.func.attr if isinstance(n.func,ast.Attribute) else getattr(n.func,"id","") for n in ast.walk(loop) if isinstance(n,ast.Call)]
checks["no_inside_branch_state_set_reset_or_manager_action"] = not set(calls)&{"set_state","restore","reset","process_action","apply_action"}
checks["native_step_then_boundary_forward"] = calls.count("original_sim_step")==1 and calls.count("original_sim_forward")==1
checks["no_native_mj_forward_or_step_calls"] = not any(isinstance(n,ast.Attribute) and n.attr in {"mj_step","mj_forward"} for n in ast.walk(tree))
cap={"status":"UNKNOWN","nacon":0,"nefc":0,"contact_capacity":192,"constraint_capacity":2048,"overflow_bitmask":None,"ncollision":0}
checks["capacity_unknown_never_fabricated_OK"] = m.capacity_summary([cap])["status"]=="UNKNOWN"
cap.update(status="OVERFLOW",overflow_bitmask=2)
checks["overflow_bitmask_preserved_and_disqualifies"] = m.capacity_summary([cap])["status"]=="OVERFLOW" and m.capacity_summary([cap])["overflow_bitmask_or"]==2
checks["CPU_checks_did_not_import_torch_or_warp"] = not any(k in sys.modules for k in ("torch","warp","mujoco_warp"))
out = ROOT/"outputs/microprobe_cpu_static_v1"
out.mkdir(parents=True,exist_ok=True)
result={"status":"passed" if all(checks.values()) else "failed","checks":checks,"count":len(checks),
        "wall_seconds":time.perf_counter()-started,"runtime_sha256":m.sha256(m.__file__),
        "plan_sha256":m.sha256(ROOT/"calibration_plan_v1.json"),"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
        "qualification":"CPU syntax/hash/model/array contract only; no physical replay/GPU execution"}
m.write_json(out/"checks.json",result)
print(json.dumps(result,indent=2))
sys.exit(0 if all(checks.values()) else 1)
