"""Correct approach and withdrawal wrist targets on the development carton using the actual table."""

import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_hocap_carton import evaluate
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.ego.source_stage_contract import score_frozen_source_stages
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/hocap_dev_release_stability_v1"
OUT = ROOT / "runs/hocap_carton_table_clear_v1"


def contacts(model):
    box, table = model.geom("carton_collision").id, model.geom("table").id
    hand = {int(k) for pair in zip(model.pair_geom1, model.pair_geom2) if box in pair for k in pair} - {box, table}
    if len(hand) != 21:
        raise ValueError("Unexpected hand collision layout")
    return table, hand


def desired_gaps(model, wrists, fingers):
    table, hand = contacts(model)
    d = mujoco.MjData(model)
    finger_dofs = model.jnt_qposadr[model.actuator_trnid[:,0]]
    gaps = []
    for wrist, q in zip(wrists, fingers, strict=True):
        d.qpos[:3] = wrist[:3,3]
        d.qpos[3:7] = Rotation.from_matrix(wrist[:3,:3]).as_quat(scalar_first=True)
        d.qpos[finger_dofs] = q
        mujoco.mj_kinematics(model,d)
        gaps.append(min(mujoco.mj_geomDistance(model,d,table,g,1.,None) for g in hand))
    return np.array(gaps), d.geom_xmat[table].reshape(3,3)[:,2].copy()


def audit(model, tr):
    table, hand = contacts(model)
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:], d.ctrl[:] = tr["sim_qpos"][0], tr["sim_qvel"][0], tr["finger_ctrl"][0]
    mujoco.mj_forward(model,d)
    mujoco.mj_forward(model,d)
    samples = []
    err = 0.
    def penetration():
        return max([0.] + [-c.dist for c in d.contact if
                   (c.geom1==table and c.geom2 in hand) or (c.geom2==table and c.geom1 in hand)])
    for frame,(lo,hi) in enumerate(zip(tr["frame_end_substep"][:-1],tr["frame_end_substep"][1:]),1):
        for k in range(lo,hi):
            d.ctrl[:] = tr["substep_finger_ctrl"][k]
            d.qfrc_applied[:6] = tr["substep_wrist_generalized_wrench"][k]
            mujoco.mj_step(model,d)
            samples.append(penetration())
        mujoco.mj_forward(model,d)
        err = max(err, np.max(np.abs(d.qpos-tr["sim_qpos"][frame])), np.max(np.abs(d.qvel-tr["sim_qvel"][frame])))
    samples.append(penetration())
    return float(err), float(max(samples)*1000), int(np.count_nonzero(samples))


def main():
    OUT.mkdir(exist_ok=True)
    release = json.loads((PARENT/"report.json").read_bytes())
    parent = PARENT / f"shift_{release['chosen']:+d}" / "report.json"
    desc = json.loads(parent.read_bytes())
    stage = verify_file_pin(release["source_stage_definition"])
    contract = json.loads(stage.read_bytes())
    masks = stage.with_name("source_intervals.npz")
    if file_pin(masks)["sha256"] != contract["intervals_sha256"]:
        raise ValueError("Changed source stages")
    with np.load(masks, allow_pickle=False) as z:
        source_arrays = dict(z)
    with np.load(verify_file_pin(desc["trace"]), allow_pickle=False) as z:
        base = dict(z)
    model_path = verify_file_pin(desc["mjb"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    gaps, normal = desired_gaps(model, base["target_wrist"], base["target_finger_q"])
    t = base["source_nominal_time_s"]
    q0 = model.qpos0[:3].copy()
    weight = model.body_mass[model.body("carton").id] * np.linalg.norm(model.opt.gravity)
    protocol = ROOT/"configs/hocap_carton_evaluation.json"
    record = dict(parent=file_pin(parent), source=file_pin(verify_file_pin(desc["source_report"])),
                  stage=file_pin(stage), script=file_pin(__file__), protocol=file_pin(protocol),
                  affected_command="Robot wrist translation along the compiled inclined table normal only.",
                  selection="First in the stated order passing all original and release gates with zero hand-table overlap in the recorded substeps.",
                  tested=[], chosen=None)
    start = perf_counter()
    for margin in (1, 2, 4, 6, 8, 12):
        for speed in (.4, .2, .6):
            folder = OUT / f"margin_{margin}mm_rate_{int(speed*100)}cm_s"
            folder.mkdir(exist_ok=True)
            needed = np.maximum(0., margin/1000-gaps)
            shift = np.maximum(0., (needed[:,None]-speed*np.abs(t[:,None]-t[None,:])).max(axis=0))
            wrists = base["target_wrist"].copy()
            wrists[:,:3,3] += shift[:,None]*normal
            model.qpos0[:3] = q0 + shift[0]*normal
            tr = control.replay(model, wrists, base["target_finger_q"], t, desc["wrist"])
            err, peak, violating = audit(model,tr)
            if err != 0.:
                raise ValueError(f"Independent control replay differs by {err}")
            for name in ("world_from_carton_source","source_right_hand21","source_nominal_time_s","target_finger_q"):
                tr[name] = base[name]
            tr.update(target_wrist=wrists,table_clearance_offset_m=shift)
            np.savez_compressed(folder/"trace.npz",**tr)
            run = {k:desc[k] for k in ("source_report","wrist","physics_dt_s","nominal_fps","table","mjb")}
            run.update(parent=file_pin(parent), trace=file_pin(folder/"trace.npz"),
                       independent_substep_action_replay_max_state_error=err,
                       max_hand_table_overlap_mm=peak, hand_table_overlap_samples=violating)
            write_json(folder/"report.json",run)
            evaluate(folder/"report.json",verify_file_pin(desc["source_report"]),protocol,folder/"phase_evaluation.json")
            legacy = json.loads((folder/"phase_evaluation.json").read_bytes())
            nominal_force_time = weight*np.diff(t)
            stages, detail = score_frozen_source_stages(source_arrays,contract,contract["config"],
                tr["world_from_carton_sim"],tr["interval_hand_carton_normal_impulse_Ns"][1:]/nominal_force_time,
                tr["interval_table_carton_normal_impulse_Ns"][1:]/nominal_force_time,tr["interval_max_hand_contacts"][1:])
            ok = bool(legacy["passed_numeric_development_gates"] and all(stages.values()) and violating==0)
            result = dict(folder=folder.name,passed=ok,margin_mm=margin,offset_speed_m_s=speed,
                          max_hand_table_overlap_mm=peak,hand_table_overlap_samples=violating,
                          changed_target_frames=np.flatnonzero(shift>0).tolist(),
                          max_wrist_offset_mm=float(1000*max(shift)),legacy_gates=legacy["gates"],
                          physical_metrics=legacy["metrics"],source_stages=stages,source_stage_metrics=detail)
            record["tested"].append(result)
            if ok:
                # Store the repaired starting wrist in the exported model as well as the command bundle.
                scene = folder/"scene.mjb"
                mujoco.mj_saveModel(model,str(scene),None)
                run["mjb"] = file_pin(scene)
                write_json(folder/"report.json",run)
                evaluate(folder/"report.json",verify_file_pin(desc["source_report"]),protocol,folder/"phase_evaluation.json")
                loaded = mujoco.MjModel.from_binary_path(str(scene))
                if audit(loaded,tr) != (0.,0.,0) or not json.loads((folder/"phase_evaluation.json").read_bytes())["passed_numeric_development_gates"]:
                    raise ValueError("Saved model failed post-export replay")
                record["chosen"] = file_pin(folder/"report.json")
            record["elapsed_s"] = perf_counter()-start
            write_json(OUT/"comparison.json",record)
            print(folder.name, ok, peak, violating, flush=True)
            if ok:
                return


if __name__ == "__main__":
    main()
