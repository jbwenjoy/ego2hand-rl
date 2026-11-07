"""Reconstruct existing rest-hold contacts/wrenches on CPU; never solve dynamics.

Solver row k uses saved pre-state k-1 and post-state k. Startup contacts are
reported separately and never counted twice as an impulse. Free-fall curves
are arithmetic references, not newly simulated trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_NORMAL_N = 1e-6  # Same descriptive active criterion as saved capture helper.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "outputs/rest_hold_frame0_v1")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/rest_contact_diagnostic_v1")
    args = ap.parse_args()
    src, out = args.input.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan_path = src.with_name(src.name + "_watchdog") / "plan_snapshot.json"
    plan = read(plan_path)
    evaluation, runtime, cap = (read(src / n) for n in ("evaluation.json", "runtime.json", "capacity_audit.json"))
    assert sha(plan_path) == runtime["plan_sha256"] == evaluation["plan_sha256"]
    assert runtime["status"] == "completed_engineering_new_rest_diagnostic"
    assert evaluation["physics_inference_qualified"] and evaluation["complete"]
    assert cap["status"] == "OK" and cap["read_count"] == 1 and cap["read_after_step"] == 60
    source_checks = {k: sha(v["path"]) == v["sha256"] for k, v in plan["source_pins"].items()}
    assert all(source_checks.values())
    z = dict(np.load(src / "raw_states.npz", allow_pickle=False))
    solver = read(src / "solver_samples.json")
    raw = [json.loads(line) for line in (src / "raw_contacts.jsonl").read_text().splitlines() if line.strip()]
    assert len(solver) == 60 and np.array_equal(z["index"], np.arange(61))
    assert runtime["physics_steps"] == runtime["step_calls"] == 60
    assert runtime["startup_Warp_forward_calls"] == runtime["overflow_reads"] == 1
    assert runtime["postinit_state_writes"] == runtime["postinit_ctrl_writes"] == 0
    model = mujoco.MjModel.from_binary_path(plan["model"]["path"])
    obj = model.body("object/cube").id
    joint = model.joint("object/cube_freejoint").id
    qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert np.array_equal(z["object_pose_w"], z["full_qpos"][:, qa:qa + 7])
    assert np.array_equal(z["object_twist_native"], z["full_qvel"][:, va:va + 6])
    assert np.count_nonzero(model.body_ipos[obj]) == 0  # COM = free-joint xyz.
    assert not np.any(model.dof_damping[va:va + 6])
    assert not np.any(model.dof_frictionloss[va:va + 6])
    assert not np.any(model.dof_armature[va:va + 6])
    assert model.body_gravcomp[obj] == model.opt.density == model.opt.viscosity == 0
    actual = dict(np.load(src / "actual_context_applied.npz", allow_pickle=False))
    expected = dict(np.load(src / "expected_context_fields.npz", allow_pickle=False))
    readbacks = dict(np.load(src / "actual_context_readbacks.npz", allow_pickle=False))
    context_checks = {}
    for field, value in actual.items():
        context_checks[field] = all(np.array_equal(value, x) for x in (
            expected[field], readbacks["after_copy__" + field], readbacks["after_set_const_0__" + field]))
    assert all(context_checks.values())
    mass = float(actual["body_mass"][obj])
    dt = float(plan["dt_s"])
    dt32 = float(np.float32(model.opt.timestep))
    gravity_nominal = np.array(model.opt.gravity)
    gravity = gravity_nominal.astype(np.float32).astype(np.float64)
    weight = mass * np.linalg.norm(gravity)
    object_geoms = set(np.flatnonzero(model.geom_bodyid == obj).tolist())
    hand_geoms = {i for i in range(model.ngeom) if re.fullmatch(
        r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(i).name)
        and (model.geom_contype[i] or model.geom_conaffinity[i])}
    names = read(plan["firstframe"]["path"])["joint_names"]
    jids = np.array([model.joint(n).id for n in names])
    aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j).item() for j in jids])
    assert np.all(model.actuator_forcelimited[aids])
    force_limits = np.max(np.abs(model.actuator_forcerange[aids]), axis=1)
    by_step = {k: [] for k in range(61)}
    for c in raw:
        assert c["step"] in by_step
        by_step[c["step"]].append(c)
    ncontact = np.zeros(60, dtype=int)
    positive_contacts = np.zeros(60, dtype=int)
    positive_pairs = np.zeros(60, dtype=int)
    fingers = np.zeros(60, dtype=int)
    force = np.zeros((60, 3))
    moment_arm_torque = np.zeros((60, 3))
    direct_torque = np.zeros((60, 3))
    force_other = np.zeros((60, 3))
    max_frame_error = max_capture_force_error = max_capture_direct_error = max_com_error = 0.0
    events = []
    solver_t = np.zeros(60)
    for row, s in enumerate(solver):
        k = row + 1
        assert s["step"] == k and s["prestate_index"] == k - 1 and s["poststate_index"] == k
        assert s["phase"] == "just_completed_solver_preintegration"
        assert s["solver_t_s"] == z["raw_time_s"][k - 1] and s["poststate_t_s"] == z["raw_time_s"][k]
        assert z["force_prestate_index"][k] == k - 1
        com = np.asarray(s["solver_object_COM_w"])
        max_com_error = max(max_com_error, float(np.abs(com - z["object_pose_w"][k - 1, :3]).max()))
        solver_t[row] = s["solver_t_s"]
        capture_f, capture_tau = [], []
        pairs, fid = set(), set()
        for c in by_step[k]:
            assert c["phase"] == s["phase"] and c["solver_t_s"] == s["solver_t_s"]
            g1, g2 = c["geom"]
            if g1 not in object_geoms and g2 not in object_geoms:
                continue
            sign = 1.0 if g2 in object_geoms else -1.0
            frame = np.asarray(c["frame_rows_w"], dtype=float)
            cf = np.asarray(c["force6_contact_frame"], dtype=float)
            fw, tw = sign * frame.T @ cf[:3], sign * frame.T @ cf[3:]
            max_frame_error = max(max_frame_error, float(np.abs(frame @ frame.T - np.eye(3)).max()))
            hand = g1 if g1 in hand_geoms else g2 if g2 in hand_geoms else None
            if hand is None:
                force_other[row] += fw
                continue
            ncontact[row] += 1
            force[row] += fw
            moment_arm_torque[row] += np.cross(np.asarray(c["pos_w"]) - com, fw)
            direct_torque[row] += tw
            capture_f.append(fw)
            capture_tau.append(tw)
            positive = c["efc_address"] >= 0 and cf[0] > ACTIVE_NORMAL_N
            if positive:
                positive_contacts[row] += 1
                pairs.add(tuple(c["geom"]))
                match = re.search(r"right_finger([1-5])_", model.geom(hand).name)
                if match:
                    fid.add(int(match[1]))
            events.append({"step": k, "solver_t_s": float(solver_t[row]), "prestate_index": k - 1,
                           "poststate_index": k, "geom_names": [model.geom(g1).name, model.geom(g2).name],
                           "normal_force_N": float(cf[0]), "positive_force": bool(positive),
                           "force_on_object_w_N": fw.tolist(), "direct_torque_on_object_w_Nm": tw.tolist(),
                           "moment_arm_torque_on_object_w_Nm": np.cross(np.asarray(c["pos_w"]) - com, fw).tolist()})
        positive_pairs[row], fingers[row] = len(pairs), len(fid)
        saved = s["measured_hand_object"]
        assert len(saved["force_w_N"]) == ncontact[row]
        if capture_f:
            max_capture_force_error = max(max_capture_force_error, float(np.abs(np.asarray(capture_f) - saved["force_w_N"]).max()))
            max_capture_direct_error = max(max_capture_direct_error, float(np.abs(np.asarray(capture_tau) - saved["torque_at_contact_w_Nm"]).max()))
        assert int(np.count_nonzero(saved["active"])) == positive_contacts[row]
        assert s["object_environment_contacts"] == 0
    assert max_capture_force_error < 1e-12 and max_capture_direct_error < 1e-12
    assert max_com_error < 1e-7 and not np.any(force_other)
    torque = moment_arm_torque + direct_torque
    vel = z["object_twist_native"][:, :3]
    pos = z["object_pose_w"][:, :3]
    acceleration = np.diff(vel, axis=0) / dt32
    linear_balance_residual = mass * acceleration - (force + mass * gravity)
    impulse = dt32 * np.cumsum(force, axis=0)
    velocity_from_measured_forces = vel[0] + np.vstack((np.zeros(3), impulse / mass)) + np.arange(61)[:, None] * dt32 * gravity
    impulse_residual = mass * (vel[-1] - vel[0] - 60 * dt32 * gravity) - impulse[-1]
    # Arithmetic semi-implicit reference: v_k=v0+k*dt*g, p_k=p0+dt*sum(v_1..v_k).
    steps = np.arange(61, dtype=float)
    t = steps * dt
    freefall_v = vel[0] + steps[:, None] * dt32 * gravity
    freefall_p_discrete = pos[0] + steps[:, None] * dt32 * vel[0] + (steps * (steps + 1) / 2)[:, None] * dt32**2 * gravity
    freefall_p_continuous = pos[0] + t[:, None] * vel[0] + .5 * t[:, None]**2 * gravity_nominal
    # For this API, free-joint xyz is updated directly by its first3 qvel; no
    # rotation is applied. Check that mapping independently against saved data.
    position_update_residual = np.diff(pos, axis=0) - dt32 * vel[1:]
    qdelta = np.max(np.abs(z["joint_rad"] - z["joint_rad"][0]), axis=1)
    utilization = np.max(np.abs(z["actuator_force_native"][1:]) / force_limits, axis=1)
    positive_rows = np.flatnonzero(positive_contacts)
    last_row = int(positive_rows[-1]) if len(positive_rows) else None
    api_source = Path(mujoco.__file__).parent.parent / "mujoco_warp/_src/forward.py"
    api_text = api_source.read_text(encoding="utf-8")
    assert "qpos_new = qpos_pos + timestep * qvel_lin" in api_text
    identity_files = ["raw_states.npz", "raw_contacts.jsonl", "solver_samples.json", "evaluation.json", "runtime.json", "capacity_audit.json", "actual_context_applied.npz", "actual_context_readbacks.npz", "expected_context_fields.npz"]
    report = {
        "status": "COMPLETE_ARTIFACT_ONLY_DIAGNOSTIC",
        "input": str(src), "input_sha256": {name: sha(src / name) for name in identity_files},
        "plan_sha256": sha(plan_path), "analysis_source_sha256": sha(__file__),
        "source_pin_checks": source_checks, "actual8_context_readback_checks": context_checks,
        "recorded_rest_hold_pass": evaluation["rest_hold_pass"], "recorded_gates": evaluation["gates"],
        "raw_coverage": {"states": 61, "solver_intervals": 60, "startup_sample_in_impulse": False},
        "timing": "solver[k] preintegration state[k-1] -> postintegration state[k]; no refreshed final0.3s contacts",
        "linear_velocity_convention": {
            "definition": "Free-joint first3 qvel directly differentiates world xyz; object COM offset is exactly0",
            "installed_API": str(api_source), "installed_API_sha256": sha(api_source),
            "lines": "_next_position 78-82; direct xyz + dt*qvel_lin", "integrator": "IMPLICITFAST",
            "object_qposadr": qa, "object_qveladr": va,
            "position_update_max_abs_residual_m": float(np.abs(position_update_residual).max()),
            "angular_velocity": "Native angular3 are retained in rawdata; no unverified world-angular conversion or Euler angular balance is attempted"},
        "object_physics": {"actual_mass_kg": mass, "gravity_nominal_w": gravity_nominal.tolist(), "gravity_Warp_FP32_w": gravity.tolist(), "weight_N": weight, "body_COM_offset_m": model.body_ipos[obj].tolist(), "actual_principal_inertia_kgm2": actual["body_inertia"][obj].tolist(), "actual_inertia_quaternion_wxyz": actual["body_iquat"][obj].tolist(), "object_passive_damping_friction_armature_gravcomp_fluid": "All0, independently checked from frozenMJB", "semantic_dt_s": dt, "Warp_FP32_dt_s": dt32},
        "contact_summary": {
            "generated_hand_object_samples": int(np.count_nonzero(ncontact)), "positive_force_hand_object_samples": int(np.count_nonzero(positive_contacts)),
            "no_hand_object_contact_samples": int(np.count_nonzero(ncontact == 0)),
            "max_contact_count_per_sample": int(ncontact.max()), "max_positive_force_pairs": int(positive_pairs.max()),
            "max_distinct_positive_force_fingers": int(fingers.max()), "active_normal_N_diagnostic_threshold": ACTIVE_NORMAL_N,
            "last_positive_solver_t_s": None if last_row is None else float(solver_t[last_row]),
            "last_positive_solver_step": None if last_row is None else last_row + 1,
            "max_upward_contact_force_N": float(force[:, 2].max()), "max_upward_force_over_weight": float(force[:, 2].max() / weight),
            "actual_contacts": events,
            "query_interpretation": "The initial nativequery contact has a matching measured pair and positive solved force. It is not a demonstrated ghost. A generated contact/positive force alone does not establish load-bearing grasp; ungenerated geometric gaps remainUNKNOWN."},
        "wrench": {"convention": "Force-on-object uses +frame.T for objectgeom2 and -frame.T for objectgeom1; torque_COM=sum((contactpos-COM)xF + signed frame.T direct_contact_torque)",
                   "recomputed_capture_force_max_abs_error_N": max_capture_force_error,
                   "recomputed_capture_direct_torque_max_abs_error_Nm": max_capture_direct_error,
                   "contact_frame_orthonormal_max_abs_error": max_frame_error,
                   "solver_COM_vs_prestate_max_abs_error_m": max_com_error,
                   "max_abs_direct_contact_torque_Nm": float(np.abs(direct_torque).max()),
                   "contact_impulse_world_Ns": impulse[-1].tolist(),
                   "max_abs_torque_COM_Nm": float(np.abs(torque).max())},
        "linear_balance": {"equation": "m*(v_post-v_pre)/dt = sumF_contact + m*g; COM0/no passive-object or applied forces in this frozen model",
                           "max_abs_interval_force_residual_N": float(np.abs(linear_balance_residual).max()),
                           "rms_interval_force_residual_N": float(np.sqrt(np.mean(linear_balance_residual**2))),
                           "cumulative_impulse_residual_Ns": impulse_residual.tolist(),
                           "measured_force_integrated_velocity_max_abs_error_m_s": float(np.abs(velocity_from_measured_forces - vel).max()),
                           "final_actual_velocity_world_m_s": vel[-1].tolist(),
                           "final_freefall_velocity_world_m_s": freefall_v[-1].tolist()},
        "freefall_reference": {"continuous_displacement_at0p3_m": (freefall_p_continuous[-1] - pos[0]).tolist(),
                               "semiimplicit_discrete_displacement_at60_m": (freefall_p_discrete[-1] - pos[0]).tolist(),
                               "actual_displacement_at60_m": (pos[-1] - pos[0]).tolist(),
                               "actual_minus_discrete_freefall_position_m": (pos[-1] - freefall_p_discrete[-1]).tolist(),
                               "scope": "Arithmetic references at same initialposition/zero velocity; not newphysics or new measured trajectories"},
        "hand_control": {"max_joint_drift_rad": float(qdelta.max()), "max_joint_speed_rad_s": float(np.abs(z["joint_velocity_rad_s"]).max()), "max_actuator_force_limit_fraction": float(utilization.max()), "actuator_force_saturation_samples": int(np.count_nonzero(utilization >= 1 - 1e-6)), "constant_u_exact": bool(z["ctrl_exact"].all()), "C2_clamped_components": evaluation["C2_clamped_components"]},
        "interpretation": ["This designed zero-initial-velocity rest candidate fails holding: one weak finger2 contact lasts only the first3 solver intervals; no hand-object contacts remain for the other57intervals", "The net force/velocity and free-fall comparisons diagnose gravity-dominated loss of support in this rest intervention; they do not establish a unique causal explanation or a general contact-model failure", "No initial multi-finger load-bearing grasp is evidenced. Prioritize explicit contact-intent/geometric initialization design over changing PD damping on this evidence", "Restfailure does not prove the candidate motion infeasible: motion may require nonzero designed initial velocity/momentum", "Any future candidate finite-difference initial velocities must be separately frozen as designed kinematic derivatives, never video-measured velocities", "N=1 designed rest diagnostic, no trainingseed replication, observation-scale/frame qualification, continuous safety, realrobot result, or teacher/reference force labels"],
        "GPU_used": False, "new_physics_steps": 0, "native_forward_calls": 0,
    }
    arrays = dict(solver_time_s=solver_t, prestate_index=np.arange(60), poststate_index=np.arange(1, 61),
                  contact_count=ncontact, positive_force_contact_count=positive_contacts, positive_force_pair_count=positive_pairs,
                  distinct_positive_force_fingers=fingers, contact_force_on_object_world_N=force,
                  moment_arm_torque_world_Nm=moment_arm_torque, direct_contact_torque_world_Nm=direct_torque,
                  net_contact_torque_COM_world_Nm=torque, linear_balance_residual_N=linear_balance_residual,
                  state_time_s=t, actual_position_world_m=pos, actual_velocity_world_m_s=vel,
                  velocity_integrated_measured_contact_force_world_m_s=velocity_from_measured_forces,
                  freefall_velocity_world_m_s=freefall_v, freefall_position_semiimplicit_world_m=freefall_p_discrete,
                  freefall_position_continuous_world_m=freefall_p_continuous, max_joint_drift_rad=qdelta,
                  actuator_force_limit_fraction=utilization)
    np.savez_compressed(out / "diagnostic_arrays.npz", **arrays)
    with (out / "intervals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "solver_prestate_t_s", "poststate_t_s", "generated_hand_object_contacts", "positive_force_contacts", "positive_force_pairs", "positive_force_fingers", "Fx_N", "Fy_N", "Fz_N", "tau_COM_x_Nm", "tau_COM_y_Nm", "tau_COM_z_Nm", "Fx_balance_residual_N", "Fy_balance_residual_N", "Fz_balance_residual_N"])
        for i in range(60):
            writer.writerow([i + 1, solver_t[i], z["raw_time_s"][i + 1], ncontact[i], positive_contacts[i], positive_pairs[i], fingers[i], *force[i], *torque[i], *linear_balance_residual[i]])
    render(out, arrays, weight, pos[0])
    report["figures_sha256"] = {n: sha(out / n) for n in ("rest_diagnostic.png", "rest_diagnostic.svg", "contact_onset.png", "contact_onset.svg")}
    report["visual_QA"] = "pending_actual_view"
    write(out / "report.json", report)
    lines = ["CPU contact/wrench diagnostic: COMPLETE", "N=1 designed zero-velocity0.3s rest; no newphysics.",
             f"Actual contact samples: {np.count_nonzero(ncontact)}/60; maximum active fingers: {fingers.max()}; last positive solver_t: {report['contact_summary']['last_positive_solver_t_s']}s.",
             f"Peak vertical hand force/weight: {100 * force[:, 2].max() / weight:.4f}%.",
             f"Actual displacement: {(pos[-1] - pos[0]).tolist()}m; discrete freefall: {(freefall_p_discrete[-1] - pos[0]).tolist()}m.",
             f"Max interval force-balance residual: {np.abs(linear_balance_residual).max():.9g}N; integrated velocity error: {np.abs(velocity_from_measured_forces - vel).max():.9g}m/s.",
             "Torque includes both moment-arm and direct contact torque. Startup sample is not counted twice.",
             *report["interpretation"]]
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out), "contact_summary": {k: v for k, v in report["contact_summary"].items() if k != "actual_contacts"}, "linear_balance": report["linear_balance"]}, indent=2))


def render(out, a, weight, p0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, s = a["state_time_s"], a["solver_time_s"]
    f, axes = plt.subplots(3, 2, figsize=(12.8, 10.5))
    ax = axes[0, 0]
    ax.plot(t, 1e3 * (a["actual_position_world_m"][:, 2] - p0[2]), label="Actual post-state")
    ax.plot(t, 1e3 * (a["freefall_position_semiimplicit_world_m"][:, 2] - p0[2]), "--", label="Discrete free fall")
    ax.plot(t, 1e3 * (a["freefall_position_continuous_world_m"][:, 2] - p0[2]), ":", label="Continuous free fall")
    ax.set_ylabel("Object vertical displacement (mm)"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(t, a["actual_velocity_world_m_s"][:, 2], label="Actual vz")
    ax.plot(t, a["velocity_integrated_measured_contact_force_world_m_s"][:, 2], "--", label="Integrated measured F + gravity")
    ax.plot(t, a["freefall_velocity_world_m_s"][:, 2], ":", label="Free-fall vz")
    ax.set_ylabel("World vertical velocity (m/s)"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for key, label, style in (("contact_count", "Generated contacts", "-"), ("positive_force_pair_count", "Positive-force pairs", "--"), ("distinct_positive_force_fingers", "Positive-force fingers", ":")):
        ax.step(s, a[key], where="post", linestyle=style, label=label)
    ax.set_ylim(-.05, 1.2); ax.set_ylabel("Hand-object count"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.step(s, 100 * a["contact_force_on_object_world_N"][:, 2] / weight, where="post")
    ax.set_ylabel("Upward hand force / weight (%)")
    ax.text(.98, .92, "Peak <6%; full support would be100%", transform=ax.transAxes, ha="right", fontsize=9)
    ax = axes[2, 0]
    for i, lab in enumerate(("x", "y", "z")):
        ax.plot(s, 1e3 * a["net_contact_torque_COM_world_Nm"][:, i], label=lab)
    ax.set_ylabel("Contact torque about COM (mN m)"); ax.legend(fontsize=8)
    ax = axes[2, 1]
    ax.plot(t, 1e3 * a["max_joint_drift_rad"])
    ax.set_ylabel("Max joint drift from initial (mrad)")
    for ax in axes.ravel():
        ax.set_xlabel("Time (s)"); ax.grid(alpha=.23); ax.set_xlim(0, .3)
    f.suptitle("One designed rest-hold: saved61states /60solver intervals; no new physics\nSolver forces use pre-state; positions/velocities use post-state; startup force excluded from impulse", fontsize=12)
    f.tight_layout(rect=(0, 0, 1, .94))
    for suffix in ("png", "svg"):
        f.savefig(out / ("rest_diagnostic." + suffix), dpi=160, bbox_inches="tight")
    plt.close(f)
    f, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    ax = axes[0, 0]
    for i, lab in enumerate(("Fx", "Fy", "Fz")):
        ax.step(s, 1e3 * a["contact_force_on_object_world_N"][:, i], where="post", label=lab)
    ax.set_ylabel("Force on object (mN)"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    for i, lab in enumerate(("tau x", "tau y", "tau z")):
        ax.step(s, 1e3 * a["net_contact_torque_COM_world_Nm"][:, i], where="post", label=lab)
    ax.set_ylabel("Torque about COM (mN m)"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(t, 1e3 * (a["actual_position_world_m"][:, 2] - p0[2]), label="Object dz")
    ax.plot(t, 1e3 * (a["freefall_position_semiimplicit_world_m"][:, 2] - p0[2]), "--", label="Discrete free-fall dz")
    ax.set_ylabel("Object vertical displacement (mm)"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(t, 1e3 * a["max_joint_drift_rad"], label="Largest joint drift")
    ax.set_ylabel("Max joint drift (mrad)"); ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlim(0, .03); ax.set_xlabel("Time (s), first30ms"); ax.grid(alpha=.23)
        ax.axvline(.015, color="grey", ls=":", lw=1)
    f.suptitle("Contact onset detail: only finger2-link3 contacts object in solver intervals0–5–10ms\nNo hand-object contacts from solver15ms onward; not a multi-finger load-bearing grasp", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, .92))
    for suffix in ("png", "svg"):
        f.savefig(out / ("contact_onset." + suffix), dpi=160, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    main()
