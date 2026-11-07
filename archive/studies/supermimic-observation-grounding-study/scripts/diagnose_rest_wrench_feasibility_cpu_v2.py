"""CPU object-wrench existence screen for the saved step1 contact set.

No collision query, forward, step, force inference from observations, source
robot labels, altered rawdata, or GPU. LP uses the actual native pyramidal
contact basis with no upper normal-force cap; object wrench balance alone
does not impose the robot's joint equilibrium or guarantee dynamic holding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import mujoco
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
TORQUE_LENGTH_SCALE_M = .01  # Numerical row scaling only; not a physical cap.


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def native_pyramid_rays(frame_on_object, point, com, friction, condim):
    """Columns map nonnegative native edge weights to force/COM torque."""
    assert condim in (1, 3, 4, 6)
    local = []
    if condim == 1:
        local.append(np.array([1., 0, 0, 0, 0, 0]))
    else:
        for i in range(condim - 1):
            for sign in (1., -1.):
                ray = np.zeros(6)
                ray[0] = 1.
                ray[i + 1] = sign * friction[i]
                local.append(ray)
    world = []
    for ray in local:
        force, direct = frame_on_object.T @ ray[:3], frame_on_object.T @ ray[3:]
        world.append(np.r_[force, np.cross(point - com, force) + direct])
    return np.array(world).T, np.array(local).T


def solve(contacts, com, target):
    blocks = [native_pyramid_rays(c["frame"], c["point"], com, c["friction"], c["condim"])[0] for c in contacts]
    matrix = np.concatenate(blocks, axis=1)
    scaling = np.r_[np.ones(3), np.ones(3) / TORQUE_LENGTH_SCALE_M]
    # Each native edge has unit compressive-normal coefficient. Minimize the
    # sum of those coefficients, but there is deliberately no normal uppercap.
    lp = linprog(np.ones(matrix.shape[1]), A_eq=scaling[:, None] * matrix,
                 b_eq=scaling * target, bounds=(0., None), method="highs",
                 options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    result = {"solver_status": int(lp.status), "solver_message": lp.message,
              "feasible_in_native_object_wrench_screen": bool(lp.success),
              "normal_force_upper_cap_N": None, "upper_cap_scope": "None/unbounded; not an actuator-force estimate",
              "native_edges": int(matrix.shape[1]), "ray_sampling": "Exactly2*(condim-1) installed-native pyramidal edges, no extra angular discretization",
              "torque_row_length_scale_m": TORQUE_LENGTH_SCALE_M}
    if lp.success:
        result["edge_weights_N"] = lp.x.tolist()
        result["force_balance_max_abs_residual_N"] = float(np.abs((matrix @ lp.x - target)[:3]).max())
        result["torque_balance_max_abs_residual_Nm"] = float(np.abs((matrix @ lp.x - target)[3:]).max())
        offset = 0
        values = []
        for c, block in zip(contacts, blocks):
            n = block.shape[1]
            values.append({"contact_id": c["id"], "normal_force_N": float(lp.x[offset:offset+n].sum()),
                           "object_wrench_world": (block @ lp.x[offset:offset+n]).tolist()})
            offset += n
        result["per_contact_candidate_wrench"] = values
    return result, matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "outputs/rest_hold_frame0_v1")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/rest_wrench_feasibility_v2")
    a = ap.parse_args(); src, out = a.input.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    planfile = src.with_name(src.name + "_watchdog") / "plan_snapshot.json"
    plan, evaluation, cap = read(planfile), read(src / "evaluation.json"), read(src / "capacity_audit.json")
    assert cap["status"] == "OK" and evaluation["complete"] and evaluation["physics_inference_qualified"]
    assert sha(planfile) == evaluation["plan_sha256"]
    model = mujoco.MjModel.from_binary_path(plan["model"]["path"])
    assert sha(plan["model"]["path"]) == plan["model"]["sha256"]
    actual = dict(np.load(src / "actual_context_applied.npz", allow_pickle=False))
    raw = [json.loads(l) for l in (src / "raw_contacts.jsonl").read_text().splitlines() if l.strip()]
    s = read(src / "solver_samples.json")[0]
    assert s["step"] == 1 and s["prestate_index"] == 0 and s["solver_t_s"] == 0.
    obj = model.body("object/cube").id
    og = set(np.flatnonzero(model.geom_bodyid == obj).tolist())
    contacts_raw = [c for c in raw if c["step"] == 1 and any(g in og for g in c["geom"])]
    assert len(contacts_raw) == 1 and s["object_environment_contacts"] == 0
    c = contacts_raw[0]; g1, g2 = c["geom"]
    # No explicitpair overrides or adhesion; use installed contact_material_params.
    assert model.npair == 0 and not model.flg_adhesion and not np.any(model.geom_adhesion)
    p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
    if p1 != p2:
        selected = g1 if p1 > p2 else g2
        condim = int(model.geom_condim[selected]); f3 = actual["geom_friction"][selected]
        material_rule = "Strict higher geompriority selects its condim/friction"
    else:
        selected = None
        condim = int(max(model.geom_condim[g1], model.geom_condim[g2]))
        f3 = np.maximum(actual["geom_friction"][g1], actual["geom_friction"][g2])
        material_rule = "Equal priority uses maxcondim and componentwise maxfriction"
    friction = np.array([f3[0], f3[0], f3[1], f3[2], f3[2]], dtype=float)
    assert np.array_equal(friction, s["measured_hand_object"]["friction"][0])
    assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and condim == 4
    sign = 1. if g2 in og else -1.
    frame = sign * np.array(c["frame_rows_w"], dtype=float)
    point, com = np.array(c["pos_w"]), np.array(s["solver_object_COM_w"])
    mass = float(actual["body_mass"][obj]); gravity = model.opt.gravity.astype(np.float32).astype(float)
    required_force = -mass * gravity
    target = np.r_[required_force, np.zeros(3)]
    normal = frame[0] / np.linalg.norm(frame[0])
    required_local_force = np.linalg.solve(frame.T, required_force)
    arm_torque = np.cross(point - com, required_force)
    best_unbounded_torsion = float(normal @ (-arm_torque))
    unachievable_torque = -arm_torque - best_unbounded_torsion * normal
    actual_contact = {"id": "saved_step1_actual", "frame": frame, "point": point, "friction": friction, "condim": condim}
    lp_result, matrix = solve([actual_contact], com, target)
    assert not lp_result["feasible_in_native_object_wrench_screen"]
    # Independent separating witness: every admissible native edge has positive
    # n dot force, but the required support has negative n dot force.
    witness_edges = normal @ matrix[:3]
    witness_target = float(normal @ required_force)
    assert witness_edges.min() > .99999 and witness_target < -.1

    # A finite, explicitly hypothetical surface construction on the known main
    # object cylinder. These points/normals are not claimed visible/reachable.
    geom = next(g for g in og if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert np.array_equal(actual["geom_pos"][geom], np.zeros(3))
    rotation = quat_matrix(s["solver_object_body_pose_w"][3:]) @ quat_matrix(model.geom_quat[geom])
    axis = rotation[:, 2]
    up = -gravity / np.linalg.norm(gravity)
    inward = up - axis * (axis @ up); inward /= np.linalg.norm(inward)
    radius, half_length = map(float, actual["geom_size"][geom, :2])
    axial_offset = float(.5 * half_length)  # Fixed geometry illustration, not tuned.
    frame_hyp = np.array([inward, axis, np.cross(inward, axis)])
    hypotheses = [{"id": "hypothetical_lower_side_minus", "frame": frame_hyp, "point": com - radius * inward - axial_offset * axis, "friction": friction, "condim": condim},
                  {"id": "hypothetical_lower_side_plus", "frame": frame_hyp, "point": com - radius * inward + axial_offset * axis, "friction": friction, "condim": condim}]
    # An explicit analytical witness, independent of the LP result.
    fn_total, fa_total = float(required_force @ inward), float(required_force @ axis)
    fn_plus = .5 * (fn_total - radius * fa_total / axial_offset)
    fn_minus = fn_total - fn_plus
    witness_forces = [fn_minus * (inward + fa_total / fn_total * axis), fn_plus * (inward + fa_total / fn_total * axis)]
    witness_wrench = sum((np.r_[f, np.cross(h["point"] - com, f)] for h, f in zip(hypotheses, witness_forces)), np.zeros(6))
    assert min(fn_minus, fn_plus) > 0 and abs(fa_total / fn_total) < friction[0]
    assert np.abs(witness_wrench - target).max() < 1e-12
    hypothesis_lp, hypothesis_matrix = solve([actual_contact] + hypotheses, com, target)
    assert hypothesis_lp["feasible_in_native_object_wrench_screen"]
    source_root = Path(mujoco.__file__).parent.parent / "mujoco_warp/_src"
    source_refs = [(source_root / "collision_core.py", 352, 397, "Contactpriority/condim/friction mix"),
                   (source_root / "constraint.py", 3137, 3223, "Nativepyramid force/torque basis and normal-axis torsion")]
    refs = [{"path": str(p), "sha256": sha(p), "start_line": start, "end_line": end, "claim": claim} for p, start, end, claim in source_refs]
    report = {
        "status": "COMPLETE_STRICT_NECESSARY_CONDITIONS_FAIL",
        "scope": "Object-only static-wrench screen of savedstep1 prestate; no newphysics/contactquery/GPU",
        "analysis_source_sha256": sha(__file__), "plan_sha256": sha(planfile),
        "input_sha256": {n: sha(src / n) for n in ("raw_contacts.jsonl", "solver_samples.json", "raw_states.npz", "actual_context_applied.npz", "evaluation.json", "capacity_audit.json")},
        "installed_primary_sources": refs,
        "official_crosschecks": ["https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters", "https://mujoco.readthedocs.io/en/stable/computation/index.html#condim", "https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones"],
        "source_semantics": "Highpriority geom suppliescondim/friction. Effectivecondim4 allows2tangentialforces and directtorsion aboutnormal, not rollingmoments abouttangents. Geometryfriction3 expands to contactfriction5. Nativecone here ispyramidal, with6nonnegativeedges.",
        "contact_material": {"geom_ids": [g1, g2], "geom_names": [model.geom(g1).name, model.geom(g2).name], "priorities": [p1, p2], "geom_condim": [int(model.geom_condim[g1]), int(model.geom_condim[g2])], "selected_geom": selected, "mix_rule": material_rule, "effective_condim": condim, "actual_contact_friction5": friction.tolist(), "mu_torsion_units": "m", "adhesion": "None: flagfalse/allgeomadhesion0/noexplicitpairs", "normal_force_upper_cap_N": None},
        "saved_contact_geometry": {"point_world_m": point.tolist(), "COM_world_m": com.tolist(), "r_point_minus_COM_m": (point - com).tolist(), "normal_force_on_object_world": normal.tolist(), "frame_rows_force_on_object_world": frame.tolist()},
        "gravity_support": {"mass_kg": mass, "gravity_Warp_FP32_world_m_s2": gravity.tolist(), "required_force_world_N": required_force.tolist(), "required_COM_torque_Nm": [0., 0., 0.]},
        "strict_proof_unilateral": {"necessary_condition": "With this single nonadhesive contact, F=-mg and contactframe fn>=0. Requiredfn is negative, so equilibrium is impossible independent of friction magnitude, upper forcecaps, or torque availability.", "required_force_contactframe_N": required_local_force.tolist(), "required_force_dot_unit_normal_N": witness_target, "native_edge_dot_normal_min": float(witness_edges.min()), "native_edge_dot_normal_max": float(witness_edges.max()), "condition_passed": False},
        "strict_proof_moment_span": {"relaxation": "Even granting arbitrary worldF=-mg and unbounded signed normal torsion, condim4 cannot cancel moment components perpendicular to its normal. This is stronger than imposing the actual torsionalfriction limit.", "r_cross_required_F_Nm": arm_torque.tolist(), "best_unbounded_direct_torsion_scalar_Nm": best_unbounded_torsion, "uncancellable_COM_torque_vector_Nm": unachievable_torque.tolist(), "uncancellable_COM_torque_norm_Nm": float(np.linalg.norm(unachievable_torque)), "condition_passed": False},
        "actual_contact_native_pyramid_LP": lp_result,
        "hypothetical_addition": {"scope": "Two added lateral-cylinder surface targets atfixed ±half_length/2, radial lower-side normal from gravityprojection; same material as actualhand contact for geometry-only isolation. Not observedcontacts or robotreachable/IKvalidated targets.", "existing_contact_retained": True, "object_geom_id": geom, "radius_m": float(radius), "half_length_m": float(half_length), "fixed_axial_offsets_m": [-axial_offset, axial_offset], "cylinder_axis_world": axis.tolist(), "proposed_inward_normal_world": inward.tolist(), "points_world_m": [h["point"].tolist() for h in hypotheses], "analytical_candidate_forces_world_N": [f.tolist() for f in witness_forces], "analytical_force_witness_normal_N": [fn_minus, fn_plus], "tangential_over_normal_ratio": abs(fa_total / fn_total), "analytical_wrench_residual_max_abs_mixedunits": float(np.abs(witness_wrench - target).max()), "native_pyramid_LP": hypothesis_lp, "reachability_tested": False, "actual_contacts_created": False, "force_reference_labels_created": False},
        "interpretation_limits": ["Actualcontact equilibrium fails by exact necessaryconditions, not merely by a sampledcone/cap choice", "Thehypothetical positiveLP is only an objectwrench existence witness in the chosen nativepyramid geometry/material; no actuator/Jacobian/hand equilibrium or softcontact constitutive state is enforced", "Finiteforce allocation values are mathematical witnesses, not measured forces or video labels", "A prospective retargeting candidate should seek additional inwardnormals/positions whose wrenchcone contains gravitysupport and supplies opposing moment arms; the2lower-side targets illustrate that condition but do not establish robotreachability", "Static rest failure does not establish candidate motion infeasibility; designed nonzero initial momentum is a distinctcondition", "No originalego contactlabels, force supervision, simulationteacher, dynamic stability, continuous safety or realrobot qualification"],
        "scipy_version": scipy.__version__, "GPU_used": False, "new_physics_steps": 0,
    }
    np.savez_compressed(out / "wrench_matrices.npz", actual_native_wrench_matrix=matrix,
                        added_hypotheses_wrench_matrix=hypothesis_matrix, required_world_wrench=target)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = f"""Saved frame0/step1 static-wrench diagnostic\n\nThe actualcontact set is infeasible for gravity plus zeroCOMtorque by2independent necessaryconditions.\nEffectivecondim={condim}, not3: handpriority1 beats objectpriority0, selecting handfriction/condim. Actualcontactfriction5={friction.tolist()}.\nRequiredcompressivefn={required_local_force[0]:.9f}N is negative. A nonadhesive contact cannot supply this force.\nEven allowing unbounded directtorsion alongnormal, uncancellable moment={1000*np.linalg.norm(unachievable_torque):.9f}mNm. Actualcondim4 is not a no-moment contact.\nExactnative6-edge pyramidalLP (no normaluppercap): infeasible={not lp_result['feasible_in_native_object_wrench_screen']}.\n\nTwo explicitly hypothetical lower-side cylinder contacts at±{axial_offset:.9f}m, inwardnormal={inward.tolist()}, admit an objectwrench balance with nativepyramid material. Their construction is not an observedcontact/reachability certificate. See report.json for exactpoints/normalforces/wrenchwitness.\n\nPrimarymixsource: {source_root / 'collision_core.py'}:352-397\nPrimarybasissource: {source_root / 'constraint.py'}:3137-3223\nOfficialcontactparameters: https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters\nOfficialcondim/cones: https://mujoco.readthedocs.io/en/stable/computation/index.html#condim\n\nNoGPU/newphysics. Noactualforce/reference labels are created. This objectonly existence screen is necessary for staticbalance, not a dynamicgrasp or robotfeasibility certificate. Restfailure does not imply the entire candidate motion is infeasible.\n"""
    (out / "REPORT.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "effective_condim": condim, "required_fn_N": float(required_local_force[0]), "uncancellable_moment_Nm": float(np.linalg.norm(unachievable_torque)), "actual_LP_feasible": lp_result["feasible_in_native_object_wrench_screen"], "hypothetical_addition_LP_feasible": hypothesis_lp["feasible_in_native_object_wrench_screen"]}, indent=2))


if __name__ == "__main__":
    main()
