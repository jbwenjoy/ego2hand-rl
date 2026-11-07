"""Object-only static screens of both preserved frame0v1 geometric candidates.

No dynamics, collision refresh, GPU, source robot labels, or newly measured
contact forces. Reuse the pinned native-pyramid LP and independently construct
a force-only continuous-cone separating certificate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import mujoco
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
INTENT = Path("D:/workspace/supermimic-observation-contact-intent-study")
LP_SOURCE = ROOT / "scripts/diagnose_rest_wrench_feasibility_cpu_v2.py"
LP_SHA = "ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca"
CERTIFICATE_Y = np.array([-1., 1., -.5])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/initial_candidate_wrench_screen_v1")
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    assert sha(LP_SOURCE) == LP_SHA
    spec = importlib.util.spec_from_file_location("pinned_native_pyramid_wrench_screen", LP_SOURCE)
    lp_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(lp_module)
    results = {}
    for variant in ("h23", "h123"):
        recipe_path = INTENT / f"configs/frame0-{variant}-v1.json"
        directory = INTENT / f"outputs/frame0_{variant}_v1"
        recipe, geometric_report = read(recipe_path), read(directory / "report.json")
        assert sha(recipe_path) == geometric_report["recipe_sha256"]
        assert sha(directory / "initialization_candidate.npz") == geometric_report["output_npz_sha256"]
        pins = {k: sha(v["path"]) == v["sha256"] for k, v in recipe["pins"].items()}
        assert all(pins.values())
        assert geometric_report["physical_reference_qualified"] is False
        assert geometric_report["geometry_candidate_accepted_for_dynamics"] is False
        model = mujoco.MjModel.from_binary_path(recipe["pins"]["model"]["path"])
        obj = model.body("object/cube").id
        object_geoms = set(np.flatnonzero(model.geom_bodyid == obj).tolist())
        assert model.npair == 0 and not model.flg_adhesion and not np.any(model.geom_adhesion)
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        with np.load(recipe["pins"]["context"]["path"], allow_pickle=False) as ctx:
            mass = float(ctx["body_mass"][0, obj])
            geom_friction = ctx["geom_friction"][0].astype(float)
            assert np.array_equal(ctx["body_mass"], np.broadcast_to(ctx["body_mass"][:1], ctx["body_mass"].shape))
            assert np.array_equal(ctx["geom_friction"], np.broadcast_to(ctx["geom_friction"][:1], ctx["geom_friction"].shape))
        with np.load(directory / "initialization_candidate.npz", allow_pickle=False) as z:
            pose = z["mapped_object_pose_w"][0].copy()
            selfmask = (z["geometry_pair_kind"] == "hand_self") & z["generated_contact_mask"][0]
            selfdepth = np.maximum(-z["selected_native_contact_distance_m"][0, selfmask], 0)
        com = pose[:3] + lp_module.quat_matrix(pose[3:]) @ model.body_ipos[obj]
        gravity = model.opt.gravity.astype(np.float32).astype(float)
        force_required = -mass * gravity
        target = np.r_[force_required, np.zeros(3)]
        records = read(directory / "native_contacts.json")
        contacts, per_contact, finger_ids, pairs = [], [], set(), set()
        for i, c in enumerate(records):
            assert c["phase"] == "native_geometry_query_not_solved_force"
            g1, g2 = c["geom"]
            assert (g1 in object_geoms) != (g2 in object_geoms)
            hand_geom = g1 if g2 in object_geoms else g2
            assert hand_geom == c["hand_geom"]
            p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
            if p1 == p2:
                condim = int(max(model.geom_condim[g1], model.geom_condim[g2]))
                friction3 = np.maximum(geom_friction[g1], geom_friction[g2])
            else:
                selected = g1 if p1 > p2 else g2
                condim, friction3 = int(model.geom_condim[selected]), geom_friction[selected]
            friction5 = np.array([friction3[0], friction3[0], friction3[1], friction3[2], friction3[2]])
            assert condim == c["effective_condim"] and np.array_equal(friction5, c["friction5"])
            sign = 1. if g2 in object_geoms else -1.
            frame = sign * np.array(c["frame_rows_geom0_to_geom1_w"])
            assert np.max(np.abs(frame @ frame.T - np.eye(3))) < 1e-12
            point = np.array(c["position_w"])
            contacts.append({"id": f"native_geometry_contact_{i}", "point": point, "frame": frame, "condim": condim, "friction": friction5})
            # For any force in the relaxed continuous Coulomb force cone,
            # y dot F >= fn*(yn - norm(mu1*yt1,mu2*yt2)). Torsional coupling
            # is deliberately ignored, so this is an outer relaxation.
            local_y = frame @ CERTIFICATE_Y
            lower = float(local_y[0] - np.linalg.norm(friction5[:2] * local_y[1:]))
            per_contact.append({"contact_index": i, "geom": [g1, g2], "names": c["names"],
                                "normal_force_on_object_world": frame[0].tolist(),
                                "point_world_m": point.tolist(), "condim": condim,
                                "actual_friction5": friction5.tolist(), "distance_m": c["distance_m"],
                                "continuous_force_cone_y_dot_F_lower_per_unit_fn": lower})
            import re
            match = re.search(r"right_finger([1-5])_", model.geom(hand_geom).name)
            assert match; finger_ids.add(int(match[1])); pairs.add(tuple(sorted((g1, g2))))
        full_lp, matrix = lp_module.solve(contacts, com, target)
        force_lp = linprog(np.ones(matrix.shape[1]), A_eq=matrix[:3], b_eq=force_required,
                           bounds=(0., None), method="highs",
                           options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        min_continuous = min(c["continuous_force_cone_y_dot_F_lower_per_unit_fn"] for c in per_contact)
        min_native = float(np.min(CERTIFICATE_Y @ matrix[:3]))
        target_dot = float(CERTIFICATE_Y @ force_required)
        assert min_continuous > .25 and min_native > 0 and target_dot < -.14
        assert not full_lp["feasible_in_native_object_wrench_screen"] and not force_lp.success
        selfmax = float(selfdepth.max()) if len(selfdepth) else None
        assert len(selfdepth) == geometric_report["native_summary"]["hand_self"]["generated_pairs"]
        summary = {
            "status": "OBJECT_WRENCH_INFEASIBLE_BY_SEPARATING_CERTIFICATE",
            "candidate": recipe["hypothesis_id"], "input_directory": str(directory),
            "recipe_sha256": sha(recipe_path), "source_pin_checks": pins,
            "input_sha256": {n: sha(directory / n) for n in ("initialization_candidate.npz", "native_contacts.json", "report.json")},
            "contact_phase": "Native geometric collision queries; not solved contact forces",
            "contact_count": len(contacts), "unique_geom_pairs": len(pairs), "fingers": sorted(finger_ids),
            "per_contact": per_contact, "required_world_force_N": force_required.tolist(),
            "COM_world_m": com.tolist(), "mass_kg": mass,
            "native_pyramid_force_only_LP": {"feasible": bool(force_lp.success), "status": int(force_lp.status), "message": force_lp.message},
            "native_pyramid_full_wrench_LP": full_lp,
            "force_only_separating_certificate": {
                "y_world": CERTIFICATE_Y.tolist(), "y_dot_required_force_N": target_dot,
                "native_ray_min_y_dot_force": min_native,
                "continuous_relaxed_cone_min_lower_per_fn": min_continuous,
                "proof": "Every allowed contact force has ydotF>=fn*strict_positive>=0, even in the relaxed continuous forcecone. Every sum therefore has ydotSumF>=0. The required gravitysupport has negativeydotF, so forcebalance is impossible, independently of COMtorque or forcecaps.",
                "not_a_ray_discretization_result": True,
                "selection_scope": "Posthoc algebraic certificate for the preserved geometry; not a controllerparameter or experimental acceptancegate"},
            "hand_self_contacts": {"generated_pair_count": int(len(selfdepth)), "max_depth_m": selfmax,
                                   "whole_hand_equilibrium_claimed": False,
                                   "scope": "Not ignored as solvedself forces; no selfcontact force equilibrium is inferred. Objectbalance alreadyfails a necessarycondition."},
            "PD_preload_interface": {"prepared": False, "reason": "Requested conditional objectwrench feasibility prerequisite fails; no preload solution is implied"},
            "geometric_material_goal_error_m": geometric_report["material_point_goal_error_m"],
            "physical_reference_qualified": False, "measured_or_reference_force_labels_created": False,
        }
        results[variant] = summary
        np.savez_compressed(out / f"{variant}_wrench_matrix.npz", native_world_wrench_matrix=matrix,
                            required_world_wrench=target, separating_y_world=CERTIFICATE_Y)
    # Minimal positive/negative LP contract on a synthetic under-COM point;
    # this checks solver/sign plumbing, not any candidate performance.
    friction = np.array([.7, .7, .005, .0001, .0001])
    support = {"id": "synthetic_under_COM", "point": np.array([0., 0., -.01]),
               "frame": np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), "condim": 4, "friction": friction}
    positive, _ = lp_module.solve([support], np.zeros(3), np.array([0., 0., .2943, 0., 0., 0.]))
    support["frame"] = -support["frame"]
    negative, _ = lp_module.solve([support], np.zeros(3), np.array([0., 0., .2943, 0., 0., 0.]))
    assert positive["feasible_in_native_object_wrench_screen"] and not negative["feasible_in_native_object_wrench_screen"]
    report = {
        "status": "COMPLETE_CPU_STATIC_SCREEN", "source_sha256": sha(__file__),
        "native_LP_source": {"path": str(LP_SOURCE), "sha256": LP_SHA}, "cases": results,
        "plumbing_contract": {"synthetic_upward_contact_feasible": True, "synthetic_downward_contact_infeasible": True},
        "scope": "Both preserved v1candidates, exactnative pyramid/no forceuppercap plus continuous-cone separating proof; no newcontactforces or physics",
        "source_semantics": "Material priority/condim/friction checked from pinned originalMJB+actualcontext, matching saved nativegeometric contact records",
        "primary_sources": ["https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters", "https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones"],
        "limits": ["Even a positive object-wrench screen would not establish joint/PD/selfcontact/compliance/dynamic feasibility", "Selfcontact evidence forH123 retained; no wholehand equilibrium claim", "Failure of these v1contact geometries is not a proof about future corrected v2source-material mappings or the entire candidate motion", "No historical candidates removed or acceptance thresholds changed"],
        "GPU_used": False, "new_physics_steps": 0, "new_collision_queries": 0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["CPU static object-wrench screen, both preserved v1 candidates", "No normal upperforce cap. Actual native pyramid LP and morepermissive continuous forcecone both excluded by a separatingcertificate."]
    for h, r in results.items():
        cert = r["force_only_separating_certificate"]
        lines.append(f"{h}: contacts={r['contact_count']}, pairs={r['unique_geom_pairs']}, fingers={r['fingers']}; force/fullLP infeasible. y={cert['y_world']}, ydotF_required={cert['y_dot_required_force_N']:.9f}N; mincontinuous lower/fn={cert['continuous_relaxed_cone_min_lower_per_fn']:.9f}; self={r['hand_self_contacts']}. No PDpreload branch.")
    lines += ["Extra contact count/fingers do not resolve same-side wrench direction limitations. Material-anchor mapping errors remain separate preserved geometry evidence.", *report["limits"]]
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({h: {"contacts": r["contact_count"], "force_LP_feasible": r["native_pyramid_force_only_LP"]["feasible"], "wrench_LP_feasible": r["native_pyramid_full_wrench_LP"]["feasible_in_native_object_wrench_screen"], "continuous_certificate_lower": r["force_only_separating_certificate"]["continuous_relaxed_cone_min_lower_per_fn"], "self": r["hand_self_contacts"]} for h, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
