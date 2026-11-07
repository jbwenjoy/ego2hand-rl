"""Independent saved-array arithmetic for phone Warp G2-G8; no engine imports.

This is an assessment of the existing runner's captures. The assessor also
authored the Warp recorder; independence here means arithmetic/IO replay without
importing the producer or a physics backend, not independent experimental authors.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PILOT = BASE.parent
MAP = {"drop": "drop", "settle": "table_settling", "push_release": "force_push_release"}
PROTO_SHA = "8981e89aec0993887aad65db1722f80d27470d6553a7bd25760a6ba7556d0505"
WARP_SHA = "4be344d6cbae06e29a4e96f538b117c0a362203fac2c9df27529070514311550"


def pin(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(item):
    actual = pin(item["path"])
    if actual["sha256"] != item["sha256"]:
        raise ValueError(f"Immutable pin mismatch: {item['path']}")
    return actual


def load(item):
    verify(item)
    with np.load(item["path"], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def norm(v):
    return np.sqrt(np.sum(np.asarray(v, dtype=np.float64)**2, axis=-1))


def rms(v):
    return math.sqrt(math.fsum(float(x*x) for x in np.asarray(v).ravel())/len(v))


def orient(q, r):
    q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    q, r = q/np.linalg.norm(q), r/np.linalg.norm(r)
    return float(2*math.acos(min(1., abs(math.fsum(float(x*y) for x, y in zip(q, r))))))


def state_geometry(qpos, qvel, center, inertia, mass):
    # Separate explicit matrix construction, never import shared producer helper.
    p, v = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    rotation = np.empty((len(p), 3, 3), dtype=float)
    for i, (w, x, y, z) in enumerate(p[:, 3:]):
        rotation[i] = ((1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w),
                       (2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w),
                       (2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y))
    offsets = np.stack([R @ center for R in rotation])
    omega = np.stack([R @ w for R, w in zip(rotation, v[:, 3:])])
    com = p[:, :3]+offsets
    vc = v[:, :3]+np.cross(omega, offsets)
    potential = mass*9.81*com[:, 2]
    kinetic = .5*mass*np.sum(vc**2, axis=1)+.5*np.sum(v[:, 3:]**2*inertia, axis=1)
    return com, vc, omega, np.stack((potential, kinetic), axis=1)


def case_key(row, warp=False):
    test = row["protocol_test_name"] if warp else row["test"]
    return (float(row["mass_kg"]), float(row["sliding_friction"]), float(row["dt_s"]), test)


def compare_states(a, b, dt, gates):
    # Inputs already aligned by integer tick. No clock->index conversion.
    delta = a["com"]-b["com"]
    rms_key = "shared_tick_COM_trajectory_RMS_difference_m" if "shared_tick_COM_trajectory_RMS_difference_m" in gates else "same_tick_COM_trajectory_RMS_difference_m"
    m = {"final_COM_position_difference_m": float(norm(delta[-1])),
         "final_orientation_difference_deg": math.degrees(orient(a["qpos"][-1, 3:], b["qpos"][-1, 3:])),
         "final_COM_velocity_vector_difference_m_s": float(norm(a["vcom"][-1]-b["vcom"][-1])),
         "final_angular_velocity_vector_difference_rad_s": float(norm(a["omega"][-1]-b["omega"][-1])),
         rms_key: rms(delta),
         "first_contact_time_difference_s": abs(a["first_contact_tick"]-b["first_contact_tick"])*dt}
    checks = {k: bool(value <= gates[k]) for k, value in m.items()}
    return {"metrics": m, "checks": checks, "passed": all(checks.values()),
            "sample_count": len(delta), "alignment": "same integer tick; no interpolation or clock rounding"}


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def assess(output):
    protocol = read(PILOT/"protocols/phone_physics_stage1_v1.json")
    pp = pin(PILOT/"protocols/phone_physics_stage1_v1.json")
    wp_report = pin(BASE/"warp_v1/result.json")
    if pp["sha256"] != PROTO_SHA or wp_report["sha256"] != WARP_SHA:
        raise ValueError("Not the fixed authorized protocol/terminal run")
    w, c = read(wp_report["path"]), read(BASE/"cpu_v1/result.json")
    verify(w["protocol"]); verify(c["protocol"])
    for item in [*w["sources"].values(), *w["asset_pins"].values(), *w["relevant_backend_sources"].values()]:
        verify(item)
    group_map = {(g["group"]["mass_kg"], g["group"]["mu"], g["group"]["dt_s"]): g for g in w["groups"]}
    cm = {case_key(row): row for row in c["cases"]}
    wm = {case_key(row, True): row for row in w["cases"]}
    expected = set(itertools.product([.1, .2, .3], [.3, .6, .9], [.0025, .00125], list(MAP.values())))
    if set(cm) != expected or set(wm) != expected or len(w["cases"]) != 54 or len(c["cases"]) != 54:
        raise ValueError("Missing, duplicated, unexpected, or incorrectly mapped case")
    thresholds = protocol["engineering_gates_predeclared_not_measured"]
    per_case, by_key, cpu_raw = [], {}, {}
    arrays_read = 0
    identity_checks = {"all54_cases_each_backend": True, "same_protocol_hash": w["protocol"]["sha256"] == c["protocol"]["sha256"],
                       "all18_groups": len(group_map) == 18,
                       "world_step_total": w["world_steps_returned"] == 97200,
                       "batch_step_total": w["batch_steps_returned"] == 32400,
                       "final_forward_count": w["extra_forward_sampling_calls"] == 18}
    all_g2_numeric = True
    for key, row in wm.items():
        mass, mu, dt, test = key
        cr = cm[key]
        a, ct, ca = load(row["trajectory"]), load(row["contact_table"]), load(cr["trajectory"])
        arrays_read += 3
        g = group_map[key[:3]]
        opts = g["CPU_compiled_model_options"]
        n = round(3/dt)
        ticks = np.arange(n+1)
        structural = []
        def require(condition, message):
            if not condition:
                structural.append(message)
        require(row["test"] in MAP and MAP[row["test"]] == test, "protocol test mapping")
        require(a["qpos"].shape == (n+1, 7) and a["qvel"].shape == (n+1, 6), "state shape")
        require(np.array_equal(a["integer_tick"], ticks), "integer tick order/count")
        require(np.array_equal(ca["tick"], ticks), "CPU integer tick order/count")
        require(np.array_equal(a["tick_derived_requested_time_s"], ticks*dt), "requested tick times")
        effective = float(np.float32(dt))
        require(row["dt_effective_s"] == effective, "effective dt cast")
        require(np.array_equal(a["tick_derived_effective_time_s"], ticks*effective), "effective tick times")
        expected_time = np.zeros(n+1, dtype=np.float32)
        for i in range(n):
            expected_time[i+1] = np.float32(expected_time[i]+np.float32(dt))
        clock_recurrence = np.array_equal(a["time_s"].view(np.uint32), expected_time.view(np.uint32))
        require(clock_recurrence, "FP32 repeated-addition clock differs")
        require(np.array_equal(a["qpos"][0].view(np.uint32), ca["qpos"][0].astype(np.float32).view(np.uint32)), "shared initial qpos")
        require(np.array_equal(a["qvel"][0].view(np.uint32), ca["qvel"][0].astype(np.float32).view(np.uint32)), "shared initial qvel")
        require(row["model_xml_sha256"] == cr["model_xml_sha256"], "canonical compiled XML SHA")
        require(opts == cr["compiled_model_options"], "compiled CPU model options equality")
        for name in ("solver", "integrator", "cone", "iterations", "ls_iterations", "enableflags", "disableflags"):
            require(g["GPU_effective_options"][name] == opts["option"][name], "GPU option "+name)
        verify(row["compiled_model_XML"])
        tree = ET.parse(row["compiled_model_XML"]["path"])
        body = tree.find("./worldbody/body[@name='hot3d_phone24']")
        inertial = body.find("inertial")
        center = np.fromstring(inertial.get("pos"), sep=" ")
        inertia = np.fromstring(inertial.get("diaginertia"), sep=" ")
        assert center.shape == (3,) and inertia.shape == (3,)
        com, vc, omega, energy = state_geometry(a["qpos"], a["qvel"], center, inertia, mass)
        # Generated contact table preserves every original slot, no force filter.
        wi = TESTS_INDEX[row["test"]]
        require(np.all(ct["worldid"] == wi), "contact world owner")
        require(ct["state_tick"].dtype.kind in "iu" and np.all((ct["state_tick"] >= 0) & (ct["state_tick"] <= n)), "contact tick range")
        names = opts["geom_names"]
        allowed_pair = {names.index("phone24_collision_aabb"), names.index("bench_support")}
        require(all(set(pair) == allowed_pair for pair in ct["geom"]), "generated phone-plane pair identities")
        require(np.all((ct["original_Warp_contact_slot"] >= 0) & (ct["original_Warp_contact_slot"] < g["capacity_naconmax"])), "contact source slot range")
        pairs = list(zip(ct["state_tick"].tolist(), ct["original_Warp_contact_slot"].tolist()))
        require(len(set(pairs)) == len(pairs), "duplicate contact slot within a tick")
        counts = np.bincount(ct["state_tick"], minlength=n+1)
        mins = np.full(n+1, np.inf)
        np.minimum.at(mins, ct["state_tick"], ct["dist"])
        empty = counts == 0
        depth = np.where(empty, 0., np.maximum(0., -mins))
        require(np.array_equal(counts, a["contact_count"]), "contact counts from complete long table")
        require(np.array_equal(empty, a["empty_generated_contact_set"]), "empty contact masks")
        require(np.isnan(a["minimum_generated_contact_distance_m"][empty]).all(), "valid empty minimum must be NaN")
        require(np.array_equal(mins[~empty], a["minimum_generated_contact_distance_m"][~empty]), "nonempty generated minima")
        require(np.array_equal(depth, a["penetration_m"]), "observed-candidate depth")
        state_finite = all(np.isfinite(a[k]).all() for k in ("time_s", "qpos", "qvel", "com_world_m", "com_velocity_world_m_s", "omega_world_rad_s", "energy_potential_kinetic_J", "applied_com_force_world_x_N"))
        raw_contacts_finite = all(np.isfinite(ct[k]).all() for k in ("dist", "pos", "frame"))
        qnorm = norm(a["qpos"][:, 3:])
        capacity_ok = (not np.any(a["overflow_flags"]) and not np.any(g["final_overflow_flags"]) and
                       np.all(a["pre_solver_nefc_after_step"] <= g["capacity_njmax_per_world"]) and
                       all(x <= g["capacity_njmax_per_world"] for x in g["final_nefc"]))
        numeric_ok = state_finite and raw_contacts_finite and bool(np.all(qnorm > 0)) and bool(np.all(energy[:, 1] >= 0)) and capacity_ok
        all_g2_numeric &= numeric_ok
        rest = ticks*dt >= 2.5-dt/4
        vs, ws = norm(vc), norm(omega)
        expected_force = np.zeros(n+1, dtype=np.float32)
        if test == "force_push_release":
            previous_tick = np.arange(n)
            expected_force[1:] = np.where((previous_tick >= round(1/dt)) & (previous_tick < round(1.2/dt)), 2., 0.)
        force_schedule = np.array_equal(a["applied_com_force_world_x_N"].view(np.uint32), expected_force.view(np.uint32))
        requested_impulse = math.fsum(float(x) for x in a["applied_com_force_world_x_N"][1:])*dt
        effective_impulse = math.fsum(float(x) for x in a["applied_com_force_world_x_N"][1:])*effective
        expected_impulse = .4 if test == "force_push_release" else 0.
        impulse_roundoff = 64*np.finfo(float).eps*max(1., abs(expected_impulse))
        e = energy.sum(1)
        energy_windows = []
        for start, stop in ([(0., 1.), (1.2, 3.)] if test == "force_push_release" else [(0., 3.)]):
            first, last = round(start/dt), round(stop/dt)
            gain = float(np.max(e[first:last+1]-e[first]))
            limit = max(.0005, .10*max(abs(float(e[first])), .005))
            energy_windows.append({"start_tick": first, "end_tick_inclusive": last, "start_energy_J": float(e[first]), "max_positive_gain_J": gain, "limit_J": limit, "passed": gain <= limit})
        work = math.fsum(float(f*dx) for f, dx in zip(a["applied_com_force_world_x_N"][1:], np.diff(com[:, 0])))
        first_contact = int(np.flatnonzero(counts > 0)[0]) if np.any(counts > 0) else None
        metrics = {"completed_steps": n, "state_rows": n+1, "empty_generated_rows": int(empty.sum()),
            "final_time_accumulated_F32_s": float(a["time_s"][-1]), "requested_dt_s": dt, "effective_dt_F32_s": effective,
            "max_clock_drift_vs_tick_effective_dt_s": float(np.max(np.abs(a["time_s"].astype(float)-ticks*effective))),
            "clock_repeated_FP32_addition_bit_exact": bool(clock_recurrence),
            "max_penetration_m": float(depth.max()), "rest_max_penetration_m": float(depth[rest].max()),
            "rest_COM_speed_RMS_m_s": rms(vs[rest]), "rest_COM_speed_max_m_s": float(vs[rest].max()),
            "rest_angular_speed_RMS_rad_s": rms(ws[rest]), "rest_angular_speed_max_rad_s": float(ws[rest].max()),
            "first_generated_contact_tick": first_contact, "first_generated_contact_tick_time_s": first_contact*dt if first_contact is not None else None,
            "requested_grid_force_impulse_N_s": requested_impulse, "effective_grid_force_impulse_N_s": effective_impulse,
            "force_active_intervals": int(np.count_nonzero(a["applied_com_force_world_x_N"][1:])),
            "force_tick_schedule_exact": bool(force_schedule), "known_external_work_J": work,
            "energy_initial_J": float(e[0]), "energy_final_J": float(e[-1]), "energy_max_J": float(e.max()),
            "passive_energy_windows": energy_windows,
            "quaternion_norm_min": float(qnorm.min()), "quaternion_norm_max": float(qnorm.max()),
            "saved_COM_descriptor_maxabs_error_m": float(np.max(np.abs(com-a["com_world_m"]))),
            "saved_COM_v_descriptor_maxabs_error_m_s": float(np.max(np.abs(vc-a["com_velocity_world_m_s"]))),
            "saved_world_omega_descriptor_maxabs_error_rad_s": float(np.max(np.abs(omega-a["omega_world_rad_s"]))),
            "saved_rigid_energy_descriptor_maxabs_error_J": float(np.max(np.abs(energy-a["energy_potential_kinetic_J"])))}
        gr = thresholds["G4_rest_motion"]
        G3 = metrics["max_penetration_m"] <= thresholds["G3_penetration"]["max_observed_candidate_penetration_all_times_m"] and metrics["rest_max_penetration_m"] <= thresholds["G3_penetration"]["max_observed_candidate_penetration_rest_window_m"]
        G4 = metrics["rest_COM_speed_RMS_m_s"] <= gr["RMS_COM_speed_m_s"] and metrics["rest_COM_speed_max_m_s"] <= gr["max_COM_speed_m_s"] and metrics["rest_angular_speed_RMS_rad_s"] <= gr["RMS_angular_speed_rad_s"] and metrics["rest_angular_speed_max_rad_s"] <= gr["max_angular_speed_rad_s"]
        G6 = bool(np.any(counts[rest] > 0)) and force_schedule and abs(requested_impulse-expected_impulse) <= impulse_roundoff
        gates = {"G2_numeric_and_capacity": bool(numeric_ok), "G2_complete_warning_ledger": None,
                 "G3_penetration": bool(G3), "G4_rest_motion": bool(G4),
                 "G5_energy": all(x["passed"] for x in energy_windows), "G6_contacts_and_input": bool(G6)}
        by_key[key] = {"qpos": a["qpos"], "com": com, "vcom": vc, "omega": omega, "first_contact_tick": first_contact}
        cpu_raw[key] = {"qpos": ca["qpos"], "com": ca["com_world_m"], "vcom": ca["com_velocity_world_m_s"],
                        "omega": ca["angular_velocity_world_rad_s"],
                        "first_contact_tick": int(np.flatnonzero(ca["contact_count"] > 0)[0])}
        parity = compare_states(by_key[key], cpu_raw[key], dt, thresholds["G8_CPU_Warp_parity"])
        per_case.append({"case": row["case"], "CPU_case": cr["case"], "mass_kg": mass, "sliding_friction": mu,
                         "protocol_test_name": test, "dt_requested_s": dt, "dt_effective_s": effective,
                         "trajectory": row["trajectory"], "contact_table": row["contact_table"], "CPU_trajectory": cr["trajectory"],
                         "structural_failures": structural, "metrics": metrics, "gates": gates, "G8_parity": parity})
    timestep_rows = []
    for mass, mu, test in itertools.product([.1, .2, .3], [.3, .6, .9], list(MAP.values())):
        coarse = by_key[(mass, mu, .0025, test)]
        fine = by_key[(mass, mu, .00125, test)]
        sampled = {k: (v[::2] if isinstance(v, np.ndarray) else v/2) for k, v in fine.items()}
        comparison = compare_states(coarse, sampled, .0025, thresholds["G7_timestep_pair"])
        timestep_rows.append({"mass_kg": mass, "sliding_friction": mu, "protocol_test_name": test, **comparison})
    counts = {name: {"pass": sum(row["gates"][name] is True for row in per_case),
                      "fail": sum(row["gates"][name] is False for row in per_case),
                      "unknown": sum(row["gates"][name] is None for row in per_case)} for name in per_case[0]["gates"]}
    geometry = read(w["asset_pins"]["geometry_approximation_audit"]["path"])
    max_pen_row = max(per_case, key=lambda x:x["metrics"]["max_penetration_m"])
    worst_parity = {name: max(({"case": x["case"], "value": x["G8_parity"]["metrics"][name]} for x in per_case), key=lambda x:x["value"]) for name in per_case[0]["G8_parity"]["metrics"]}
    assessment = {"schema": "phone_physics_stage1_Warp_saved_assessment:1", "status": "COMPLETED_SAVED_ASSESSMENT_WITH_FAILURES",
        "created_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(), "assessor": pin(__file__),
        "protocol": pp, "Warp_producer_report": wp_report, "CPU_producer_report": pin(BASE/"cpu_v1/result.json"),
        "geometry_audit": w["asset_pins"]["geometry_approximation_audit"],
        "independence_scope": "Separate saved-array arithmetic without producer/physics imports; assessor authored Warp recorder, so not an independent author/experimental execution.",
        "identity_checks": identity_checks, "structural_failure_case_count": sum(bool(r["structural_failures"]) for r in per_case),
        "gate_case_counts": counts, "G7_timestep_passed": sum(r["passed"] for r in timestep_rows), "G7_timestep_total": 27,
        "G8_parity_passed": sum(r["G8_parity"]["passed"] for r in per_case), "G8_parity_total": 54,
        "maximum_Warp_observed_candidate_penetration": {"case": max_pen_row["case"], "m": max_pen_row["metrics"]["max_penetration_m"]},
        "worst_G8_metrics": worst_parity, "case_results": per_case, "timestep_comparisons": timestep_rows,
        "Warp_bench_qualified": False, "backend_numeric_parity_qualified": all(r["G8_parity"]["passed"] for r in per_case) and not any(r["structural_failures"] for r in per_case),
        "stage1_qualified": False, "real_parameter_calibrated": False, "ego_retargeting_qualified": False, "hand_task_qualified": False,
        "not_exported": ["Warp engine-native energy", "Warp decoded native contact forces", "complete Warp solver-warning ledger"],
        "limits": ["G3 absolute penetration failure survives even when G7/G8 comparisons pass.",
                   "A valid empty generated set has minimum-distance NaN and observed-candidate depth0; global clearance remains unknown.",
                   "Clock drift is reported from FP32 repeated addition; all comparisons use integer ticks, never accumulated time as an index.",
                   "Warp contact tick i is native PRE-forward with input for interval i; CPU extra POST-forward at tick i retains input for interval i-1. Compare geometry, not a contact-force waveform.",
                   "CPU97200 additional post-forward calls vsWarp18 final-forward calls are different sampling API counts; no whole-hidden-state or general backend equivalence certificate.",
                   "Backend/dt float precision differences are preserved; no physics rerun, solver retuning, mass/friction selection, hand, Actor, or training."],
        "actual_NPZ_reads": arrays_read, "new_physics_or_model_execution": False}
    cpu_review = BASE/"cpu_saved_assessment_v1/report.json"
    if cpu_review.exists():assessment["separate_CPU_saved_assessment"] = pin(cpu_review)
    save_json(output/"report.json", assessment)
    write_csv(output/"all54_case_gates.csv", [{"case": r["case"], "mass_kg": r["mass_kg"], "sliding_friction": r["sliding_friction"], "test": r["protocol_test_name"], "dt_s": r["dt_requested_s"], "gates": r["gates"], "metrics": r["metrics"], "G8": r["G8_parity"], "structural_failures": r["structural_failures"]} for r in per_case])
    write_csv(output/"all27_timestep_comparisons.csv", timestep_rows)
    print(json.dumps({"report": pin(output/"report.json"), "gate_counts": counts,
                      "G7": [assessment["G7_timestep_passed"], 27], "G8": [assessment["G8_parity_passed"], 54],
                      "max_penetration": assessment["maximum_Warp_observed_candidate_penetration"],
                      "structural_failed_cases": assessment["structural_failure_case_count"]}))


TESTS_INDEX = {"drop": 0, "settle": 1, "push_release": 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "SAVED_ONLY_DEFAULT_NO_ARRAY_READS", "assessor": pin(__file__)}))
    else:
        out = Path(__file__).resolve().parent
        if (out/"report.json").exists() or (out/"failure.json").exists():
            raise FileExistsError("Fresh assessment result required")
        try:
            assess(out)
        except Exception as exc:
            save_json(out/"failure.json", {"status": "FAILED_SAVED_ASSESSMENT_NO_RETRY", "error": repr(exc), "assessor": pin(__file__)})
            raise
