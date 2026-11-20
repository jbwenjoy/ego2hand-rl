"""Replay frozen HOT3D mug commands under isolated mass and sliding-friction changes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


CASES = [
    ("nominal", 1.0, 1.0),
    ("mass_085", 0.85, 1.0),
    ("mass_115", 1.15, 1.0),
    ("sliding_080", 1.0, 0.8),
    ("sliding_120", 1.0, 1.2),
]


def read_json(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def load_arrays(pin):
    with np.load(verify_file_pin(pin), allow_pickle=False) as data:
        return dict(data)


def replay(model_path, binding, reference, initial_ctrl, commands, mass_scale, mu_scale):
    model = mujoco.MjModel.from_binary_path(str(model_path))
    body = binding["mug_body_id"]
    qadr = binding["mug_free_qpos"]
    table = model.geom("mug_task_table").id
    hand = [model.geom(name).id for name in binding["hand_collision_names"]]
    mug = binding["mug_part_geom_ids"]
    is_hand = np.isin(np.arange(model.ngeom), hand)
    is_mug = np.isin(np.arange(model.ngeom), mug)
    a, b = model.pair_geom1, model.pair_geom2
    pairs = (is_hand[a] & is_mug[b]) | (is_mug[a] & is_hand[b])
    assert len(set(zip(a[pairs], b[pairs]))) == len(hand) * len(mug)
    model.pair_friction[pairs, :2] *= mu_scale
    expected_mu = np.unique(model.pair_friction[pairs, :2], axis=0)
    assert expected_mu.shape == (1, 2)
    assert np.all(model.body_gravcomp == 0) and model.neq == 0 and model.nu == 20
    assert model.jnt_type[model.body_jntadr[body]] == mujoco.mjtJoint.mjJNT_FREE
    assert np.all(model.actuator_trnid[:, 0] < model.body_jntadr[body])
    data = mujoco.MjData(model)
    if mass_scale != 1:
        model.body_mass[body] *= mass_scale
        model.body_inertia[body] *= mass_scale
        mujoco.mj_setConst(model, data)
    data.qpos[:] = reference["initial_qpos27"]
    data.qvel[:] = reference["initial_qvel26"]
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)

    times = reference["elapsed_s"]
    dt = times[1] - times[0]
    sub = round(dt / model.opt.timestep)
    np.testing.assert_allclose(np.diff(times), sub * model.opt.timestep, rtol=0, atol=1e-12)
    assert sub == 64
    qpos = []
    table_peak = []
    loaded_count = 0
    solver_mu = set()
    contact_force = np.empty(6)
    for command in commands:
        data.ctrl[:] = command
        peak = 0.
        for _ in range(sub):
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            mujoco.mj_step(model, data)
            contacts = data.contact
            g, h = contacts.geom.T
            hand_mug = (is_hand[g] & is_mug[h]) | (is_mug[g] & is_hand[h])
            table_mug = ((g == table) & is_mug[h]) | ((h == table) & is_mug[g])
            # Read solver contact values after collision detection and pair mixing.
            for index in np.flatnonzero(hand_mug | table_mug):
                mujoco.mj_contactForce(model, data, index, contact_force)
                if hand_mug[index]:
                    if contact_force[0] > 0.1:
                        loaded_count += 1
                        solver_mu.add(tuple(contacts[index].friction[:2]))
                else:
                    peak = max(peak, float(contact_force[0]))
        mujoco.mj_kinematics(model, data)
        qpos.append(data.qpos.copy())
        table_peak.append(peak)
    assert solver_mu == {tuple(expected_mu[0])}
    np.testing.assert_allclose(data.time, len(commands) * dt, rtol=0, atol=1e-12)
    qpos = np.array(qpos)
    steps = slice(1, len(commands) + 1)
    rotation = Rotation.from_quat(qpos[:, qadr + np.array([4, 5, 6, 3])])
    ref_rotation = Rotation.from_quat(reference["quaternion_wxyz"][steps][:, [1, 2, 3, 0]])
    com = qpos[:, qadr:qadr + 3] + rotation.apply(reference["COM_BODY_m"])
    mm = 1000 * np.linalg.norm(com - reference["COM_world_m"][steps], axis=1)
    angle = np.rad2deg((rotation * ref_rotation.inv()).magnitude())
    assert np.isfinite(qpos).all() and np.isfinite(mm).all() and np.isfinite(angle).all()
    table_peak = np.array(table_peak)
    invalid = (mm > 10) | (angle > 10) | (table_peak > 0.1)
    prefix = int(np.argmax(invalid)) if invalid.any() else len(qpos)
    first = None
    if prefix < len(qpos):
        reasons = []
        if mm[prefix] > 10:
            reasons.append("com_mm")
        if angle[prefix] > 10:
            reasons.append("so3_deg")
        if table_peak[prefix] > 0.1:
            reasons.append("table_normal_N")
        first = dict(time_s=float(times[prefix + 1]), reasons=reasons,
                     com_mm=float(mm[prefix]), so3_deg=float(angle[prefix]),
                     table_normal_N=float(table_peak[prefix]))
    detail = dict(mass_kg=float(model.body_mass[body]), inertia_kg_m2=model.body_inertia[body].tolist(),
                  explicit_hand_mug_pairs=int(pairs.sum()), active_solver_mu=sorted(map(list, solver_mu)),
                  loaded_hand_mug_contact_samples=loaded_count, native_steps=len(commands) * sub,
                  applied_wrenches_zero_each_step=True, valid_s=float(times[prefix]),
                  right_censored=first is None, first_breach=first,
                  end_com_mm=float(mm[-1]), end_so3_deg=float(angle[-1]))
    trace = dict(qpos=qpos, ctrl=commands, time_s=times[steps], com_mm=mm, so3_deg=angle,
                 table_peak_N=table_peak)
    return detail, trace


def run(parent, output):
    report_pin = file_pin(parent)
    old = read_json(report_pin)
    ff = read_json(read_json(old["intent"])["parent"])
    trial = read_json(read_json(ff["intent"])["parent"])
    contract = read_json(read_json(trial["dynamic"])["intent"])
    binding, ref = read_json(contract["binding"]), read_json(contract["reference"])
    reference = load_arrays(ref["output"])
    model_path = verify_file_pin(ref["model"])
    ff_trace = load_arrays(ff["results"]["interval_lp"]["trace"])
    old_rows = {r["mode"]: r for r in old["rows"] if r["seed"] == 7411}
    frozen = {name: load_arrays(row["trace"]) for name, row in old_rows.items()}
    steps = min(len(frozen[k]["ctrl"]) for k in ("tracking", "support"))
    same = np.array_equal(frozen["support"]["ctrl"], frozen["tracking"]["ctrl"])
    streams = {"support": frozen["support"]["ctrl"][:steps]}
    if not same:
        streams["tracking"] = frozen["tracking"]["ctrl"][:steps]
    streams["replay"] = ff_trace["ctrl"][1:steps + 1]
    for k, v in frozen.items():
        source = streams["support"] if k == "tracking" and same else streams[k]
        overlap = min(steps, len(v["ctrl"]))
        np.testing.assert_array_equal(v["ctrl"][:overlap], source[:overlap])
    output.mkdir(exist_ok=False, parents=True)
    intent = dict(parent=report_pin, source=file_pin(Path(__file__)), cases=CASES, seed=7411,
                  simulator="native CPU MuJoCo " + mujoco.__version__, fixed="initial qpos/qvel, initial ctrl, actuator position commands, gravity, geometry, clock",
                  nominal_hand_servo="same fixed hand commands, actual hand position evolves dynamically",
                  cases_definition="mass and principal inertia scaled together; sliding friction alone scales both tangential coefficients on explicit hand-mug pairs",
                  prefix_rule="all consecutive control endpoints through first COM >10 mm, SO(3) >10 deg, or any substep table-mug normal >0.1 N in the interval; failing endpoint excluded",
                  clock="50 Hz controls, 64 native steps per control in every case",
                  support_tracking_identical_entire_controls=same, common_controls=steps,
                  frozen_shooting_lengths={k: len(frozen[k]["ctrl"]) for k in ("tracking", "support")},
                  full_reference_s=float(reference["elapsed_s"][-1]),
                  censoring="post-breach states remain in common-window endpoint comparison. A breach-free window is right-censored at the last available command.",
                  inputs={"model": ref["model"], "reference": ref["output"], "binding": contract["binding"],
                          "base": ff["results"]["interval_lp"]["trace"],
                          "frozen": {k: row["trace"] for k, row in old_rows.items()}})
    write_json(output / "intent.json", intent)
    results = []
    for mode, commands in streams.items():
        for name, mass, mu in CASES:
            detail, trace = replay(model_path, binding, reference, ff_trace["ctrl"][0], commands, mass, mu)
            verified = {}
            if name == "nominal":
                modes = ["tracking", "support"] if mode == "support" and same else [mode]
                for k in modes:
                    n = min(len(frozen[k]["qpos"]), steps)
                    np.testing.assert_array_equal(trace["qpos"][:n], frozen[k]["qpos"][:n])
                    np.testing.assert_array_equal(trace["table_peak_N"][:n] > 0.1,
                                                  [r["table"] for r in old_rows[k]["actual"][:n]])
                    np.testing.assert_allclose(trace["com_mm"][:n], [r["mm"] for r in old_rows[k]["actual"][:n]], rtol=0, atol=1e-10)
                    np.testing.assert_allclose(trace["so3_deg"][:n], [r["deg"] for r in old_rows[k]["actual"][:n]], rtol=0, atol=1e-10)
                    assert detail["valid_s"] == old_rows[k]["valid_s"]
                    verified[k] = n
                if mode == "replay":
                    np.testing.assert_array_equal(trace["qpos"], ff_trace["qpos"][:steps])
                    verified["full_feedforward"] = steps
            path = output / f"{mode}_{name}.npz"
            np.savez_compressed(path, **trace)
            results.append(dict(mode=mode, case=name, archived_exact_qpos_steps=verified,
                                trace=file_pin(path), **detail))
            print(mode, name, detail["valid_s"], detail["end_com_mm"], detail["end_so3_deg"], flush=True)
    write_json(output / "report.json", dict(intent=file_pin(output / "intent.json"), rows=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
