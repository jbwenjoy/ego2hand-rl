"""Export action-conditioned 20 ms collision labels and an independently archived W1 test set."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1] / "runs"
THRESHOLD = np.array([2., .2])
QUOTAS = {"parent": 54, "shared": 2, "specialist_001": 36, "specialist_003": 17, "base": 40}
K = 21


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def catalog():
    reports = {k: file_pin(ROOT / path / "report.json") for k, path in {
        "001": "pen_multiclip_ppo_native_v1/clip_001",
        "003_fit": "pen_start_prefix_balanced_w1_v1/base/base_0",
        "003_test": "pen_prefix_digit_handoff_v1",
        "002": "pen_dev_init_compare_v1/refit002",
    }.items()}
    fit = [dict(clip="clip_001", arm=row["arm"], rep=0, case=row["case"])
           for row in read(reports["001"])["rows"] if row["rep"] == 0]
    fit.append(dict(clip="clip_003", arm="base", rep=0, case=reports["003_fit"]))
    test = [dict(clip="clip_003", arm=arm, rep=row["rep"], case=row["case"])
            for arm, value in read(reports["003_test"])["arms"].items() for row in value["cases"]]
    test.extend(dict(clip="clip_002", arm=row["arm"], rep=row["rep"], case=row["case"])
                for row in read(reports["002"])["rows"] if row["arm"] == "support" and row["rep"] == 0)
    return reports, fit, test


def split_for(clip, phase):
    width = 32 if clip == "clip_001" else 12
    return "val" if ((phase - 1) // width) % 4 == 2 else "train"


def selected_phases(case, count, is_test):
    clip = case["clip"]
    if is_test:
        return np.arange(9 if clip == "clip_003" else 1, count + 1)
    # Skip the original clip001 entry. Its preload solver memory is not in the trace.
    start = 2 if clip == "clip_001" else 1
    eligible = [p for p in range(start, count + 1)
                if split_for(clip, max(p - 1, 1)) == split_for(clip, p) == split_for(clip, p + 1)]
    n = min(QUOTAS[case["arm"]], len(eligible))
    return np.asarray(eligible)[np.linspace(0, len(eligible) - 1, n).round().astype(int)]


def geometry(model):
    body = model.body("object/cube").id
    pen = model.geom_bodyid == body
    groups = np.array([normal_group(model.geom(i).name or "") for i in range(model.ngeom)])
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    pairs = [[(int(h), int(o)) for h in np.flatnonzero(active & (groups == g))
              for o in np.flatnonzero(pen & active)] for g in range(6)]
    assert all(pairs)
    return pen, groups >= 0, pairs


def depth(data, pen, hand):
    a, b = data.contact.geom.T
    d = np.maximum(0., -data.contact.dist)
    return 1000 * np.array([d[pen[a] ^ pen[b]].max(initial=0.),
                            d[hand[a] & hand[b]].max(initial=0.)])


def prepare(model, data, qpos, qvel, warm, mount, time):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qacc_warmstart[:] = warm
    data.mocap_pos[0] = mount[:3]
    data.mocap_quat[0] = mount[3:]
    data.time = time


def boundaries(model, data, geom):
    pen, hand, pairs = geom
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)
    distances = [1000 * min(mujoco.mj_geomDistance(model, data, h, o, .03, None) for h, o in group)
                 for group in pairs]
    return np.r_[distances, depth(data, pen, hand)]


def commands(model, actual, rng, is_test):
    actual = actual.astype(float)
    lo, hi = model.actuator_ctrlrange.T
    assert np.all((actual >= lo) & (actual <= hi))
    if is_test:
        return [actual]
    ring = np.isin(np.arange(20), np.arange(12, 16))
    result = [actual]
    for mask, amplitude in [(ring, a) for a in (.025, .075, .15, .24)] + [
            (~ring, a) for a in (.025, .075, .15)] + [(np.ones(20, bool), a) for a in (.025, .075, .15)]:
        delta = np.zeros(20)
        delta[mask] = rng.uniform(-1, 1, mask.sum())
        delta *= amplitude / np.abs(delta).max()
        result.extend([np.clip(actual + delta, lo, hi), np.clip(actual - delta, lo, hi)])
    assert len(result) == K and len({v.tobytes() for v in result}) == K
    return result


def load_source(case, anchors):
    report = read(case["case"])
    folder = verify_file_pin(case["case"]).parent
    if (folder / "cpu_export/report.json").exists():
        receipt = read(file_pin(folder / "cpu_export/report.json"))
        assert receipt["model"]["sha256"] == anchors["model"]["sha256"]
        assert receipt["physical_context"]["sha256"] == anchors["physics"]["sha256"]
    else:
        assert file_pin(folder / "model.mjb")["sha256"] == anchors["source"]["sha256"]
        assert file_pin(folder / "physical_context.npz")["sha256"] == anchors["physics"]["sha256"]
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as archive:
        keys = ("pre_qpos", "pre_qvel", "post_qacc_warmstart", "post_mocap_pos", "post_mocap_quat",
                "post_ctrl", "pre_penetration_m", "pre_self_penetration_m", "post_GPU_time")
        trace = {k: archive[k] for k in keys}
        assert not archive["post_xfrc_applied"].any() and not archive["post_qfrc_applied"].any()
    assert len(trace["pre_qpos"]) == report["controls"] * 8
    for field in ("post_mocap_pos", "post_mocap_quat"):
        arr = trace[field]
        np.testing.assert_array_equal(arr, np.broadcast_to(arr[0], arr.shape))
    assert report["rep"] == case["rep"] and report["arm"] == case["arm"] or case["arm"] in ("hold", "abrupt")
    return report, trace


def pack(cases, is_test, model, geom, anchors, state_table, fit_hashes):
    data = mujoco.MjData(model)
    rows, sources = [], []
    for case in cases:
        report, trace = load_source(case, anchors)
        count = report["controls"]
        phases = selected_phases(case, count, is_test)
        mount = np.r_[trace["post_mocap_pos"][0,0], trace["post_mocap_quat"][0,0]]
        per_source = dict(case=case, selected=[], controls=count, stop=report["stop"], trace=report["trace"])
        for phase_np in phases:
            phase = int(phase_np)
            tick = 8 * (phase - 1)
            q, v = trace["pre_qpos"][tick], trace["pre_qvel"][tick]
            warm = trace["post_qacc_warmstart"][tick - 1] if tick else np.zeros(26)
            time = float(trace["post_GPU_time"][tick - 1]) if tick else 0.
            pd = trace["post_ctrl"][tick]
            np.testing.assert_array_equal(trace["post_ctrl"][tick:tick+8], np.broadcast_to(pd, (8,20)))
            if not tick:
                assert case["clip"] != "clip_001" and np.all(v == 0)
            key = hashlib.sha256(np.r_[q, v, warm, mount].astype("<f8").tobytes()).hexdigest()
            if not is_test and key in fit_hashes:
                continue
            if is_test and key in fit_hashes:
                raise ValueError("Exact test state was seen in fit")
            if not is_test:
                fit_hashes.add(key)
            state_id = state_table.setdefault(key, len(state_table))
            per_source["selected"].append(phase)
            prepare(model, data, q, v, warm, mount, time)
            boundary = boundaries(model, data, geom)
            s = native_state55(q, v, np.arange(7, 27), np.arange(6, 26), 0, 0)
            oracle = 1000 * np.array([trace[k][tick:tick+8].max() for k in
                                     ("pre_penetration_m", "pre_self_penetration_m")], dtype=float)
            np.testing.assert_allclose(oracle / 1000, report["checks"][phase-1]["geometry"][:2], rtol=0, atol=1e-9)
            rng = np.random.default_rng(np.random.SeedSequence([8301, int(key[:8], 16), int(key[8:16], 16)]))
            for action_id, command in enumerate(commands(model, pd, rng, is_test)):
                prepare(model, data, q, v, warm, mount, time)
                data.ctrl[:] = command
                peaks = np.zeros(2)
                for _ in range(8):
                    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                    mujoco.mj_step(model, data)
                    peaks = np.maximum(peaks, depth(data, geom[0], geom[1]))
                gold = oracle if is_test else peaks
                rows.append(dict(split="test" if is_test else split_for(case["clip"], phase), clip=case["clip"],
                                 arm=case["arm"], rep=case["rep"], phase=phase, state_id=state_id, action_id=action_id,
                                 state55=s, warm26=warm, pd20=command, boundary_mm=boundary, y_mm=gold, cpu_mm=peaks,
                                 exceeds=gold > THRESHOLD, qpos27=q, qvel26=v, mount7=mount, time_s=time))
        sources.append(per_source)
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    for arr in arrays.values():
        if arr.dtype.kind == "f":
            assert np.isfinite(arr).all()
    for clip in np.unique(arrays["clip"]):
        train, val = [np.unique(arrays["phase"][(arrays["split"] == name) & (arrays["clip"] == clip)]) for name in ("train", "val")]
        if not is_test:
            assert len(train) and len(val) and np.min(np.abs(train[:, None]-val)) >= 3
    arrays["threshold_mm"] = THRESHOLD
    return arrays, sources


def run(out):
    reports, fit, test = catalog()
    refit = file_pin(ROOT / "pen_refit_catalog_v2/refit.json")
    anchors = read(refit)
    phys = ROOT / "pen_multiclip_ppo_native_v1/clip_001/parent_0/cpu_export/report.json"
    anchors["physics"] = read(file_pin(phys))["physical_context"]
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors["model"])))
    verify_file_pin(anchors["physics"])
    assert model.neq == model.na == 0 and model.nv == 26 and model.nq == 27 and model.nmocap == 1
    assert model.opt.timestep == .0025 and np.all(model.body_gravcomp == 0)
    np.testing.assert_array_equal(model.jnt_qposadr[model.actuator_trnid[:,0]], np.arange(7,27))
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.body_mass[model.body("object/cube").id] > 0.
    out.mkdir(exist_ok=False, parents=True)
    intent = dict(sources=reports, fit=fit, test=test, refit=refit, physics=anchors["physics"], script=file_pin(__file__),
                  actions=K, substeps=8, dt_s=.0025, thresholds_mm=THRESHOLD.tolist(), seed=8301,
                  frozen_sampling="For each fit actor(rep0), uniformly subsample phase numbers on the predeclared allowed grid, including eligible terminal controls. Drop clip001 first control (unobserved entry warmstart). Deduplicate exact physical start bytes across fit cases. clip003 base starts from its own policy with no prefix.",
                  quotas=QUOTAS, split="Same phase block assignment across actors of a clip. Consecutive 32-control blocks on 001, 12-control blocks on 003, every 3rd of each 4 blocks is validation. Omit both phases touching each split transition, including across actors.",
                  labels="CPU training peaks and original Warp test peaks use eight pre-integration collision configurations (contact arrays left by mj_step). Original PD row is action 0; test has only observed PD. CPU test comparator uses the same measured start and command. Free pen, no external force/equality aid.",
                  inputs="state55 in canonical pen record chart; previous native warm26, actual commanded pd20; six closest signed rigid distances (fingers 1..5,palm, capped at 30 mm) and instantaneous object and hand-self depths from prestate only. No future observed trajectory in features.",
                  test_scope="All four previously observed clip003 handoffs, phase>=9, all three archival numerical replicates; only held-out support actor rep0 on distinct clip002. Test cases are held out of this fit, not novel unseen research clips. No test labels used to tune sampling.")
    write_json(out / "intent.json", intent)
    geom = geometry(model)
    state_table, fit_hashes, result = {}, set(), {}
    for label, cases in [("fit", fit), ("test", test)]:
        arrays, sources = pack(cases, label == "test", model, geom, anchors, state_table, fit_hashes)
        file = out / f"{label}.npz"
        np.savez_compressed(file, **arrays)
        stats = {}
        for split in np.unique(arrays["split"]):
            mask = arrays["split"] == split
            stats[str(split)] = dict(rows=int(mask.sum()), unique_states=len(np.unique(arrays["state_id"][mask])),
                                    exceed=arrays["exceeds"][mask].sum(axis=0).tolist())
        result[label] = dict(file=file_pin(file), sources=sources, stats=stats)
        print(label, stats, flush=True)
    write_json(out / "report.json", dict(intent=file_pin(out / "intent.json"), datasets=result,
                                         cpu_native_ticks=sum(sum(r["rows"] for r in entry["stats"].values()) * 8 for entry in result.values()),
                                         fit_excludes_test_state_bytes=True, min_train_val_phase_separation=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out.resolve())
