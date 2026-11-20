"""Generate frozen actions with different planning models and cross-replay them on one HOT3D clip."""
import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import evaluate_mug_frozen_sensitivity as physics
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


SEEDS = (8201, 8202)
STEPS = 30
HORIZON = 3
GENERATORS = {"nominal": (1., 1.), "heavy_low": (1.15, 0.8)}
EVALUATORS = {**GENERATORS, "intermediate": (1.075, 0.9)}


def commands_for_seed(seed, prior, base, preload, limits):
    rng = np.random.default_rng(seed)
    preload = preload / np.max(np.abs(preload))
    grid = []
    for phase in range(STEPS):
        center = prior[phase:phase + HORIZON]
        archived = base[phase:phase + HORIZON]
        directions = rng.normal(size=(3, center.shape[1]))
        directions /= np.sqrt(np.mean(directions ** 2, axis=1, keepdims=True))
        blocks = [archived, center]
        blocks.extend(center + gain * preload for gain in (-.10, -.05, .05, .10))
        for direction in directions:
            blocks.extend([center + .04 * direction, center - .04 * direction])
        # The same absolute command tensor is loaded by both planning models.
        grid.append(np.clip(np.asarray(blocks), limits[:, 0], limits[:, 1]))
    return np.stack(grid)


def plan(path, binding, z, initial_ctrl, base, grid, mass_scale, mu_scale):
    m = mujoco.MjModel.from_binary_path(str(path))
    body = binding["mug_body_id"]
    qa = binding["mug_free_qpos"]
    hand = np.isin(np.arange(m.ngeom), [m.geom(n).id for n in binding["hand_collision_names"]])
    mug = np.isin(np.arange(m.ngeom), binding["mug_part_geom_ids"])
    a, b = m.pair_geom1, m.pair_geom2
    pairs = (hand[a] & mug[b]) | (mug[a] & hand[b])
    assert pairs.sum() == hand.sum() * mug.sum()
    m.pair_friction[pairs, :2] *= mu_scale
    actual = mujoco.MjData(m)
    if mass_scale != 1.:
        m.body_mass[body] *= mass_scale
        m.body_inertia[body] *= mass_scale
        mujoco.mj_setConst(m, actual)
    actual.qpos[:] = z["initial_qpos27"]
    actual.qvel[:] = z["initial_qvel26"]
    actual.ctrl[:] = initial_ctrl
    mujoco.mj_forward(m, actual)
    test = mujoco.MjData(m)
    dt = .02
    substeps = round(dt / m.opt.timestep)
    assert substeps == 64 and m.neq == 0 and not m.body_gravcomp.any()
    np.testing.assert_allclose(np.diff(z["elapsed_s"]), dt, atol=1e-12, rtol=0)
    table = m.geom("mug_task_table").id
    force = np.empty(6)
    quaternion_indices = qa + np.array([4, 5, 6, 3])
    ref_rotation = Rotation.from_quat(z["quaternion_wxyz"][:, [1, 2, 3, 0]])

    def interval(data, control, phase):
        data.ctrl[:] = control
        assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
        hit_table = False
        for _ in range(substeps):
            mujoco.mj_step(m, data)
            g, h = data.contact.geom.T
            idx = np.flatnonzero(((g == table) & mug[h]) | ((h == table) & mug[g]))
            for ic in idx:
                mujoco.mj_contactForce(m, data, ic, force)
                hit_table |= force[0] > .1
        mujoco.mj_kinematics(m, data)
        rot = Rotation.from_quat(data.qpos[quaternion_indices])
        com = data.qpos[qa:qa + 3] + rot.apply(z["COM_BODY_m"])
        mm = 1000 * np.linalg.norm(com - z["COM_world_m"][phase + 1])
        deg = np.rad2deg((rot * ref_rotation[phase + 1].inv()).magnitude())
        cmd_penalty = .01 * np.mean(((control - base[phase]) / .12) ** 2)
        return (mm / 10) ** 2 + (deg / 10) ** 2 + 100 * hit_table + cmd_penalty

    chosen, costs, qpos, commands = [], [], [], []
    for i, candidates in enumerate(grid):
        phase_costs, first_qpos = [], []
        for candidate in candidates:
            mujoco.mj_copyData(test, m, actual)
            cost = 0.
            for j, control in enumerate(candidate):
                cost += interval(test, control, i + j)
                if j == 0:
                    first_qpos.append(test.qpos.copy())
            phase_costs.append(cost)
        idx = int(np.argmin(phase_costs))
        costs.append(phase_costs)
        chosen.append(idx)
        commands.append(candidates[idx, 0].copy())
        interval(actual, commands[-1], i)
        np.testing.assert_array_equal(actual.qpos, first_qpos[idx])
        qpos.append(actual.qpos.copy())
    return dict(ctrl=np.array(commands), chosen=np.array(chosen), costs=np.array(costs), qpos=np.array(qpos))


def generate(source, out):
    pin = file_pin(source)
    old = physics.read_json(pin)
    original = physics.read_json(old["parent"])
    prior = next(row["trace"] for row in original["rows"] if row["mode"] == "support" and row["seed"] == 7412)
    inputs = {k: old["inputs"][k] for k in ("base", "reference", "binding", "model")}
    model = verify_file_pin(inputs["model"])
    binding = physics.read_json(inputs["binding"])
    z = physics.load_arrays(inputs["reference"])
    ff = physics.load_arrays(inputs["base"])["ctrl"]
    center = physics.load_arrays(prior)["ctrl"]
    assert len(center) >= STEPS + HORIZON - 1
    limits = mujoco.MjModel.from_binary_path(str(model)).actuator_ctrlrange.copy()
    out.mkdir(exist_ok=False, parents=True)
    intent = dict(source=pin, prior=prior, script=file_pin(__file__), physics=file_pin(physics.__file__),
                  inputs=inputs, generators=GENERATORS, evaluators=EVALUATORS, seeds=SEEDS,
                  steps=STEPS, native_substeps=64, horizon=HORIZON, candidates=12,
                  methods="Both use identical absolute controls precomputed per seed, phase and branch: FF, archived nominal 7412-support, four signed fixed preload offsets about that prior, and three antithetic Gaussian pairs about that prior. No post-selection tuning.",
                  score="Sum over three future controls of (COM_mm/10)^2+(SO3_deg/10)^2+100*any_table_normal_gt_0.1N+.01*mean(((u-current_phase_ff)/.12)^2); execute lowest-cost branch's first action; tie by candidate index.",
                  protocol="All 30 replans execute even after first breach. No resetting, state injection or recovery credit after a breach. Original reference has 100 controls. Intermediate held out from generation. CPU MuJoCo " + mujoco.__version__,
                  prefix_rule=old["prefix_rule"])
    intent_path = out / "intent.json"
    write_json(intent_path, intent)
    runs = []
    for seed in SEEDS:
        grid = commands_for_seed(seed, center, ff[1:], ff[0] - z["initial_qpos27"][:20], limits)
        assert grid.shape == (STEPS, 12, HORIZON, 20)
        path = out / f"grid_{seed}.npz"
        np.savez_compressed(path, ctrl=grid)
        grid_pin = file_pin(path)
        for name, (mass, mu) in GENERATORS.items():
            commands = physics.load_arrays(grid_pin)["ctrl"]
            start = perf_counter()
            trajectory = plan(model, binding, z, ff[0], ff[1:], commands, mass, mu)
            elapsed = perf_counter() - start
            path = out / f"frozen_{name}_{seed}.npz"
            np.savez_compressed(path, **trajectory)
            runs.append(dict(seed=seed, model=name, grid=grid_pin, trace=file_pin(path),
                             planning_steps=STEPS * 12 * HORIZON * 64, executed_steps=STEPS * 64,
                             wall_s=elapsed))
            write_json(out / "progress.json", dict(completed=runs))
            print(seed, name, "frozen", round(elapsed, 2), "s", flush=True)
    write_json(out / "frozen.json", dict(intent=file_pin(intent_path), rows=runs))


def evaluate(freeze, out):
    pin = file_pin(freeze)
    frozen = physics.read_json(pin)
    intent = physics.read_json(frozen["intent"])
    verify_file_pin(intent["script"])
    verify_file_pin(intent["physics"])
    inputs = intent["inputs"]
    path = verify_file_pin(inputs["model"])
    binding = physics.read_json(inputs["binding"])
    z = physics.load_arrays(inputs["reference"])
    ff = physics.load_arrays(inputs["base"])["ctrl"]
    out.mkdir(exist_ok=False, parents=True)
    rows = []
    for run in frozen["rows"]:
        archive = physics.load_arrays(run["trace"])
        grid = physics.load_arrays(run["grid"])["ctrl"]
        np.testing.assert_array_equal(archive["ctrl"], grid[np.arange(len(grid)), archive["chosen"], 0])
        for name, (mass, mu) in intent["evaluators"].items():
            detail, trace = physics.replay(path, binding, z, ff[0], archive["ctrl"], mass, mu)
            same_model = name == run["model"]
            if same_model:
                np.testing.assert_array_equal(trace["qpos"], archive["qpos"])
            file = out / f'{run["seed"]}_{run["model"]}_on_{name}.npz'
            np.savez_compressed(file, **trace)
            rows.append(dict(seed=run["seed"], generator=run["model"], evaluator=name,
                             exact_generator_qpos=same_model, trace=file_pin(file), **detail))
            print(run["seed"], run["model"], "on", name, detail["valid_s"], detail["right_censored"], flush=True)
    write_json(out / "report.json", dict(freeze=pin, rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "evaluate"])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (generate if args.phase == "generate" else evaluate)(args.source.resolve(), args.out.resolve())
