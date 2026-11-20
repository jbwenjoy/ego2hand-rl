"""Select from archived mug controls on single factors, then test frozen winners on paired factors."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

import evaluate_mug_frozen_sensitivity as physics
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


CORNERS = [
    ("light_low_mu", 0.85, 0.8), ("light_high_mu", 0.85, 1.2),
    ("heavy_low_mu", 1.15, 0.8), ("heavy_high_mu", 1.15, 1.2),
]


def evaluate(inputs, controls, cases, out, expected_qpos=None):
    reference = physics.load_arrays(inputs["reference"])
    binding = physics.read_json(inputs["binding"])
    model = verify_file_pin(inputs["model"])
    initial_ctrl = physics.load_arrays(inputs["base"])["ctrl"][0]
    records = []
    for label, commands in controls.items():
        for name, mass_scale, mu_scale in cases:
            start = perf_counter()
            result, trace = physics.replay(model, binding, reference, initial_ctrl, commands, mass_scale, mu_scale)
            elapsed = perf_counter() - start
            exact = []
            if name == "nominal" and expected_qpos is not None:
                for alias, expected in expected_qpos[label]:
                    np.testing.assert_array_equal(trace["qpos"], expected)
                    exact.append(alias)
            metric = np.mean((trace["com_mm"] / 10) ** 2 + (trace["so3_deg"] / 10) ** 2)
            path = out / f"{label}_{name}.npz"
            np.savez_compressed(path, **trace)
            records.append(dict(label=label, case=name, mass_scale=mass_scale, sliding_scale=mu_scale,
                                pose_cost=float(metric), replay_wall_s=elapsed, exact_nominal_qpos=exact,
                                trace=file_pin(path), **result))
    return records


def rank(candidates, rows, cases):
    scores = []
    for label in candidates:
        members = [r for r in rows if r["label"] == label and r["case"] in cases]
        assert len(members) == len(cases)
        scores.append(dict(label=label, min_valid_s=min(r["valid_s"] for r in members),
                           worst_pose_cost=max(r["pose_cost"] for r in members)))
    return sorted(scores, key=lambda r: (-r["min_valid_s"], r["worst_pose_cost"], r["label"]))


def screen(source, out):
    parent = file_pin(source)
    prior = physics.read_json(parent)
    old = physics.read_json(prior["parent"])
    inputs = {k: prior["inputs"][k] for k in ("model", "reference", "binding", "base")}
    by_label = {(r["seed"], r["mode"]): r for r in old["rows"] if r["mode"] in {"support", "tracking"}}
    ordered = [by_label[seed, mode] for seed in (7411, 7412, 7413) for mode in ("tracking", "support")]
    raw = [(f"{r['mode']}_{r['seed']}", r["trace"], physics.load_arrays(r["trace"])) for r in ordered]
    steps = min(len(trace["ctrl"]) for _, _, trace in raw)
    controls, aliases, catalog, expected_qpos = {}, {}, {}, {}
    for ordinal, (name, pin, trace) in enumerate(raw):
        commands = trace["ctrl"][:steps]
        assert commands.dtype == np.dtype("<f8") and commands.shape == (steps, 20)
        block = commands.tobytes("C")
        if block not in aliases:
            label = f"c{ordinal + 1:02d}"
            aliases[block] = label
            controls[label] = commands
            expected_qpos[label] = []
            catalog[label] = dict(sha256_ctrl_C_float64=hashlib.sha256(block).hexdigest(), archives=[])
        label = aliases[block]
        catalog[label]["archives"].append(dict(name=name, trace=pin, original_steps=len(trace["ctrl"])))
        expected_qpos[label].append((name, trace["qpos"][:steps]))
    out.mkdir(parents=True, exist_ok=False)
    intent = dict(source=parent, script=file_pin(Path(__file__)), physics=file_pin(physics.__file__), inputs=inputs,
                  candidates=catalog, controls=steps, common_s=0.02*steps, screen_cases=physics.CASES,
                  test_only_cases=CORNERS, tie_order="maximize minimum valid prefix over selection cases, then minimize maximum across cases of mean over ALL common time steps of (COM_mm/10)^2+(SO3_deg/10)^2, then ascending fixed cNN",
                  prefix_rule=prior["prefix_rule"], censoring=prior["censoring"],
                  scope="One known HOT3D trajectory and fixed frozen commands with a common prefix; test parameter combinations excluded from selection. Commands were originally planned with nominal dynamics on this trajectory.")
    intent_path = out / "intent.json"
    write_json(intent_path, intent)
    rows = evaluate(inputs, controls, physics.CASES, out, expected_qpos)
    groups = {"nominal": {"nominal"}, "robust": {c[0] for c in physics.CASES}}
    ranks = {group: rank(controls, rows, cases) for group, cases in groups.items()}
    costs = {}
    for group, cases in groups.items():
        members = [r for r in rows if r["case"] in cases]
        costs[group] = dict(replays=len(members), native_steps=sum(r["native_steps"] for r in members),
                            replay_wall_s=sum(r["replay_wall_s"] for r in members))
    costs["feedforward"] = dict(replays=0, native_steps=0)
    screened = out / "screen.json"
    write_json(screened, dict(rows=rows, ranks=ranks, selection_costs=costs,
                              cost_note="Nominal rows are reused in robust screening. Reported costs are per-method replay costs after constructing the same archived candidate library."))
    winners = {group: ranked[0]["label"] for group, ranked in ranks.items()}
    selected = {group: controls[label] for group, label in winners.items()}
    selected["feedforward"] = physics.load_arrays(inputs["base"])["ctrl"][1:steps+1]
    path = out / "frozen_commands.npz"
    np.savez_compressed(path, **selected)
    write_json(out / "frozen_selection.json", dict(intent=file_pin(intent_path), screen=file_pin(screened),
                                                   winners=winners, commands=file_pin(path)))
    print("Frozen choices", winners, "common window", intent["common_s"], "s", flush=True)


def test(freeze, out):
    pin = file_pin(freeze)
    selected = physics.read_json(pin)
    intent = physics.read_json(selected["intent"])
    verify_file_pin(intent["script"])
    verify_file_pin(intent["physics"])
    screen_report = physics.read_json(selected["screen"])
    choices = physics.load_arrays(selected["commands"])
    assert {x[0] for x in intent["screen_cases"]}.isdisjoint(x[0] for x in intent["test_only_cases"])
    assert {tuple(x[1:]) for x in intent["screen_cases"]}.isdisjoint(tuple(x[1:]) for x in intent["test_only_cases"])
    for group, label in selected["winners"].items():
        assert screen_report["ranks"][group][0]["label"] == label
        assert hashlib.sha256(choices[group].tobytes("C")).hexdigest() == intent["candidates"][label]["sha256_ctrl_C_float64"]
    out.mkdir(exist_ok=False, parents=True)
    rows = evaluate(intent["inputs"], choices, intent["test_only_cases"], out)
    cases = {c[0] for c in intent["test_only_cases"]}
    scores = rank(choices, rows, cases)
    for row in scores:
        group = [r for r in rows if r["label"] == row["label"]]
        row["mean_valid_s"] = float(np.mean([r["valid_s"] for r in group]))
        row["censored_corners"] = sum(r["right_censored"] for r in group)
    write_json(out / "report.json", dict(freeze=pin, rows=rows, summary=scores, selection_costs=screen_report["selection_costs"],
                                         eval_native_steps_each=sum(r["native_steps"] for r in rows if r["label"] == "nominal")))
    print("Corner results", scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "test"))
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (screen if args.phase == "screen" else test)(args.source.resolve(), args.out.resolve())
