"""Plot the frozen held-out comparisons using archived physical scores and charged device time."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from plot_wuji_development_cost import host_path
from reconcile_wuji_heldout_cost import S5_MAX_S


ARMS = {
    "cold_single": ("逐条冷启动", "#245B84", "o"),
    "recovery_shared_mf": ("共享策略", "#A9572C", "s"),
    "recovery_shared_wm": ("共享策略＋目标新学动力学", "#096D57", "^"),
    "recovery_shared_wm_warm_dynamics": ("共享策略＋迁移配对动力学", "#794F88", "D"),
}
TARGET_BUDGET = 418_560


def read(path):
    return json.loads(path.read_bytes())


def verified_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Digest mismatch for {path}")
    return data


def pinned(record):
    return json.loads(verified_bytes(host_path(record["path"]), record["sha256"]))


def prefix_contacts(receipt, stage, report, actor):
    """Describe only the intervals actually executed. Missing sensor data stays unknown."""
    contract = pinned(stage["contract"])
    path = host_path(stage["contract"]["path"]).with_name(contract["intervals_file"])
    masks = verified_bytes(path, contract["intervals_sha256"])
    executed = actor["source_intervals"] if actor else report["numerical_failure"]["applied_intervals"]
    with np.load(io.BytesIO(masks), allow_pickle=False) as source:
        engaged = source["engaged_moving_intervals"][:executed]
        air = source["airborne_intervals"][:executed]
    counts = dict(engaged_source_total=contract["engaged_intervals"], engaged_reached=int(engaged.sum()),
                  engaged_with_hand_load=None, airborne_source_total=contract["airborne_intervals"],
                  airborne_reached=int(air.sum()), airborne_supported=None)
    if actor is None or not report["force_validation"]["validated"]:
        return counts
    trace_path = host_path(receipt["path"]).parent / "raw" / "trace.npz"
    trace = verified_bytes(trace_path, actor["trace_sha256"])
    with np.load(io.BytesIO(trace), allow_pickle=False) as arrays:
        samples = arrays["substep_contacts"]
    if samples.shape != (executed * 40, 6):
        raise ValueError("Unexpected contact trace length for the frozen 1200/30 Hz physics")
    samples = samples.reshape(executed, 40, 6)
    force = samples[:, :, :2].mean(axis=1) / (report["identity"]["mass_kg"] * 9.81)
    config = contract["config"]
    loaded = (force[:, 0] >= config["minimum_normal_load_bodyweights"]) & (samples[:, :, 2].max(axis=1) > 0)
    supported = loaded & (force[:, 1] < config["maximum_airborne_table_load_bodyweights"])
    counts.update(engaged_with_hand_load=int(loaded[engaged].sum()), airborne_supported=int(supported[air].sum()))
    return counts


def measure(arm, result, entries, stages, common_card_s, unknown_common_wall_s=0.):
    cost = common_card_s
    lower, upper = result["pretraining_exclusive_card_s_range"]
    curve = {(p["subject"], p["target_step"]): p for p in result["event_curve"]}
    seen = set()
    points = []
    for event in entries:
        if event["kind"] == "disturbance":
            continue
        if event["card"]:
            if event["unknown_cost"]:
                raise ValueError(f"GPU job of unknown duration in {arm}; x-axis needs an open bound")
            cost += event["known_wall_s"]
        if event["kind"] != "evaluate" or not event["outcome"]:
            continue
        subject, budget = event["subject"], event["steps"]
        meta = stages[subject]
        report = pinned(event["receipt"])
        state = report["status"]
        if (state != event["outcome"]["status"] or report["parquet_sha256"] != meta["parquet"]["sha256"]
                or report["contract_sha256"] != meta["contract"]["sha256"]):
            raise ValueError("Score is bound to a different outcome or source")
        source = pinned(meta["source"])
        start, end = source["frames"]
        actor = None
        if report["realizations"]:
            actor, = (r for r in report["realizations"] if r["id"] == "actor")
            executed = actor["source_intervals"]
            reasons = actor["terminal"].get("termination_causes", [])
        else:
            failure = report["numerical_failure"]
            executed = failure["applied_intervals"]
            reasons = [failure["kind"]]
            if failure["requested_intervals"] != end - start or failure["rejected_action_applied"]:
                raise ValueError("Invalid physical prefix for numerical stop")
        if not 0 <= executed <= end - start:
            raise ValueError("Recorded prefix outside the source window")
        if state == "passed":
            if not event["outcome"]["full_repeated"] or executed != end - start:
                raise ValueError("Unverified complete trajectory")
            seen.add(subject)
        point = curve[subject, budget]
        if point["distinct_verified"] != len(seen):
            raise ValueError("Score order differs from the frozen accounting")
        wall = point["end_to_end_known_s"]
        points.append(dict(subject=subject, target_steps=budget, status=state,
                           sequence=source["sequence"], object=source["object_id"],
                           first_frame=start, last_frame=end, requested=end-start, executed=executed,
                           observed_contacts=prefix_contacts(event["receipt"], meta, report, actor),
                           causes=reasons, including_pretraining_gpu_h=[(lower+cost)/3600, (upper+cost)/3600],
                           known_target_end_to_end_s=wall,
                           including_pretraining_active_wall_h=[(lower+wall)/3600, (upper+wall+unknown_common_wall_s)/3600],
                           distinct_verified=len(seen), receipt=event["receipt"]))
    if len(points) != len(curve) or abs(cost/3600-result["target_gpu_hours_known"]) > 1e-8:
        raise ValueError(f"Score or GPU ledger did not reconcile for {arm}")
    return points


def collect(runs):
    original = runs / "wuji_heldout_formal_v1"
    paired_dir = runs / "wuji_heldout_recovered_warm_formal_v1"
    close = read(paired_dir / "complete.json")
    paired = pinned(close["report"])
    inputs = pinned(close["inputs"])
    primary = pinned(inputs["primary_report"])
    final_warm = pinned(inputs["final_warm_report"])
    recovered = pinned(inputs["recovered_control_report"])
    stage_path = original / "registered_stages.json"
    if hashlib.sha256(stage_path.read_bytes()).hexdigest() != primary["registered_stages_sha256"]:
        raise ValueError("Changed source stages")
    stages = read(stage_path)["subjects"]
    common = [e for e in primary["entries"] if e["method"] is None and e["card"]]
    if any(e["unknown_cost"] for e in common):
        raise ValueError("Common GPU cost is unbounded")
    missing = [e["key"] for e in primary["entries"] if e["unknown_cost"]]
    if missing != ["subject_5_cert_cpu"]:
        raise ValueError(f"Recheck the wall-time bound for {missing}")
    common_s = sum(e["known_wall_s"] for e in common)
    arms = {"cold_single": (primary["methods"]["cold_single"],
                            [e for e in primary["entries"] if e["method"] == "cold_single"])}
    for name, data in recovered["recovery_methods"].items():
        events = [e for e in recovered["unique_recovery_jobs"] if e["method"] == data["primary_method"]]
        arms[name] = data["standalone"], events
    arms[paired["method"]] = paired["standalone"], paired["reused_zero_shot_entries"] + paired["unique_entries"]
    unavailable = [v for k, v in primary["methods"].items() if k != "cold_single"] + [final_warm["standalone"]]
    if any(v["assigned_but_not_scored_invalid_teacher"] != len(stages) for v in unavailable):
        raise ValueError("Fixed-endpoint methods differ from the stated chart scope")
    output = {}
    for arm in ARMS:
        result, entries = arms[arm]
        if not result["target_branches_accounted"] or result["assigned"] != len(stages):
            raise ValueError("Cannot plot an open experiment as final")
        if any(e["unknown_cost"] for e in entries):
            raise ValueError("Unbounded method-specific time in the wall-time curve")
        output[arm] = measure(arm, result, entries, stages, common_s, S5_MAX_S)
    return output


def draw(points, output):
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.7, 7.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.13, 1])
    fig.suptitle("真实第一视角留出轨迹 · Wuji Hand 2 / RTX 5090", fontsize=17, weight="bold")
    subjects = {"subject_2": "糖果盒 · 主样本", "subject_5": "茶盒", "subject_6": "DVD 盒"}
    top.suptitle("从源首帧实际执行的控制区间比例", fontsize=12)
    axs = top.subplots(1, 3, sharey=True)
    for ax, (subject, title) in zip(axs, subjects.items()):
        for arm, rows in points.items():
            label, color, marker = ARMS[arm]
            case = [p for p in rows if p["subject"] == subject]
            ax.plot([p["target_steps"]/TARGET_BUDGET for p in case], [p["executed"]/p["requested"] for p in case],
                    label=label, marker=marker, lw=1.3, ms=5, color=color, alpha=.90)
        ax.set(title=title, xticks=[0, .25, .5, .75, 1], ylim=(-.04, 1.04), xlim=(-.04, 1.04))
        ax.set_xticklabels(["0", "1/4", "1/2", "3/4", "全额"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.22)
    top.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
    bottom.suptitle("累计通过的独立轨迹数与占卡时间", fontsize=12)
    axes = bottom.subplots(1, 4, sharex=True, sharey=True)
    maximum = max(p["including_pretraining_gpu_h"][1] for rows in points.values() for p in rows)
    for ax, (arm, rows) in zip(axes, points.items()):
        label, color, marker = ARMS[arm]
        lower = [p["including_pretraining_gpu_h"][0] for p in rows]
        upper = [p["including_pretraining_gpu_h"][1] for p in rows]
        counts = [p["distinct_verified"] for p in rows]
        ax.step(lower, counts, where="post", lw=1, color=color, alpha=.6)
        ax.hlines(counts, lower, upper, color=color, lw=1.2)
        ax.scatter(lower, counts, marker=marker, s=15, color=color)
        ax.plot(upper, counts, "|", color=color, ms=4)
        ax.set(title=label, ylim=(-.2, 3.1), yticks=range(4), xlabel="占卡小时", xlim=(0, maximum*1.045))
        ax.grid(axis="y", alpha=.22)
    fig.supxlabel("每条适配上限 418,560 步。冷启动糖果盒的全额策略无有效动作，末点留空。茶盒的冻结初态已超过穿透门限。\n配对动力学两组复用同一次零步评分。占卡含各自预训练、目标训练与物理评测。横向短线为中断计时范围。", fontsize=9)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def draw_wall_times(points, output):
    fig, axes = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(12.7, 3.5), layout="constrained")
    fig.suptitle("累计通过的留出轨迹数与活跃作业墙钟", fontsize=15, weight="bold")
    maximum = max(p["including_pretraining_active_wall_h"][1] for rows in points.values() for p in rows)
    for ax, (arm, rows) in zip(axes, points.items()):
        label, color, marker = ARMS[arm]
        bounds = [p["including_pretraining_active_wall_h"] for p in rows]
        lower, upper = np.asarray(bounds).T
        counts = [p["distinct_verified"] for p in rows]
        ax.step(lower, counts, where="post", color=color, lw=1)
        ax.hlines(counts, lower, upper, lw=1.2, color=color)
        ax.scatter(lower, counts, marker=marker, s=16, color=color)
        ax.plot(upper, counts, "|", color=color, ms=4)
        ax.set(title=label, xlabel="累计作业小时", xlim=(0, maximum*1.045), ylim=(-.2, 3.1), yticks=range(4))
        ax.grid(axis="y", alpha=.22)
    fig.supxlabel("含该方案的共享预训练、来源准备、物理校准、适配和评分。横线给出计时区间。各独立方案均按三个目标摊销。", fontsize=9)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = collect(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    draw(records, args.out / "heldout_cost_and_coverage.png")
    draw_wall_times(records, args.out / "heldout_active_wall_time.png")
    (args.out / "scored_points.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
