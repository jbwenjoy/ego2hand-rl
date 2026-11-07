"""CPU-only, reproducible seed0 paired-pilot audit, report and scientific plots.

Reads immutable physical rollouts and frozen criteria; never runs physics,
updates a checkpoint, changes a threshold or uses an on-policy contact target.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chord_study.acceptance import projected_shaft_progress, protocol_digest
from chord_study.contacts import support_reward
from chord_study.evaluation import evaluate
from chord_study.geometry import quat_matrix
from chord_study.reference import Reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identical_report(actual, recorded, prefix=""):
    """Verify every recomputed metric, retaining the original evaluator reports."""
    for key, value in actual.items():
        require(key in recorded, f"missing recorded metric {prefix + key}")
        other = recorded[key]
        if isinstance(value, dict):
            identical_report(value, other, prefix + key + ".")
        elif isinstance(value, (float, np.floating)):
            require(np.isclose(value, other, rtol=1e-10, atol=1e-10), f"metric mismatch {prefix + key}")
        else:
            require(value == other, f"metric mismatch {prefix + key}")


def physical_score(run, targets, manifest):
    a, reference, meta = run["arrays"], run["reference"], run["metadata"]
    require(meta["model_sha256"] == manifest["model_sha256"], "contact target model differs")
    require(np.array_equal(a["time_s"], targets["time_s"]), "contact target clocks differ")
    require(np.allclose(a["support_basis"], targets["support_basis"], atol=1e-7, rtol=0), "512 bases differ")
    for key, target_key in (("object_pose_w", "input_object_pose_w"), ("wrist_pose_w", "input_wrist_pose_w")):
        require(np.allclose(reference.arrays[key], targets[target_key], atol=1e-7, rtol=0), f"intent reference differs: {key}")
    require(a["live_support512"].shape == (151, 512), "need full physical 512-direction contacts")
    actual = torch.from_numpy(a["live_support512"]).to(torch.float64)[:, None, None, :]
    score = support_reward(actual, torch.from_numpy(targets["support512"]).to(torch.float64)).numpy()
    phase = targets["phase"]
    weights = np.where(np.isin(phase, [2, 3]), 1., np.where(phase == 4, .5, 0.)) * targets["contact_known"].astype(bool)
    require(not np.any((weights > 0) & targets["known_no_contact"].astype(bool)), "known-no-contact phase would attract")
    require(weights.sum() > 0, "no known contact-intent frames")
    result = {"schema": "supermimic_chord_physical_contact_diagnostic:1",
              "rollout_sha256": digest(run["directory"] / "rollout.npz"),
              "fixed_target_sha256": manifest["target_npz_sha256"], "frames": len(score),
              "weighted_actual_support_match512": float(np.sum(score * weights) / weights.sum()),
              "known_contact_frames": int((weights > 0).sum()),
              "phase_mean_scores": {str(int(i)): float(score[phase == i].mean()) for i in np.unique(phase)},
              "actual_solver_contact_frames": int(a["live_contact_active"].any(-1).sum()),
              "score_framewise": score.tolist(), "phase_weights": weights.tolist(),
              "scope": "Auxiliary solver-active contact geometry versus fixed kinematic intent; not force matching or task success",
              "no_finger_identity_or_extra_contact_failure": True}
    return result


def load_run(root, folder, label, train_folder, expected_iteration):
    directory = root / "outputs" / folder
    reference = Reference.load(directory / "reference.npz")
    arrays = npz(directory / "rollout.npz")
    meta, recorded = read(directory / "rollout.json"), read(directory / "evaluation.json")
    require(meta["checkpoint_iteration"] == expected_iteration, "wrong checkpoint iteration")
    require(recorded["frames"] == recorded["expected_frames"] == 151, "incomplete evaluation artifact")
    require(meta["rollout_censored"] is False, "censored evaluation")
    require(meta["resets_during_rollout"] == meta["object_state_writes_after_initialization"] == 0, "assisted/reset rollout")
    require(digest(directory / "reference.npz") == meta["reference_bundle_sha256"], "reference hash mismatch")
    require(digest(directory / "model.mjb") == meta["model_sha256"], "model hash mismatch")
    require(protocol_digest(reference.meta["task_acceptance"]) == recorded["task_protocol_sha256"], "criterion hash mismatch")
    recomputed = evaluate(reference, arrays, meta)
    identical_report(recomputed, recorded)
    train = root / "outputs" / train_folder
    index = [json.loads(line) for line in (train / "compiled/checkpoint_index.jsonl").read_text().splitlines()]
    completed = next(x for x in index if x["checkpoint_sha256"] == meta["checkpoint_sha256"])
    require(digest(Path(completed["checkpoint"])) == completed["checkpoint_sha256"], "checkpoint changed")
    summary = read(train / "summary.json")
    cost = summary["total_wall_seconds_including_failed_attempts"]
    return {"label": label, "directory": directory, "reference": reference, "arrays": arrays,
            "metadata": meta, "evaluation": recorded, "training_directory": train,
            "checkpoint_completion": completed, "training_summary": summary,
            "recorded_training_process_seconds": cost}


def trajectories(run):
    a, reference = run["arrays"], run["reference"]
    axis = quat_matrix(a["object_pose_w"][:, 3:])[:, :, 0]
    ref_axis = quat_matrix(reference.arrays["object_pose_w"][:, 3:])[:, :, 0]
    normal = reference.meta["task_acceptance"]["progress_frame"]["normal_w"]
    progress, _ = projected_shaft_progress(axis, normal)
    target, _ = projected_shaft_progress(ref_axis, normal)
    return {"time_s": a["time_s"].tolist(),
            "position_error_mm": (1000 * np.linalg.norm(a["object_pose_w"][:, :3] - reference.arrays["object_pose_w"][:, :3], axis=-1)).tolist(),
            "shaft_axis_error_deg": np.rad2deg(np.arccos(np.clip((axis * ref_axis).sum(-1), -1, 1))).tolist(),
            "projected_shaft_progress_deg": progress.tolist(), "reference_progress_deg": target.tolist(),
            "progress_abs_error_deg": np.abs(progress - target).tolist(),
            "penetration_mm": (1000 * a["penetration_m"]).tolist()}


def plot(runs, output, protocol):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10), sharex=True, constrained_layout=True)
    colors = ["#1764ab", "#d55e00", "#8854a4"]
    styles = ["-", "-", "--"]
    for run, color, style in zip(runs, colors, styles):
        v, score = run["trajectories"], run["contact_score"]
        time = v["time_s"]
        for ax, key in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]],
                           ["position_error_mm", "shaft_axis_error_deg", "projected_shaft_progress_deg", "progress_abs_error_deg", "penetration_mm"]):
            ax.plot(time, v[key], color=color, linestyle=style, linewidth=1.7, label=run["label"])
        values = np.where(np.asarray(score["phase_weights"]) > 0, score["score_framewise"], np.nan)
        axes[2, 1].plot(time, values, color=color, linestyle=style, linewidth=1.5, label=run["label"])
    axes[0, 0].axhline(protocol["position_max_m"] * 1000, color="0.3", linestyle=":", label="Frozen maximum")
    axes[0, 1].axhline(protocol["shaft_angle_max_deg"], color="0.3", linestyle=":")
    axes[1, 0].plot(runs[0]["trajectories"]["time_s"], runs[0]["trajectories"]["reference_progress_deg"], color="0.15", linestyle=":", linewidth=2, label="Frozen reference")
    axes[1, 1].axhline(protocol["progress_abs_error_max_deg"], color="0.3", linestyle=":", label="All-frame cap 30 deg")
    axes[1, 1].scatter([3.], [protocol["progress_endpoint_error_max_deg"]], color="0.15", marker="D", s=28, label="Endpoint cap 20 deg", zorder=6)
    axes[2, 0].axhline(protocol["penetration_max_m"] * 1000, color="0.3", linestyle=":")
    for ax, title, ylabel in zip(axes.flat,
            ["Object position tracking", "Directed shaft axis tracking", "Observable shaft progress (excludes axial roll)", "Progress error", "Logged penetration (all categories)", "Auxiliary actual contact-geometry score"],
            ["Position error (mm)", "Shaft error (deg)", "Projected progress (deg)", "Absolute error (deg)", "Penetration (mm)", "Frozen-intent support match, 512 directions"]):
        ax.set(title=title, ylabel=ylabel, xlim=(0, 3)); ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[2, 1].set_ylim(0, 1.03)
    for ax in axes[-1]:
        ax.set_xlabel("Control time (s), 151 samples; no smoothing")
    progress_min = min(min(r["trajectories"]["projected_shaft_progress_deg"]) for r in runs)
    axes[1, 0].set_ylim(bottom=min(0., progress_min))
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    axes[1, 1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Single-seed physical pilot: A1L passes; A2 misses the frozen shaft-axis limit\n"
                 "Primary: 2000 iterations each. Secondary: A2 checkpoint 1700 was available before 1200 s; A1L already finished at 923 s.", fontsize=12)
    fig.savefig(output / "tracking.png", dpi=180)
    fig.savefig(output / "tracking.svg")
    plt.close(fig)


def metric_row(row):
    e = row["evaluation"]
    return (f"| {row['label']} | {row['checkpoint_iteration']} | {'PASS' if e['task_success'] else 'FAIL'} | "
            f"{1000*e['position_error_m']['mean']:.2f} / {1000*e['position_error_m']['max']:.2f} | "
            f"{e['orientation_error_deg']['max']:.3f} | {e['rotation_progress_error_deg']['endpoint']:.3f} | "
            f"{row['contact_score']['weighted_actual_support_match512']:.6f} |")


def report(data, output):
    runs = data["runs"]; a1, a2, secondary = runs
    head = "| 方法 | iter | task | 位置均值 / 最大 mm | shaft 最大 ° | progress 终点误差 ° | support512 ↑ |\n|---|---:|---|---:|---:|---:|---:|"
    lines = ["# Seed0 A1L / A2 配对 pilot 对比", "",
             "**本次单 seed 结果不支持 A2 改善冻结物理任务验收：A1L 通过；A2 在等更新预算和等时可用 checkpoint 两种口径下均失败。**", "",
             "## 等采样、等更新主预算", "", head, metric_row(a1), metric_row(a2), "",
             "两臂均为 2000 iter、648192 个真实 transition、WM/failure-head 各 4000、actor 1800、critic 7200 次更新。"
             f"记录的进程总耗时分别为 **{a1['training_process_seconds']:.2f}s** 和 **{a2['training_process_seconds']:.2f}s**（A2 为 A1L 的 {data['cost_comparison']['A2_to_A1L_process_time_ratio']:.3f} 倍）；这项比较不是等计算成本。", "",
             f"A2 的位置最大误差较低，但 shaft 最大误差 {a2['evaluation']['orientation_error_deg']['max']:.3f}° 超过冻结 20°，"
             f"progress 终点误差 {a2['evaluation']['rotation_progress_error_deg']['endpoint']:.3f}° 超过冻结 20°。"
             "三次 rollout 的无掉落、位置、穿透、执行目标与实际关节限位、负载及日志合同均通过；主预算 A2 的失败项如实保留。", "",
             "## 1200 秒内可获得 checkpoint 的次预算", "", head, metric_row(a1), metric_row(secondary), "",
             "该次预算来自各自 **2000 iter 上限实验** 中截止前已完整保存的 checkpoint，未延长 A1L 训练来耗满 1200 秒。"
             "A1L 在约 923 秒已达 2000 iter，沿用同一 final；A2 选择 model_1700。选择仅使用完成时间和完整性，不使用评估质量。", "",
             f"A2 model_1700 在 OS 创建后 **{data['secondary_selection']['selected']['os_creation_elapsed_seconds']:.6f}s** 完成，"
             "具备 552192 transitions、WM/failure-head 各 3400、actor 1500、critic 6000 次更新；1800 checkpoint 已超时。"
             f"该模型 progress 终点误差 {secondary['evaluation']['rotation_progress_error_deg']['endpoint']:.3f}° 通过，"
             f"但 shaft 最大误差 {secondary['evaluation']['orientation_error_deg']['max']:.3f}° 仍超过 20°，因此 task 仍失败。", "",
             "## 接触诊断与适用范围", "",
             f"主预算 A1L / A2 的辅助 support512 得分分别为 {a1['contact_score']['weighted_actual_support_match512']:.7f} / "
             f"{a2['contact_score']['weighted_actual_support_match512']:.7f}，差值 {data['primary_difference_A2_minus_A1L']['support512']:+.7f}。"
             "分数使用真实 Warp solver-active 手物接触几何、物体 COM、同一固定 512 方向 basis，与独立冻结的运动学接触意图比较。"
             "这不是接触力真值匹配，也不是新的任务成功条件；分数略升同时出现任务失败，不构成物理接触收益证据。", "",
             "A2 是 imagination-only 的 soft geometric prior / anticipation shaping，目标独立冻结。"
             "训练采用 64 个固定 stride8 方向、每碰撞 geom 64 点共 1344 点；本报告接触评估采用完整 512 方向。"
             "先验实际执行为 eager，原生 actor/WM 子网编译；此前 standalone Inductor compiled-precision 门槛失败的记录仍保留，"
             "不能把训练完成改写为该精度门槛通过。旧局部 contact predictor / dynamics-gradient 资格失败同样保留。", "",
             "这是一条 upstream_kinematic clip007[200:351] 的固定腕任务重设计，原始视频父 ID 未知，"
             "不是新 ego 视频、独立新动作或跨物体泛化。仅 seed0，没有置信区间或总体有效性结论。"
             "A2 同时增加 support 吸引和 penetration prior；即使未来有收益，也需 penalty-only 消融才能区分来源。", "",
             "## 输入和审计", "",
             "三次评估 reference NPZ/JSON、model、任务协议、评估器 SHA 完全相同；151 帧规范时间、实际初态 q/qd、物体姿态与线/角速度、腕、初始执行目标均逐值相同。"
             "本脚本从原 NPZ 重新计算验收指标并与已保存 evaluation.json 核对；完整性和哈希核查保存在 comparison.json。", "",
             "成本仅统计对应完整正式训练进程，不含此前中断、工程 smoke、失败编译尝试及独立评估。"
             f"记录的 GPU peak allocated：A1L {a1['training_peak_allocated_gib']:.3f} GiB，A2 {a2['training_peak_allocated_gib']:.3f} GiB。"
             "该值不是峰值 reserved 或全卡显存占用。", "",
             "![原始跟踪曲线](tracking.png)", "",
             "## 复现", "", "```powershell", f"& '{data['python_executable']}' scripts/summarize_paired_pilot.py", "```", "",
             "产物：comparison.json、三个独立 contact_score_*.json、tracking.png 与 tracking.svg。图中不做平滑，"
             "仅接触诊断对 unknown/no-contact 阶段不画吸引分数；所有原始逐帧分数和 phase 权重保留于 JSON。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); output = (args.output or root / "outputs/paired_pilot_seed0_comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    specifications = [
        ("a1l_paired_v2_frozen_eval_seed0", "A1L 2000", "a1l_paired_v2_clip007_2000iter_seed0_restart1", 2000),
        ("a2_paired_v2_frozen_eval_seed0", "A2 2000", "a2_paired_v2_clip007_2000iter_seed0", 2000),
        ("a2_secondary1200_frozen_eval_seed0", "A2 1700 (<1200s)", "a2_paired_v2_clip007_2000iter_seed0", 1700)]
    runs = [load_run(root, *spec) for spec in specifications]
    base = runs[0]; initial = ["joint_pos_rad", "joint_vel_rad_s", "object_pose_w", "object_lin_vel_w_m_s",
                              "object_ang_vel_w_rad_s", "wrist_pose_w", "joint_target_rad"]
    checks = {}
    for run in runs:
        name = run["label"]; arrays = run["arrays"]
        checks[name] = {"initial_state_exact": {key: np.array_equal(arrays[key][0], base["arrays"][key][0]) for key in initial},
                        "canonical_times_exact": np.array_equal(arrays["time_s"], base["arrays"]["time_s"]),
                        "raw_physics_times_exact": np.array_equal(arrays["physics_time_s"], base["arrays"]["physics_time_s"]),
                        "reference_npz_sha_equal": digest(run["directory"] / "reference.npz") == digest(base["directory"] / "reference.npz"),
                        "reference_json_sha_equal": digest(run["directory"] / "reference.json") == digest(base["directory"] / "reference.json"),
                        "model_sha_equal": run["metadata"]["model_sha256"] == base["metadata"]["model_sha256"],
                        "protocol_equal": run["reference"].meta["task_acceptance"] == base["reference"].meta["task_acceptance"],
                        "evaluator_sha_equal": run["metadata"]["script_sha256"] == base["metadata"]["script_sha256"],
                        "recomputed_evaluation_matches_recorded": True}
        require(all(checks[name]["initial_state_exact"].values()) and all(v for k, v in checks[name].items() if k != "initial_state_exact"), f"unmatched physical inputs: {name}")
    targets_path = root / "outputs/reference_contact_targets_v1/contact_targets.npz"
    targets, manifest = npz(targets_path), read(targets_path.parent / "manifest.json")
    require(digest(targets_path) == manifest["target_npz_sha256"], "target hash mismatch")
    records = []
    for index, run in enumerate(runs):
        run["contact_score"] = physical_score(run, targets, manifest)
        run["trajectories"] = trajectories(run)
        train = run["training_summary"]["attempts"][0]
        record = {"label": run["label"], "evaluation_directory": str(run["directory"]),
                  "training_directory": str(run["training_directory"]),
                  "checkpoint_iteration": run["metadata"]["checkpoint_iteration"],
                  "checkpoint_sha256": run["metadata"]["checkpoint_sha256"],
                  "checkpoint_completion": run["checkpoint_completion"],
                  "evaluation": run["evaluation"], "contact_score": run["contact_score"],
                  "trajectories": run["trajectories"],
                  "training_process_seconds": run["recorded_training_process_seconds"],
                  "training_peak_allocated_gib": train["torch_peak_allocated_gib"],
                  "artifact_sha256": {file: digest(run["directory"] / file) for file in ["rollout.npz", "rollout.json", "evaluation.json", "reference.npz", "reference.json", "model.mjb"]}}
        records.append(record)
        write(output / f"contact_score_{['a1l2000', 'a22000', 'a21700'][index]}.json", run["contact_score"])
    selection = read(root / "outputs/a2_paired_v2_clip007_2000iter_seed0/health_final_root/secondary_budget_selection.json")
    require(selection["selected"]["checkpoint_sha256"] == runs[2]["metadata"]["checkpoint_sha256"], "secondary selection differs")
    require(runs[0]["checkpoint_completion"]["process_elapsed_seconds_at_completion"] < 1200, "A1L final not available in wall budget")
    primary_steps = records[0]["checkpoint_completion"]["actual_optimizer_steps"]
    require(primary_steps == records[1]["checkpoint_completion"]["actual_optimizer_steps"], "primary optimizer budgets differ")
    require(records[0]["checkpoint_completion"]["real_simulator_transitions"] == records[1]["checkpoint_completion"]["real_simulator_transitions"] == 648192, "primary real sample budgets differ")
    first_train, second_train = runs[0]["training_directory"], runs[1]["training_directory"]
    first_inventory, second_inventory = read(first_train / "source/inventory.json"), read(second_train / "source/inventory.json")
    training_checks = {
        "training_hyperparameters_exact": read(first_train / "input_config.json")["training"] == read(second_train / "input_config.json")["training"],
        "paired_protocol_sha_equal": digest(first_train / "source/paired_protocol.json") == digest(second_train / "source/paired_protocol.json"),
        "frozen_training_script_sha_equal": first_inventory["script_sha256"] == second_inventory["script_sha256"],
        "frozen_dex_source_hashes_exact": first_inventory["files"] == second_inventory["files"],
        "training_seed_zero": all(r["metadata"]["seed"] == 0 for r in runs)}
    require(all(training_checks.values()), "primary training/control/protocol inputs differ")
    data = {"schema": "supermimic_chord_seed0_paired_comparison:1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable, "gpu_executed_by_summarizer": False,
            "script_sha256": digest(__file__), "matplotlib_version": matplotlib.__version__,
            "metric_source_sha256": {str(p.relative_to(root)): digest(p) for p in
                                      [root / "src/chord_study/evaluation.py", root / "src/chord_study/acceptance.py", root / "src/chord_study/contacts.py"]},
            "claim_scope": "single-seed single-clip physical pilot; auxiliary geometry intent score is not a force or dynamics qualification",
            "frozen_reference_sha256": digest(base["directory"] / "reference.npz"),
            "frozen_model_sha256": base["metadata"]["model_sha256"],
            "frozen_task_protocol_sha256": base["evaluation"]["task_protocol_sha256"],
            "fixed_contact_target_sha256": manifest["target_npz_sha256"],
            "input_and_metric_checks": checks,
            "primary_training_checks": training_checks,
            "primary_comparison": ["A1L 2000", "A2 2000"],
            "secondary_comparison": ["A1L 2000", "A2 1700 (<1200s)"],
            "secondary_interpretation": "latest complete checkpoint available before1200s within2000-iteration-capped runs; A1L finished early, neither arm is claimed to consume exactly1200s",
            "secondary_selection": selection, "runs": records,
            "primary_difference_A2_minus_A1L": {"support512": records[1]["contact_score"]["weighted_actual_support_match512"] - records[0]["contact_score"]["weighted_actual_support_match512"],
                                               "shaft_max_deg": records[1]["evaluation"]["orientation_error_deg"]["max"] - records[0]["evaluation"]["orientation_error_deg"]["max"]},
            "cost_comparison": {"A2_to_A1L_process_time_ratio": records[1]["training_process_seconds"] / records[0]["training_process_seconds"],
                                "primary_equal_updates_not_equal_compute": True,
                                "excluded_costs": "earlier interrupted runs, engineering probes/smokes, failed standalone compilation, and independent evaluation"},
            "retained_negative_evidence": [str(root / "outputs/a2_cuda_pointmax_final_20261102/result.json"),
                                           str(root / "outputs/contact_geometry_cpu_exterior_v3/report.json")],
            "plot_contact_score_mask": "draw known reference-contact phases only; all framewise scores and weights remain in JSON"}
    write(output / "comparison.json", data)
    report(data, output)
    plot(runs, output, base["reference"].meta["task_acceptance"])
    print(json.dumps({"output": str(output), "task_success": {r["label"]: r["evaluation"]["task_success"] for r in records},
                      "support512": {r["label"]: r["contact_score"]["weighted_actual_support_match512"] for r in records}}, indent=2))


if __name__ == "__main__":
    main()
