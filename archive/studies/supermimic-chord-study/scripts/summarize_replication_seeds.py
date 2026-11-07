"""CPU-only descriptive paired-seed report; pending artifacts never become failures.

The statistical units are the three training seeds, under one common evaluation
seed/initial condition. Seed0 pilot data are excluded. No training or physics.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from summarize_paired_pilot import (ROOT, Reference, digest, evaluate, identical_report,
                                    npz, physical_score, read, require, trajectories, write)

PRIMARY = "primary_equal_updates_2000"
SECONDARY = "pre1200_availability_with_2000_cap"
GATES = ["complete_clip", "unassisted", "fixed_wrist_ok", "task_tracking_ok", "task_penetration_ok",
         "task_joint_limits_ok", "joint_state_limits_ok", "joint_target_limits_ok", "progress_ok", "drop_ok", "load_ok", "target_rate_ok", "task_logging_contract_ok"]
INITIAL = ["joint_pos_rad", "joint_vel_rad_s", "object_pose_w", "object_lin_vel_w_m_s",
           "object_ang_vel_w_rad_s", "wrist_pose_w", "joint_target_rad"]
REQUIRED = ["rollout.npz", "rollout.json", "evaluation.json", "reference.npz", "reference.json", "model.mjb", "condition.json"]


def summary(values):
    x = np.asarray(values, dtype=float)
    require(x.shape == (3,) and np.isfinite(x).all(), "descriptive unit must be exactly three complete training seeds")
    return {"n_training_seeds": 3, "values_seed123": x.tolist(), "mean": float(x.mean()),
            "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def readiness(plan, manager):
    missing = []
    if manager is None or manager.get("status") != "completed_all_evaluations":
        missing.append("evaluation manager has not completed all evaluations")
    for job in plan["jobs"]:
        for file in REQUIRED:
            path = Path(job["output"]) / file
            if not path.is_file():
                missing.append(str(path))
        for path in [Path(job["checkpoint"]), Path(job["training_run"]) / "input_config.json",
                     Path(job["cost_output"]) / "cost.json"]:
            if not path.is_file(): missing.append(str(path))
    return missing


def load_job(job, plan, targets, manifest, selection, base):
    directory, train = Path(job["output"]), Path(job["training_run"])
    recorded, meta, condition = read(directory / "evaluation.json"), read(directory / "rollout.json"), read(directory / "condition.json")
    config = read(train / "input_config.json")
    raw = torch.load(Path(job["checkpoint"]), map_location="cpu", weights_only=False)
    checkpoint_config = raw["config"]
    checks = {
        "training_seed_matches_plan_input_raw_checkpoint_and_metadata":
            job["training_seed"] == config["seed"] == checkpoint_config["seed"] == meta["checkpoint_training_config"]["seed"] == condition["training_seed"],
        "evaluation_seed_zero_separate_from_training_seed": meta["seed"] == plan["evaluation_seed"] == condition["evaluation_seed"] == 0,
        "arm_from_frozen_input_not_legacy_actor_label": config["method"] == job["method"] == condition["experimental_method"],
        "condition_budget_and_hash": condition["budget"] == job["budget"] and condition["checkpoint_sha256"] == job["checkpoint_sha256"],
        "checkpoint_hash_binding": digest(Path(job["checkpoint"])) == job["checkpoint_sha256"] == meta["checkpoint_sha256"],
        "checkpoint_iteration_binding": raw["iteration"] == meta["checkpoint_iteration"],
        "frozen_reference_binding": digest(directory / "reference.npz") == plan["reference_sha256"] == meta["reference_bundle_sha256"],
        "model_binding": digest(directory / "model.mjb") == meta["model_sha256"] == manifest["model_sha256"],
        "evaluator_binding": meta["script_sha256"] == plan["evaluator_sha256"],
        "full3s_transport": recorded["frames"] == recorded["expected_frames"] == 151 and meta["rollout_censored"] is False,
        "zero_reset_assist_noise_randomization": meta["resets_during_rollout"] == meta["object_state_writes_after_initialization"] == 0 and
            all(meta[k] is False for k in ["domain_randomization", "observation_corruption", "external_disturbances"])}
    require(all(checks.values()), f"provenance mismatch: {job}: {checks}")
    require(isinstance(recorded.get("task_success"), bool), "PENDING: task_success unavailable, cannot count as false")
    require(all(isinstance(recorded.get(k), bool) for k in GATES), "PENDING: required task gate unavailable")
    reference, arrays = Reference.load(directory / "reference.npz"), npz(directory / "rollout.npz")
    identical_report(evaluate(reference, arrays, meta), recorded)
    checks["independent_metric_recomputation_matches"] = True
    if base:
        checks.update({"same_reference_json": digest(directory / "reference.json") == digest(base["directory"] / "reference.json"),
                       "same_model": meta["model_sha256"] == base["metadata"]["model_sha256"],
                       "same_frozen_protocol": reference.meta["task_acceptance"] == base["reference"].meta["task_acceptance"],
                       "same_canonical_time": np.array_equal(arrays["time_s"], base["arrays"]["time_s"]),
                       "same_raw_physics_time": np.array_equal(arrays["physics_time_s"], base["arrays"]["physics_time_s"])})
        checks["same_initial_state_exact"] = {k: np.array_equal(arrays[k][0], base["arrays"][k][0]) for k in INITIAL}
        require(all(checks["same_initial_state_exact"].values()), "initial-state mismatch")
        require(all(v for k, v in checks.items() if k != "same_initial_state_exact"), "common physics/reference mismatch")
    index = [json.loads(x) for x in (train / "compiled/checkpoint_index.jsonl").read_text().splitlines()]
    completed = next(x for x in index if x["checkpoint_sha256"] == job["checkpoint_sha256"])
    selected = next(x for x in selection["selected"] if x["method"] == job["method"] and x["seed"] == job["training_seed"])
    if job["budget"] == SECONDARY:
        require(selected["sha256"] == job["checkpoint_sha256"], "secondary checkpoint differs from audited OS selection")
    elif job["method"] == "A1L":
        require(selected["sha256"] == job["checkpoint_sha256"], "A1L secondary must alias the exact primary final")
    require(all(raw["modules"][name] for name in raw["modules"]), "empty checkpoint module")
    run = {"directory": directory, "reference": reference, "arrays": arrays, "metadata": meta}
    score = physical_score(run, targets, manifest)
    curve = trajectories(run)
    a = arrays
    violation = np.maximum(np.maximum(a["joint_lower_rad"] - a["joint_pos_rad"], a["joint_pos_rad"] - a["joint_upper_rad"]), 0).max(-1)
    curve["joint_state_violation_mrad"] = (1000 * violation).tolist()
    cost, train_summary = read(Path(job["cost_output"]) / "cost.json"), read(train / "summary.json")
    checks["cost_job_completed"] = cost["exit_code"] == 0 and cost["timed_out"] is False
    require(checks["cost_job_completed"], "engineering evaluation did not complete")
    row = {"id": f"{job['method'].lower()}_seed{job['training_seed']}_{'primary' if job['budget']==PRIMARY else 'secondary1200'}",
           "method": job["method"], "training_seed": job["training_seed"], "evaluation_seed": 0,
           "budget": job["budget"], "iteration": raw["iteration"], "checkpoint": job["checkpoint"],
           "checkpoint_sha256": job["checkpoint_sha256"], "evaluation_directory": str(directory),
           "task_success": recorded["task_success"], "gates": {k: recorded[k] for k in GATES},
           "failed_task_gates": [k for k in GATES if not recorded[k]],
           "metrics": {"position_mean_mm": 1000*recorded["position_error_m"]["mean"],
                       "position_max_mm": 1000*recorded["position_error_m"]["max"],
                       "shaft_max_deg": recorded["orientation_error_deg"]["max"],
                       "shaft_mean_deg": recorded["orientation_error_deg"]["mean"],
                       "progress_endpoint_error_deg": recorded["rotation_progress_error_deg"]["endpoint"],
                       "joint_state_violation_mrad": 1000*recorded["joint_state_violation_max_rad"],
                       "joint_target_violation_mrad": 1000*recorded["joint_target_violation_max_rad"],
                       "penetration_max_mm": 1000*float(a["penetration_m"].max()),
                       "load_ratio_max": recorded["load_ratio_max"], "load_charge_max": recorded["load_charge_max"],
                       "support512": score["weighted_actual_support_match512"]},
           "contact_diagnostic": score, "trajectories": curve, "checks": checks,
           "source_evaluation": recorded, "selected_checkpoint_counts": completed,
           "secondary_A1L_alias_of_this_final": job["method"] == "A1L",
           "training_process_recorded_seconds": train_summary["total_wall_seconds_including_failed_attempts"],
           "training_peak_allocated_gib": train_summary["attempts"][0]["torch_peak_allocated_gib"],
           "evaluation_cost": cost,
           "input_sha256": {file: digest(directory/file) for file in REQUIRED},
           "raw_checkpoint_config": checkpoint_config}
    del raw
    return row, run


def group_statistics(rows):
    require(len(rows) == 3 and sorted(x["training_seed"] for x in rows) == [1, 2, 3], "three paired seeds required")
    rows = sorted(rows, key=lambda x:x["training_seed"])
    return {"n_training_seeds":3, "success_count":sum(x["task_success"] for x in rows),
            "observed_task_fraction":sum(x["task_success"] for x in rows)/3,
            "gate_pass_counts":{k:sum(x["gates"][k] for x in rows) for k in GATES},
            "descriptive_metrics":{k:summary([r["metrics"][k] for r in rows]) for k in rows[0]["metrics"]}}


def paired(a1, a2):
    pairs = [{"training_seed":seed, "A1L_id":a1[seed]["id"], "A2_id":a2[seed]["id"],
              "task_difference_A2_minus_A1L":int(a2[seed]["task_success"])-int(a1[seed]["task_success"]),
              "metric_differences_A2_minus_A1L":{k:a2[seed]["metrics"][k]-a1[seed]["metrics"][k] for k in a1[seed]["metrics"]}} for seed in [1,2,3]]
    return {"pairs":pairs,"difference_convention":"A2 minus A1L; lower is better for errors/violations, higher for support score",
            "descriptive_differences":{k:summary([p["metric_differences_A2_minus_A1L"][k] for p in pairs]) for k in pairs[0]["metric_differences_A2_minus_A1L"]}}


def figures(primary_a1, primary_a2, time_a2, output, protocol):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none"})
    fig, axes = plt.subplots(4,3,figsize=(13.5,11),sharex=True,sharey="row",constrained_layout=True)
    keys = ["position_error_mm","shaft_axis_error_deg","progress_abs_error_deg","joint_state_violation_mrad"]
    labels = ["Position error (mm)","Shaft axis error (deg)","Progress error (deg)","Joint state limit excess (mrad)"]
    thresholds = [protocol["position_max_m"]*1000,protocol["shaft_angle_max_deg"],protocol["progress_abs_error_max_deg"],protocol["joint_limit_margin_rad"]*1000]
    for col, seed in enumerate([1,2,3]):
        for row, key in enumerate(keys):
            ax = axes[row,col]
            for run,color,style,label in [(primary_a1[seed],"#1764ab","-","A1L 2000"),
                                           (primary_a2[seed],"#d55e00","-","A2 2000"),
                                           (time_a2[seed],"#8854a4","--","A2 pre1200 checkpoint")]:
                ax.plot(run["trajectories"]["time_s"],run["trajectories"][key],color=color,linestyle=style,label=label,linewidth=1.5)
            ax.axhline(thresholds[row],color="0.25",linestyle=":",linewidth=1)
            if row==2: ax.scatter([3],[protocol["progress_endpoint_error_max_deg"]],color="0.2",marker="D",s=17,zorder=6)
            ax.grid(alpha=.18); ax.set_xlim(0,3); ax.set_ylim(bottom=0)
            if col==0:ax.set_ylabel(labels[row])
        axes[0,col].set_title(f"Training seed {seed}; evaluation seed 0")
        axes[-1,col].set_xlabel("Control time (s); 151 recorded samples")
    axes[0,0].legend(fontsize=8,loc="upper left")
    # Explicit shared limits after every seed has been plotted: setting one
    # shared axis early can disable autoscaling and hide a later seed's spike.
    for row,key in enumerate(keys):
        peak=max(thresholds[row], max(max(r["trajectories"][key])
                 for arm in [primary_a1,primary_a2,time_a2] for r in arm.values()))
        axes[row,0].set_ylim(0,peak*1.08)
    fig.suptitle("Paired training seeds 1-3: all physical rollouts retained\nDotted lines are frozen gates; progress endpoint cap is the diamond. No smoothing or frame-based inference.",fontsize=12)
    for extension in ["png","svg"]:fig.savefig(output/f"tracking.{extension}",dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    specs = [("shaft_max_deg","Maximum shaft error (deg)",20),
             ("progress_endpoint_error_deg","Endpoint progress error (deg)",20),
             ("joint_state_violation_mrad","Maximum joint-state excess (mrad)",1),
             ("support512","Auxiliary support match (512)",None)]
    for row, other in enumerate([primary_a2,time_a2]):
        for col,(key,title,threshold) in enumerate(specs):
            ax=axes[row,col]
            for seed,color,marker in zip([1,2,3],["#1764ab","#d55e00","#15845c"],["o","s","^"]):
                ax.plot([0,1],[primary_a1[seed]["metrics"][key],other[seed]["metrics"][key]],color=color,marker=marker,label=f"Seed {seed}",alpha=.9)
            if threshold is not None:ax.axhline(threshold,color="0.25",linestyle=":")
            ax.set_xticks([0,1],["A1L final", "A2 final" if row==0 else "A2 pre1200"])
            ax.set_xlim(-.18,1.18);ax.grid(alpha=.18);ax.set_title(title);ax.set_ylim(bottom=0)
            if key=="support512":
                values=[arm[s]["metrics"][key] for arm in [primary_a1,other] for s in [1,2,3]]
                ax.set_ylim(max(0,min(values)-.015),min(1,max(values)+.015))
                ax.set_title("Auxiliary support score (512)\nZoomed y-axis")
            if col==0:ax.set_ylabel("Equal updates" if row==0 else "Pre1200 availability, 2000 cap")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Paired descriptive effects: each line is one training seed (n=3 pairs)\nErrors/violations: lower is better. Support score: higher is better; it is not task success.",fontsize=12)
    for extension in ["png","svg"]:fig.savefig(output/f"paired_effects.{extension}",dpi=180)
    plt.close(fig)


def markdown(data, output):
    rows=data["runs"]; groups=data["groups"]; get=lambda method,budget,seed:next(r for r in rows if r["method"]==method and r["budget"]==budget and r["training_seed"]==seed)
    primary_a1={s:get("A1L",PRIMARY,s) for s in [1,2,3]}; primary_a2={s:get("A2",PRIMARY,s) for s in [1,2,3]};time_a2={s:get("A2",SECONDARY,s) for s in [1,2,3]}
    a,b,c=groups["primary_A1L"]["success_count"],groups["primary_A2"]["success_count"],groups["secondary_A2"]["success_count"]
    axis_improved=sum(primary_a2[s]["metrics"]["shaft_max_deg"]<primary_a1[s]["metrics"]["shaft_max_deg"] for s in [1,2,3])
    q_failed=sum(not primary_a2[s]["gates"]["joint_state_limits_ok"] for s in [1,2,3])
    bounded=sum(primary_a2[s]["gates"]["joint_target_limits_ok"] for s in [1,2,3])
    text=["# Seeds 1–3 配对复制结果", "", f"**等更新主预算：A1L {a}/3，A2 {b}/3；1200 秒内可获得 checkpoint：A1L {a}/3，A2 {c}/3。**", "",
          "这里的 N=3 是训练随机种子1/2/3；所有评估固定 seed0、同一物理初态。seed0 pilot 不计入复制统计，151帧只是每条轨迹的采样点。仅作配对描述性汇总，不作总体显著性、p值或置信区间推断。", "",
          "## 逐 seed 全部结果", "", "|预算|方法|训练 seed|评估 seed|iter|task|shaft max °|progress endpoint °|state超限 mrad|target超限 mrad|support512|失败门槛|",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        m=row["metrics"];text.append(f"|{'主' if row['budget']==PRIMARY else '次'}|{row['method']}|{row['training_seed']}|0|{row['iteration']}|{'PASS' if row['task_success'] else 'FAIL'}|{m['shaft_max_deg']:.3f}|{m['progress_endpoint_error_deg']:.3f}|{m['joint_state_violation_mrad']:.4f}|{m['joint_target_violation_mrad']:.5f}|{m['support512']:.6f}|{', '.join(row['failed_task_gates']) or '—'}|")
    text += ["", f"主预算有 **{axis_improved}/3 对 seed 的 shaft 最大误差降低**，这是描述性跟踪信号；整体任务通过数为上述 {a}/3 与 {b}/3，不能把局部指标改善等同整体任务收益。"
             f"主预算A2有 {q_failed}/3 条实际 joint state 超过冻结1mrad允许量，而执行 target 限幅 {bounded}/3 通过；保持原状态和门槛，没有clamp状态或放宽标准。"
             "平均轴误差、位置误差与最大轴误差分开列示，避免把最大值改善泛称所有跟踪指标改善。", "",
             "次预算中的 A1L 是上述同一 final 的 alias，没有重复评估或新增独立样本。A1L 三次均在1200秒前完成2000迭代；A2三次按 OS CreationDate 规则选1700，未按评估质量选 checkpoint。不能称两臂均耗满1200秒。", "",
             "主预算两臂每seed均2000 iter、648192真实 transitions、WM/failure-head各4000、actor1800、critic7200更新；次预算A2为552192 transitions、WM/failure-head各3400、actor1500、critic6000。", "",
             "## 配对差值与描述性范围", "", "差值均为 A2−A1L；误差/越限越低越好，辅助接触分数越高越好。", "",
             "|预算|指标|seed1差值|seed2差值|seed3差值|均值|中位数|最小 / 最大|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for budget in ["primary","secondary"]:
        for key in ["position_mean_mm","position_max_mm","shaft_mean_deg","shaft_max_deg","progress_endpoint_error_deg","joint_state_violation_mrad","support512"]:
            v=data["paired_effects"][budget]["descriptive_differences"][key];x=v["values_seed123"]
            text.append(f"|{budget}|{key}|{x[0]:+.5f}|{x[1]:+.5f}|{x[2]:+.5f}|{v['mean']:+.5f}|{v['median']:+.5f}|{v['min']:+.5f} / {v['max']:+.5f}|")
    text += ["", "所有任务门槛、每组连续指标的mean/median/min/max及逐seed原值均保存在 comparison.json，CSV列出9条唯一物理rollout。没有把缺失项转为0或失败。任务总门槛须全部满足，某个跟踪指标或辅助分数改善不抵消实际关节状态越限。", "",
             "## 成本口径", "", "|方法|seed|训练记录秒|queue GPU slot秒（含CPU health）|peak allocated GiB|", "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["budget"]==PRIMARY:
            cost=data["training_costs"][f"{row['method']}_seed{row['training_seed']}"]
            text.append(f"|{row['method']}|{row['training_seed']}|{cost['training_recorded_seconds']:.3f}|{cost['queue_reserved_including_CPU_health_seconds']:.3f}|{cost['peak_allocated_gib']:.3f}|")
    text += ["", "等更新不等于等compute。GPU slot是单卡预留墙钟，不是利用率或CUDA计算时间；训练记录与queue含健康检查口径分别列示。9次独立评估成本单列于JSON，secondary alias不重复计费；此前工程smoke、失败编译和seed0 pilot费用不混入此复制实验成本，也不声称为研究总费用。缓存按冻结queue连续复用，warm history随顺序变化。", "",
             "## 证据与适用边界", "", "九条rollout均核对计划、原训练输入、raw checkpoint config、condition及metadata；training_seed与evaluation_seed分开保存。reference/model/protocol、规范time、raw physics clock和初态q/qd/object pose/twists/wrist/初始target逐值核对；验收从原NPZ重算，512分数由真实solver-active手物接触几何对独立固定意图计算。", "",
             "A2是soft geometric prior/anticipation，而非合格的接触力预测器。训练64方向与物理评估512方向分开；局部代理/动力学梯度资格失败和standalone compiled-precision失败记录均保留，实际prior走eager，原生子网编译。该辅助score既非受力真值也非任务成功，不能据其变化作物理接触收益结论。", "",
             "A2同时加入support吸引与penetration prior，没有support-only或penalty-only消融，不能把任何变化只归因support。本实验只有clip007[200:351]固定腕重设计、同一物体、同一初态；无真实无本体ego输入，原视频父ID未知。三训练seed不构成跨视频、跨物体或初态泛化。门槛是50Hz模拟任务标准，不是连续时间/硬件安全认证。", "",
             f"保留的独立失败证据：[compiled precision]({(ROOT / 'outputs/a2_cuda_pointmax_final_20261102/result.json').as_posix()})、"
             f"[geometry/dynamics proxy qualification]({(ROOT / 'outputs/contact_geometry_cpu_exterior_v3/report.json').as_posix()})。", "",
             "![全部跟踪轨迹](tracking.png)", "", "![逐seed配对图](paired_effects.png)", "", "## 复现", "", "```powershell",
             f"& '{sys.executable}' '{Path(__file__).resolve()}'", "```", ""]
    (output/"REPORT.md").write_text("\n".join(text),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,default=ROOT/"configs/replication-seeds123-evaluation-v1.json");p.add_argument("--output",type=Path,default=ROOT/"outputs/replication_seeds123_comparison_v1");args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);plan=read(args.plan);manager_path=Path(plan["output"])/"state.json";manager=read(manager_path) if manager_path.exists() else None
    pending=readiness(plan,manager)
    base_data={"schema":"supermimic_chord_replication_comparison:1","created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":digest(__file__),"gpu_executed":False,"plan_sha256":digest(args.plan),"training_seeds":[1,2,3],"evaluation_seed":0,"seed0_pilot_excluded":True,"statistical_unit":"paired training seed; n=3, not frames"}
    if pending:
        write(output/"comparison.json",{**base_data,"status":"PENDING","pending":pending,"runs":[],"aggregates":None})
        (output/"REPORT.md").write_text("# PENDING\n\n九个评估及manager完成状态尚未齐全；不计算率、不把缺失当FAIL/0。\n\n"+"\n".join(pending)+"\n",encoding="utf-8");print("PENDING");return
    require(manager["plan_sha256"]==digest(args.plan)==digest(Path(plan["output"])/"plan.json"),"evaluation plan changed")
    require(len(plan["jobs"])==len(manager["jobs"])==9,"exactly nine planned unique evaluations required")
    torch.set_num_threads(2)
    for job,observed in zip(plan["jobs"],manager["jobs"]):
        require(all(observed.get(k)==v for k,v in job.items()) and observed["status"]=="evaluation_recorded","manager job mismatch")
    targets_path=ROOT/"outputs/reference_contact_targets_v1/contact_targets.npz";targets=npz(targets_path);manifest=read(targets_path.parent/"manifest.json");require(digest(targets_path)==manifest["target_npz_sha256"],"target hash changed")
    selection=read(ROOT/"outputs/replication_seeds123_audit_v1/secondary1200_manifest.json");require(selection["status"]=="passed","secondary audit pending/failed")
    rows=[];base=None
    try:
        for job in plan["jobs"]:
            row,run=load_job(job,plan,targets,manifest,selection,base)
            if base is None:base=run
            rows.append(row)
    except ValueError as error:
        state="PENDING" if str(error).startswith("PENDING:") else "INVALID"
        write(output/"comparison.json",{**base_data,"status":state,"error":str(error),"runs":[],"aggregates":None})
        (output/"REPORT.md").write_text(f"# {state}\n\n{error}\n\n未计算率或把缺失视为FAIL。\n",encoding="utf-8");raise
    primary_a1={r["training_seed"]:r for r in rows if r["method"]=="A1L"};primary_a2={r["training_seed"]:r for r in rows if r["method"]=="A2" and r["budget"]==PRIMARY};time_a2={r["training_seed"]:r for r in rows if r["budget"]==SECONDARY}
    groups={"primary_A1L":group_statistics(list(primary_a1.values())),"primary_A2":group_statistics(list(primary_a2.values())),"secondary_A1L":group_statistics(list(primary_a1.values())),"secondary_A2":group_statistics(list(time_a2.values()))}
    queue=read(ROOT/"outputs/replication_seeds123_queue_v1/state.json");costs={}
    for job in queue["jobs"]:
        train=read(Path(job["output"])/"summary.json");costs[f"{job['method']}_seed{job['seed']}"]={"training_recorded_seconds":train["total_wall_seconds_including_failed_attempts"],"queue_reserved_including_CPU_health_seconds":job["reserved_wall_seconds_including_CPU_health"],"peak_allocated_gib":train["attempts"][0]["torch_peak_allocated_gib"]}
    data={**base_data,"status":"COMPLETE","manager_state_sha256":digest(manager_path),"manager_status":manager["status"],"unique_physical_rollouts":9,"runs":rows,"groups":groups,
          "paired_effects":{"primary":paired(primary_a1,primary_a2),"secondary":paired(primary_a1,time_a2)},
          "secondary_A1L_aliases":{str(s):primary_a1[s]["id"] for s in [1,2,3]},"secondary_selection_manifest_sha256":digest(ROOT/"outputs/replication_seeds123_audit_v1/secondary1200_manifest.json"),
          "training_costs":costs,"evaluation_reserved_gpu_hours":sum(r["evaluation_cost"]["reserved_gpu_hours"] for r in rows),
          "helper_script_sha256":digest(ROOT/"scripts/summarize_paired_pilot.py"),"fixed_target_sha256":manifest["target_npz_sha256"],
          "retained_limitations":["single reference/object/initial condition, no raw ego data","local proxy/dynamics gradient qualification failed","standalone compiled precision gate failed; actual prior eager","support and penetration prior bundled; no support-only attribution","descriptive n3, seed0 pilot excluded, no population significance"]}
    write(output/"comparison.json",data);shutil.copy2(args.plan,output/"plan_snapshot.json");shutil.copy2(manager_path,output/"manager_state_snapshot.json")
    with (output/"逐seed.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        flat=[{**{k:r[k] for k in ["id","method","training_seed","evaluation_seed","budget","iteration","task_success"]},**r["metrics"],**r["gates"],"failed_task_gates":";".join(r["failed_task_gates"]),"A1L_secondary_alias":r["secondary_A1L_alias_of_this_final"]} for r in rows];writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for r in rows:write(output/f"contact_score_{r['id']}.json",r["contact_diagnostic"])
    markdown(data,output);figures(primary_a1,primary_a2,time_a2,output,base["reference"].meta["task_acceptance"])
    print(json.dumps({"status":"COMPLETE","groups":{k:{"n":3,"success":v["success_count"]} for k,v in groups.items()},"output":str(output)},indent=2))


if __name__=="__main__":main()
