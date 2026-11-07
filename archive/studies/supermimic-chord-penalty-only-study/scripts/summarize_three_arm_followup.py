"""CPU-only three-arm follow-up summary; historical controls, n=3 seed pairs.

No GPU, training, new physics, threshold changes or quality-based selection.
Only complete manager-bound evaluation sets are summarized. Missing results
remain PENDING; integrity errors become INVALID, never task failures or zeros.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT.parent/"supermimic-chord-study"
sys.path.insert(0,str(MAIN/"scripts"))
import summarize_replication_seeds as shared
from summarize_paired_pilot import digest, npz, read, require, write
# The historical helper sets an empty mask; override before any CUDA query.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch
import mujoco
from matplotlib import pyplot as plt
from chord_study.contacts import wrench_support

PRIMARY,SECONDARY=shared.PRIMARY,shared.SECONDARY
METHODS=["A1L","P_ONLY","A2"]


def contrast(left,right,label):
    pairs=[{"training_seed":s,"left":left[s]["id"],"right":right[s]["id"],
            "task_difference":int(right[s]["task_success"])-int(left[s]["task_success"]),
            "differences":{k:right[s]["metrics"][k]-left[s]["metrics"][k] for k in left[s]["metrics"]}} for s in [1,2,3]]
    return {"label":label,"n_training_seed_pairs":3,"pairs":pairs,
            "descriptive_differences":{k:shared.summary([p["differences"][k] for p in pairs]) for k in pairs[0]["differences"]}}


def contact_integrity(run,model,targets):
    a=run["arrays"];active=a["live_contact_active"].astype(bool)
    object_body=model.body("object/cube").id
    object_geoms=set(np.flatnonzero(model.geom_bodyid==object_body).tolist())
    hand={i for i in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)",model.geom(i).name)}
    pairs=a["live_contact_geom_pair"][active]
    checks={"active_count_matches_raw":np.array_equal(active.sum(-1),a["hand_object_active_contact_count"]),
            "active_pairs_are_hand_object_no_mount":all((int(x) in object_geoms and int(y) in hand) or (int(y) in object_geoms and int(x) in hand) for x,y in pairs),
            "raw_geometry_and_forces_finite":all(np.isfinite(a[k]).all() for k in ["live_contact_pos_o_m","live_contact_pos_com_o_m","live_contact_normal_o","live_contact_force_w_N"]),
            "active_normal_unit":bool(np.allclose(np.linalg.norm(a["live_contact_normal_o"][active],axis=-1),1,atol=1e-6,rtol=0)),
            "COM_shift_matches_model":bool(np.allclose(a["live_contact_pos_com_o_m"][active],a["live_contact_pos_o_m"][active]-model.body_ipos[object_body],atol=1e-10,rtol=0)),
            "basis_matches_frozen512":bool(np.allclose(a["support_basis"],targets["support_basis"],atol=1e-7,rtol=0))}
    recomputed=wrench_support(torch.from_numpy(a["live_contact_pos_com_o_m"]),torch.from_numpy(a["live_contact_normal_o"]),
                             torch.from_numpy(active),torch.from_numpy(a["support_basis"]),run["reference"].meta["object_radius_m"]).numpy()
    error=float(np.abs(recomputed-a["live_support512"]).max())
    checks["support512_recomputed_from_raw_contacts"]=error<=1e-10
    require(all(checks.values()),f"raw physical-contact transport mismatch: {checks}")
    return {"checks":checks,"support_recompute_max_abs_error":error,
            "actual_contact_frames":int(active.any(-1).sum()),"frames":len(active),
            "max_solver_active_contacts":int(active.sum(-1).max()),
            "max_logged_contact_force_norm_N":float(np.linalg.norm(a["live_contact_force_w_N"],axis=-1).max()),
            "scope":"unit-cone geometric support from solver-active hand-object contacts; force amplitudes are logged but are not the reference target"}


def figures(groups,out,protocol):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none"})
    colors={"A1L":"#1764ab","P_ONLY":"#15845c","A2":"#d55e00"}
    metrics=[("position_error_mm","Position error (mm)",protocol["position_max_m"]*1000),
             ("shaft_axis_error_deg","Shaft axis error (deg)",protocol["shaft_angle_max_deg"]),
             ("progress_abs_error_deg","Progress error (deg)",protocol["progress_abs_error_max_deg"]),
             ("joint_state_violation_mrad","Actual joint-state excess (mrad)",protocol["joint_limit_margin_rad"]*1000)]
    for budget in ["primary","secondary"]:
        fig,axes=plt.subplots(4,3,figsize=(13.5,11),sharex=True,sharey="row",constrained_layout=True)
        for col,seed in enumerate([1,2,3]):
            for row,(key,label,limit) in enumerate(metrics):
                ax=axes[row,col]
                for method in METHODS:
                    r=groups[budget][method][seed];ax.plot(r["trajectories"]["time_s"],r["trajectories"][key],color=colors[method],label=f"{method} {r['iteration']}",linewidth=1.5)
                ax.axhline(limit,color="0.2",linestyle=":",linewidth=1)
                if row==2:ax.scatter([3],[protocol["progress_endpoint_error_max_deg"]],color="0.2",marker="D",s=18,zorder=6)
                ax.grid(alpha=.18);ax.set_xlim(0,3)
                if col==0:ax.set_ylabel(label)
            axes[0,col].set_title(f"Training seed {seed}; evaluation seed 0")
            axes[-1,col].set_xlabel("Control time (s); 151 samples, no smoothing")
        for row,(key,label,limit) in enumerate(metrics):
            peak=max(limit,max(max(r["trajectories"][key]) for arm in groups[budget].values() for r in arm.values()))
            axes[row,0].set_ylim(0,1.08*peak)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(("Primary equal-update historical three-arm follow-up" if budget=="primary" else "Secondary checkpoint availability before1200s, within2000-iteration caps")+
                     "\nFrozen gates unchanged; actual state excess is separate from bounded commands. Each column is one training seed.",fontsize=12)
        for ext in ["png","svg"]:fig.savefig(out/f"tracking_{budget}.{ext}",dpi=170)
        plt.close(fig)
    specs=[("shaft_max_deg","Maximum shaft error (deg)",20),
           ("progress_endpoint_error_deg","Endpoint progress error (deg)",20),
           ("joint_state_violation_mrad","Actual joint-state excess (mrad)",1),
           ("support512","Auxiliary support score (512)\nZoomed y-axis",None)]
    fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    for row,budget in enumerate(["primary","secondary"]):
        for col,(key,title,limit) in enumerate(specs):
            ax=axes[row,col]
            for seed,color,marker in zip([1,2,3],["#1764ab","#d55e00","#8854a4"],["o","s","^"]):
                ax.plot(range(3),[groups[budget][method][seed]["metrics"][key] for method in METHODS],color=color,marker=marker,label=f"Seed {seed}")
            ax.set_xticks(range(3),METHODS);ax.set_xlim(-.15,2.15);ax.set_title(title);ax.grid(alpha=.18);ax.set_ylim(bottom=0)
            if limit is not None:
                ax.axhline(limit,color="0.2",linestyle=":")
                peak=max(limit,max(r["metrics"][key] for arm in groups[budget].values() for r in arm.values()))
                ax.set_ylim(0,peak*1.08)
            else:
                values=[r["metrics"][key] for arm in groups[budget].values() for r in arm.values()]
                ax.set_ylim(max(0,min(values)-.015),min(1,max(values)+.015))
            if col==0:ax.set_ylabel("Primary: 2000 each" if row==0 else "Pre1200: 2000 / 1800 / 1700")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Descriptive paired three-arm follow-up (n=3 training seeds; historical controls)\nErrors/excess: lower is better; support score: higher is better, not task success. No population inference.",fontsize=12)
    for ext in ["png","svg"]:fig.savefig(out/f"paired_effects.{ext}",dpi=180)
    plt.close(fig)


def markdown(data,groups,out):
    lines=["# 三臂历史对照后续探索：A1L / P_ONLY / A2", "",
           "本报告仅使用训练seeds1/2/3；evaluation seed统一0、同一物理初态。N=3训练seed对，不是151帧，也不包含seed0 pilot。仅报告描述性结果，不计算总体显著性或置信区间。", "",
           "|预算|A1L通过|P_ONLY通过|A2通过|", "|---|---:|---:|---:|"]
    for budget in ["primary","secondary"]:
        counts=[data["group_statistics"][budget][m]["success_count"] for m in METHODS]
        lines.append(f"|{budget}|{counts[0]}/3|{counts[1]}/3|{counts[2]}/3|")
    for budget in ["primary","secondary"]:
        lines += ["",f"## {budget} 逐seed全部结果", "", "|方法|训练seed|评估seed|iter|task|位置mean/max mm|轴mean/max °|进度终点 °|qstate超限 mrad|target超限 mrad|support512|实际接触帧|失败门槛|",
                  "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for seed in [1,2,3]:
            for method in METHODS:
                r=groups[budget][method][seed];m=r["metrics"]
                lines.append(f"|{method}{' (alias)' if budget=='secondary' and method=='A1L' else ''}|{seed}|0|{r['iteration']}|{'PASS' if r['task_success'] else 'FAIL'}|{m['position_mean_mm']:.2f}/{m['position_max_mm']:.2f}|{m['shaft_mean_deg']:.3f}/{m['shaft_max_deg']:.3f}|{m['progress_endpoint_error_deg']:.3f}|{m['joint_state_violation_mrad']:.4f}|{m['joint_target_violation_mrad']:.5f}|{m['support512']:.6f}|{r['physical_contact_integrity']['actual_contact_frames']}/151|{', '.join(r['failed_task_gates']) or '—'}|")
        success=[s for s in [1,2,3] if groups[budget]["P_ONLY"][s]["task_success"]]
        lines += ["",f"P_ONLY 的 {budget} 成功seed为 {success}；按本预算独立计数。没有跨预算取每seed最好结果，也没有按评估结果选择checkpoint。"]
    lines += ["", "## 目标对照与配对描述", "",
              "P_ONLY−A1L 是不含support吸引时新增penalty的目标对照；A2−P_ONLY 是在同一penalty系数下增加support的条件对照。下表差值为右臂减左臂。误差/超限降低与分数提高分别解读，不能只看峰值而忽略时间均值或整体门槛。", "",
              "|预算|对照|指标|seed1|seed2|seed3|均值|中位数|最小/最大|", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for budget,contrasts in data["contrasts"].items():
        for name,c in contrasts.items():
            for key in ["position_mean_mm","position_max_mm","shaft_mean_deg","shaft_max_deg","progress_endpoint_error_deg","joint_state_violation_mrad","support512"]:
                v=c["descriptive_differences"][key];x=v["values_seed123"]
                lines.append(f"|{budget}|{name}|{key}|{x[0]:+.5f}|{x[1]:+.5f}|{x[2]:+.5f}|{v['mean']:+.5f}|{v['median']:+.5f}|{v['min']:+.5f}/{v['max']:+.5f}|")
    axis_pairs=sum(groups["primary"]["A2"][s]["metrics"]["shaft_max_deg"]<groups["primary"]["P_ONLY"][s]["metrics"]["shaft_max_deg"] for s in [1,2,3])
    q_pairs=sum(groups["primary"]["A2"][s]["metrics"]["joint_state_violation_mrad"]>groups["primary"]["P_ONLY"][s]["metrics"]["joint_state_violation_mrad"] for s in [1,2,3])
    lines += ["",f"本次主预算中，A2相对P_ONLY有 {axis_pairs}/3 对seed的最大轴误差降低，同时 {q_pairs}/3 对实际关节状态超限增大。这里是观测到的配对信号，不宣称support有无条件因果主效应；每条失败门槛仍如表保留。",
              "P_ONLY / A2 / A1L 的不同预算结果不可拼成一组更高成功率。没有support-only(1,0)臂，无法识别完整2×2交互或无条件support效应。", "",
              "## 预算、历史顺序与成本", "",
              "主预算各臂每seed均2000iter、648192真实transitions、WM/failure-head各4000、actor1800、critic7200更新；这不等于等compute。次预算是2000上限实验中strictpre1200可获得的checkpoint：A1L2000(648192/WM4000/actor1800/critic7200)、P_ONLY1800(584192/3600/1600/6400)、A21700(552192/3400/1500/6000)。A1L提前完成，不称耗满1200秒。次预算训练量不同，仅作时间可用性比较，不能作为纯系数机制分离。", "",
              "P_ONLY是在既有A1L/A2及seed0结果已知后冻结的探索性follow-up，不是原复制实验预注册的一臂。A1L/A2是不可变历史控制，P_ONLY三run全部发生在之后；三臂未同期随机排序，时间、系统/缓存历史与batch混杂仍在，不能把这些目标对照当作无偏因果估计。不同臂会产生不同on-policy训练数据，并非逐样本相同。", "",
              "|方法|seed|训练记录秒|GPU slot秒(含CPU health)|peak allocated GiB|", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for seed in [1,2,3]:
            c=data["training_costs"][f"{method}_seed{seed}"]
            lines.append(f"|{method}|{seed}|{c['training_recorded_seconds']:.3f}|{c['queue_GPU_slot_seconds']:.3f}|{c['peak_allocated_gib']:.3f}|")
    lines += ["",f"15条唯一物理评估的独立预留成本合计 {data['evaluation_reserved_gpu_hours']:.6f} GPUh；A1L次预算alias不重复计费。GPU slot不是利用率或训练计算时间；此前smoke、工程/编译失败、seed0及CPU报告费用未混入上表，不声称为研究总费用。P_ONLY保留完整A2前向用于最小共享P，但反向可能更省；没有填充计算来制造等成本。", "",
              "## 输入、原始接触与审计边界", "",
              "15条唯一rollout均绑定各自计划和completed manager状态，核对原训练输入、raw checkpoint训练seed、evaluation seed0、condition和checkpoint SHA；所有reference/model/任务协议、规范time/raw physics clock、t0 q/qd/object pose/twists/wrist/初始target一致。所有验收从原NPZ重算，与保存报告相符。P_ONLY三个final和三个1800各有独立52项健康审计；A1L/A2历史审计原样保留。", "",
              "512分数逐条从真实solver-active手物接触对固定运动学意图独立计算，并从raw contact位置/法向/active重新验证support512、COM移位、单位法向及无mount混入。该支持函数使用单位摩擦锥几何，不是测量力幅值拟合，也不是整体任务成功标准。actual joint state与bounded command分别报告，冻结1mrad允许量不变，没有clamp记录的状态。", "",
              "P_ONLY复用的P仅是采样手物几何penetration惩罚，不含显式actual-q joint-limit barrier。关节合规是本次不同学习轨迹的观测，不能据此证明P直接约束关节或已识别其力学/因果作用链。", "",
              "A2/P_ONLY的几何先验没有通过既有contact predictor/dynamics-gradient资格；standalone compiled-precision失败仍保留，实际prior走eager，原生子网编译。旧CPU精确penalty tensor检查是计算合同验证，不是动力学有效性。只有一个clip007[200:351]固定腕重设计、同一物体/初态、每checkpoint一次物理回放；无真实无本体ego输入、未知原视频父ID，训练seed并非跨动作/视频泛化。50Hz采样门槛也不是连续时间或硬件安全认证。", "",
              f"保留证据：[P_ONLY独立审计]({(ROOT/'outputs/penalty_only_seeds123_audit_v2/manifest.json').as_posix()})、"
              f"[strictpre1200选择]({(ROOT/'outputs/penalty_only_seeds123_audit_v2/secondary1200_manifest.json').as_posix()})、"
              f"[compiled precision失败]({(MAIN/'outputs/a2_cuda_pointmax_final_20261102/result.json').as_posix()})、"
              f"[proxy qualification失败]({(MAIN/'outputs/contact_geometry_cpu_exterior_v3/report.json').as_posix()})。", "",
              "![主预算原始轨迹](tracking_primary.png)", "", "![次预算原始轨迹](tracking_secondary.png)", "", "![三臂配对图](paired_effects.png)", "",
              "## 复现", "", "```powershell",f"& '{sys.executable}' '{Path(__file__).resolve()}'", "```", ""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"outputs/three_arm_seed123_comparison_v1");a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    header={"schema":"three_arm_penalty_followup_comparison:1","created_utc":datetime.now(timezone.utc).isoformat(),"gpu_executed":False,"script_sha256":digest(__file__),"statistical_unit":"three paired training seeds1/2/3, evaluation seed0, seed0 pilot excluded"}
    try:
        plan_paths=[MAIN/"configs/replication-seeds123-evaluation-v1.json",MAIN/"configs/penalty-only-seeds123-evaluation-v1.json"]
        packages=[];pending=[]
        for path in plan_paths:
            plan=read(path);mp=Path(plan["output"])/"state.json";manager=read(mp) if mp.exists() else None;pending+=shared.readiness(plan,manager);packages.append((path,plan,mp,manager))
        if pending:
            write(out/"comparison.json",{**header,"status":"PENDING","missing":pending,"runs":[],"aggregates":None});(out/"REPORT.md").write_text("# PENDING\n\n未齐全，不计FAIL/0或成功率。\n"+"\n".join(pending),encoding="utf-8");return
        torch.set_num_threads(2);require(not torch.cuda.is_available() and torch.cuda.device_count()==0 and not torch.cuda.is_initialized(),"CPU mask -1 not effective")
        targetpath=MAIN/"outputs/reference_contact_targets_v1/contact_targets.npz";targets=npz(targetpath);targetmanifest=read(targetpath.parent/"manifest.json")
        canonical=read(MAIN/"configs/paired-contact-pilot-v2.json")
        require(digest(targetpath)==targetmanifest["target_npz_sha256"]==canonical["contact_target_npz_sha256"],"canonical target mismatch")
        audit_paths=[MAIN/"outputs/replication_seeds123_audit_v1/manifest.json",ROOT/"outputs/penalty_only_seeds123_audit_v2/manifest.json"]
        audits=[read(x) for x in audit_paths];require(all(x["status"]=="passed" for x in audits),"PENDING: training audits not passed")
        selection_paths=[MAIN/"outputs/replication_seeds123_audit_v1/secondary1200_manifest.json",ROOT/"outputs/penalty_only_seeds123_audit_v2/secondary1200_manifest.json"]
        selected={"selected":[item for path in selection_paths for item in read(path)["selected"]]}
        auditrows={(x["method"],x["seed"]):x for audit in audits for x in audit["runs"]}
        rows=[];base=None;model=None;jobs=[];planrecords=[]
        for path,plan,mp,manager in packages:
            require(manager["plan_sha256"]==digest(path)==digest(Path(plan["output"])/"plan.json"),"frozen evaluation plan changed")
            require(len(plan["jobs"])==len(manager["jobs"]),"manager job count differs")
            planrecords.append({"path":str(path),"sha256":digest(path),"manager":str(mp),"manager_sha256":digest(mp),"manager_status":manager["status"]})
            for job,observed in zip(plan["jobs"],manager["jobs"]):
                require(all(observed.get(k)==v for k,v in job.items()) and observed["status"]=="evaluation_recorded","manager job differs")
                r,run=shared.load_job(job,plan,targets,targetmanifest,selected,base)
                require(r["source_evaluation"]["task_protocol_sha256"]==canonical["task_acceptance_sha256"],"canonical task gates differ")
                ar=auditrows[(job["method"],job["training_seed"])];require(ar["status"]=="passed","training health unavailable")
                if job["budget"]==PRIMARY:require(ar["final_sha256"]==job["checkpoint_sha256"],"final training audit binding differs")
                else:require(ar["selected"]["checkpoint_sha256"]==job["checkpoint_sha256"],"selected audit binding differs")
                if job["method"]=="P_ONLY":
                    health=read(Path(ar["final_health"] if job["budget"]==PRIMARY else ar["secondary_health"]))
                    require(len(health["checks"])==52 and health["status"]=="passed","P_ONLY52-check health unavailable")
                    r["P_ONLY_independent_health"]={"path":ar["final_health"] if job["budget"]==PRIMARY else ar["secondary_health"],"checks":52}
                if base is None:base=run;model=mujoco.MjModel.from_binary_path(str(run["directory"]/"model.mjb"))
                r["physical_contact_integrity"]=contact_integrity(run,model,targets);rows.append(r);jobs.append(job)
        expected={(method,seed,PRIMARY) for method in METHODS for seed in [1,2,3]}|{(method,seed,SECONDARY) for method in ["A2","P_ONLY"] for seed in [1,2,3]}
        require(len(rows)==15 and {(r["method"],r["training_seed"],r["budget"]) for r in rows}==expected,"not the frozen15 unique evaluations")
        groups={budget:{method:{} for method in METHODS} for budget in ["primary","secondary"]}
        for r in rows:groups["primary" if r["budget"]==PRIMARY else "secondary"][r["method"]][r["training_seed"]]=r
        groups["secondary"]["A1L"]=groups["primary"]["A1L"]
        descriptions={b:{m:shared.group_statistics(list(arm.values())) for m,arm in arms.items()} for b,arms in groups.items()}
        contrasts={b:{"P_ONLY_minus_A1L":contrast(arms["A1L"],arms["P_ONLY"],"penalty increment without support"),
                      "A2_minus_P_ONLY":contrast(arms["P_ONLY"],arms["A2"],"support increment conditional on shared penalty")} for b,arms in groups.items()}
        queues=[read(MAIN/"outputs/replication_seeds123_queue_v1/state.json"),read(ROOT/"outputs/penalty_only_seeds123_queue_v1/state.json")];costs={}
        for q in queues:
            for j in q["jobs"]:
                s=read(Path(j["output"])/"summary.json");costs[f"{j['method']}_seed{j['seed']}"]={"training_recorded_seconds":s["total_wall_seconds_including_failed_attempts"],"queue_GPU_slot_seconds":j["reserved_wall_seconds_including_CPU_health"],"peak_allocated_gib":s["attempts"][0]["torch_peak_allocated_gib"]}
        data={**header,"status":"COMPLETE","unique_evaluations":15,"runs":rows,"group_statistics":descriptions,"contrasts":contrasts,
              "plans":planrecords,"training_costs":costs,"training_audits":[{"path":str(p),"sha256":digest(p)} for p in audit_paths],
              "secondary_A1L_aliases":{str(s):groups["primary"]["A1L"][s]["id"] for s in [1,2,3]},
              "evaluation_reserved_gpu_hours":sum(r["evaluation_cost"]["reserved_gpu_hours"] for r in rows),
              "shared_helper_sha256":digest(MAIN/"scripts/summarize_replication_seeds.py"),"paired_protocol_sha256":digest(MAIN/"configs/paired-contact-pilot-v2.json"),
              "contact_target_sha256":digest(targetpath),"followup_protocol_sha256":digest(ROOT/"configs/penalty-only-followup-v1.json"),
              "historical_control_limitation":"P_ONLY frozen and run after known A1L/A2 outcomes; no contemporaneously randomized three-arm batch; method/order/cache batch confounding",
              "secondary_interpretation":"A1L2000 alias, P_ONLY1800, A21700 before1200s within2000 caps; unequal training amounts, not pure coefficient mechanism",
              "retained_failures":[str(MAIN/"outputs/a2_cuda_pointmax_final_20261102/result.json"),str(MAIN/"outputs/contact_geometry_cpu_exterior_v3/report.json")],
              "CUDA_initialized_after":torch.cuda.is_initialized()}
        require(data["CUDA_initialized_after"] is False,"CUDA unexpectedly initialized")
        write(out/"comparison.json",data)
        for r in rows:write(out/f"contact_score_{r['id']}.json",r["contact_diagnostic"])
        with (out/"逐seed.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            flat=[{**{k:r[k] for k in ["id","method","training_seed","evaluation_seed","budget","iteration","task_success"]},**r["metrics"],**r["gates"],"actual_contact_frames":r["physical_contact_integrity"]["actual_contact_frames"],"failed_task_gates":";".join(r["failed_task_gates"])} for r in rows];writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        for i,(p,plan,mp,manager) in enumerate(packages):shutil.copy2(p,out/f"plan_{i+1}.json");shutil.copy2(mp,out/f"manager_state_{i+1}.json")
        markdown(data,groups,out);figures(groups,out,base["reference"].meta["task_acceptance"])
        print(json.dumps({"status":"COMPLETE","success_counts":{b:{m:d["success_count"] for m,d in arms.items()} for b,arms in descriptions.items()},"unique_evaluations":15,"output":str(out)},indent=2))
    except (OSError,KeyError,StopIteration,ValueError,TypeError) as error:
        status="PENDING" if isinstance(error,FileNotFoundError) or str(error).startswith("PENDING:") else "INVALID"
        write(out/"comparison.json",{**header,"status":status,"error":str(error),"runs":[],"aggregates":None})
        (out/"REPORT.md").write_text(f"# {status}\n\n{error}\n\n没有把缺失或完整性错误计为任务失败。\n",encoding="utf-8")
        raise


if __name__=="__main__":main()
