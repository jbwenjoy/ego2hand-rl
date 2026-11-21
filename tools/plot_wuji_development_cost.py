"""Plot only closed, receipt-verified development scores against each method's charged GPU time."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from summarize_wuji_tranche import apply_interruption_bounds, fold


STYLES = {
    "single": ("逐条 FlashCHORD", "#245B84", "o"),
    "shared_mf": ("多参考 · 无世界模型", "#BB622F", "s"),
    "shared_wm": ("多参考 · 含世界模型", "#007E70", "^"),
}


def host_path(path):
    """Resolve the ledger's Linux archive paths on the Windows workstation as well."""
    if os.name == "nt":
        if path.startswith("/mnt/") and path[5:6].isalpha() and path[6:7] == "/":
            return Path(path[5] + ":/" + path[7:])
        if path.startswith("/"):
            return Path(r"\\wsl.localhost\Ubuntu-24.04") / path[1:]
    return Path(path)


def points(tranche, bounds_path, limit):
    ledger_bytes = (tranche / "ledger.json").read_bytes()
    plan = json.loads((tranche / "plan.json").read_bytes())
    all_rows = json.loads(ledger_bytes)
    if limit is not None and not 1 <= limit <= len(all_rows):
        raise ValueError(f"Only {len(all_rows)} scored segments are currently closed")
    rows = all_rows[:limit]
    cost = fold(plan, rows, verify=False)
    bounds = json.loads(bounds_path.read_bytes())["attempts"] if bounds_path else {}
    apply_interruption_bounds(cost, rows, bounds)
    pos = {m: 0 for m in STYLES}
    refs = {r["sequence"]: r["sha256"] for r in plan["references"]}
    plotted = []
    for row in rows:
        method = "single" if row["stream"].startswith("single") else row["stream"]
        for check in row["checks"]:
            event = cost["methods"][method]["events"][pos[method]]
            pos[method] += 1
            report_bytes = host_path(check["report"]).read_bytes()
            raw = json.loads(report_bytes)
            if not (hashlib.sha256(report_bytes).hexdigest() == check["report_sha256"]
                    and raw["checkpoint"] == row["checkpoint"]
                    and raw["sequence"] == check["sequence"] == event["sequence"]
                    and raw["parquet_sha256"] == refs[raw["sequence"]]
                    and raw["protocol_sha256"] == plan["protocol_sha256"]
                    and raw["gates"] == check["gates"]
                    and raw["passed_numeric_development_gates"] == check["passed"]):
                raise ValueError(f"Mismatched score receipt for {row['stream']} / {check['sequence']}")
            prefix, requested = raw["physical_intervals"], raw["requested_intervals"]
            if not (0 <= prefix <= requested and requested > 0 and (not check["passed"] or prefix == requested)):
                raise ValueError("Invalid recorded physical horizon")
            hi = event["card_s_upper_bound"]
            plotted.append(dict(method=method, stream=row["stream"], round=row["round"], sequence=raw["sequence"],
                executed=prefix, requested=requested, fraction=prefix / requested, strict_pass=check["passed"],
                recorded_numeric_failure=raw.get("status") == "failed_nonfinite_action",
                lower_min=event["card_s"]/60, upper_min=None if hi is None else hi/60,
                lower_bound_only=bool(event["cost_is_lower_bound"]), unknown_intervals=event["unknown_intervals"],
                strict_count=event["qualified_distinct"], physical_repeat_count=event["physical_repeat_distinct"],
                score_path=check["report"], score_sha256=check["report_sha256"],
                retry=bool(row.get("numerical_retry"))))
    return plan, plotted, len(rows), hashlib.sha256(ledger_bytes).hexdigest()


def draw(points, plan, count, path, overlap):
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if candidate.is_file():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams.update({"font.size":10, "axes.unicode_minus":False})
    fig, axs = plt.subplots(2,3,figsize=(12.4,7.2))
    fig.subplots_adjust(left=.074,right=.977,bottom=.17,top=.746,hspace=.92,wspace=.22)
    maximum = max([p["upper_min"] if p["upper_min"] is not None else p["lower_min"] for p in points] + [1.])
    margin = .032*maximum
    xmax = maximum * 1.085
    refs = [r["sequence"] for r in plan["references"]]
    for j, ax in enumerate(axs.flat):
        for spine in ("top","right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom","left"):
            ax.spines[spine].set_color("#ADB8BD")
        ax.grid(axis="y",color="#DDE4E5",linewidth=.75,zorder=0)
        ax.tick_params(colors="#465862",length=3)
        ax.set_xlim(0,xmax)
        ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
        if j<3:
            ax.set_ylim(-.06,1.04)
            ax.set_yticks([0,.25,.5,.75,1]); ax.yaxis.set_major_formatter(PercentFormatter(1))
            total=next((p["requested"] for p in points if p["sequence"]==refs[j]),None)
            title=refs[j].replace("subject","S",1).replace("_"," · ",1)
            ax.set_title(title+(f"    {total} 个源控制窗" if total else ""),loc="left",fontsize=11)
        else:
            ax.set_ylim(-.21,len(refs)+.16);ax.set_yticks(range(len(refs)+1))
            m=tuple(STYLES)[j-3]
            ax.set_title(STYLES[m][0],loc="left",color=STYLES[m][1],fontsize=11.5)
    def point(ax,x,y,p):
        _,color,shape=STYLES[p["method"]]
        hi=p["upper_min"]
        if p["lower_bound_only"] and hi is None:
            ax.annotate("",(x+margin,y),(x,y),arrowprops=dict(arrowstyle="->",color=color,lw=1,
                        shrinkA=0,shrinkB=0,mutation_scale=9),zorder=2)
        elif hi is not None and hi-x>1e-6:
            ax.plot([x,hi],[y,y],color=color,lw=1.2,zorder=2)
            ax.plot(hi,y,"|",color=color,markersize=6,zorder=2)
        marker="X" if p["recorded_numeric_failure"] else shape
        ax.plot(x,y,marker=marker,linestyle="none",mec="white",mew=.65,mfc=color,
                markersize=8 if marker=="X" else 6.2,zorder=4)
    for p in points:
        point(axs[0,refs.index(p["sequence"])],p["lower_min"],p["fraction"],p)
        point(axs[1,tuple(STYLES).index(p["method"])],p["lower_min"],p["strict_count"],p)
    fig.text(.07,.947,"训练成本与动作复现结果",fontsize=20,fontweight="bold",color="#182C39")
    fig.text(.07,.901,f"{count} 个已完成评分的训练段  ·  {len(points)} 次单条动作评估  ·  三条开发演示",fontsize=10.5,color="#4B606B")
    handles=[Line2D([],[],marker=s,ls="none",mfc=c,mec="white",markersize=8,label=n) for n,c,s in STYLES.values()]
    handles.append(Line2D([],[],marker="X",ls="none",color="#384B55",markersize=8,label="非有限动作"))
    fig.legend(handles=handles,ncol=4,loc="upper left",bbox_to_anchor=(.06,.883),frameon=False,
               handletextpad=.35,columnspacing=1.8,fontsize=10)
    fig.text(.07,.805,"每条演示实际执行的控制窗比例",fontsize=12.5,fontweight="bold",color="#182C39")
    fig.text(.07,.435,"累计通过全部严格门限的不同演示数",fontsize=12.5,fontweight="bold",color="#182C39")
    fig.text(.52,.093,"各方法累计计费的独占 GPU 时间（分钟）",ha="center",fontsize=11.5)
    fig.text(.07,.047,"每个点是一次已评分的检查点。短横线标示时间上下界，向右箭头表示仅有下界。"
             "未执行的控制窗不补齐，已执行的前缀不等于通过物理验收。",fontsize=9,color="#4B606B")
    if overlap:
        fig.text(.07,.018,f"第三轮多参考无模型训练期间曾运行约 {overlap['external_cmd_wall_s']:.1f} 秒的 OpenGL 渲染。"
                 "训练墙钟全额计入，显卡占用未按进程拆分。",fontsize=8.5,color="#4B606B")
    fig.savefig(path,dpi=165,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tranche",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--interruption-bounds",type=Path)
    parser.add_argument("--max-closed",type=int,help="Keep only this many closed ledger rows, in recorded order")
    args=parser.parse_args()
    plan,records,n,ledger_sha=points(args.tranche,args.interruption_bounds,args.max_closed)
    if not records:raise ValueError("At least one closed score required")
    args.output.mkdir(parents=True,exist_ok=False)
    dest=args.output/"cost_and_source_qualification.png"
    overlap_file=args.tranche/"r3_shared_mf_render_overlap.json"
    overlap=(json.loads(overlap_file.read_bytes()) if overlap_file.is_file() and
             any(p["method"]=="shared_mf" and p["round"]==3 for p in records) else None)
    draw(records,plan,n,dest,overlap)
    (args.output/"plotted_points.json").write_text(json.dumps(dict(
        closed_segments=n,evaluations=len(records),ledger_sha256=ledger_sha,
        plot_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        bounds_path=str(args.interruption_bounds) if args.interruption_bounds else None,
        bounds_sha256=hashlib.sha256(args.interruption_bounds.read_bytes()).hexdigest() if args.interruption_bounds else None,
        x_axis="Per-method charged card minutes at each closed evaluation; lower endpoints plotted.",
        y_axes=["executed physical intervals / requested source intervals", "distinct original strict passes accumulated"],
        concurrent_render_note=overlap,records=records),ensure_ascii=False,allow_nan=False,indent=2),encoding="utf8")
