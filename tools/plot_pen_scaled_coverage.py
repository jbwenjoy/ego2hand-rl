"""Plot sampling coverage and strict evaluation as a function of training transitions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def render(run, target):
    counts, budgets, iterations = [], [], {}
    with (run / "iterations.jsonl").open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            hist = np.asarray(row["phase_histogram_valid"], dtype=np.int64)
            if hist.shape != (369,) or hist.sum() != row["valid_count"] or hist[0] or (hist < 0).any():
                raise ValueError("Unexpected reference histogram")
            counts.append(hist[1:].reshape(46, 8).sum(axis=1))
            budgets.append(row["transitions"])
            iterations[row["iteration"]] = row["transitions"]
    if not budgets:
        raise ValueError("No completed rollouts")
    # One column per 16 rollouts. Normalize partial trailing columns to the
    # same per-transition unit so a partial block does not look under-sampled.
    edges = [0]
    density = []
    for start in range(0, len(counts), 16):
        block = counts[start:start+16]
        stop = budgets[start+len(block)-1]
        density.append(np.sum(block, axis=0)/(stop-edges[-1]))
        edges.append(stop)
    fig, (ax, ev) = plt.subplots(2, 1, figsize=(9,6), sharex=True,
                                  gridspec_kw={"height_ratios":[3,1]}, layout="constrained")
    z = np.asarray(density).T
    mesh = ax.pcolormesh(np.asarray(edges)/1e6, np.arange(0,369,8)*.02,
                         np.ma.masked_equal(z,0), norm=LogNorm(vmin=1e-6,vmax=1), cmap="viridis")
    ax.set_facecolor("#eeeeee")
    ax.axhline(3.68, linestyle="--", color="#be5014", linewidth=1.3, label="Second half begins")
    ax.axhline(5.52, linestyle=":", color="#ac291c", linewidth=1.3, label="Last quarter begins")
    ax.set(ylabel="Reference time (s)", ylim=(0,7.36), title="Wuji Hand 2 pen spin | 256-world residual PPO")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(mesh,ax=ax,label="Valid transitions in phase bin / all transitions")
    seen = 0
    for p in sorted(run.glob("eval_*/report.json")):
        report=json.loads(p.read_bytes())
        if report["iteration"] not in iterations:
            continue
        x=iterations[report["iteration"]]/1e6
        ev.scatter(x,report["successes"]/report["total"],color="#273754",zorder=3)
        ev.annotate(f"{report['successes']}/{report['total']}", (x,report["successes"]/report["total"]),
                    xytext=(5,5),textcoords="offset points",fontsize=9)
        seen+=1
    ev.set(ylabel="Strict success rate",xlabel="Training environment steps (millions)",
           ylim=(-.08,1.2),yticks=[0,.5,1],xlim=(0,edges[-1]/1e6))
    ev.grid(alpha=.2)
    ax.set_title("Gray bins contain zero valid samples",loc="left",fontsize=8,pad=5)
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target,dpi=150)
    plt.close(fig)
    return dict(rollouts=len(counts),transitions=budgets[-1],completed_evaluations=seen)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("target",type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.run,args.target)))
