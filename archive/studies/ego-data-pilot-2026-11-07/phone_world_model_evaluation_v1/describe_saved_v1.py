"""Pure saved evaluation description/figures. No Torch or neural replay."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
R = H/"completed_v1"
OUT = H/"saved_description_v1"

def pin(p):
    p = Path(p).resolve(strict=True); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

def read(p):return json.loads(Path(p).read_text())

def main():
    OUT.mkdir()
    report = read(R/"report.json"); protocol = read(H/"protocol_v1.json")
    acq = read(protocol["inputs"]["acquisition_report"]["path"])
    limits = np.asarray(next(iter(acq["model_audits"].values()))["joint_range20x2"])
    qdescriptors = []
    for method in report["methods"]:
        with np.load(R/(method+"_recursive.npz"), allow_pickle=False) as z:
            p = z["prediction58"].copy(); t = z["target58"].copy(); mask = z["target_available_mask"].copy()
        pe = np.maximum(limits[:, 0]-p[:, :, :20], 0)+np.maximum(p[:, :, :20]-limits[:, 1], 0)
        te = np.maximum(limits[:, 0]-t[:, :, :20], 0)+np.maximum(t[:, :, :20]-limits[:, 1], 0)
        for j in range(10):
            qdescriptors.append(dict(method=method,horizon=j+1,predicted_episode_count=len(p),
                predicted_outside_count=int(np.any(pe[:,j]>0,-1).sum()),predicted_max_excursion_rad=float(np.max(pe[:,j])),
                available_true_targets=int(mask[:,j].sum()),true_outside_count=int(np.any(te[mask[:,j],j]>0,-1).sum()),
                true_max_excursion_rad=float(np.max(te[mask[:,j],j])) if np.any(mask[:,j]) else None))
    methods = report["methods"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    short = ["WM final5000", "TRAIN mean", "Constant velocity", "Ballistic COM"]
    for method, label in zip(methods, short):
        rows = [v for v in report["recursive_ALL"] if v["method"] == method]
        for axis, key in zip(ax, ["COM_vector_RMS_mm", "full_SO3_geodesic_RMS_deg"]):
            common = [v for v in rows if v["horizon"] < 10]
            line, = axis.plot([v["horizon"]*.02 for v in common], [v["error_metrics"][key] for v in common], marker="o", label=label)
            last = next(v for v in rows if v["horizon"] == 10)
            axis.scatter([.2], [last["error_metrics"][key]], facecolors="none", edgecolors=line.get_color(), marker="s", s=65)
    ax[0].set_ylabel("COM vector RMS (mm)"); ax[1].set_ylabel("Full-body SO(3) RMS (degrees)")
    for axis in ax:
        axis.set_xlabel("Prediction horizon (s)"); axis.grid(alpha=.25)
    ax[0].legend(fontsize=8)
    fig.suptitle("Fixed final5000 TEST: recursive errors, all 12 episode starts\nConnected markers: 12 targets; open H10 squares: only 6 targets, not the same denominator", fontsize=10)
    fig.savefig(OUT/"recursive_COM_and_full_orientation.png", dpi=180); plt.close(fig)
    curve = read(R/"all_training_and_saved_validation_curves.json")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot([v["iteration"] for v in curve["training_rows"]], [v["losses"]["total"] for v in curve["training_rows"]], lw=.8, label="TRAIN all 5000 rows")
    ax.plot([v["iteration"] for v in curve["validation_checkpoints"]], [v["validation"]["total"] for v in curve["validation_checkpoints"]], "o-", label="Saved VALIDATION (250/1000/5000)")
    ax.set_xlabel("Optimizer updates"); ax.set_ylabel("Declared normalized objective"); ax.grid(alpha=.25); ax.legend()
    ax.set_title("Fixed final5000 retained despite validation deterioration")
    fig.savefig(OUT/"all_training_and_saved_validation_curve.png", dpi=180); plt.close(fig)
    summary = dict(schema="phone_fixed_final_saved_author_description:1",producer_report=pin(R/"report.json"),protocol=pin(H/"protocol_v1.json"),
        process_terminal=pin(H/"process_v1/terminal.json"),source=pin(__file__),joint_excursions=qdescriptors,
        actual_new_NN_Torch_native_physics=0,plots={f.name:pin(f) for f in OUT.glob("*.png")},
        scope="Author saved description only; does not claim an independent neural replay or physical result")
    with (OUT/"report.json").open("x") as f:json.dump(summary,f,indent=2,allow_nan=False)
    text = """# Fixed final5000 phone WM evaluation

The once-only CPU evaluation completed with one checkpoint load, one constructor,
114 one-step predictions and120 recursive predictions (11 neural calls total).
No gradients, optimizer, native model or physics were run. All120 TEST rows remain,
including6 masked partial targets. Fixed5000 was chosen before TEST; no earlier
checkpoint was evaluated on TEST.

| One-step ALL114 | COM RMS mm | Full SO3 RMS degrees |
|---|---:|---:|
| WM final5000 | 2.252 | 1.440 |
| TRAIN mean | 9.246 | 3.642 |
| Constant velocity | 1.653 | 0.874 |
| Continuous ballistic COM | 0.766 | 0.874 |

The strong kinematic baselines prevent calling the learned model a one-step pose
winner. The learned coverage Brier is .03421 versus TRAIN mean .11820, while its
nonpad-force maximum RMSE is30.568N versus mean2.345N. Force channels are distinct
PRE summaries; neither Brier nor pose accuracy establishes true contact forces.

True coverage groups contain63 no-pad,16 one-pad and35 both-pad-observed intervals.
The learned COM RMS is2.717/1.827/1.306mm respectively; ballistic is.130/.666/1.296mm.
Both-pad coverage means each pad was seen during the interval, not simultaneous
support. No-pad coverage does not exclude weak, nonpad or table contacts.

Recursive learned RMS at H1/H2/H5/H9 is .665/1.448/6.919/18.554mm and
.954/1.909/5.495/13.660 degrees, all12 targets. H10 has only6 genuine targets:
29.550mm/12.880 degrees; the other6 partial targets are not fabricated. H10 must
not be interpreted as a monotonic curve on the same population. All predictions
were finite and nearSO3. Raw true-joint excursions for both predictions and
available native targets are listed separately in report.json, without clipping.

Saved validation total loss worsens1.135→1.915→2.459 at250/1000/5000, while its
coverage Brier worsens.00732→.01258→.02497. All5000 training log rows and all three
validation checkpoints are retained, with no TEST checkpoint selection.

This is local simulated heldout-family evaluation around two mappings of one ego
frame under assumed physical parameters. Hidden solver/history state is absent.
There is no independent ego dynamics, robust holding, true-Jacobian, Actor or
CHORD benefit qualification.
"""
    (OUT/"report.md").open("x",encoding="utf-8").write(text)
    print(json.dumps({"report":pin(OUT/"report.json"),"markdown":pin(OUT/"report.md")}))

if __name__ == "__main__":main()
