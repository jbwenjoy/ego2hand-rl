"""Artifact-only onset zoom and explicit norm/vertical free-fall accounting."""
import hashlib
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rest_contact_diagnostic_v1"
SRC = ROOT / "outputs/rest_hold_frame0_v1"
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

report = read(OUT / "report.json")
a = dict(np.load(OUT / "diagnostic_arrays.npz", allow_pickle=False))
p = read(SRC.with_name(SRC.name + "_watchdog") / "plan_snapshot.json")
with np.load(p["candidate"]["path"], allow_pickle=False) as candidate:
    pick = (candidate["geometry_pair_kind"] == "hand_object") & candidate["generated_contact_mask"][0]
    query_pairs = candidate["geometry_pair_indices"][pick]
    query_distance = candidate["selected_native_contact_distance_m"][0, pick]
first_contacts = [json.loads(x) for x in (SRC / "raw_contacts.jsonl").read_text().splitlines()]
first_contacts = [x for x in first_contacts if x["step"] == 1 and sorted(x["geom"]) in query_pairs.tolist()]
assert {tuple(sorted(c["geom"])) for c in first_contacts} == {tuple(x) for x in query_pairs.tolist()}
query_matches = []
for pair, distance in zip(query_pairs, query_distance):
    match = [c for c in first_contacts if sorted(c["geom"]) == pair.tolist()]
    query_matches.append({"geom_pair_sorted": pair.tolist(), "cached_CPU_query_distance_m": float(distance),
                          "step1_Warp_distance_m": [c["distance_m"] for c in match],
                          "max_distance_difference_m": float(max(abs(c["distance_m"] - distance) for c in match))})
t, s = a["state_time_s"], a["solver_time_s"]
pos, ff = a["actual_position_world_m"], a["freefall_position_semiimplicit_world_m"]
dt = report["object_physics"]["Warp_FP32_dt_s"]
predicted_position = pos[0] + np.vstack((np.zeros(3), dt * np.cumsum(a["velocity_integrated_measured_contact_force_world_m_s"][1:], axis=0)))
dz, ffdz = (pos[:, 2] - pos[0, 2]) * 1000, (ff[:, 2] - pos[0, 2]) * 1000
mask = t <= .030000000001
f, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
for i, label in enumerate(("Fx", "Fy", "Fz")):
    axes[0, 0].step(s, 1000 * a["contact_force_on_object_world_N"][:, i], where="post", label=label)
axes[0, 0].set_ylabel("Force on object (mN)")
for i, label in enumerate(("tau x", "tau y", "tau z")):
    axes[0, 1].step(s, 1000 * a["net_contact_torque_COM_world_Nm"][:, i], where="post", label=label)
axes[0, 1].set_ylabel("Torque about COM (mN m)")
axes[1, 0].plot(t, dz, label="Actual object dz")
axes[1, 0].plot(t, ffdz, "--", label="Discrete free-fall dz")
axes[1, 0].set_ylim(min(dz[mask].min(), ffdz[mask].min()) - .3, .3)
axes[1, 0].set_ylabel("Object vertical displacement (mm)")
axes[1, 1].plot(t, 1000 * a["max_joint_drift_rad"], label="Largest joint drift")
axes[1, 1].set_ylim(-.05, float(1000 * a["max_joint_drift_rad"][mask].max()) * 1.12 + .03)
axes[1, 1].set_ylabel("Max joint drift (mrad)")
for ax in axes.ravel():
    ax.set_xlim(0, .03); ax.set_xlabel("Time (s), first30ms"); ax.grid(alpha=.23)
    ax.axvline(.015, color="grey", ls=":", lw=1); ax.legend(fontsize=8)
f.suptitle("First30ms: one finger2-link3 contact, positive force only in the first3solver intervals\nDotted15ms marks first solver sample with no hand-object contact; axis ranges show this window", fontsize=11)
f.tight_layout(rect=(0, 0, 1, .92))
for suffix in ("png", "svg"):
    path = OUT / ("contact_onset_zoomed." + suffix)
    assert not path.exists(), "Preserve previous artifact"
    f.savefig(path, dpi=160, bbox_inches="tight")
plt.close(f)
actual_norm = float(np.linalg.norm(pos[-1] - pos[0]))
assert abs(actual_norm - read(SRC / "evaluation.json")["max_object_position_error_m"]) < 1e-12
supplement = {
    "status": "COMPLETE_ARTIFACT_ONLY_SUPPLEMENT", "source_sha256": sha(Path(__file__)),
    "diagnostic_report_sha256": sha(OUT / "report.json"), "diagnostic_arrays_sha256": sha(OUT / "diagnostic_arrays.npz"),
    "query_frame0_pair_matches": query_matches,
    "query_scope": "Compare frozen candidate native contact pairs/distances to step1prestate only. Same pair with smallFP32/narrowphase depth difference is not an inferredforce or a guarantee for other states; no newquery/physics was called.",
    "freefall": {
        "continuous_0p3s_drop_mm": 1000 * abs(report["freefall_reference"]["continuous_displacement_at0p3_m"][2]),
        "semiimplicit_canonical_dt_drop_mm": 1000 * .5 * 9.81 * .005**2 * 60 * 61,
        "semiimplicit_Warp_FP32_drop_mm": float(-ffdz[-1]),
        "actual_3D_displacement_norm_mm": actual_norm * 1000,
        "actual_vertical_drop_mm": float(-dz[-1]),
        "actual_lateral_displacement_mm": ((pos[-1, :2] - pos[0, :2]) * 1000).tolist(),
        "discrete_drop_minus_actual_vertical_drop_mm": float(dz[-1] - ffdz[-1]),
        "discrete_drop_minus_actual_3D_norm_mm": float(-ffdz[-1] - actual_norm * 1000),
        "upward_displacement_predicted_by_timeweighted_measured_Fz_mm": float((predicted_position[-1, 2] - ff[-1, 2]) * 1000),
        "force_integrated_position_max_abs_residual_m": float(np.abs(predicted_position - pos).max()),
        "interpretation": "Measured upward impulse offsets vertical fall by about2.205mm; lateral impulse adds lateral displacement, so3Dnorm gap is about2.128mm. Continuous free-fall441.45mm is a differentintegration reference: its gap is not evidence of extra downward force. Allreferences are arithmetic, not newphysics."
    },
    "figures_sha256": {n: sha(OUT / n) for n in ("contact_onset_zoomed.png", "contact_onset_zoomed.svg")},
    "earlier_plot_retained": "contact_onset.png/svg use fulltrajectory y-limits; use the new zoomed version for onset detail. No earlier artifact overwritten.",
    "visual_QA": "pending_actual_view", "GPU_used": False, "new_physics_steps": 0
}
target = OUT / "artifact_supplement.json"
assert not target.exists()
target.write_text(json.dumps(supplement, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(supplement["freefall"], indent=2))
