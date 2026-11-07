"""Render saved geometric distance diagnostics; no model load or new queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(source, output):
    with np.load(source / "pair_distances.npz", allow_pickle=False) as z:
        time, distance = z["time_s"], z["pair_signed_distance_m"]
    depth = np.maximum(-distance.min(axis=1), 0) * 1000
    count = (distance < -.003).sum(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].plot(time, depth, color="#be3d2a")
    axes[0].axhline(3, color="black", ls="--", label="3mm descriptive screen")
    axes[0].set_ylabel("Deepest pair depth (mm)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower right")
    axes[1].step(time, count, where="mid", color="#355b91")
    axes[1].set_ylabel("Pairs with depth >3mm")
    axes[1].set_xlabel("Candidate time (s)")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Palm-root v2: geometric hand-object incompatibility under current mapping\n151 frames; 26 hand x 3 object geoms; no dynamics or inferred contact labels", fontsize=12)
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "contact_screen.png", dpi=160)
    fig.savefig(output / "contact_screen.svg")
    plt.close(fig)
    metadata = {"source_sha256": hashlib.sha256((source / "pair_distances.npz").read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": "Artifact-only rendering. Negative pair depths retained; positive/zero distance anomalies are separately recorded in query review. Not a dynamics result."}
    (output / "render.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    run(a.source, a.output)
