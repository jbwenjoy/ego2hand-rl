"""Audit source hand approach, clearance and release using joints, scan and raw depth frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from audit_hocap_carton_support import backproject, fit_table, spans
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.strict_json import write_json


def audit(dataset, report, output):
    ref = json.loads(report.read_bytes())
    seq = dataset / ref["sequence"]
    meta = yaml.safe_load((seq / "meta.yaml").read_text())
    mesh = trimesh.load_mesh(verify_file_pin(ref["geometry"]["source"]), process=False)
    with np.load(verify_file_pin(ref["source_poses"]), allow_pickle=False) as a:
        frames, hand, obj = a["frames"], a["world_right_hand21"], a["world_from_carton"]
    pts = (hand[:, [4, 8, 12, 16, 20]] - obj[:, None, :3, 3]) @ obj[:, :3, :3]
    query = SurfaceQuery(mesh.vertices, mesh.faces)
    tip = 1000 * np.stack([query.nearest(points)[1] for points in pts])
    extrinsics = dataset / "calibration/extrinsics" / meta["extrinsics"]
    extrinsic = yaml.safe_load(extrinsics.read_text())["extrinsics"]
    def transform(key):
        return np.vstack((np.array(extrinsic[key]).reshape(3, 4), [0, 0, 0, 1]))

    # The interior check uses the frame of greatest translation relative to this source window.
    samples = [0, int(np.argmax(np.linalg.norm(obj[:, :3, 3]-obj[0, :3, 3], axis=1))), len(frames)-1]
    table_rows = []
    for serial in ("105322251564", "043422252387"):
        calib = dataset / f"calibration/intrinsics/{serial}.yaml"
        k = yaml.safe_load(calib.read_text())["color"]
        world_cam = np.linalg.solve(transform("tag_1"), transform(serial))
        for ix in samples:
            depth = seq / serial / f"depth_{frames[ix]:06d}.png"
            plane, stats = fit_table(backproject(depth, k, world_cam))
            table_rows.append(dict(camera=serial, frame=int(frames[ix]), depth=file_pin(depth), calib=file_pin(calib),
                                   **stats, estimated_table_z_at_initial_object_mm=float(1000*np.r_[obj[0,:2,3],1]@plane)))
    heights = np.array([row["estimated_table_z_at_initial_object_mm"] for row in table_rows])
    output.mkdir(parents=True, exist_ok=False)
    datafile = output / "signals.npz"
    np.savez_compressed(datafile, source_frames=frames, tip_joint_distance_mm=tip)
    write_json(output / "report.json", dict(
        source=file_pin(report), script=file_pin(__file__), reused_table_fit=file_pin(Path(__file__).with_name("audit_hocap_carton_support.py")),
        checks=table_rows, extrinsics=file_pin(extrinsics), signals=file_pin(datafile),
        target_depth_table_z_at_initial_xy_range_mm=[float(heights.min()), float(heights.max())],
        fingertip_min_distance_mm_at_start_end=tip[[0, -1]].min(1).tolist(),
        near_frames_abs={str(t): [[int(frames[a]),int(frames[b])] for a,b in spans(tip.min(1)<t)] for t in (10,20,40)},
        scope="Source-only post-check. Near refers to tip-joint distance to the scanned surface. Table depth fits validate the same-frame scene; local depth does not calibrate gravity or force.",
    ))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("source_report", type=Path)
    p.add_argument("output", type=Path)
    args=p.parse_args()
    audit(args.dataset.resolve(), args.source_report.resolve(), args.output.resolve())
