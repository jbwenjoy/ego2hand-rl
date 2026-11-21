"""Measure tabletop support in HO-Cap's tag-1 world using recorded RealSense depth."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
import yaml

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.strict_json import write_json


DEPTH_FRAMES = (0, 60, 140)
TOOLKIT = "https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab"
ANNOTATION = "https://github.com/IRVLUTD/HO-Cap-Annotation/blob/f554ecdfdd1230f277a4dae443b104914caeb5a7"


def backproject(depth_path, intrinsics, transform):
    with Image.open(depth_path) as image:
        depth = np.asarray(image)
    v, u = np.mgrid[:depth.shape[0]:3, :depth.shape[1]:3]
    z = depth[v, u] / 1000.0
    x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
    points = np.stack((x, y, z), axis=-1)[z > 0] @ transform[:3, :3].T + transform[:3, 3]
    # A broad fixed region of the table around the object, including space above/below it.
    mask = ((abs(points[:, 0] - .1) < .25) & (abs(points[:, 1] - .1) < .25)
            & (points[:, 2] > -.12) & (points[:, 2] < .30))
    return points[mask]


def fit_table(points):
    rng = np.random.default_rng(701)
    xyz = points if len(points) < 8000 else points[rng.choice(len(points), 8000, replace=False)]
    design = np.column_stack((xyz[:, :2], np.ones(len(xyz))))
    best_count, best_plane = 0, None
    for _ in range(160):
        ids = rng.choice(len(xyz), 3, replace=False)
        if abs(np.linalg.det(design[ids])) < .0008:
            continue
        coef = np.linalg.solve(design[ids], xyz[ids, 2])
        if np.linalg.norm(coef[:2]) > np.tan(np.deg2rad(35)):
            continue
        count = np.count_nonzero(abs(design @ coef - xyz[:, 2]) < .006)
        if count > best_count:
            best_plane, best_count = coef, count
    if best_plane is None:
        raise RuntimeError("No near-horizontal plane supported by depth")
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    for _ in range(4):
        mask = abs(design @ best_plane - points[:, 2]) < .006
        best_plane = np.linalg.lstsq(design[mask], points[mask, 2], rcond=None)[0]
    residual = abs(design @ best_plane - points[:, 2])
    inlier = residual < .006
    return best_plane, dict(points=len(points), inliers=int(inlier.sum()),
                            inlier_fraction=float(inlier.mean()), coef_z_xy1=best_plane.tolist(),
                            residual_p95_mm_inliers=float(np.quantile(residual[inlier], .95) * 1000))


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)).tolist()


def run(dataset, source_report, out):
    source = json.loads(source_report.read_bytes())
    sequence = dataset / source["sequence"]
    poses = verify_file_pin(source["source_poses"])
    asset = verify_file_pin(source["geometry"]["source"])
    mesh = trimesh.load_mesh(asset, process=False)
    with np.load(poses, allow_pickle=False) as record:
        objects = record["world_from_carton"]
        hands = record["world_right_hand21"]
        frames = record["frames"]
    assert np.array_equal(frames, np.arange(141))
    meta = yaml.safe_load((sequence / "meta.yaml").read_text())
    extrinsic_file = dataset / "calibration/extrinsics" / meta["extrinsics"]
    extrinsics = yaml.safe_load(extrinsic_file.read_text())["extrinsics"]
    def matrix(key):
        return np.vstack((np.asarray(extrinsics[key]).reshape((3, 4)), [0, 0, 0, 1]))

    per_frame, camera_medians, coefficients, pins = [], [], [], [file_pin(extrinsic_file)]
    for serial in meta["realsense"]["serials"]:
        camera_fits = []
        intrinsic_file = dataset / "calibration/intrinsics" / f"{serial}.yaml"
        intrinsics = yaml.safe_load(intrinsic_file.read_text())["color"]
        pins.append(file_pin(intrinsic_file))
        transform = np.linalg.solve(matrix("tag_1"), matrix(serial))
        for frame in DEPTH_FRAMES:
            path = sequence / serial / f"depth_{frame:06d}.png"
            pins.append(file_pin(path))
            points = backproject(path, intrinsics, transform)
            if len(points) < 1000:
                per_frame.append(dict(camera=serial, frame=frame, points=len(points), used=False,
                                      reason="fewer than 1000 subsampled points in tabletop crop"))
                continue
            plane, stats = fit_table(points)
            # Require distributed support, not a wall patch or an object top at one location.
            mask = abs(points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])) < .006
            xy_span = np.ptp(np.quantile(points[mask, :2], [.05, .95], axis=0), axis=0)
            use = stats["inliers"] >= 800 and np.min(xy_span) > .18
            if use:
                coefficients.append(plane)
                camera_fits.append(plane)
            per_frame.append(dict(camera=serial, frame=frame, used=bool(use),
                                  central90_xy_span_m=xy_span.tolist(), **stats))
        if camera_fits:
            camera_medians.append((serial, np.median(camera_fits, axis=0)))

    # Equal weight per camera and frame prevents large pixel coverage dominating the result.
    if len(camera_medians) < 2:
        raise RuntimeError("Fewer than two independent views support a table estimate")
    view_planes = np.array([p for _, p in camera_medians])
    coef = np.median(view_planes, axis=0)
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    plane_point = np.array([0.0, 0.0, coef[2]])
    vertices = mesh.vertices @ objects[:, :3, :3].transpose(0, 2, 1) + objects[:, None, :3, 3]
    gaps = ((vertices - plane_point) @ normal).min(axis=1)
    tilt = np.rad2deg(np.arccos(np.clip(objects[:, 2, 1], -1, 1)))
    heights = np.column_stack((objects[:, :2, 3], np.ones(len(objects)))) @ np.array(coefficients).T
    initial_heights = np.r_[objects[0, :2, 3], 1] @ view_planes.T

    query = SurfaceQuery(mesh.vertices, mesh.faces)
    local = (hands - objects[:, None, :3, 3]) @ objects[:, :3, :3]
    distances = np.stack([query.nearest(row)[1] for row in local[:, [4, 8, 12, 16, 20], :]])
    out.mkdir(parents=True, exist_ok=False)
    signals = out / "source_support.npz"
    np.savez_compressed(signals, frames=frames, nominal_time_s=frames / 30.0,
                        mesh_bottom_plane_gap_m=gaps, fingertip_joint_to_scan_m=distances,
                        local_y_to_world_z_deg=tilt, source_world_normal=normal, source_plane_point=plane_point)
    write_json(out / "report.json", dict(
        sequence=source["sequence"], source=file_pin(source_report), raw_inputs=pins, code=file_pin(__file__),
        sources=dict(depth=TOOLKIT + "/hocap_toolkit/loaders/sequence_loader.py",
                     timing=ANNOTATION + "/tools/01_video_segmentation.py#L253",
                     joints=TOOLKIT + "/hocap_toolkit/utils/mano_info.py",
                     dataset="https://irvlutd.github.io/HOCap/"),
        timing=dict(nominal_hz=30, last_sample_s=140/30, source_sensor_timestamps_available=False,
                    basis="Author annotation export uses fps=30; sequence archive supplies frame indices only."),
        table=dict(equation="n dot (p - plane_point) = 0", normal_world=normal.tolist(),
                   plane_point_world_m=plane_point.tolist(), coefficients_z_from_xy1=coef.tolist(),
                   normal_vs_world_z_deg=float(np.rad2deg(np.arccos(normal[2]))),
                   gravity_scope="Use -world Z provisionally; tabletop normal does not measure the gravity vector.",
                   mesh_bottom_signed_gap_at_key_frames_mm={str(i):float(gaps[i]*1000) for i in (0, 30, 40, 60, 80, 105, 110, 120, 140)},
                   max_mesh_bottom_clearance_mm=float(gaps.max()*1000),
                   init_plane_z_by_camera_mm={k:float(z*1000) for (k, _),z in zip(camera_medians, initial_heights, strict=True)},
                   max_between_fit_z_span_at_object_xy_mm=float(np.max(np.ptp(heights,axis=1))*1000),
                   plane_fits=per_frame),
        geometry=dict(positive_local_y_vs_world_z_deg_at_frame0=float(tilt[0]),
                      positive_local_y_vs_world_z_deg_range=[float(tilt.min()),float(tilt.max())]),
        events=dict(
            support_clearance_thresholds_mm={str(mm):spans(gaps > mm/1000) for mm in (5, 10, 20, 50)},
            fingertip_joint_near_thresholds_mm={str(mm):spans(distances.min(axis=1)<mm/1000) for mm in (5, 10, 15)},
            contact_scope="Unsigned tip-joint proximity measures geometry, not skin contact, pressure, or release force."),
        signals=file_pin(signals),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.dataset.resolve(), args.source_report.resolve(), args.output.resolve())
