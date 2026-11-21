"""Export a right-hand HO-Cap development window in the original shared world frame."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from review_hocap_carton import CAMERA, transforms
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def static_collision(mesh, object_id, output):
    hull = output / "carton_collision.obj"
    mesh.convex_hull.export(hull)
    scene = output / "carton_geometry.xml"
    scene.write_text(
        f'<mujoco model="{object_id}"><asset><mesh name="carton" file="{hull.name}"/></asset>'
        '<worldbody><geom name="collision" type="mesh" mesh="carton"/></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    converted = model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
    np.testing.assert_allclose(mesh.bounds, [converted.min(0), converted.max(0)], atol=1e-6, rtol=0)
    return dict(hull=file_pin(hull), static_mjcf=file_pin(scene), size_mm=(mesh.extents * 1000).tolist(),
                geometry_only=True, dynamics_calibrated=False)


def box_mask_in_scan_outline(mesh_vertices, camera_pose, intrinsics, mask):
    in_camera = mesh_vertices @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    if np.any(in_camera[:, 2] <= 0):
        raise ValueError("Object extends behind the fixed audit camera")
    pixel = in_camera @ intrinsics.T
    pixel = pixel[:, :2] / pixel[:, 2, None]
    hull = pixel[ConvexHull(pixel).vertices]
    canvas = Image.new("1", (mask.shape[1], mask.shape[0]))
    ImageDraw.Draw(canvas).polygon([tuple(x) for x in hull], fill=1)
    return np.count_nonzero(np.asarray(canvas) & mask)


def stage(dataset, sequence, object_id, first, last, output):
    seq = dataset / sequence
    meta_path = seq / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    if meta["task_id"] != 1 or meta["mano_sides"] != ["right"] or not (0 <= first < last < meta["num_frames"]):
        raise ValueError("Expected a right-hand pick-and-place reference with valid frame bounds")
    obj_idx = meta["object_ids"].index(object_id)
    pv = meta["hololens"]["serial"]
    frames = np.arange(first, last + 1)
    pose_paths = [seq / f"poses_{k}.npy" for k in ("o", "m", "pv")]
    obj, mano, camera = (np.load(p, allow_pickle=False) for p in pose_paths)
    world_obj = transforms(obj[obj_idx, frames])
    mesh_path = dataset / "models" / object_id / "cleaned_mesh_10000.obj"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = mesh.convex_hull.vertices
    cal = dataset / "calibration"
    ext_path = cal / "extrinsics" / meta["extrinsics"]
    exts = yaml.safe_load(ext_path.read_text())["extrinsics"]
    def ext(k):
        return np.vstack((np.asarray(exts[k]).reshape(3, 4), [0, 0, 0, 1]))

    world_cam = np.linalg.solve(ext("tag_1"), ext(CAMERA))
    hand, mask_pixels, masked_overlap, hand_pixels, file_rows = [], [], [], [], []
    for i, ref in zip(frames, world_obj, strict=True):
        rgb = seq / pv / f"color_{i:06d}.jpg"
        label = seq / CAMERA / f"label_{i:06d}.npz"
        with Image.open(rgb) as image:
            image.load()
            assert image.size == (meta["hololens"]["pv_width"], meta["hololens"]["pv_height"])
        with np.load(label, allow_pickle=False) as anno:
            assert list(anno["obj_class_names"]) == [*meta["object_ids"], "RIGHT_HAND"]
            points = anno["hand_joints_3d"][0]
            assert (points[:, 2] > 0).all() and points.shape == (21, 3)
            hand.append(points @ world_cam[:3, :3].T + world_cam[:3, 3])
            obj_cam = anno["obj_poses"][obj_idx]
            np.testing.assert_allclose(world_cam @ obj_cam, ref, atol=1e-6, rtol=0)
            mask = anno["seg_mask"] == obj_idx + 1
            mask_pixels.append(np.count_nonzero(mask))
            hand_pixels.append(np.count_nonzero(anno["seg_mask"] == len(meta["object_ids"]) + 1))
            masked_overlap.append(box_mask_in_scan_outline(vertices, obj_cam, anno["cam_K"], mask))
        file_rows.append(dict(frame=int(i), ego_rgb=file_pin(rgb), camera_label=file_pin(label)))
    hand, mask_pixels, hand_pixels = np.asarray(hand), np.asarray(mask_pixels), np.asarray(hand_pixels)
    overlap = np.divide(masked_overlap, mask_pixels, out=np.full(len(frames), np.nan), where=mask_pixels>0)
    output.mkdir(parents=True, exist_ok=False)
    geometry = static_collision(mesh, object_id, output)
    geometry["source"] = file_pin(mesh_path)
    src = output / "source_carton.npz"
    np.savez_compressed(src, frames=frames, world_from_carton=world_obj, world_right_hand21=hand,
                        right_hand_mano=mano[0,frames], world_from_egocamera=transforms(camera[frames]))
    mask_evidence = output / "mask_checks.npz"
    np.savez_compressed(mask_evidence, frames=frames, box_visible_pixels=mask_pixels,
                        right_hand_visible_pixels=hand_pixels, box_mask_inside_outline_fraction=overlap)
    write_json(output / "input_frames.json", file_rows)
    movement = np.linalg.norm(world_obj[:, :3, 3] - world_obj[0, :3, 3], axis=1)
    rot = Rotation.from_matrix(world_obj[:, :3, :3])
    rotation_deg = np.rad2deg((rot[0].inv()*rot).magnitude())
    z_alignment = world_obj[0, 2, :3]
    most_vertical = np.argmax(abs(z_alignment))
    sign = int(np.sign(z_alignment[most_vertical]))
    angle = np.rad2deg(np.arccos(np.clip(sign*world_obj[:, 2, most_vertical], -1, 1)))
    others = np.delete(obj[:, frames, 4:], obj_idx, axis=0)
    cal_paths = [ext_path, cal / "intrinsics" / f"{pv}.yaml", cal / "intrinsics" / f"{CAMERA}.yaml",
                 cal / "mano" / f"{meta['subject_id']}.yaml"]
    write_json(output / "report.json", dict(
        role="development reference; eligible for training and tuning", sequence=sequence,
        frames=[first,last], source_frame_count=len(frames), object_id=object_id, side="right",
        source_project="https://irvlutd.github.io/HOCap/", license="CC BY 4.0",
        code=file_pin(__file__), source_inputs=[file_pin(p) for p in [meta_path,*pose_paths,*cal_paths]],
        source_poses=file_pin(src), input_frames=file_pin(output/"input_frames.json"), geometry=geometry,
        mask_checks=file_pin(mask_evidence), box_visible_frames=int(np.sum(mask_pixels>0)),
        hand_visible_frames=int(np.sum(hand_pixels>0)), box_mask_pixels_min=int(mask_pixels.min()),
        scan_outline_coverage_visible_box_mask_min=float(np.nanmin(overlap)),
        max_excursion_mm=float(1000*movement.max()), peak_rotation_deg=float(rotation_deg.max()),
        max_other_object_displacement_mm=float(1000*np.linalg.norm(others-others[:,:1,:],axis=-1).max()),
        initial_most_world_vertical_local_axis=("+" if sign>0 else "-")+"XYZ"[most_vertical],
        initial_rotation_world_from_object=world_obj[0,:3,:3].tolist(),
        that_axis_tilt_degrees=[float(angle.min()),float(angle.max())],
        orientation_policy="Original mesh frame, full six-DOF reference, no object-axis renormalization. Dominant initial vertical axis describes pose, not a verified label for the printed top.",
        timing="Absolute published frame IDs. 30 Hz nominal for HO-Cap author playback; no sensor timestamps in selected files.",
    ))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("object_id")
    parser.add_argument("first", type=int)
    parser.add_argument("last", type=int)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    stage(args.dataset.resolve(), args.sequence, args.object_id, args.first, args.last, args.output.resolve())
