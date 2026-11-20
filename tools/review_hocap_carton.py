"""Extract a markerless HO-Cap pick-and-place window and check its reference frames."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


SEQUENCE = Path("subject_1/20231025_170650")
OBJECT_ID = "G04_2"
CAMERA = "105322251564"
FRAME_COUNT = 141
SOURCE = "https://irvlutd.github.io/HOCap/"


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def transforms(poses):
    result = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(poses[:, :4]).as_matrix()
    result[:, :3, 3] = poses[:, 4:]
    return result


def check_geometry(mesh_path, out):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    convex = mesh.convex_hull
    obj = out / "carton_collision.obj"
    convex.export(obj)
    xml = out / "carton_geometry.xml"
    xml.write_text(
        '<mujoco model="hocap_G04_2"><asset><mesh name="carton" file="carton_collision.obj"/>'
        '</asset><worldbody><geom name="collision" type="mesh" mesh="carton"/></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = model.geom("collision").id
    world_mesh = model.mesh_vert @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    np.testing.assert_allclose(
        [world_mesh.min(axis=0), world_mesh.max(axis=0)], mesh.bounds, rtol=0, atol=1e-6
    )
    return mesh, dict(
        source=file_pin(mesh_path), hull=file_pin(obj), static_mjcf=file_pin(xml),
        bounds_m=mesh.bounds.tolist(), size_mm=(mesh.extents * 1000).tolist(),
        source_vertices=len(mesh.vertices), geometry_only=True, dynamics_calibrated=False,
    )


def camera_projection(points, camera_to_world, intrinsics):
    local = (points - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    if np.any(local[:, 2] <= 0):
        raise ValueError("Selected overlay contains a point behind the HoloLens camera")
    return [
        (float(x / z * intrinsics["fx"] + intrinsics["ppx"]),
         float(y / z * intrinsics["fy"] + intrinsics["ppy"]))
        for x, y, z in local
    ]


def make_preview(images, objects, cameras, joints, mesh, intrinsics, path):
    frames = [0, 40, 80, 110, 140]
    panel = Image.new("RGB", (275 * len(frames), 338), "#17232c")
    caption = ImageDraw.Draw(panel)
    # Project the scan's convex outline and the source joints on a second copy.
    for col, frame in enumerate(frames):
        with Image.open(images[frame]) as rgb:
            raw = rgb.convert("RGB")
        overlay = raw.copy()
        draw = ImageDraw.Draw(overlay)
        pose = objects[frame]
        vertices = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
        pixels = np.array(camera_projection(vertices, cameras[frame], intrinsics))
        outline = pixels[ConvexHull(pixels).vertices].tolist()
        draw.line([tuple(p) for p in outline + outline[:1]], fill="#f5cd4b", width=5)
        for x, y in camera_projection(joints[frame], cameras[frame], intrinsics):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#36e4eb")
        caption.text((275 * col + 5, 4), f"HO-Cap   |   frame {frame}", fill="white")
        panel.paste(raw.resize((275, 155), Image.Resampling.LANCZOS), (275 * col, 20))
        panel.paste(overlay.resize((275, 155), Image.Resampling.LANCZOS), (275 * col, 180))
    panel.save(path)


def review(dataset, out):
    out.mkdir(parents=True, exist_ok=False)
    seq = dataset / SEQUENCE
    meta = load_yaml(seq / "meta.yaml")
    if meta["mano_sides"] != ["right"]:
        raise ValueError("Expected right-hand-only annotations for the selected sequence")
    obj_index = meta["object_ids"].index(OBJECT_ID)
    source_o = np.load(seq / "poses_o.npy", allow_pickle=False)
    source_m = np.load(seq / "poses_m.npy", allow_pickle=False)
    source_pv = np.load(seq / "poses_pv.npy", allow_pickle=False)
    objects = transforms(source_o[obj_index, :FRAME_COUNT])
    cameras = transforms(source_pv[:FRAME_COUNT])
    hand_mano = source_m[0, :FRAME_COUNT]

    calib = dataset / "calibration"
    ext_path = calib / "extrinsics" / meta["extrinsics"]
    ext = load_yaml(ext_path)["extrinsics"]

    def to_matrix(serial):
        return np.vstack((np.asarray(ext[serial]).reshape(3, 4), [0, 0, 0, 1]))

    world_from_rs = np.linalg.inv(to_matrix("tag_1")) @ to_matrix(CAMERA)
    pv = meta["hololens"]["serial"]
    pv_calib = calib / "intrinsics" / f"{pv}.yaml"
    intrinsics = load_yaml(pv_calib)["color"]
    joints, dup_objects, images, members = [], [], [], []
    for frame in range(FRAME_COUNT):
        rgb = seq / pv / f"color_{frame:06d}.jpg"
        label = seq / CAMERA / f"label_{frame:06d}.npz"
        with Image.open(rgb) as picture:
            picture.load()
            assert picture.size == (intrinsics["width"], intrinsics["height"])
        with np.load(label, allow_pickle=False) as ref:
            assert list(ref["obj_class_names"]) == [*meta["object_ids"], "RIGHT_HAND"]
            points = ref["hand_joints_3d"][0]
            assert points.shape == (21, 3) and (points[:, 2] > 0).all()
            joints.append(points @ world_from_rs[:3, :3].T + world_from_rs[:3, 3])
            dup_objects.append(world_from_rs @ ref["obj_poses"][obj_index])
        images.append(rgb)
        members.append(dict(frame=frame, ego_rgb=file_pin(rgb), camera_label=file_pin(label)))
    joints, dup_objects = np.stack(joints), np.stack(dup_objects)
    for array in (objects, cameras, joints, hand_mano):
        assert np.isfinite(array).all()
    np.testing.assert_allclose(dup_objects, objects, rtol=0, atol=1e-6)
    mesh, geometry = check_geometry(dataset / "models" / OBJECT_ID / "cleaned_mesh_10000.obj", out)
    array_file = out / "source_carton.npz"
    np.savez_compressed(
        array_file, frames=np.arange(FRAME_COUNT), world_from_carton=objects,
        world_from_egocamera=cameras, world_right_hand21=joints, right_hand_mano=hand_mano,
    )
    preview = out / "source_carton_rgb.jpg"
    make_preview(images, objects, cameras, joints, mesh, intrinsics, preview)
    inputs = [seq / name for name in ("meta.yaml", "poses_m.npy", "poses_o.npy", "poses_pv.npy")]
    inputs.extend([ext_path, pv_calib, calib / "mano/subject_1.yaml"])
    motion = np.linalg.norm(objects[:, :3, 3] - objects[0, :3, 3], axis=1) * 1000
    write_json(out / "input_frames.json", members)
    write_json(out / "report.json", dict(
        dataset_source=SOURCE, author="Wang et al., HO-Cap (NeurIPS 2025)", license="CC BY 4.0",
        archives="https://github.com/IRVLUTD/HO-Cap/blob/main/config/hocap_recordings.yaml",
        tool=file_pin(__file__), sequence=SEQUENCE.as_posix(), frames=[0, FRAME_COUNT - 1],
        object_id=OBJECT_ID, side="right", source_inputs=[file_pin(p) for p in inputs],
        input_frames=file_pin(out / "input_frames.json"), object_start_to_max_mm=float(motion.max()),
        object_start_to_last_mm=float(motion[-1]), frame_of_max_displacement=int(motion.argmax()),
        max_world_object_cross_export_m=float(np.linalg.norm(
            objects[:, :3, 3] - dup_objects[:, :3, 3], axis=1).max()),
        geometry=geometry, source_poses=file_pin(array_file), preview=file_pin(preview),
        capture_time_scope="Frame indices preserved. Sensor timestamps not supplied by these selected members.",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.dataset.resolve(), args.output.resolve())
