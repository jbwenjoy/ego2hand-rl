"""Compare saved physical states to the source, using the moving HO-Cap ego camera."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


WIDTH = 640
HEIGHT = 360
GOLD = "#ffd76b"
MEASURE = "#e7eef4"


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def font(size):
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def use_source_camera(scene, pose, lens):
    """Set both GL eyes to the monoscopic calibrated OpenCV camera."""
    near = 0.005
    for camera in scene.camera:
        camera.pos[:] = pose[:3, 3]
        camera.forward[:] = pose[:3, 2]
        camera.up[:] = -pose[:3, 1]
        camera.frustum_near = near
        camera.frustum_far = 10
        camera.frustum_center = (0.5 * lens["width"] - lens["ppx"]) * near / lens["fx"]
        camera.frustum_width = 0.5 * lens["width"] * near / lens["fx"]
        camera.frustum_top = lens["ppy"] * near / lens["fy"]
        camera.frustum_bottom = (lens["ppy"] - lens["height"]) * near / lens["fy"]


def source_outline(vertices, world_from_object, world_from_camera, lens):
    world = vertices @ world_from_object[:3, :3].T + world_from_object[:3, 3]
    cam = (world - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    if np.any(cam[:, 2] <= 0):
        raise ValueError("Reference carton extends behind the camera near plane")
    x = (lens["fx"] * cam[:, 0] / cam[:, 2] + lens["ppx"]) * WIDTH / lens["width"]
    y = (lens["fy"] * cam[:, 1] / cam[:, 2] + lens["ppy"]) * HEIGHT / lens["height"]
    pixels = np.column_stack((x, y))
    hull = pixels[ConvexHull(pixels).vertices]
    return [tuple(p) for p in np.vstack((hull, hull[0]))]


def audit_pose(model, data, body, trace_pose):
    pose = np.eye(4)
    pose[:3, 3] = data.xpos[body]
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    np.testing.assert_allclose(pose, trace_pose, rtol=0, atol=1e-6,
                               err_msg=f"Frame state disagrees with saved {model.body(body).name} pose")


def contact_counts(model, data, carton, palm):
    # Count geometric contacts active at the displayed state, without inferring a force.
    hand = {palm}
    for body in range(palm + 1, model.nbody):
        if model.body_parentid[body] in hand:
            hand.add(body)
    hand_points = support_points = 0
    for contact in data.contact:
        if contact.efc_address < 0:
            continue
        body1, body2 = model.geom_bodyid[contact.geom]
        if carton not in (body1, body2):
            continue
        other = body2 if body1 == carton else body1
        if other in hand:
            hand_points += 1
        elif model.body_weldid[other] == 0:
            support_points += 1
    return hand_points, support_points


def measure_errors(source, actual):
    delta_mm = 1000 * np.linalg.norm(source[:, :3, 3] - actual[:, :3, 3], axis=1)
    relative = source[:, :3, :3].transpose(0, 2, 1) @ actual[:, :3, :3]
    angle_deg = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    return delta_mm, angle_deg


def render(source_dir, run_dir, output, fps):
    source_report = read_json(source_dir / "report.json")
    frame_rows = read_json(verify_file_pin(source_report["input_frames"]))
    with np.load(verify_file_pin(source_report["source_poses"]), allow_pickle=False) as source:
        frame_ids = source["frames"]
        boxes = source["world_from_carton"]
        cameras = source["world_from_egocamera"]
    lens_path = next(p for p in source_report["source_inputs"] if "intrinsics/hololens" in p["path"].replace("\\", "/"))
    lens = yaml.safe_load(verify_file_pin(lens_path).read_text("utf-8"))["color"]
    if lens["width"] * HEIGHT != lens["height"] * WIDTH:
        raise ValueError("Source and rendered frame aspect ratios differ")
    scan = trimesh.load_mesh(verify_file_pin(source_report["geometry"]["hull"]), process=False)
    vertices = np.asarray(scan.vertices)
    rgb_paths = [verify_file_pin(row["ego_rgb"]) for row in frame_rows]
    np.testing.assert_array_equal(frame_ids, [row["frame"] for row in frame_rows])

    run_report = read_json(run_dir / "report.json")
    model_path = verify_file_pin(run_report["mjb"])
    trace_path = verify_file_pin(run_report["trace"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = trace["sim_qpos"]
        np.testing.assert_allclose(boxes, trace["world_from_carton_source"], atol=1e-8, rtol=0)
        physical_boxes = trace["world_from_carton_sim"]
    if qpos.shape != (len(frame_ids), model.nq) or physical_boxes.shape != boxes.shape:
        raise ValueError("Need exactly one saved full physical state for each source frame")
    delta_mm, angle_deg = measure_errors(boxes, physical_boxes)
    model.vis.global_.offwidth = max(WIDTH, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(HEIGHT, model.vis.global_.offheight)
    model.vis.headlight.ambient[:] = 0.52
    model.vis.headlight.diffuse[:] = 0.78
    model.vis.headlight.specular[:] = 0.18
    carton = model.body("carton").id
    palm = model.body("right_palm_link").id
    model.geom_rgba[model.geom_bodyid == carton] = [0.12, 0.55, 0.87, 1.0]
    data = mujoco.MjData(model)
    output.mkdir(parents=True, exist_ok=False)
    video = output / "source_vs_physics.mp4"
    sheet_frames = [0, 40, int(source_report["frame_of_max_displacement"]), 110, len(frame_ids) - 1]
    counts = []
    title, small = font(21), font(17)
    with mujoco.Renderer(model, width=WIDTH, height=HEIGHT) as renderer, imageio.get_writer(
        video, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"],
    ) as writer:
        for i, src_id in enumerate(frame_ids):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            audit_pose(model, data, carton, physical_boxes[i])
            renderer.update_scene(data)
            use_source_camera(renderer.scene, cameras[i], lens)
            physics = Image.fromarray(renderer.render().copy())
            contour = source_outline(vertices, boxes[i], cameras[i], lens)
            ImageDraw.Draw(physics).line(contour, fill=GOLD, width=2)
            n_hand, n_support = contact_counts(model, data, carton, palm)
            counts.append([n_hand, n_support])
            canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + 112), "#12222c")
            with Image.open(rgb_paths[i]) as rgb:
                canvas.paste(rgb.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS), (0, 40))
            canvas.paste(physics, (WIDTH, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 5), "HO-Cap  |  第一视角原片", font=title, fill=MEASURE)
            draw.text((WIDTH + 12, 5), "Wuji Hand 2  |  自由纸盒动力学回放", font=title, fill=MEASURE)
            draw.text((12, 408), f"原片帧 {src_id:03d} / {frame_ids[-1]}     按 {fps:g} 帧/秒回放", font=small, fill=MEASURE)
            draw.text((12, 440), "金色轮廓为源纸盒姿态，蓝色实体为仿真纸盒。", font=small, fill=GOLD)
            draw.text((WIDTH + 12, 408), f"物体位置误差 {delta_mm[i]:.1f} mm     旋转误差 {angle_deg[i]:.1f}°", font=small, fill=MEASURE)
            draw.text((WIDTH + 12, 440), f"当前接触点数     纸盒与手 {n_hand}     纸盒与固定支撑 {n_support}", font=small, fill=MEASURE)
            writer.append_data(np.asarray(canvas))

    # Each column groups one source frame with its decoded simulated counterpart.
    cell_width, cell_height = 300, 169
    sheet = Image.new("RGB", (len(sheet_frames) * cell_width, cell_height * 2 + 77), "#12222c")
    captions = ImageDraw.Draw(sheet)
    with imageio.get_reader(video) as reader:
        if reader.count_frames() != len(frame_ids) or abs(reader.get_meta_data()["fps"] - fps) > 0.001:
            raise ValueError("Frame count or playback frequency changed during encoding")
        for index, frame in enumerate(sheet_frames):
            decoded = Image.fromarray(reader.get_data(frame))
            x = index * cell_width
            captions.text((x + 8, 3), f"原片   |   第 {frame} 帧", font=small, fill=MEASURE)
            for col, y in [(0, 31), (1, 34 + cell_height)]:
                view = decoded.crop((WIDTH * col, 40, WIDTH * (col + 1), 40 + HEIGHT))
                sheet.paste(view.resize((cell_width, cell_height), Image.Resampling.LANCZOS), (x, y))
            captions.text((x + 8, 41 + 2 * cell_height),
                          f"仿真误差  {delta_mm[frame]:.0f} mm  /  {angle_deg[frame]:.0f}°",
                          font=small, fill=GOLD)
    preview = output / "source_vs_physics.jpg"
    sheet.save(preview, quality=94)
    write_json(output / "report.json", dict(
        source=file_pin(source_dir / "report.json"), trace=file_pin(trace_path), model=file_pin(model_path),
        renderer=file_pin(__file__), source_frames=frame_ids.tolist(), preview_frames=sheet_frames,
        playback_fps=fps, position_error_mm=delta_mm.tolist(), rotation_error_deg=angle_deg.tolist(),
        hand_and_fixed_support_active_contact_counts=counts,
        camera="Per-frame HO-Cap world_from_egocamera and pinhole intrinsics. Same world frame as rollout.",
        video=file_pin(video), preview=file_pin(preview),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    render(args.source.resolve(), args.run.resolve(), args.output.resolve(), args.fps)
