"""Render the recorded robot and free object alongside the synchronized HoloLens frames on CPU."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
import yaml
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from render_hocap_carton import WIDTH, HEIGHT, GOLD, MEASURE, source_outline, use_source_camera


def read(path):
    return json.loads(path.read_bytes())


def verify(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"File changed: {path}")
    return path


def render(job, bundle, out):
    study = job.parents[2]
    ledger = read(job)
    if ledger["kind"] != "evaluate" or not ledger["success"]:
        raise ValueError("An archived evaluation attempt is required")
    score_path = verify(Path(ledger["receipt"]["path"]), ledger["receipt"]["sha256"])
    score = read(score_path)
    actor, = (r for r in score["realizations"] if r["id"] == "actor")
    trace = verify(score_path.parent / "raw/trace.npz", actor["trace_sha256"])
    source_dir = study / "source_jobs" / f"{ledger['subject']}_prepare/attempt_001/source"
    source_path = verify(source_dir / "source.json", score["identity"]["raw_source_report_sha256"])
    source = read(source_path)
    source_data = verify(source_dir / "source_carton.npz", source["source_poses"]["sha256"])
    ik = read(source_dir / "direct_ik.json")
    model_path = verify(source_dir / "kinematics.mjb", ik["mjb"]["sha256"])
    mesh = trimesh.load_mesh(verify(source_dir / "carton_collision.obj", source["geometry"]["hull"]["sha256"]), process=False)
    reservation = verify(bundle.parent / "reservation.json", source["source_reservation"]["sha256"])
    verify(bundle, read(reservation)["source_bundle"]["sha256"])
    with np.load(source_data, allow_pickle=False) as raw:
        frames, objects, cameras = (raw[k] for k in ("frames", "world_from_carton", "world_from_egocamera"))
    with np.load(trace, allow_pickle=False) as raw:
        joints, bodies = raw["joint_q"], raw["body_q"]
        np.testing.assert_array_equal(raw["source_frames"], frames[:len(joints)])
        np.testing.assert_array_equal(raw["raw_object_pose"], objects)
    model = mujoco.MjModel.from_binary_path(str(model_path))
    mapping = read(study.parent / "hocap_subject1_learned_native_visual_v1/body_mapping.json")
    ids = np.flatnonzero(np.linalg.norm(bodies[0, :, :3] - objects[0, :3, 3], axis=1) < 2e-6)
    candidates = [i for i in ids if (Rotation.from_quat(bodies[0,i,3:]).inv() *
                  Rotation.from_matrix(objects[0,:3,:3])).magnitude() < 1e-5]
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely identify the recorded object")
    body = candidates[0]
    verified = [(i, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
                for i, name in enumerate(mapping["body_labels"])]
    verified = [(i, j) for i, j in verified if j >= 0] + [(body, model.body("carton").id)]
    traces, native = np.asarray(verified).T
    data = mujoco.MjData(model)
    delta = np.linalg.norm(bodies[:,body,:3] - objects[:len(joints),:3,3], axis=1) * 1000
    angle = np.rad2deg((Rotation.from_quat(bodies[:,body,3:]).inv() *
                       Rotation.from_matrix(objects[:len(joints),:3,:3])).magnitude())
    model.geom_rgba[model.geom_bodyid == model.body("carton").id] = [.12, .55, .87, 1.]
    model.vis.headlight.ambient[:] = .52
    model.vis.headlight.diffuse[:] = .78
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    out.mkdir(parents=True, exist_ok=True)
    video = out / "ego_vs_policy.mp4"
    typeface = "/mnt/c/Windows/Fonts/msyh.ttc"
    title, caption = ImageFont.truetype(typeface, 20), ImageFont.truetype(typeface, 17)
    max_position_m = max_rotation_deg = 0.
    with ZipFile(bundle) as archive, mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        gpu = GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in gpu.lower():
            raise RuntimeError(f"A CPU renderer is required alongside the serial 5090 study: {gpu}")
        seq = source["sequence"]
        serial = yaml.safe_load(archive.read(seq + "/meta.yaml"))["hololens"]["serial"]
        lens = yaml.safe_load(archive.read(f"calibration/intrinsics/{serial}.yaml"))["color"]
        if lens["width"]*HEIGHT != lens["height"]*WIDTH:
            raise ValueError("Mismatched source and renderer aspect ratio")
        with imageio.get_writer(video, fps=30., codec="libx264", pixelformat="yuv420p", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error",
                                output_params=["-threads","1","-movflags","+faststart"]) as writer:
            for i, frame in enumerate(frames[:len(joints)]):
                # The kinematic reconstruction is audited against every visual hand link below.
                hand = joints[i]
                box = bodies[i,body]
                data.qpos[:3], data.qpos[3:7] = hand[:3], hand[[6,3,4,5]]
                for name, idx in mapping["finger_q_indices"].items():
                    data.qpos[model.joint(name).qposadr[0]] = hand[idx]
                data.qpos[-7:-4], data.qpos[-4:] = box[:3], box[[6,3,4,5]]
                mujoco.mj_forward(model, data)
                pos = np.max(np.linalg.norm(data.xpos[native] - bodies[i,traces,:3], axis=1))
                rot = np.rad2deg((Rotation.from_matrix(data.xmat[native].reshape(-1,3,3)).inv() *
                                  Rotation.from_quat(bodies[i,traces,3:])).magnitude()).max()
                if pos > 1.e-5 or rot > .001:
                    raise ValueError("Rendered bodies disagree with the recorded physical state")
                max_position_m, max_rotation_deg = max(max_position_m, pos), max(max_rotation_deg, rot)
                renderer.update_scene(data)
                use_source_camera(renderer.scene, cameras[i], lens)
                physical = Image.fromarray(renderer.render())
                outline = source_outline(mesh.vertices, objects[i], cameras[i], lens)
                ImageDraw.Draw(physical).line(outline, fill=GOLD, width=2)
                rgb = Image.open(io.BytesIO(archive.read(f"{seq}/{serial}/color_{frame:06}.jpg"))).convert("RGB")
                canvas = Image.new("RGB", (WIDTH*2, HEIGHT+98), "#14232c")
                canvas.paste(rgb.resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS),(0,36))
                canvas.paste(physical,(WIDTH,36))
                draw = ImageDraw.Draw(canvas)
                draw.text((12,4), "HO-Cap 第一视角原片", font=title, fill=MEASURE)
                draw.text((WIDTH+12,4), f"Wuji 手物理回放   ·   {ledger['steps']:,} 训练步", font=title, fill=MEASURE)
                draw.text((12,HEIGHT+41), f"源帧 {frame} / {frames[-1]}     {(frame-frames[0])/30:.2f} 秒",
                          font=caption, fill=MEASURE)
                draw.text((WIDTH+12,HEIGHT+41), f"物体位置误差 {delta[i]:.1f} mm     姿态误差 {angle[i]:.1f}°",
                          font=caption, fill=MEASURE)
                draw.text((WIDTH+12,HEIGHT+68), "蓝色为实际物体，黄线为同帧源物体的轮廓", font=caption, fill=GOLD)
                writer.append_data(np.asarray(canvas))
    with imageio.get_reader(video) as decoder:
        if decoder.count_frames() != len(joints) or abs(decoder.get_meta_data()["fps"]-30.) > .001:
            raise ValueError("Encoded film changed the temporal alignment")
        worst = int(np.argmax(angle))
        Image.fromarray(decoder.get_data(worst)).save(out / "maximum_rotation_error.jpg", quality=93)
    record = dict(subject=ledger["subject"], method=ledger["method"], steps=ledger["steps"], status=score["status"],
                  source=source_path.as_posix(), source_sha256=score["identity"]["raw_source_report_sha256"],
                  score=ledger["receipt"], trace_sha256=actor["trace_sha256"], original_frames=source["frames"],
                  encoded_frames=len(joints), fps=30, max_rotation_source_frame=int(frames[worst]),
                  max_rotation_error_deg=float(angle[worst]), max_render_body_origin_error_mm=max_position_m*1000,
                  max_render_body_angle_error_deg=max_rotation_deg, verified_body_count=len(native),
                  software_renderer=gpu, video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (out / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    render(args.job.resolve(), args.bundle.resolve(), args.out.resolve())
