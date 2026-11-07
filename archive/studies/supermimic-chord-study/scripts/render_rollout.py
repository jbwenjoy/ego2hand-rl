"""Headless native-MuJoCo rendering of an already recorded physical trajectory.

Only logged joint positions and object poses are assigned to MjData. Kinematics
updates the drawing state; mj_step/mj_forward are never called. The fixed wrist
is checked against the log. No policy, dynamics replay, GUI or asset changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=ROOT / "outputs" / "a0_clip001_3s_seed0")
    parser.add_argument("--overwrite", action="store_true", help="replace only prior renderer outputs")
    parser.add_argument("--label", default="A0")
    parser.add_argument("--output-stem",default="rollout_a0")
    parser.add_argument("--detail-distance",type=float,default=.34)
    args = parser.parse_args()
    folder = args.rollout_dir.resolve()
    if any(c in args.output_stem for c in ("/","\\")) or args.output_stem in (".","..",""):raise ValueError("output stem must be a plain filename")
    movie = folder / (args.output_stem+".mp4")
    if movie.exists() and not args.overwrite:
        raise FileExistsError(f"render already exists: {movie}; use --overwrite to regenerate")
    # MuJoCo's GLFW context uses an invisible window. On Windows EGL is not a
    # supported backend; this affects only this subprocess, never desktop UI.
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "glfw"
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    started = time.perf_counter()
    model_path = folder / "model.mjb"
    logged_path = folder / "rollout_a0.npz"
    if not logged_path.exists():logged_path=folder/"rollout.npz"
    metadata_path = logged_path.with_suffix(".json")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(logged_path, allow_pickle=False) as archive:
        logged = {key: archive[key].copy() for key in archive.files}
    times = logged["time_s"]
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-8) or dt <= 0:
        raise ValueError("render expects uniformly sampled logged time")
    fps = 1 / dt
    frame_count = len(times)
    joint_ids = np.asarray([model.joint("robot/" + name).id for name in metadata["joint_names"]])
    joint_qadr = model.jnt_qposadr[joint_ids]
    object_body = model.body("object/cube").id
    object_joint = model.joint("object/cube_freejoint").id
    object_qadr = int(model.jnt_qposadr[object_joint])
    wrist_body = model.body("robot/right_palm_link").id
    if model.body_parentid[object_body] != 0:
        raise ValueError("renderer requires the logged object free body to be attached to world")
    covered_qpos = set(joint_qadr.tolist()) | set(range(object_qadr, object_qadr + 7))
    if covered_qpos != set(range(model.nq)):
        raise ValueError("not all model generalized positions are available in the log")
    if logged["joint_pos_rad"].shape != (frame_count, len(joint_ids)):
        raise ValueError("joint log shape does not match the model mapping")

    def apply_logged_state(index):
        data.qpos[joint_qadr] = logged["joint_pos_rad"][index]
        data.qpos[object_qadr:object_qadr + 7] = logged["object_pose_w"][index]
        data.time = float(times[index])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        object_error = float(np.max(np.abs(data.xpos[object_body] - logged["object_pose_w"][index, :3])))
        wrist_error = float(np.max(np.abs(data.xpos[wrist_body] - logged["wrist_pose_w"][index, :3])))
        logged_q = logged["wrist_pose_w"][index, 3:]
        quat_error = float(min(np.max(np.abs(data.xquat[wrist_body] - logged_q)),
                               np.max(np.abs(data.xquat[wrist_body] + logged_q))))
        if max(object_error, wrist_error, quat_error) > 1e-7:
            raise ValueError("drawing state does not reproduce the logged object/wrist pose")
        return object_error, wrist_error, quat_error

    def camera(lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    # Both cameras remain fixed throughout, so the fall cannot be hidden by
    # following the pen. The right panel preserves a close view of the fingers.
    overview = camera([-.105, .065, .28], .88, 135, -22)
    if args.detail_distance<=0:raise ValueError("positive camera distance required")
    detail = camera([-.10, .035, .51], args.detail_distance, 135, -40)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
        small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    selected = sorted(set([0, min(10, frame_count-1), min(25, frame_count-1),
                           min(50, frame_count-1), min(100, frame_count-1), frame_count-1]))
    stills = []
    maximum_errors = np.zeros(3)
    writer = imageio.get_writer(str(movie), fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=16)
    try:
        with mujoco.Renderer(model, height=640, width=640) as renderer:
            for index in range(frame_count):
                maximum_errors = np.maximum(maximum_errors, apply_logged_state(index))
                renderer.update_scene(data, camera=overview, scene_option=option)
                left = renderer.render().copy()
                renderer.update_scene(data, camera=detail, scene_option=option)
                right = renderer.render().copy()
                canvas = Image.new("RGB", (1280, 720), (14, 20, 29))
                canvas.paste(Image.fromarray(left), (0, 48))
                canvas.paste(Image.fromarray(right), (640, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 8), f"{args.label} | RECORDED PHYSICS STATES | fixed-wrist Wuji Hand 2", font=font, fill=(233, 240, 249))
                draw.text((16, 58), "OVERVIEW / fixed camera", font=small, fill=(244, 244, 244), stroke_width=1, stroke_fill=(0, 0, 0))
                draw.text((656, 58), "HAND DETAIL / fixed camera", font=small, fill=(244, 244, 244), stroke_width=1, stroke_fill=(0, 0, 0))
                contacts = int(logged["live_contact_active"][index].sum()) if "live_contact_active" in logged else int(logged["hand_object_active_contact_count"][index])
                draw.text((16, 695), f"t={times[index]:.2f}s  pen z={logged['object_pose_w'][index,2]:.3f}m  solved hand contacts={contacts}",
                          font=small, fill=(207, 225, 244))
                draw.text((805, 695), "Drawing only; no new dynamics simulation", font=small, fill=(183, 198, 219))
                writer.append_data(np.asarray(canvas))
                if index in selected:
                    still = folder / f"render_frame_{index:03d}.png"
                    canvas.save(still)
                    stills.append((index, canvas.copy(), still.name))
    finally:
        writer.close()
    # A compact exported artifact for visual QA and environments without video playback.
    sheet = Image.new("RGB", (1920, 720), (14, 20, 29))
    for tile, (_, frame, _) in enumerate(stills[:6]):
        frame.thumbnail((640, 360), Image.Resampling.LANCZOS)
        sheet.paste(frame, ((tile % 3) * 640, (tile // 3) * 360))
    sheet_path = folder / "render_contact_sheet.png"
    sheet.save(sheet_path)
    # Decode actual output frames, rather than only checking a container exists.
    reader = imageio.get_reader(str(movie), format="ffmpeg")
    try:
        decoded_count = reader.count_frames()
        first, last = np.asarray(reader.get_data(0)), np.asarray(reader.get_data(decoded_count-1))
        decode_stats = {"decoded_frame_count": decoded_count, "shape": list(first.shape),
                        "first_frame_pixel_std": float(first.std()), "last_frame_pixel_std": float(last.std()),
                        "first_last_mean_abs_pixel_difference": float(np.abs(first.astype(float)-last.astype(float)).mean())}
    finally:
        reader.close()
    if decoded_count != frame_count or first.std() < 5 or last.std() < 5:
        raise ValueError("encoded movie failed frame count/nonblank verification")
    render_meta = {
        "scope": "offline rendering of logged physical states; no new simulation, resets, policy or object assistance",
        "label":args.label,
        "python": sys.executable, "mujoco_version": mujoco.__version__, "gl_context": mujoco.GLContext.__module__,
        "model_sha256": digest(model_path), "rollout_sha256": digest(logged_path),
        "fps": fps, "logged_frames": frame_count, "logged_duration_s": float(times[-1]-times[0]),
        "encoded_duration_s": frame_count/fps, "resolution": [1280, 720],
        "position_error_max_m": {"object": float(maximum_errors[0]), "wrist": float(maximum_errors[1])},
        "wrist_quaternion_component_error_max": float(maximum_errors[2]),
        "kinematics_calls_only": ["mj_kinematics", "mj_comPos"], "mj_step_calls": 0, "mj_forward_calls": 0,
        "cameras": [{"lookat": cam.lookat.tolist(), "distance": cam.distance, "azimuth": cam.azimuth, "elevation": cam.elevation}
                    for cam in (overview, detail)],
        "video": movie.name, "contact_sheet": sheet_path.name,
        "keyframes": [{"index": index, "time_s": float(times[index]), "path": name} for index, _, name in stills],
        "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(), "decode_check": decode_stats,
        "elapsed_seconds": time.perf_counter()-started,
    }
    (folder / "render_metadata.json").write_text(json.dumps(render_meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"video": str(movie), "contact_sheet": str(sheet_path), "frames": decoded_count,
                      "position_errors_m": render_meta["position_error_max_m"], "elapsed_seconds": render_meta["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
