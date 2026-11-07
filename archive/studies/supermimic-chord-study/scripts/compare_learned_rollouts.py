"""Compose A1L/A2 recorded-state renders with fixed cameras; no new physics."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from compare_recorded_rollouts import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    m0, v0, r0, p0, a0, reader0 = load(args.left, "rollout_a1l.mp4")
    m1, v1, r1, p1, a1, reader1 = load(args.right, "rollout_a2.mp4")
    if any(m0[key] != m1[key] for key in ("model_sha256", "reference_bundle_sha256", "script_sha256")):
        raise ValueError("Evaluation model/reference/script differ")
    if v0["cameras"] != v1["cameras"] or not np.array_equal(r0["time_s"], r1["time_s"]):
        raise ValueError("Camera or clock mismatch")
    for key in ("joint_pos_rad", "joint_vel_rad_s", "object_pose_w", "object_lin_vel_w_m_s", "object_ang_vel_w_rad_s", "wrist_pose_w"):
        if not np.array_equal(r0[key][0], r1[key][0]):
            raise ValueError(f"Initial state mismatch: {key}")
    args.output.mkdir(parents=True)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
    small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
    movie = args.output / "a1l_vs_a2_seed0.mp4"
    writer = imageio.get_writer(movie, fps=50, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    stills = []
    try:
        for k, t in enumerate(r0["time_s"]):
            canvas = Image.new("RGB", (1280, 720), (14, 20, 29))
            canvas.paste(Image.fromarray(reader0.get_data(k)[48:688, 640:1280]), (0, 48))
            canvas.paste(Image.fromarray(reader1.get_data(k)[48:688, 640:1280]), (640, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 6), "A1L / native bounded baseline", font=font, fill=(164, 220, 255))
            draw.text((656, 6), "A2 / geometric intent prior", font=font, fill=(255, 205, 160))
            for x, pos, angle in ((16, p0[k], a0[k]), (656, p1[k], a1[k])):
                draw.text((x, 28), f"t={t:.2f}s  position={pos:.1f}mm  shaft={angle:.1f}deg", font=small, fill=(220, 227, 235))
            draw.text((16, 695), "Equal updates; all recorded states; no reset or assist.", font=small, fill=(201, 214, 229))
            draw.text((760, 695), "One seen clip / seed0; no force claim.", font=small, fill=(201, 214, 229))
            writer.append_data(np.asarray(canvas))
            if k in (0, 25, 50, 75, 100, 150):
                stills.append(canvas.copy())
    finally:
        writer.close()
        reader0.close()
        reader1.close()
    sheet = Image.new("RGB", (1920, 720), (14, 20, 29))
    for k, frame in enumerate(stills):
        frame.thumbnail((640, 360))
        sheet.paste(frame, ((k % 3) * 640, (k // 3) * 360))
    sheet.save(args.output / "comparison_contact_sheet.png")
    decoded = imageio.get_reader(movie)
    count = decoded.count_frames()
    decoded.close()
    if count != len(r0["time_s"]):
        raise ValueError("Rendered frame count differs from physical log")
    report = {"scope": "Equal-update seed0 primary models; drawing only, no new dynamics or interpolation",
              "frames": count, "initial_states_exact_equal": True, "reference_sha256": m0["reference_bundle_sha256"],
              "model_sha256": m0["model_sha256"], "left_checkpoint_sha256": m0["checkpoint_sha256"],
              "right_checkpoint_sha256": m1["checkpoint_sha256"],
              "left_rollout_sha256": hashlib.sha256((args.left / "rollout.npz").read_bytes()).hexdigest(),
              "right_rollout_sha256": hashlib.sha256((args.right / "rollout.npz").read_bytes()).hexdigest(),
              "interpretation": "A1L passes frozen task; A2 fails shaft/progress gates. Contact proxy remains unqualified; no real ego claim."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"video": str(movie), "frames": count}))


if __name__ == "__main__":
    main()
