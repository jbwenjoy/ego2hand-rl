"""Measure observed blue pixels outside the projected HO-Cap G04_2 convex outline."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import trimesh
import yaml

from render_hocap_carton import HEIGHT, WIDTH, font, read_json, source_outline
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def review(source, output):
    report = read_json(source / "report.json")
    inputs = read_json(verify_file_pin(report["input_frames"]))
    with np.load(verify_file_pin(report["source_poses"]), allow_pickle=False) as data:
        carton = data["world_from_carton"]
        camera = data["world_from_egocamera"]
    lens_pin = next(p for p in report["source_inputs"] if "intrinsics/hololens" in p["path"].replace("\\", "/"))
    lens = yaml.safe_load(verify_file_pin(lens_pin).read_text("utf-8"))["color"]
    hull = trimesh.load_mesh(verify_file_pin(report["geometry"]["hull"]), process=False)
    frames = [0, 40, report["frame_of_max_displacement"], 110, len(inputs) - 1]
    w, h = lens["width"], lens["height"]
    table = []
    sheet = Image.new("RGB", (280 * len(frames), 157 * 2 + 73), "#12222c")
    captions = ImageDraw.Draw(sheet)
    for col, index in enumerate(frames):
        if inputs[index]["frame"] != index:
            raise ValueError("Non-contiguous source frames")
        with Image.open(verify_file_pin(inputs[index]["ego_rgb"])) as picture:
            rgb = picture.convert("RGB")
        if rgb.size != (w, h):
            raise ValueError("RGB and calibration dimensions differ")
        xy = np.array(source_outline(hull.vertices, carton[index], camera[index], lens)) * [w / WIDTH, h / HEIGHT]
        x0, y0 = np.maximum(0, np.floor(xy.min(axis=0) - 70)).astype(int)
        x1, y1 = np.minimum([w, h], np.ceil(xy.max(axis=0) + 70)).astype(int)
        pixels = np.asarray(rgb)[y0:y1, x0:x1]
        r, g, b = pixels.astype(np.float32).transpose(2, 0, 1)
        blue = (b - r > 35) & (b > 1.03 * g) & (b > 75)
        polygon = [tuple(point) for point in xy - [x0, y0]]
        mask = Image.new("L", (x1 - x0, y1 - y0))
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        distance = distance_transform_edt(np.asarray(mask) == 0)
        outside = blue & (distance > 5)
        values = distance[blue]
        if not values.size:
            raise ValueError("Empty blue support")
        percentage = 100 * outside.sum() / values.size
        display = pixels.copy()
        display[outside] = np.round(display[outside] * 0.25 + np.array([255, 66, 190]) * 0.75)
        annotation = rgb.copy()
        annotation.paste(Image.fromarray(display), (x0, y0))
        ImageDraw.Draw(annotation).line([tuple(p) for p in xy], fill="#ffd76b", width=4)
        x = col * 280
        captions.text((x + 8, 4), f"第 {index} 帧", font=font(17), fill="#e7eef4")
        for image, y in [(rgb, 30), (annotation, 188)]:
            sheet.paste(image.resize((280, 157), Image.Resampling.LANCZOS), (x, y))
        captions.text((x + 8, 352), f"轮廓外蓝色像素  {percentage:.1f}%", font=font(15), fill="#e7eef4")
        table.append(dict(frame=index, original_resolution=[w, h], roi_xyxy=[x0, y0, x1, y1],
                          blue_pixels=int(values.size), blue_outside_5px=int(outside.sum()),
                          fraction_outside_5px=float(percentage / 100),
                          distance_to_closed_projection_p95_px=float(np.percentile(values, 95))))
    output.mkdir(parents=True, exist_ok=False)
    preview = output / "blue_pixels_and_source_outline.jpg"
    sheet.save(preview, quality=94)
    write_json(output / "report.json", dict(
        source=file_pin(source / "report.json"), tool=file_pin(__file__),
        projection_implementation=file_pin(Path(__file__).with_name("render_hocap_carton.py")),
        blue="B-R>35, B>1.03*G, B>75 (RGB, 8-bit); box hull bounding rectangle enlarged by 70 px",
        interpretation="Observable blue regions only. Not a segmentation label, calibrated translation error, or a correction to the source poses.",
        tolerance_px=5, frames=table, preview=file_pin(preview),
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    review(args.source.resolve(), args.output.resolve())
