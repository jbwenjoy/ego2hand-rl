"""Package a source-selected right-hand HO-Cap window for a future held-out evaluation."""

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import yaml

from review_hocap_carton import CAMERA, transforms
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def write_window(dataset, sequence, object_id, first, last, out):
    seq = dataset / sequence
    meta = yaml.safe_load((seq / "meta.yaml").read_text())
    if meta["task_id"] != 1 or meta["mano_sides"] != ["right"] or not (0 <= first < last < meta["num_frames"]):
        raise ValueError("Expected a single right-hand pick-and-place window")
    object_slot = meta["object_ids"].index(object_id)
    pv = meta["hololens"]["serial"]
    source_models = [
        dataset / "models" / object_id / "cleaned_mesh_10000.obj",
        dataset / "calibration/extrinsics" / meta["extrinsics"],
        dataset / "calibration/mano" / f"{meta['subject_id']}.yaml",
        dataset / "calibration/intrinsics" / f"{CAMERA}.yaml",
        dataset / "calibration/intrinsics" / f"{pv}.yaml",
        seq / "meta.yaml",
    ]
    ext = yaml.safe_load(source_models[1].read_text())["extrinsics"]
    def matrix(key):
        return np.vstack((np.array(ext[key]).reshape(3, 4), [0, 0, 0, 1]))

    world_cam = np.linalg.solve(matrix("tag_1"), matrix(CAMERA))
    frame = np.arange(first, last + 1)
    paths = [seq / f"poses_{k}.npy" for k in ("o", "m", "pv")]
    source_object, source_mano, source_camera = (np.load(p, allow_pickle=False) for p in paths)
    obj = transforms(source_object[object_slot, frame])
    joints = []
    sources = []
    for i, reference in zip(frame, obj, strict=True):
        rgb = seq / pv / f"color_{i:06}.jpg"
        label = seq / CAMERA / f"label_{i:06}.npz"
        with np.load(label, allow_pickle=False) as raw:
            if list(raw["obj_class_names"]) != list(meta["object_ids"]) + ["RIGHT_HAND"]:
                raise ValueError("Label class order changed")
            local = raw["hand_joints_3d"][0]
            if local.shape != (21, 3) or not np.isfinite(local).all() or not (local[:, 2] > 0).all():
                raise ValueError(f"Missing right-hand joints in camera label {i}")
            joints.append(local @ world_cam[:3, :3].T + world_cam[:3, 3])
            np.testing.assert_allclose(world_cam @ raw["obj_poses"][object_slot], reference, atol=1e-6, rtol=0)
        with Image.open(rgb) as image:
            image.load()
            assert image.size == (meta["hololens"]["pv_width"], meta["hololens"]["pv_height"])
        sources.extend([rgb, label])

    values = io.BytesIO()
    np.savez_compressed(values, frames=frame, right_hand_mano=source_mano[0, frame],
                        world_right_hand21=np.stack(joints), world_from_carton=obj,
                        world_from_egocamera=transforms(source_camera[frame]))
    out.mkdir(parents=True, exist_ok=False)
    bundle = out / "source_window.zip"
    with zipfile.ZipFile(bundle, "x") as archive:
        archive.writestr("source_window.npz", values.getvalue())
        for file in [*source_models, *sources]:
            archive.write(file, arcname=file.relative_to(dataset))
    with zipfile.ZipFile(bundle, "r") as archive:
        if archive.testzip() is not None:
            raise IOError("Failed CRC verification of packaged window")
    write_json(out / "reservation.json", dict(
        sequence=sequence, subject=meta["subject_id"], object=object_id, right_hand_only=True,
        frames=[first,last], count=len(frame), role="future unseen trajectory; excluded from development",
        selection="First eligible distinct subject right-hand pick-and-place for the selected box, using only source scene metadata and its movement between rest periods.",
        outcome_scope="Source package only. No robot retargeting, training, rollout or test score.",
        source_project="https://irvlutd.github.io/HOCap/", license="CC BY 4.0",
        creators="Jikai Wang et al., HO-Cap, NeurIPS 2025", script=file_pin(__file__),
        pose_arrays=[file_pin(p) for p in paths], source_bundle=file_pin(bundle),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("object_id")
    parser.add_argument("first", type=int)
    parser.add_argument("last", type=int)
    parser.add_argument("output", type=Path)
    choice = parser.parse_args()
    write_window(choice.dataset.resolve(), choice.sequence, choice.object_id,
                 choice.first, choice.last, choice.output.resolve())
