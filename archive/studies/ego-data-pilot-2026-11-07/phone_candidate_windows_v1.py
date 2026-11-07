"""Saved HOT3D JSON-only phone/wrist descriptors; no FK or contact inference."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import tarfile

import numpy as np

BASE = Path(__file__).resolve().parent
CLIPS = ("clip-001852", "clip-001856")
STREAMS = ("214-1", "1201-1", "1201-2")


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    assert q.shape == (4,) and np.isfinite(q).all()
    assert abs(np.linalg.norm(q) - 1) < 1e-6
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose(d):
    p = np.asarray(d["translation_xyz"], dtype=np.float64)
    assert p.shape == (3,) and np.isfinite(p).all()
    return p, rotation(d["quaternion_wxyz"])


def angle(a, b):
    # Descriptor only, never a physical acceptance gate.
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b)-1)/2, -1, 1))))


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    return {"min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def motion(p, r):
    p = np.asarray(p)
    return {
        "endpoint_translation_mm": float(np.linalg.norm(p[-1]-p[0])*1000),
        "translation_path_mm": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()*1000),
        "max_translation_from_first_mm": float(np.linalg.norm(p-p[0], axis=1).max()*1000),
        "translation_centered_RMS_mm": float(np.sqrt(np.mean(np.sum((p-p.mean(0))**2, axis=1)))*1000),
        "endpoint_rotation_deg": angle(r[0], r[-1]),
        "max_rotation_from_first_deg": max(angle(r[0], a) for a in r),
        "rotation_path_deg": sum(angle(a, b) for a, b in zip(r[:-1], r[1:])),
    }


def visibility(ob, hand):
    out = {}
    for stream in STREAMS:
        ov = ob.get("visibilities_modeled", {}).get(stream)
        pv = ob.get("visibilities_predicted", {}).get(stream)
        hv = hand.get("visibilities_modeled", {}).get(stream)
        out[stream] = {
            "phone_modeled": ov, "phone_SAM2_predicted": pv,
            "wrist_hand_modeled": hv,
            "phone_bbox_present": stream in ob.get("boxes_amodal", {}),
            "hand_bbox_present": stream in hand.get("boxes_amodal", {}),
            "phone_modal_mask_pixels": (sum(ob["masks_modal"][stream]["rle"][1::2])
                                        if stream in ob.get("masks_modal", {}) else None),
        }
    return out


def describe(rows, side):
    pp = np.array([r["phone_world_position_m"] for r in rows])
    pr = np.array([r["phone_world_rotation"] for r in rows])
    hp = np.array([r["hands"][side]["wrist_world_position_m"] for r in rows])
    hr = np.array([r["hands"][side]["wrist_world_rotation"] for r in rows])
    rp = np.array([r["hands"][side]["phone_position_in_wrist_m"] for r in rows])
    rr = np.array([r["hands"][side]["phone_rotation_in_wrist"] for r in rows])
    v = {}
    for stream in STREAMS:
        vs = [r["hands"][side]["visibility"][stream] for r in rows]
        # Missing visibility is retained as null in raw rows, separately counted here.
        v[stream] = {
            "phone_annotation_present_frames": sum(a["phone_modeled"] is not None for a in vs),
            "phone_modeled_positive_frames": sum((a["phone_modeled"] or 0) > 0 for a in vs),
            "phone_predicted_positive_frames": sum((a["phone_SAM2_predicted"] or 0) > 0 for a in vs),
            "co_visible_positive_frames": sum((a["phone_SAM2_predicted"] or 0) > 0 and
                                               (a["wrist_hand_modeled"] or 0) > 0 for a in vs),
            "phone_predicted_mean_missing_as_zero": float(np.mean([a["phone_SAM2_predicted"] or 0 for a in vs])),
            "hand_modeled_mean_missing_as_zero": float(np.mean([a["wrist_hand_modeled"] or 0 for a in vs])),
        }
    dp, dh = np.diff(pp, axis=0), np.diff(hp, axis=0)
    den = np.linalg.norm(dp)*np.linalg.norm(dh)
    return {"frames": len(rows), "start_frame": rows[0]["frame"], "end_frame_inclusive": rows[-1]["frame"],
            "start_seconds": rows[0]["time_seconds"], "end_seconds": rows[-1]["time_seconds"],
            "elapsed_seconds": rows[-1]["time_seconds"]-rows[0]["time_seconds"],
            "phone_world_motion": motion(pp, pr), "wrist_world_motion": motion(hp, hr),
            "phone_relative_to_wrist_motion": motion(rp, rr),
            "phone_origin_to_wrist_distance_mm": summary(np.linalg.norm(rp, axis=1)*1000),
            "world_translation_increment_cosine": float(np.sum(dp*dh)/den) if den > 0 else None,
            "visibility": v}


def read_clip(path):
    rows = []
    with tarfile.open(path, "r") as tar:
        ids = sorted(int(m.name.split(".")[0]) for m in tar.getmembers() if m.name.endswith(".info.json"))
        assert ids == list(range(150))
        for i in ids:
            def read(suffix):
                return json.load(tar.extractfile(f"{i:06d}.{suffix}.json"))
            objects, hands, info = read("objects"), read("hands"), read("info")
            assert len(objects["24"]) == 1
            ob = objects["24"][0]
            assert ob["object_bop_id"] == "24" and ob["object_name"] == "cellphone"
            pp, pr = pose(ob["T_world_from_object"])
            row = {"frame": i, "reference_timestamp_ns": info["ref_timestamp_ns"],
                   "sequence_id": info["sequence_id"], "participant_id": info["participant_id"],
                   "object_bop_id": 24, "object_uid": ob["object_uid"],
                   "phone_world_position_m": pp.tolist(), "phone_world_rotation": pr.tolist(), "hands": {}}
            for side in ("left", "right"):
                h = hands[side]
                hp, hr = pose(h["umetrack_pose"]["T_world_from_wrist"])
                row["hands"][side] = {
                    "wrist_world_position_m": hp.tolist(), "wrist_world_rotation": hr.tolist(),
                    "phone_position_in_wrist_m": (hr.T @ (pp-hp)).tolist(),
                    "phone_rotation_in_wrist": (hr.T @ pr).tolist(), "visibility": visibility(ob, h)}
            rows.append(row)
    t0 = rows[0]["reference_timestamp_ns"]
    for row in rows:
        row["time_seconds"] = (row["reference_timestamp_ns"]-t0)*1e-9
    assert all(b["reference_timestamp_ns"] > a["reference_timestamp_ns"] for a, b in zip(rows[:-1], rows[1:]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE/"outputs/phone_candidate_windows_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"schema": "hot3d_phone_candidate_windows_saved_JSON:1", "author_analysis_not_contact_qualification": True,
              "script": pin(__file__), "inputs": {}, "method": {
                  "object": "BOP24 cellphone, sole instance per frame",
                  "wrist": "UmeTrack T_world_from_wrist; no FK or hand-model reconstruction",
                  "relative_pose": "p_wo=R_world_wrist.T@(p_world_phone-p_world_wrist); R_wo=R_world_wrist.T@R_world_phone",
                  "rotation_descriptor": "SO(3) relative angle from normalized source quaternions; trace clamped only for roundoff",
                  "window_lengths": "30/45/60 timestamp intervals (31/46/61 samples), approximately 1/1.5/2 s",
                  "selection": "For each clip use closer-median wrist. Rank windows by RGB co-visible count/fraction, then smaller relative rotation excursion, then smaller relative translation excursion. Keep best 2s and non-overlapping best1s where RGB co-visible; no contact threshold or physical certificate.",
                  "visibility": "modeled geometry and SAM2-predicted visibility are separate, not contact labels",
                  "scope": "Only saved hands/objects/info JSON read; NumPy rigid transforms. No images, FK, mesh distance, model, Torch, or simulation."},
              "clips": {}, "candidates": []}
    for clip in CLIPS:
        path = BASE/"hot3d/train_aria"/f"{clip}.tar"
        result["inputs"][clip] = pin(path)
        rows = read_clip(path)
        whole = {s: describe(rows, s) for s in ("left", "right")}
        side = min(whole, key=lambda s: whole[s]["phone_origin_to_wrist_distance_mm"]["median"])
        windows = []
        for intervals in (30, 45, 60):
            for start in range(len(rows)-intervals):
                d = describe(rows[start:start+intervals+1], side)
                d.update({"hand": side, "intervals": intervals})
                windows.append(d)
        result["clips"][clip] = {"frame_count": len(rows), "whole_clip_by_hand": whole,
                                 "closer_median_wrist": side, "all_sliding_windows": windows, "all_frame_descriptors": rows}
        def rank(w):
            return (-w["visibility"]["214-1"]["co_visible_positive_frames"]/w["frames"],
                    w["phone_relative_to_wrist_motion"]["max_rotation_from_first_deg"],
                    w["phone_relative_to_wrist_motion"]["max_translation_from_first_mm"], w["start_frame"])
        visible = [w for w in windows if w["visibility"]["214-1"]["co_visible_positive_frames"] > 0]
        if not visible:
            result["clips"][clip]["candidate_status"] = "NO_RGB_CO_VISIBLE_PHONE_WRIST_WINDOW; annotated object not assumed manipulated"
            continue
        best2 = min([w for w in visible if w["intervals"] == 60], key=rank)
        nonoverlap = [w for w in visible if w["intervals"] == 30 and
                      (w["end_frame_inclusive"] < best2["start_frame"] or w["start_frame"] > best2["end_frame_inclusive"])]
        chosen = [best2] + ([min(nonoverlap, key=rank)] if nonoverlap else [])
        for w in chosen:
            result["candidates"].append({"clip": clip, **w, "interpretation": "Visible phone movement with wrist-relative stability supports manipulation-candidate selection; wrist relation alone cannot establish surface contact or forces."})
        result["clips"][clip]["candidate_status"] = "CANDIDATE_REQUIRES_RGB_AND_GEOMETRY_CONTACT_REVIEW"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": pin(args.output), "candidate_windows": [{"clip": w["clip"], "hand": w["hand"], "frames": [w["start_frame"], w["end_frame_inclusive"]], "seconds": [w["start_seconds"], w["end_seconds"]], "relative_motion": w["phone_relative_to_wrist_motion"]} for w in result["candidates"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
