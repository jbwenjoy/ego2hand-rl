"""Create frozen q-target-only corruption controls without running any policy.

The object, wrist, independent link/keypoint intent and base contact target
remain unchanged. This is deliberately not a geometry-consistent retargeting.
Raw q references are never clipped; the bounded controller limits execution.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from chord_study.acceptance import protocol_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def resolve(path, workspace=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(workspace)/path).resolve()


def quintic_bias_ramp(time_s, ramp_seconds=.2):
    """First two samples exactly zero; C2 smoothstep over the following .2 s."""
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 2 or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0) or ramp_seconds <= 0:
        raise ValueError("need increasing finite timestamps and positive ramp duration")
    u = np.clip((time_s-time_s[1])/ramp_seconds, 0, 1)
    weight = 6*u**5-15*u**4+10*u**3
    weight[:2] = 0.
    weight[u >= 1] = 1.
    return weight


def corrupt_q_target(q, time_s, amplitude_rad, directions):
    """Keep the catalog dtype; preserve actual post-quantization per-frame delta."""
    q = np.asarray(q)
    directions = np.asarray(directions)
    if q.shape != (len(time_s), 20) or not np.issubdtype(q.dtype, np.floating) or not np.isfinite(q).all():
        raise ValueError("expected finite floating [T,20] joint targets")
    if directions.shape != (20,) or not np.isin(directions, [-1, 1]).all() or not np.isfinite(amplitude_rad) or amplitude_rad < 0:
        raise ValueError("expected 20 fixed signed directions and nonnegative finite bias")
    ramp = quintic_bias_ramp(time_s)
    requested = ramp[:, None]*float(amplitude_rad)*directions[None]
    corrupted = (q.astype(np.float64)+requested).astype(q.dtype)
    corrupted[:2] = q[:2]
    actual = corrupted.astype(np.float64)-q.astype(np.float64)
    return corrupted, actual, requested, ramp


def rebind_protocol(parent, reference_hash):
    protocol = copy.deepcopy(parent)
    protocol["reference_npz_sha256"] = reference_hash
    return protocol, protocol_digest(protocol)


def limit_diagnostics(q, lower, upper, margin):
    excess = np.maximum(lower-q, 0)+np.maximum(q-upper, 0)
    return excess, {"scope": "unclipped reference input, not an executed target or actual robot joint",
                    "clipped_during_generation": False, "strict_limit_violation_entries": int((excess > 0).sum()),
                    "strict_limit_violation_frames": int((excess > 0).any(-1).sum()),
                    "max_reference_limit_excess_rad": float(excess.max()),
                    "entries_exceeding_frozen_margin": int((excess > margin).sum()),
                    "frozen_margin_rad": float(margin),
                    "per_joint_strict_violation_count": (excess > 0).sum(0).tolist()}


def create_variants(base_reference, output, *, contact_bundle=None, seed=20261102, workspace=ROOT):
    import mujoco
    from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe

    base_reference, output = resolve(base_reference, workspace), resolve(output, workspace)
    contact_bundle = resolve(contact_bundle or ROOT / "outputs" / "reference_contact_targets_v1", workspace)
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen variant evidence")
    metadata_path = base_reference.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    base_hash, metadata_hash = digest(base_reference), digest(metadata_path)
    parent_protocol = metadata["task_acceptance"]
    parent_protocol_hash = protocol_digest(parent_protocol)
    if parent_protocol["reference_npz_sha256"] != base_hash or metadata["task_acceptance_sha256"] != parent_protocol_hash:
        raise ValueError("base reference is not bound to its frozen acceptance protocol")
    standalone_protocol = base_reference.parent / "acceptance_protocol.json"
    if json.loads(standalone_protocol.read_text(encoding="utf-8-sig")) != parent_protocol:
        raise ValueError("standalone and embedded parent protocol differ")
    manifest_path = contact_bundle / "manifest.json"
    contact_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if contact_manifest["reference_npz_sha256"] != base_hash or contact_manifest["reference_json_sha256"] != metadata_hash:
        raise ValueError("contact intent target does not bind the given independent base reference")
    contact_target = contact_bundle / "contact_targets.npz"
    if digest(contact_target) != contact_manifest["target_npz_sha256"] or contact_manifest["target_defined_from_policy_contacts"]:
        raise ValueError("contact intent bundle is not independently frozen")
    model_path = resolve(contact_manifest["model_file"], workspace)
    if digest(model_path) != contact_manifest["model_sha256"]:
        raise ValueError("joint-limit collision model hash mismatch")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    joints = [model.joint("robot/"+name).id for name in metadata["joint_names"]]
    lower, upper = model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy()
    provenance = metadata["provenance"]
    source_path = resolve(provenance["source_file"], workspace)
    if digest(source_path) != provenance["source_sha256"]:
        raise ValueError("base derived catalog clip hash mismatch")
    index_path = source_path.parent / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    entry = next(item for item in index["clips"] if item["file"] == source_path.name)
    if entry["sha256"] != digest(source_path) or not np.isclose(index["frame_dt"], .02, atol=1e-10):
        raise ValueError("base derived catalog index/hash/clock mismatch")
    with np.load(base_reference, allow_pickle=False) as data:
        reference = {key: data[key].copy() for key in data.files}
    with np.load(source_path, allow_pickle=False) as data:
        raw = {key: data[key].copy() for key in data.files}
    if not np.array_equal(raw["reference_qpos"], reference["joint_pos_rad"]):
        raise ValueError("base catalog and frozen reference joint targets differ")
    times = reference["time_s"]
    if not np.allclose(np.diff(times), .02, rtol=0, atol=1e-10):
        raise ValueError("this variant protocol requires the frozen 50Hz sample clock")
    directions = np.random.default_rng(seed).choice(np.asarray([-1, 1], dtype=np.int8), size=20)
    source_files = {"base_reference.npz": base_reference, "base_reference.json": metadata_path,
                    "base_acceptance_protocol.json": standalone_protocol,
                    "base_catalog_clip.npz": source_path, "base_catalog_index.json": index_path,
                    "base_contact_target_manifest.json": manifest_path}
    if provenance.get("original_source_file"):
        original = resolve(provenance["original_source_file"], workspace)
        if digest(original) != provenance["original_source_sha256"]:
            raise ValueError("original upstream source NPZ hash mismatch")
        source_files["upstream_original_clip.npz"] = original
    if provenance.get("original_source_catalog"):
        source_files["upstream_original_catalog_index.json"] = resolve(provenance["original_source_catalog"], workspace)
    before = {str(path): digest(path) for path in list(source_files.values())+[contact_target, model_path]}
    output.mkdir(parents=True)
    sources_dir = output / "sources"
    sources_dir.mkdir()
    source_records = []
    for name, path in source_files.items():
        shutil.copy2(path, sources_dir / name)
        source_records.append({"original_file": str(path), "copied_file": str(sources_dir / name), "sha256": digest(path)})
    (sources_dir / "generator_source.py").write_bytes(Path(__file__).read_bytes())
    contact_reference = {"bundle_dir": str(contact_bundle), "manifest_file": str(manifest_path),
                         "manifest_sha256": digest(manifest_path), "target_file": str(contact_target),
                         "target_npz_sha256": digest(contact_target), "reference_npz_file": str(base_reference),
                         "reference_npz_sha256": base_hash, "reference_json_sha256": metadata_hash,
                         "recomputed_from_corrupted_q": False,
                         "semantics": "independent BASE-q/object geometric intent retained; not labels for the corrupted q geometry"}
    base_excess, base_limits = limit_diagnostics(reference["joint_pos_rad"], lower, upper, parent_protocol["joint_limit_margin_rad"])
    variants = []
    for amplitude in (0., .03, .06):
        label = f"bias_{int(round(amplitude*1000)):03d}"
        folder = output / label
        group = folder / "catalog" / "controlled_q_bias"
        group.mkdir(parents=True)
        local_id = f"{entry['clip_id']}_{label}"
        raw_path = group / (local_id+".npz")
        raw_variant = {key: value.copy() for key, value in raw.items()}
        q, delta, requested, ramp = corrupt_q_target(raw["reference_qpos"], times, amplitude, directions)
        raw_variant["reference_qpos"] = q
        variant_reference = {key: value.copy() for key, value in reference.items()}
        variant_reference["joint_pos_rad"] = q.astype(reference["joint_pos_rad"].dtype)
        frozen_reference = folder / "reference.npz"
        if amplitude == 0:
            shutil.copy2(source_path, raw_path)
            shutil.copy2(base_reference, frozen_reference)
        else:
            np.savez_compressed(raw_path, **raw_variant)
            np.savez_compressed(frozen_reference, **variant_reference)
        index_variant = {"schema": "wuji_pen_spin_clips:1", "group_id": "controlled_q_bias",
                         "group_name": "q-target-only controlled corruption; same original demonstration/split",
                         "frame_dt": float(index["frame_dt"]), "clip_count": 1,
                         "clips": [{"clip_id": local_id, "file": raw_path.name, "sha256": digest(raw_path)}]}
        write_json(group / "index.json", index_variant)
        recipe_path = folder / "recipe.json"
        write_json(recipe_path, {"schema": "wuji_pen_spin_recipe:1", "clips": ["controlled_q_bias/"+local_id]})
        # Resolve through the actual upstream parser, including its strict hash inventory.
        resolved = resolve_recipe(recipe_path, clips_root=folder / "catalog", verify_clip_hashes=True)
        selected = [clip for clip_set in resolved.clip_sets for clip in clip_set.clips]
        if len(selected) != 1 or selected[0].npz_path.resolve() != raw_path.resolve():
            raise ValueError("new single-clip catalog does not resolve to the intended frozen file")
        excess, limits = limit_diagnostics(variant_reference["joint_pos_rad"], lower, upper, parent_protocol["joint_limit_margin_rad"])
        np.savez_compressed(folder / "perturbation.npz", time_s=times, directions=directions, ramp_weight=ramp,
                            requested_delta_rad=requested, applied_delta_rad=delta,
                            quantization_error_rad=delta-requested, joint_lower_rad=lower, joint_upper_rad=upper,
                            reference_limit_violation=excess > 0, reference_limit_excess_rad=excess,
                            base_reference_limit_violation=base_excess > 0, base_reference_limit_excess_rad=base_excess)
        new_hash = digest(frozen_reference)
        protocol, protocol_hash = rebind_protocol(parent_protocol, new_hash)
        write_json(folder / "acceptance_protocol.json", protocol)
        perturbation = {"type": "q_target_only_additive_controlled_bias", "amplitude_rad": amplitude, "seed": int(seed),
                        "directions_in_joint_order": directions.tolist(), "joint_names": metadata["joint_names"],
                        "ramp": "first two samples zero; quintic smoothstep over next .2s then constant",
                        "ramp_begin_s": float(times[1]), "ramp_end_s": float(times[1]+.2),
                        "first_two_frames_delta_exact_zero": bool(np.array_equal(delta[:2], np.zeros((2, 20)))),
                        "catalog_q_dtype_retained": str(q.dtype), "actual_delta_file": str(folder / "perturbation.npz"),
                        "actual_delta_sha256": digest(folder / "perturbation.npz"),
                        "quantization_error_abs_max_rad": float(np.abs(delta-requested).max()),
                        "clipping_applied": False, "execution_limits": "new bounded controller, not dataset generation",
                        "object_wrist_link_mediapipe_preserved": True, "contact_intent": contact_reference,
                        "base_reference_sha256": base_hash, "base_reference_json_sha256": metadata_hash,
                        "parent_protocol_sha256": parent_protocol_hash,
                        "changed_protocol_fields": [key for key in protocol if protocol[key] != parent_protocol[key]],
                        "reference_input_limit_diagnostics": limits, "base_reference_input_limit_diagnostics": base_limits,
                        "same_split_group": metadata["split_group"], "is_independent_demo": False,
                        "original_video_parent_id": provenance.get("original_video_parent_id"),
                        "claim": "q-target corruption only; intentionally retain original independent link/object/contact intent, not whole-geometry retargeting or ego data"}
        variant_meta = copy.deepcopy(metadata)
        variant_meta.update(source_id=metadata["source_id"]+":"+label, source_kind="upstream_kinematic", npz_sha256=new_hash,
                            task_acceptance=protocol, task_acceptance_sha256=protocol_hash,
                            controlled_perturbation=perturbation, generated_utc=datetime.now(timezone.utc).isoformat())
        variant_meta["provenance"].update(source_file=str(raw_path), source_sha256=digest(raw_path),
                                           source_trajectory_id="controlled_q_bias/"+local_id,
                                           parent_frozen_reference_file=str(base_reference), parent_frozen_reference_sha256=base_hash,
                                           parent_catalog_source_file=str(source_path), parent_catalog_source_sha256=digest(source_path),
                                           parent_catalog_group_id=index["group_id"],
                                           transformation=provenance["transformation"]+"; then q-target-only controlled bias with all independent object/link/contact intents unchanged",
                                           contact_targets=contact_reference, controlled_perturbation=perturbation,
                                           claim=perturbation["claim"])
        write_json(folder / "reference.json", variant_meta)
        write_json(folder / "provenance.json", variant_meta["provenance"])
        write_json(folder / "perturbation.json", perturbation)
        immutable_ref = all(np.array_equal(variant_reference[key], reference[key]) for key in reference if key != "joint_pos_rad")
        immutable_raw = all(np.array_equal(raw_variant[key], raw[key]) for key in raw if key != "reference_qpos")
        initial = {"reference_q_first_two_exact": bool(np.array_equal(variant_reference["joint_pos_rad"][:2], reference["joint_pos_rad"][:2])),
                   "catalog_q_first_two_exact": bool(np.array_equal(q[:2], raw["reference_qpos"][:2])),
                   "reference_initial_qdot_exact": bool(np.array_equal(np.diff(variant_reference["joint_pos_rad"][:2], axis=0)/.02, np.diff(reference["joint_pos_rad"][:2], axis=0)/.02)),
                   "catalog_initial_qdot_exact": bool(np.array_equal(np.diff(q[:2], axis=0)/.02, np.diff(raw["reference_qpos"][:2], axis=0)/.02)),
                   "all_non_q_reference_arrays_exact": immutable_ref, "all_non_q_catalog_arrays_exact": immutable_raw,
                   "initial_object_pose_and_velocity_inputs_exact": bool(np.array_equal(variant_reference["object_pose_w"][:2], reference["object_pose_w"][:2])),
                   "initial_wrist_pose_and_velocity_inputs_exact": bool(np.array_equal(variant_reference["wrist_pose_w"][:2], reference["wrist_pose_w"][:2]))}
        if not all(initial.values()):
            raise AssertionError("q-only variant violated preserved initial conditions or independent targets")
        record = {"name": label, "amplitude_rad": amplitude, "folder": str(folder), "reference_file": str(frozen_reference),
                  "reference_npz_sha256": new_hash, "reference_json_sha256": digest(folder / "reference.json"),
                  "catalog_root": str(folder / "catalog"), "catalog_clip": str(raw_path), "catalog_clip_sha256": digest(raw_path),
                  "recipe_path": str(recipe_path), "recipe_sha256": digest(recipe_path),
                  "provenance_path": str(folder / "provenance.json"), "protocol_sha256": protocol_hash,
                  "initial_conditions_and_target_preservation": initial, "reference_input_limits": limits,
                  "contact_intent": contact_reference, "split_group": metadata["split_group"]}
        write_json(folder / "verification.json", record)
        variants.append(record)
    for path, expected in before.items():
        if digest(path) != expected:
            raise AssertionError("a frozen source artifact changed during controlled-data generation")
    result = {"schema": "supermimic_chord_q_only_variants:1", "source_kind": "upstream_kinematic_with_controlled_q_target_corruption",
              "base_reference_sha256": base_hash, "base_reference_json_sha256": metadata_hash,
              "parent_protocol_sha256": parent_protocol_hash, "source_files": source_records,
              "source_files_unchanged": True, "original_video_parent_id": provenance.get("original_video_parent_id"),
              "split_group": metadata["split_group"], "source_group_id": index["group_id"],
              "source_frames": provenance["source_frames"], "original_source_frames": provenance.get("original_source_frames"),
              "seed": int(seed), "directions": directions.tolist(), "variants": variants,
              "contact_intent": contact_reference,
              "scientific_boundary": "same original crop and split; q-target-only corruption, not new human/ego demonstrations, not all-geometry retargeting, no physical success claim",
              "generator_sha256": digest(__file__), "gpu_executed": False}
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reference", type=Path, default=ROOT / "outputs" / "pilot_rotating_reference_v2" / "reference.npz")
    parser.add_argument("--contact-bundle", type=Path, default=ROOT / "outputs" / "reference_contact_targets_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "controlled_q_bias_variants_v1")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    result = create_variants(args.base_reference, args.output, contact_bundle=args.contact_bundle, seed=args.seed)
    print(json.dumps({"output": str(args.output), "source_files_unchanged": result["source_files_unchanged"],
                      "variants": [{"name": row["name"], "reference_sha256": row["reference_npz_sha256"],
                                    "raw_reference_limit_violations": row["reference_input_limits"]["strict_limit_violation_entries"]}
                                   for row in result["variants"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
