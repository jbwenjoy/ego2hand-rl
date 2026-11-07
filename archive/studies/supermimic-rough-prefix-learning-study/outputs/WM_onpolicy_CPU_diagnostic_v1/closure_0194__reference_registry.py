"""CPU reference identities/initialization candidates, not a runtime reset hook."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import numpy as np
from .reader import _checked_file, _freeze, _thaw, _json, _pin, open_catalog


@dataclass(frozen=True, slots=True)
class SharedReferenceRegistry:
    path: Path
    sha256: str
    payload: Mapping

    @property
    def source_ids(self): return tuple(e["source_id"] for e in self.payload["entries"])

    def entry(self, source_id):
        rows = [e for e in self.payload["entries"] if e["source_id"] == source_id]
        if len(rows) != 1: raise ValueError("Source absent/duplicate in reference registry")
        return rows[0]

    def track(self, name):
        if name not in self.payload["tracks"]: raise ValueError("Unrecognized reference track")
        return self.payload["tracks"][name]

    def as_dict(self): return _thaw(self.payload)


def open_reference_registry(path, *, expected_sha256):
    path = _checked_file(path, expected_sha256); p = _json(path)
    if p.get("schema") != "shared_derived_reference_registry:1" or p.get("training_entry_integration") != "PENDING":
        raise ValueError("Only the explicitly pending CPU registry schema is supported")
    if p["training_authorized"] or p["physical_reference_qualified"]: raise ValueError("Registry is not execution/physical authority")
    _pin(p["upstream_batch_READY"]); _pin(p["upstream_protocol"])
    for pin in p["proposed_controlled_physics"].values(): _pin(pin)
    ready = _json(Path(p["upstream_batch_READY"]["path"]))
    enrolled = {row["source_id"]: row for row in ready["clips"]}
    if len(p["entries"]) != 10 or len({e["source_id"] for e in p["entries"]}) != 10:
        raise ValueError("Expected all10 preserved source-window identities")
    for e in p["entries"]:
        if e["source_id"] not in enrolled or e["source_frame_range_half_open"] != [0,151] or e["frames"] != 151:
            raise ValueError("Source/window mismatch; old007[200,351) is not new007[0,151)")
        old = enrolled[e["source_id"]]
        if e["canonical_native"]["sha256"] != old["native_sha256"] or e["geometry"]["sha256"] != old["geometric_sha256"]:
            raise ValueError("Registry cannot substitute another source/window's derived artifact")
        native_path, geometry_path = _pin(e["canonical_native"]), _pin(e["geometry"])
        sidecar = _json(_pin(e["native_provenance"]))
        pr = sidecar["provenance"]["strict_reader_provenance"]
        for key, expected in (("source_id",e["source_id"]),("selected_source_frame_range_half_open",[0,151]),("source_sha256",e["source_sha256"]),("export_sha256",e["observation_export_sha256"])):
            if pr[key] != expected: raise ValueError("Derived-native provenance/window mismatch: "+key)
        if pr["lineage"] != p["lineage"] or pr["split_group"] != p["split_group"]: raise ValueError("Lineage relabeling")
        with np.load(native_path,allow_pickle=False) as z, np.load(geometry_path,allow_pickle=False) as g:
            if set(z.files) != {"reference_qpos","wrist_pose_w","object_pose_w","link_pos_w"}: raise ValueError("Native ABI key mismatch")
            if not np.array_equal(z["reference_qpos"],g["joint_pos_rad"]) or not np.array_equal(z["object_pose_w"],g["mapped_object_pose_w"]): raise ValueError("Derived/native goals differ")
        _pin(e["init_requested_candidates"])
        if e["init_measured"] is not None or e["physical_initialization_status"] != "UNQUALIFIED":
            raise ValueError("CPU registry may not invent a measured/qualified initialization")
    for name,t in p["tracks"].items():
        catalog = open_catalog(t["G1_catalog"]["path"],expected_sha256=t["G1_catalog"]["sha256"])
        ids=t["source_order"]
        if list(catalog.source_ids) != ids: raise ValueError("Track sourceorder differs from approved G1 split/use")
        if t["allowed_uses"] != list(catalog.allowed_uses): raise ValueError("Track use cannot relabel heldout data as training")
        if t["frame_dt_s"] != .02 or t["trajectory_lengths"] != [151]*len(ids): raise ValueError("Track length/frequency")
        if t["trajectory_starts"] != list(range(0,151*len(ids),151)) or t["trajectory_ends"] != list(range(151,151*(len(ids)+1),151)): raise ValueError("Prefix table")
        if len(t["materialized_copies"]) != len(ids): raise ValueError("Track copies missing")
        for sid,copy in zip(ids,t["materialized_copies"]):
            if copy["source_id"] != sid: raise ValueError("Copy sourceorder")
            canonical=next(e for e in p["entries"] if e["source_id"]==sid)
            if copy["sha256"] != canonical["canonical_native"]["sha256"]: raise ValueError("Track copy changes deriveddata")
            _checked_file(copy["path"],copy["sha256"])
        _pin(t["native_recipe"]);_pin(t["native_index"])
    return SharedReferenceRegistry(path,expected_sha256,_freeze(p))


def resolve_init_requested(registry, source_id, *, velocity_design):
    """Return a DESIGN candidate; there is no measured state or executedu."""
    e=registry.entry(source_id)
    if velocity_design not in ("zero_rest","derived_native_FD"): raise ValueError("Explicit velocity design required")
    with np.load(_pin(_thaw(e["init_requested_candidates"])),allow_pickle=False) as z:
        values={k:np.array(z[k],copy=True) for k in z.files}
    return {"status":"DESIGNED_UNQUALIFIED_NOT_MEASURED", "source_id":source_id,
            "source_frame_range_half_open":list(e["source_frame_range_half_open"]),"velocity_design":velocity_design,
            "q_goal0_rad":values["q_goal0_rad"],"native_FD_goal_qd0_rad_s":values["native_FD_goal_qd0_rad_s"],
            "q_init_requested_FP32_rad":values["q_init_requested_FP32_rad"],
            "qd_init_requested_rad_s":values["zero_qd_rad_s"] if velocity_design=="zero_rest" else values["native_FD_goal_qd0_rad_s"],
            "object_pose_init_requested_FP32_w":values["object_pose_init_requested_FP32_w"],
            "object_linvel_init_requested_w_m_s":values["zero_object_velocity"] if velocity_design=="zero_rest" else values["derived_object_linvel0_w_m_s"],
            "object_angvel_init_requested_w_rad_s":values["zero_object_velocity"] if velocity_design=="zero_rest" else values["derived_object_angvel0_w_rad_s"],
            "u_executed0_rad":None,"PD_preload_design":None,"init_measured":None,"physical_qualified":False,
            "scope":"NativeFD is a kinematic design, never measuredvideo/robot velocity; realworker must explicitly convert freejointangular frame and verifyFP32readback."}


def validate_initialization_identity(registry, source_id, evidence):
    e=registry.entry(source_id); model=registry.payload["proposed_controlled_physics"]["model"]
    context=registry.payload["proposed_controlled_physics"]["context"]
    expected={"source_id":source_id,"source_frame_range_half_open":list(e["source_frame_range_half_open"]),
              "source_sha256":e["source_sha256"],"native_npz_sha256":e["canonical_native"]["sha256"],
              "model_sha256":model["sha256"],"physics_context_sha256":context["sha256"]}
    for k,v in expected.items():
        if evidence.get(k)!=v: raise ValueError("Initialization cannot migrate across source/window/artifact/physics: "+k)
    return {"identity_match":True,"physical_qualified":False,"native_training_entry_bound":False,
            "scope":"Metadata identity only, no physicalsnapshot/reset/velocity/control parity was tested"}


def require_training_ready(registry, track, *, targets_ready=False):
    registry.track(track)
    raise RuntimeError("PENDING: native training integration, measured initialization/reset parity and independent targetbundle qualification are not implemented by the CPU registry")
