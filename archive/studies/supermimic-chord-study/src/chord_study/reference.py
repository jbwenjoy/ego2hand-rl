"""Reference bundle validation. JSON metadata accompanies a non-pickled NPZ."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import rotation_error


@dataclass
class Reference:
    arrays: dict
    meta: dict

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))
        if "npz_sha256" in meta and hashlib.sha256(path.read_bytes()).hexdigest() != meta["npz_sha256"]:
            raise ValueError("reference NPZ integrity mismatch")
        return cls(arrays, meta)

    def validate(self, *, contact_required=False, wrist_position_m=.005, wrist_angle_deg=5):
        a, m = self.arrays, self.meta
        required_meta = {"schema", "source_id", "source_kind", "split_group", "joint_names", "units",
                         "quaternion_order", "world_frame", "contact_normal", "object_radius_m",
                         "object_mass_kg", "object_inertia_kg_m2", "orientation_metric", "provenance"}
        missing = required_meta-set(m)
        if missing:
            raise ValueError(f"missing metadata: {sorted(missing)}")
        if m["schema"] != "supermimic_chord_reference:1" or m["units"] != "SI" or m["quaternion_order"] != "wxyz":
            raise ValueError("unsupported schema/units/quaternion convention")
        if m["source_kind"] not in {"upstream_kinematic", "sim_teacher", "ego_reconstruction", "synthetic_test"}:
            raise ValueError("explicit source_kind required")
        if m["contact_normal"] != "force_on_object_inward":
            raise ValueError("contact normal must point inward, in the force-on-object direction")
        if not all(isinstance(m[key], str) and m[key] for key in ("source_id", "split_group", "world_frame")):
            raise ValueError("source/split/frame identity cannot be empty")
        if not isinstance(m["joint_names"],list) or len(m["joint_names"]) != 20 or not all(isinstance(s,str) and s for s in m["joint_names"]) or len(set(m["joint_names"])) != 20:
            raise ValueError("20 distinct named joints required")
        for key in ("object_radius_m", "object_mass_kg"):
            if not np.isfinite(m[key]) or m[key] <= 0:
                raise ValueError(f"invalid {key}")
        inertia = np.asarray(m["object_inertia_kg_m2"])
        if inertia.shape != (3,3) or not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0:
            raise ValueError("object inertia must be finite symmetric positive definite")
        moments = np.linalg.eigvalsh(inertia)
        if moments[-1] > moments[:2].sum() + 1e-10:
            raise ValueError("principal inertia moments violate the rigid-body triangle inequality")
        if not isinstance(m["provenance"], dict) or not m["provenance"]:
            raise ValueError("non-empty provenance with source/transform information required")
        if m["orientation_metric"] not in {"SO3", "directed_axis_x"}:
            raise ValueError("orientation metric must be explicitly SO3 or directed_axis_x")
        for key in ("time_s", "joint_pos_rad", "object_pose_w", "wrist_pose_w", "phase"):
            if key not in a:
                raise ValueError(f"missing array: {key}")
        t = a["time_s"]
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("timestamps must be finite, strictly increasing, with >=2 frames")
        n = len(t)
        for key, shape in (("joint_pos_rad", (n,20)), ("object_pose_w", (n,7)), ("wrist_pose_w", (n,7)), ("phase", (n,))):
            if a[key].shape != shape or not np.isfinite(a[key]).all():
                raise ValueError(f"invalid {key}, expected {shape}")
        for key in ("object_pose_w", "wrist_pose_w"):
            if not np.allclose(np.linalg.norm(a[key][:,3:], axis=-1), 1, atol=1e-4):
                raise ValueError(f"{key} quaternions must be unit wxyz")
        wrist = a["wrist_pose_w"]
        wrist_position = np.linalg.norm(wrist[:,:3]-wrist[0,:3], axis=-1).max()
        wrist_angle = np.rad2deg(rotation_error(wrist[:,3:], wrist[0,3:])).max()
        admissible = bool(wrist_position <= wrist_position_m and wrist_angle <= wrist_angle_deg)
        # Absence is unknown, not a zero-contact label.
        contact_fields = {"contact_pos_o_m", "contact_normal_o", "contact_active", "contact_known"}
        present = contact_fields & set(a)
        if present and present != contact_fields:
            raise ValueError("incomplete contact labels")
        if contact_required and not present:
            raise ValueError("contact-enhanced experiment requires independent contact labels")
        coverage = 0.
        if present:
            cp, cn, active, known = (a[k] for k in ("contact_pos_o_m", "contact_normal_o", "contact_active", "contact_known"))
            if cp.ndim != 3 or cp.shape[0] != n or cp.shape[-1] != 3 or cn.shape != cp.shape or active.shape != cp.shape[:-1] or known.shape != (n,):
                raise ValueError("invalid contact shapes")
            if not np.isfinite(cp).all() or not np.isfinite(cn).all() or not np.isin(active,[0,1]).all() or not np.isin(known,[0,1]).all():
                raise ValueError("invalid contact geometry/masks")
            if np.any(active.astype(bool) & (np.linalg.norm(cn,axis=-1) <= 1e-3)):
                raise ValueError("active contact without normal")
            coverage = float(known.mean())
            if contact_required and coverage == 0:
                raise ValueError("all contact labels are unknown")
        return {"frames": n, "duration_s": float(t[-1]-t[0]), "median_dt_s": float(np.median(np.diff(t))),
                "source_kind": m["source_kind"], "fixed_wrist_admissible": admissible,
                "wrist_displacement_max_m": float(wrist_position), "wrist_angle_max_deg": float(wrist_angle),
                "contact_label_coverage": coverage, "dynamics_feasibility": "requires_unassisted_physics_evaluation"}


def assign_split(group, seed=0):
    """All clips, perturbations and embodiments from one source share a split."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def assert_disjoint_split_groups(records):
    seen = {}
    for record in records:
        group, split = record["split_group"], record["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        if group in seen and seen[group] != split:
            raise ValueError(f"source trajectory leakage: {group}")
        seen[group] = split
