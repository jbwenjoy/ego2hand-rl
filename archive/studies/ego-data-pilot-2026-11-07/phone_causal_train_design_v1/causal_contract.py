"""Fail-closed geometry/contact input contract. No NN, FK or native calls.

This version has no full-H swept/coupling proof verifier. Consequently it cannot
issue a strict zero-action-path mask; an observed empty baseline is descriptive.
"""
import numpy as np

SCHEMA = "phone_current_geometry_contact_conditioning:1"
MODEL_SHA256 = "ff88ccc33fd1190328639925d82c6c11550b44dea67eec0f51c0a3cfd4339826"
FEATURE_FIELDS = {"state58", "absolute_u20", "native_h_s", "n_substeps", "pair_relative_translation_BODY_m",
    "pair_relative_rotation_BODY9", "pair_signed_gap_m", "pair_gap_lower_bound_m", "pair_geometry_known",
    "normal_on_phone_BODY3", "normal_unique_known", "pair_names", "model_sha256", "source_state_schema"}


def validate_current_geometry_inputs(batch):
    """Check a caller's current/predicted-state geometry, not create FK features.

    No target/decoded force/native future-contact value is accepted as input.
    Unknown geometry stays masked/NaN; a lower bound is not mislabeled exact SDF.
    """
    if set(batch) != FEATURE_FIELDS:
        raise ValueError("Exact current-geometry fields; no future labels/solver-force oracle inputs")
    x = np.asarray(batch["state58"])
    if x.ndim != 2 or x.shape[1] != 58 or not np.isfinite(x).all():
        raise ValueError("Finite state58 batch")
    n = len(x); k = len(batch["pair_names"])
    if k != 22 or len(set(batch["pair_names"])) != k:raise ValueError("Complete22 distinct boundary-pair names")
    R=x[:,43:52].reshape(n,3,3)
    if np.max(abs(R.transpose(0,2,1)@R-np.eye(3)),initial=0)>1e-5 or np.max(abs(np.linalg.det(R)-1),initial=0)>1e-5:
        raise ValueError("Raw state body orientation domain; no projection")
    shapes = {"absolute_u20": (n,20), "native_h_s": (n,), "n_substeps": (n,),
        "pair_relative_translation_BODY_m": (n,k,3), "pair_relative_rotation_BODY9": (n,k,9),
        "pair_signed_gap_m": (n,k), "pair_gap_lower_bound_m": (n,k), "pair_geometry_known": (n,k),
        "normal_on_phone_BODY3": (n,k,3), "normal_unique_known": (n,k)}
    for key, shape in shapes.items():
        if np.asarray(batch[key]).shape != shape:raise ValueError("Shape: "+key)
    if not np.isfinite(batch["absolute_u20"]).all() or not np.isfinite(batch["native_h_s"]).all():raise ValueError("Finite source commands/native h")
    nsub=np.asarray(batch["n_substeps"]);dt=np.asarray(batch["native_h_s"])
    if nsub.dtype.kind not in "iu" or dt.dtype.kind != "f" or np.any(nsub<=0) or np.any(dt<=0):
        raise ValueError("Positive integer substeps and floating positive native h")
    clock_tol=4*np.finfo(dt.dtype).eps*.02
    if not np.isin(nsub,[64,128]).all() or np.any(abs(nsub*dt-.02)>clock_tol):raise ValueError("Explicit64/128 substeps per nominal20ms, representation tolerance only")
    known=np.asarray(batch["pair_geometry_known"]);unique=np.asarray(batch["normal_unique_known"])
    if known.dtype != np.bool_ or unique.dtype != np.bool_ or np.any(unique & ~known):
        raise ValueError("Explicit geometry/unique-normal masks")
    for key in ("pair_relative_translation_BODY_m","pair_relative_rotation_BODY9","pair_signed_gap_m"):
        if not np.isfinite(np.asarray(batch[key])[known]).all():raise ValueError("Known geometry must be finite")
    if not np.isfinite(np.asarray(batch["normal_on_phone_BODY3"])[unique]).all():raise ValueError("Known unique normal")
    nr=np.asarray(batch["normal_on_phone_BODY3"])[unique]
    if np.max(abs(np.linalg.norm(nr,axis=-1)-1),initial=0)>1e-5:raise ValueError("Known normal unit length")
    pr=np.asarray(batch["pair_relative_rotation_BODY9"])[known].reshape(-1,3,3)
    if np.max(abs(pr.transpose(0,2,1)@pr-np.eye(3)),initial=0)>1e-5 or np.max(abs(np.linalg.det(pr)-1),initial=0)>1e-5:
        raise ValueError("Known relative rotations must be proper, not reflections")
    lower=np.asarray(batch["pair_gap_lower_bound_m"]);gap=np.asarray(batch["pair_signed_gap_m"])
    finite_lower=known & np.isfinite(lower)
    if np.any(lower[finite_lower]>gap[finite_lower]+1e-10):raise ValueError("A lower bound is not a larger exact gap")
    if batch["source_state_schema"] != "phone_COM_world_state58:1":raise ValueError("Not the old pen state/chart")
    if batch["model_sha256"] != MODEL_SHA256:raise ValueError("Fixed compiled phone model identity")
    return {"schema":SCHEMA,"rows":n,"pairs":k,"strict_zero_masks_issued":0,
        "scope":"Current-state input ABI only; does not prove collision avoidance or causal disconnection"}


def strict_object_zero_mask(state_count, actuator_count=20):
    """No v1 certificate verifier exists: retain every potential action path."""
    return np.zeros((state_count,actuator_count),dtype=np.bool_)


def loss_response_mask(available_side, full_baseline, full_side, finite_pair):
    """Eligibility for future measured finite responses; never a zero target mask."""
    a,b,c,d = map(np.asarray,(available_side,full_baseline,full_side,finite_pair))
    if any(v.dtype != np.bool_ for v in (a,b,c,d)):raise ValueError("Typed actual execution masks")
    return a & b & c & d
