"""Saved-only NumPy comparison of fixed-parent private GPU benchmarks.

Tolerance preset v1 is fixed before reading a candidate. It is an engineering
comparison envelope, not a native-gradient or policy-quality certificate.
No Torch, optimizer, native simulator, or GPU API is imported or called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

SCHEMA = "supermimic_fixed_private_GPU_update_benchmark:2"
TOLERANCES = {
    "loss": {"atol": 1e-9, "rtol": 1e-6},
    "gradient": {"atol": 1e-7, "rtol": 1e-5, "relative_L2_max": 1e-5},
    "state": {"atol": 1e-7, "rtol": 1e-6, "relative_L2_max": 1e-6},
    "normalizer_or_discrete": {"typed_raw_bytes_equal": True},
}


def pin(path):
    path = Path(path).resolve(strict=True)
    b = path.read_bytes()
    return {"path": str(path), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def checked_pin(record):
    path = Path(record["path"])
    if not path.is_absolute() or pin(path) != record:
        raise ValueError("Input pin mismatch: " + str(path))
    return path


def _content(record):
    return record["bytes"], record["sha256"]


def validate_report(report):
    if report["schema"] != SCHEMA or report["status"] != "COMPLETE_PRIVATE_TIMING_NOT_POLICY_TRAINING":
        raise ValueError("A completed fixed-parent benchmark v2 is required")
    if (report["B"], report["H"], report["NN_dtype"], report["physical_dtype"], report["AMP"], report["TF32"]) != (8, 2, "float32", "float64", False, False):
        raise ValueError("Expected B8/H2, FP32 NN/FP64 physical, AMP/TF32 disabled")
    rows = report["rows"]
    if not rows or [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Complete sequential benchmark update records required")
    if report["private_paired_updates"] != len(rows) or report["private_Adam_steps"] != 5 * len(rows):
        raise ValueError("Exactly five declared private Adam steps per paired update")
    if report["no_native_or_policy_qualification"] is not True or report["real_bank_resident"] is not True:
        raise ValueError("Wrong benchmark scope or row-bank protocol")
    micro = report["microbatch"]
    if type(micro) is not int or micro <= 0 or report["B"] % micro:
        raise ValueError("Positive divisible microbatch")
    if not any(r["phase"] == "measured" for r in rows):
        raise ValueError("No measured updates")
    modes = set(rows[0]["modes"])
    if modes != {"A1_tracking", "A2_engineered_material_proxy"}:
        raise ValueError("Both fixed phone task arms required")
    for row in rows:
        if row["phase"] not in {"warmup", "measured", "profiler"} or len(row["row_ids"]) != report["B"]:
            raise ValueError("Invalid phase or common replay batch")
        if any(type(v) is not int for v in row["row_ids"]) or set(row["modes"]) != modes:
            raise ValueError("Typed row identities and fixed arms required")
        if not np.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Finite positive measured duration")
        if any(set(v) != {"real_TD_loss", "imagined_Actor_loss"} for v in row["modes"].values()):
            raise ValueError("Exact two loss fields per arm")
    checked_pin(report["fixture"])
    config = tomllib.loads(checked_pin(report["configuration"]).read_text(encoding="utf8"))
    return config


def protocol_comparison(a, b, *, allow_microbatch_change=False):
    ca, cb = validate_report(a), validate_report(b)
    if allow_microbatch_change:
        # This explicit flag is only the planned micro2 -> micro8 comparison.
        micro_ok = (a["microbatch"], b["microbatch"]) in {(2, 2), (2, 8)}
        ca["training"].pop("microbatch", None)
        cb["training"].pop("microbatch", None)
    else:
        micro_ok = a["microbatch"] == b["microbatch"]
    fixed_fields = ["schema", "status", "device", "B", "H", "NN_dtype", "physical_dtype", "AMP", "TF32", "WM_sources", "private_paired_updates", "private_Adam_steps", "real_bank_resident", "real_bank_scope"]
    checks = {"same_" + k: a[k] == b[k] for k in fixed_fields}
    checks.update(
        same_semantic_configuration=ca == cb,
        same_parent_checkpoint=_content(a["parent_checkpoint"]) == _content(b["parent_checkpoint"]),
        same_benchmark_tool=_content(a["tool"]) == _content(b["tool"]),
        microbatch_permission=micro_ok,
        same_indices_phases_row_IDs=[(r["index"], r["phase"], r["row_ids"]) for r in a["rows"]] == [(r["index"], r["phase"], r["row_ids"]) for r in b["rows"]],
    )
    return checks


def validate_fixture(arrays):
    groups = {}
    for name, value in arrays.items():
        parts = name.split("/", 2)
        if len(parts) != 3 or parts[1] not in {"state", "pre_clip", "post_clip"} or value.dtype.kind not in "fibu":
            raise ValueError("Unsupported named fixture leaf: " + name)
        groups.setdefault((parts[0], parts[1]), set()).add(parts[2])
    expected_grad = {"WM", "A1_tracking_ACTOR", "A1_tracking_CRITIC", "A2_engineered_material_proxy_ACTOR", "A2_engineered_material_proxy_CRITIC"}
    expected_state = {"WM"} | {m + "_" + k for m in ["A1_tracking", "A2_engineered_material_proxy"] for k in ["actor", "critic", "target_critic"]}
    if {g for g, kind in groups if kind == "state"} != expected_state:
        raise ValueError("All seven WM/Actor/Critic/target state groups required")
    for kind in ["pre_clip", "post_clip"]:
        if {g for g, side in groups if side == kind} != expected_grad:
            raise ValueError("All five named pre/post gradient groups required")
    for group in expected_grad:
        if groups[group, "pre_clip"] != groups[group, "post_clip"]:
            raise ValueError("Pre/post named parameter set differs: " + group)
        state_group = group if group == "WM" else group.rsplit("_", 1)[0] + "_" + group.rsplit("_", 1)[1].lower()
        if not groups[group, "pre_clip"] <= groups[state_group, "state"]:
            raise ValueError("Gradient name has no corresponding serialized parameter")


def array_metrics(reference, candidate, *, category, exact=False):
    """Raw finite/error metrics. Zero reference norm has no fabricated ratio."""
    a, b = np.asarray(reference), np.asarray(candidate)
    base = dict(reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype), reference_shape=list(a.shape), candidate_shape=list(b.shape))
    base.update(same_dtype=a.dtype == b.dtype, same_shape=a.shape == b.shape)
    if not base["same_dtype"] or not base["same_shape"]:
        return base | {"pass": False, "reason": "dtype_or_shape_mismatch"}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    raw_equal = a.tobytes(order="C") == b.tobytes(order="C")
    base.update(finite=finite, raw_bytes_equal=raw_equal)
    if not finite:
        return base | {"pass": False, "reason": "nonfinite_leaf"}
    with np.errstate(over="ignore", invalid="ignore"):
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        delta = bb - aa
        maxabs = float(np.max(np.abs(delta), initial=0.))
        l2 = float(np.linalg.norm(delta.reshape(-1)))
        reference_l2 = float(np.linalg.norm(aa.reshape(-1)))
        relative = l2 / reference_l2 if reference_l2 > 0 else None
    if not all(np.isfinite(v) for v in [maxabs, l2, reference_l2]) or relative is not None and not np.isfinite(relative):
        return base | {"pass": False, "reason": "finite_inputs_but_metric_overflow", "max_abs": maxabs if np.isfinite(maxabs) else None, "relative_L2": None}
    base.update(max_abs=maxabs, difference_L2=l2, reference_L2=reference_l2, relative_L2=relative)
    if exact or a.dtype.kind in "ibu":
        return base | {"pass": raw_equal, "criterion": "typed_raw_bytes_exact"}
    tol = TOLERANCES[category]
    bound = tol["atol"] + tol["rtol"] * np.abs(aa)
    elementwise = bool(np.all(np.abs(delta) <= bound))
    relative_ok = relative is None or "relative_L2_max" not in tol or relative <= tol["relative_L2_max"]
    base.update(max_elementwise_tolerance_ratio=float(np.max(np.abs(delta) / bound, initial=0.)), criterion=category, elementwise_pass=elementwise, relative_L2_pass=relative_ok)
    return base | {"pass": bool(elementwise and relative_ok)}


def compare(baseline_path, candidate_path, *, allow_microbatch_change=False):
    result = dict(schema="supermimic_saved_GPU_benchmark_comparison:1", baseline={"path": str(Path(baseline_path).resolve())}, candidate={"path": str(Path(candidate_path).resolve())}, tool=pin(__file__), tolerance_preset="fixed_v1", tolerances=TOLERANCES, allow_microbatch_change=allow_microbatch_change, scope="Pure NumPy saved comparison. Declared five-Adam budgets checked; moment arrays/RNG/WM row IDs were not exported and cannot be independently compared here. No native/policy or long-training equivalence claim.")
    details = {"arrays": {}, "losses": {}}
    try:
        result.update(baseline=pin(baseline_path), candidate=pin(candidate_path))
        a = json.loads(Path(baseline_path).read_text(encoding="utf8")); b = json.loads(Path(candidate_path).read_text(encoding="utf8"))
        checks = protocol_comparison(a, b, allow_microbatch_change=allow_microbatch_change)
        result["protocol_checks"] = checks
        result["fixture_pins"] = {"baseline": a["fixture"], "candidate": b["fixture"]}
        result["source_changes"] = {"added": sorted(b["source_pins"].keys() - a["source_pins"].keys()), "removed": sorted(a["source_pins"].keys() - b["source_pins"].keys()), "changed": sorted(k for k in a["source_pins"].keys() & b["source_pins"].keys() if _content(a["source_pins"][k]) != _content(b["source_pins"][k]))}
        if not all(checks.values()):
            result.update(status="FAIL_PROTOCOL_MISMATCH", passed=False)
            return result, details
        with np.load(checked_pin(a["fixture"]), allow_pickle=False) as z: fa = {k: z[k] for k in z.files}
        with np.load(checked_pin(b["fixture"]), allow_pickle=False) as z: fb = {k: z[k] for k in z.files}
        validate_fixture(fa); validate_fixture(fb)
        result["fixture_keys"] = {"same": fa.keys() == fb.keys(), "baseline_only": sorted(fa.keys() - fb.keys()), "candidate_only": sorted(fb.keys() - fa.keys())}
        for name in sorted(fa.keys() & fb.keys()):
            _, kind, parameter = name.split("/", 2)
            exact = kind == "state" and parameter.startswith("norm.")
            details["arrays"][name] = array_metrics(fa[name], fb[name], category="state" if kind == "state" else "gradient", exact=exact)
        for ra, rb in zip(a["rows"], b["rows"]):
            losses_a = {"WM_loss": ra["WM_loss"]} | {m + "/" + k: v for m, values in ra["modes"].items() for k, v in values.items()}
            losses_b = {"WM_loss": rb["WM_loss"]} | {m + "/" + k: v for m, values in rb["modes"].items() for k, v in values.items()}
            for name in sorted(losses_a):
                details["losses"][str(ra["index"]) + "/" + name] = array_metrics(np.asarray(losses_a[name], dtype=np.float64), np.asarray(losses_b[name], dtype=np.float64), category="loss")
        all_metrics = {**details["arrays"], **{"loss/" + k: v for k, v in details["losses"].items()}}
        failed = [k for k, v in all_metrics.items() if not v["pass"]]
        result.update(passed=result["fixture_keys"]["same"] and not failed, compared_arrays=len(details["arrays"]), compared_loss_scalars=len(details["losses"]), failed_names=failed, raw_equal_arrays=sum(v.get("raw_bytes_equal", False) for v in details["arrays"].values()))
        result["status"] = "PASS_FIXED_ENGINEERING_ENVELOPE" if result["passed"] else "FAIL_NUMERICAL_COMPARISON"
        result["worst_max_abs"] = sorted(({"name": k, "max_abs": v.get("max_abs"), "relative_L2": v.get("relative_L2"), "pass": v["pass"]} for k, v in all_metrics.items() if v.get("max_abs") is not None), key=lambda v: v["max_abs"], reverse=True)[:12]
        result["timing"] = {"baseline_steady_median_s": a["steady_seconds_median"], "candidate_steady_median_s": b["steady_seconds_median"], "baseline_over_candidate_median_ratio": a["steady_seconds_median"] / b["steady_seconds_median"], "baseline_peak_allocated_bytes": a["peak_allocated_bytes"], "candidate_peak_allocated_bytes": b["peak_allocated_bytes"], "scope": "Short same-parent benchmark only; numerical status reported separately, no universal speed claim"}
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        result.update(status="FAIL_INPUT_OR_SCHEMA", passed=False, error=repr(error))
    return result, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-microbatch-change", action="store_true", help="Only permit the declared micro2 -> micro8 change; all other protocol fields remain strict")
    args = parser.parse_args(argv)
    output = args.output.resolve(); detail_path = output.with_name(output.stem + ".details.json")
    if output.exists() or detail_path.exists():
        raise FileExistsError("Comparison outputs must be fresh")
    result, details = compare(args.baseline, args.candidate, allow_microbatch_change=args.allow_microbatch_change)
    output.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("x", encoding="utf8") as f: json.dump(details, f, indent=2, allow_nan=False)
    result["details"] = pin(detail_path)
    with output.open("x", encoding="utf8") as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"report": pin(output), "status": result["status"], "passed": result["passed"], "Torch_imported": "torch" in sys.modules}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
