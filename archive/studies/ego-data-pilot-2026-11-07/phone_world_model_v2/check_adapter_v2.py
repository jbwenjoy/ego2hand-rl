"""Bounded pure-tensor and saved-loader contract; no NN or native model calls."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import hashlib
import json
import numpy as np
import phone_adapter as a
import saved_dataset as data

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
DATASET = PILOT / "phone_transition_dataset_v1"
REPORT_SHA = "98e533210bf47859db829a34a3a17c598ae0ce2cbd59b8ce523dfeb6a4757c2d"
OVERLAP_SHA = "3116ad573a92994132172c40ef12798e3928513d292f628468e225723d3b799a"


def pin(p):
    p = Path(p).resolve(strict=True)
    raw = p.read_bytes()
    return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def rejects(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError("Required guard accepted incompatible input")


def main():
    import torch
    torch.set_num_threads(4)
    out = HERE / "contract_v2.json"
    assert not out.exists(), "Fresh proof only"
    cases = {}
    kwargs = {"state_schema": a.STATE_SCHEMA, "delta_schema": a.DELTA_SCHEMA}
    base_R = a._codec().exp_world(np.array([.4, -.2, .6]))
    weights = np.array([[.2, -.4, .7], [.8, -.3, .1], [-.6, .9, .5]])
    zero_errors = {}
    for dtype in (torch.float32, torch.float64):
        x = torch.zeros(58, dtype=dtype)
        x[43:52] = torch.tensor(base_R.reshape(9), dtype=dtype)
        d = torch.zeros(52, dtype=dtype, requires_grad=True)
        y = a.decode_torch(x, d, **kwargs)
        objective = (y[43:52].reshape(3, 3) * torch.tensor(weights, dtype=dtype)).sum()
        g, = torch.autograd.grad(objective, d)
        expected = np.array([np.sum((a._codec().skew(np.eye(3)[j]) @ base_R) * weights) for j in range(3)])
        err = float(np.max(np.abs(g[43:46].numpy() - expected)))
        assert bool(torch.isfinite(g).all()) and torch.equal(y, x)
        assert err <= (2e-6 if dtype == torch.float32 else 1e-12)
        zero_errors[str(dtype)] = err
    cases["zero_rotation_analytic_derivative_F32_F64"] = {
        "passed": True, "max_rotation_gradient_abs_errors": zero_errors,
        "scope": "Pure decoder derivative, not a native-physics or learned-model gradient"}

    x = torch.zeros((2, 58), dtype=torch.float64)
    x[:, 43:52] = torch.tensor(base_R.reshape(9))
    d = torch.zeros((2, 52), dtype=torch.float64)
    d[:, 43:46] = torch.tensor([.21, .13, -.11], dtype=torch.float64)
    d.requires_grad_(True)
    y = a.decode_torch(x, d, **kwargs)
    W = torch.tensor(weights)
    g, = torch.autograd.grad((y[:, 43:52].reshape(2, 3, 3) * W).sum(), d)
    direction = torch.zeros_like(d)
    direction[:, 43:46] = torch.tensor([.7, -.4, .2], dtype=torch.float64)
    eps = 1e-6
    plus = a.decode_torch(x, d.detach() + eps * direction, **kwargs)
    minus = a.decode_torch(x, d.detach() - eps * direction, **kwargs)
    fd = float((((plus[:, 43:52] - minus[:, 43:52]) / (2 * eps)).reshape(2, 3, 3) * W).sum())
    ad = float((g * direction).sum())
    assert abs(fd - ad) < 1e-8
    left = a._codec().exp_world(np.array([.21, .13, -.11])) @ base_R
    right = base_R @ a._codec().exp_world(np.array([.21, .13, -.11]))
    assert np.max(abs(left - right)) > .05
    assert np.max(abs(y[0, 43:52].detach().numpy().reshape(3, 3) - left)) < 1e-14
    u = torch.full((2, 20), .25, dtype=torch.float64)
    calls = []
    def supplied_closed_form(features):
        calls.append(features.clone())
        return torch.zeros((2, 52), dtype=features.dtype)
    next_x, _ = a.predict_next_torch(x, u, supplied_closed_form, state_schema=a.STATE_SCHEMA,
        action_schema=a.ACTION_SCHEMA, input_schema=a.INPUT_SCHEMA, delta_schema=a.DELTA_SCHEMA)
    assert len(calls) == 1 and calls[0].shape == (2, 78) and torch.equal(calls[0][:, :58], x)
    assert torch.equal(calls[0][:, 58:], u) and torch.equal(next_x, x)
    rejects(lambda: a.decode_torch(x, d, state_schema=a.STATE_SCHEMA, delta_schema="native_WM_chart52"))
    reflected = x.clone(); reflected[:, 43:46] *= -1
    rejects(lambda: a.validate_state_torch(reflected, state_schema=a.STATE_SCHEMA))
    cases["noncommuting_world_left_decode_and_directional_FD"] = {
        "passed": True, "directional_AD": ad, "directional_FD": fd,
        "absolute_difference": abs(fd-ad), "wrong_right_decode_max_abs_difference": float(np.max(abs(left-right))),
        "external_closed_form_callback_calls": len(calls), "neural_network_calls": 0}

    ds = data.load_saved_dataset(DATASET / "dataset_v1/report.json", expected_report_sha256=REPORT_SHA,
        overlap_audit_path=DATASET / "overlap_audit_v1.json", expected_overlap_sha256=OVERLAP_SHA)
    assert len(ds.x58) == 322 and ds.branch_rows().shape == (161, 2)
    assert ds.boundary_components == (tuple(range(7)),) and len(ds.cross_boundary_alias_rows) == 6
    assert not ds.x58.flags.writeable
    assert ds.require_single_partition(["fitting_only"] * 322)["independent_holdout_qualified"] is False
    rejects(lambda: ds.require_single_partition(["train" if c < 6 else "test" for c in ds.source_boundary_index]))
    # Original row0, fixed before reading values; one real saved decoder call.
    decoded = a.decode_torch(torch.from_numpy(ds.x58[:1].copy()), torch.from_numpy(ds.physical_delta52[:1].copy()), **kwargs)
    error = float(np.max(abs(decoded.numpy() - ds.target58[:1])))
    assert error <= 2e-12
    cases["original322_loader_alias_protection_and_fixed_row0_decode"] = {
        "passed": True, "rows": 322, "H2_pairs": 161, "boundaries": 7,
        "cross_boundary_alias_pairs": [list(v) for v in ds.cross_boundary_alias_rows],
        "alias_connected_components": [list(v) for v in ds.boundary_components],
        "fixed_original_row0_decode_max_abs": error,
        "split_assigned": False, "independent_holdout_qualified": False,
        "source_pins": ds.source_pins}
    report = {"schema": "phone_world_model_v2_adapter_contract:1", "passed": True,
        "cases": cases, "source_pins": {f: pin(HERE/f) for f in ("phone_adapter.py", "saved_dataset.py", "check_adapter_v2.py")},
        "frozen_scalar_codec": pin(a.CODEC_PATH), "torch_version": torch.__version__,
        "counts": {"saved_NPZ_open": 1, "fixed_saved_rows_decoded": 1, "pure_tensor_autograd_calls": 3,
                   "external_closed_form_callback_calls": 1, "torch_load_NN_ctor_NN_forward_optimizer_FK_physics_GPU": 0},
        "CUDA_initialized": torch.cuda.is_initialized(),
        "scope": "Codec/interface and saved grouping only. No fitted model, dynamics accuracy, Markov, split, ego or policy qualification."}
    assert report["CUDA_initialized"] is False
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"report": pin(out), "passed": True, "cases": len(cases), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
