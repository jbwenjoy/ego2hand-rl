"""One bounded synthetic contract. SOURCE ONLY until Root authorizes this CLI.

No acquisition files/checkpoints are read. The real-size small MLP is evaluated
only on four invented rows, one optimizer update, and one own temporary save/load.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import train_phone_wm as m


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-synthetic", action="store_true")
    a = p.parse_args()
    if not a.run_synthetic:
        print(json.dumps({"status": "SOURCE_ONLY", "planned_synthetic_updates": 1, "actual_execution": 0})); return
    import torch
    torch.set_num_threads(4); torch.set_default_dtype(torch.float32); torch.manual_seed(7)
    n = 4
    x = np.zeros((n, 58), dtype=np.float64); x[:, 43:52] = np.eye(3).reshape(9)
    y = x.copy(); y[:, 40] = [.001, .002, .003, np.nan]
    delta = np.zeros((n, 52)); delta[:, 40] = [.001, .002, .003, np.nan]
    # One retained partial, deliberately invalid full-H target; only its mask excludes it.
    y[-1] = np.nan; delta[-1] = np.nan
    raw = {"x58_F64": x, "u20_F64": np.zeros((n, 20)), "y58_F64": y, "target_delta52_F64": delta,
        "native_dt_s_F64": np.array([.00015625] * n), "n_substeps_i64": np.array([128] * n, dtype=np.int64),
        "executed_substeps_i64": np.array([128, 128, 128, 2], dtype=np.int64),
        "complete_H_bool": np.array([True, True, True, False]), "training_eligible_bool": np.array([True, True, True, False]),
        "row_id_i64": np.arange(n, dtype=np.int64), "family_id_i64": np.zeros(n, dtype=np.int64),
        "episode_id_i64": np.zeros(n, dtype=np.int64), "interval_index_i64": np.arange(n, dtype=np.int64),
        "contact_summary_F64": np.array([[0, 1, 0, .1, .2, .3, .4], [.2, .8, .1, .2, .3, .4, .5], [.4, .6, .2, .3, .4, .5, .6], [0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)}
    with tempfile.TemporaryDirectory(prefix="phone_wm_synthetic_") as tmp:
        h = Path(tmp); f = h/"invented_train.npz"; np.savez_compressed(f, **raw)
        train = m.load_partition(m.pin(f), [0]); assert train["selection"]["used_row_ids"] == [0, 1, 2]
        assert train["selection"]["excluded_rows"][0]["row_id"] == 3
        norm = m.fit_normalizers(train)
        assert norm["x80_mean"][78] == .00015625 and norm["x80_mean"][79] == 128
        assert norm["x80_std"][0] == 1e-6
        assert abs(norm["physical_world_delta52_mean"][40] - .002) < 1e-15
        assert abs(norm["physical_world_delta52_std"][40] - np.sqrt(2e-6/3)) < 1e-15
        model = m.make_model(norm, torch); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
        batch = m.tensors(train, torch)
        before_rng = m.rng_state(torch)
        prediction = model(batch["x80"]); loss = m.loss_terms(model, prediction, batch, torch)
        assert torch.equal(model.physical_world_delta52_mean, torch.tensor(norm["physical_world_delta52_mean"], dtype=torch.float32))
        loss["total"].backward()
        assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
        optimizer.step(); m._finite_parameters_and_adam(model, optimizer, 1, torch)
        with torch.no_grad():
            saved_prediction = model(batch["x80"])
            metrics = m.validation_metrics(model, batch, saved_prediction, torch)
        path = h/"synthetic.pt"
        m.atomic_checkpoint(path, {"schema": m.ARTIFACT_SCHEMA, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, torch)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert m.raw_equal(model.state_dict(), loaded["model_state"], torch)
        assert m.raw_equal(optimizer.state_dict(), loaded["optimizer_state"], torch)
        model.load_state_dict(loaded["model_state"], strict=True)
        with torch.no_grad():
            recovered = model(batch["x80"])
        assert all(torch.equal(saved_prediction[k], recovered[k]) for k in recovered)
        assert m.raw_equal(before_rng, m.rng_state(torch), torch) and not torch.cuda.is_initialized()
    print(json.dumps({"status": "PASS_ONE_SYNTHETIC_CONTRACT_NOT_ACTUAL_DATA_OR_TRAINING_QUALIFICATION",
        "synthetic_updates": 1, "synthetic_forward_calls": 3, "synthetic_backward": 1,
        "own_temporary_checkpoint_loads": 1, "actual_acquisition_arrays_or_old_weights_native_GPU": 0,
        "partial_row_retained_but_not_fit": True, "finite": True, "metrics": metrics}))


if __name__ == "__main__":
    main()
