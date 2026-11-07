"""Fresh phone WM source. Default metadata only; no inherited pen model.

Torch/model/data-array execution occurs only in run_training after an audited,
explicitly authorized protocol. TEST is hash-bound but never opened as an NPZ.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import random
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT.parent / "phone_world_model_v2/phone_adapter.py"
ADAPTER_SHA = "a257739157b631c7dc997c869125ba64550e514c0cf3b986b59f5ce7d09ada88"
SCHEMA = "phone_world_model_training_protocol:1"
DATASET_SCHEMA = "phone_episode_acquisition_result:1"
DESIGN_SHA = "d3f4d8e0a4c44844fc8d51fb261be32f893ced20aa97a8e6404d1b08f7c229c4"
FAMILY_PLAN_SHA = "7336ae9243290158c061791a28f6f5b668cc59091e6c1cbdfa51a0f8f4704739"
ARTIFACT_SCHEMA = "phone_physical_world_model_checkpoint:1"
INPUT_SCHEMA = "phone_COM_world58_absolute_u20_h_s_nsubsteps80:1"
OUTPUT_SCHEMA = "phone_normalized_world_delta52_pad_coverage2_logforce5:1"
RECIPE = {
    "seed": 1729, "cpu_threads": 4, "dtype": "float32", "device": "cpu",
    "hidden": [128, 128], "activation": "SiLU", "updates": 5000,
    "checkpoint_updates": [250, 1000, 5000], "batch": "all_complete_TRAIN_rows",
    "optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0.,
    "betas": [.9, .999], "adam_epsilon": 1e-8, "foreach": False,
    "clip_grad_norm": 1., "input_std_floor": 1e-6,
    "physical_delta_std_floor": 1e-6, "logforce_std_floor": .001,
    "head_initialization": "zero_last_weights_delta_mean_coverage_mean_logforce_mean",
    "coverage_initial_probability_clip": .001,
    "lambda_delta": 1., "lambda_coverage": 1., "lambda_logforce": 1.,
    "background_continue_to_final": True,
}
FIELD_MAP = {"state58": "x58_F64", "action20": "u20_F64", "next_state58": "y58_F64",
    "physical_world_delta52": "target_delta52_F64", "native_h_s": "native_dt_s_F64",
    "substep_count": "n_substeps_i64", "family_id": "family_id_i64", "interval_index": "interval_index_i64"}
ARRAY_KEYS = set(FIELD_MAP.values()) | {"executed_substeps_i64", "complete_H_bool", "training_eligible_bool",
    "row_id_i64", "episode_id_i64", "contact_summary_F64"}
INPUTS = {"dataset_report", "dataset_saved_review", "acquisition_protocol"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    p = Path(path).resolve(strict=True)
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def checked(record):
    p = Path(record["path"])
    require(p.is_absolute(), "Absolute pinned path required")
    a = pin(p)
    require(a["bytes"] == record["bytes"] and a["sha256"] == record["sha256"], "Pinned input/source changed")
    return p.resolve(strict=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def declaration():
    return {"protocol_schema": SCHEMA, "dataset_schema": DATASET_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA, "input_schema": INPUT_SCHEMA,
        "output_schema": OUTPUT_SCHEMA, "recipe": RECIPE, "inputs": sorted(INPUTS),
        "dataset_array_keys": sorted(ARRAY_KEYS), "actual_execution": 0,
        "scope": "New phone model source; no inherited pen weights/chart/normalizer or policy qualification"}


def metadata_preflight(protocol_path):
    p = read(protocol_path)
    require(p.get("schema") == SCHEMA and p.get("role") == "fresh_phone_family_state_and_contact_summary", "New phone training namespace")
    require(type(p.get("CPU_run_authorized")) is bool and p.get("GPU_run_authorized") is False, "Explicit CPU-only authorization")
    require(p.get("recipe") == RECIPE and set(p.get("inputs", {})) == INPUTS, "Fixed recipe and exact inputs")
    require(p.get("input_schema") == INPUT_SCHEMA and p.get("output_schema") == OUTPUT_SCHEMA, "Physical phone input/output schema")
    out = Path(p["output"])
    require(out.is_absolute() and out.parent.resolve() == ROOT.resolve() and not out.exists(), "Fresh exact training output")
    for r in [*p["inputs"].values(), *p["source_pins"].values()]:
        checked(r)
    require(checked(p["source_pins"]["trainer"]).samefile(__file__)
        and checked(p["source_pins"]["adapter"]).samefile(ADAPTER)
        and pin(ADAPTER)["sha256"] == ADAPTER_SHA, "Current trainer and frozen left-world decoder")
    raw = read(checked(p["inputs"]["dataset_report"]))
    require(raw.get("schema") == DATASET_SCHEMA and raw.get("status") == "COMPLETE_GROUPED_NATIVE_ACQUISITION"
        and raw.get("partition_qualified") is True and raw.get("cross_partition_alias_groups") == []
        and raw.get("no_deduplication_or_reassignment") is True, "Complete grouped acquisition without cross-split aliases")
    require(raw.get("protocol") == p["inputs"]["acquisition_protocol"]
        and raw["design"]["sha256"] == DESIGN_SHA and raw["family_plan"]["sha256"] == FAMILY_PLAN_SHA,
        "Exact pre-acquisition design/family plan, not a retrospective split")
    design = read(checked(raw["design"]))
    plan = read(checked(raw["family_plan"]))
    require(plan.get("schema") == "phone_episode_family_plan:1" and len(plan["families"]) == 48
        and design["family_plan"] == raw["family_plan"], "Prior literal48-family plan")
    # This is an in-memory interface map, not a new data export or qualification.
    d = {"schema": "phone_H20ms_family_training_dataset:1", "state_schema": raw["state_schema"],
        "physical_delta_schema": raw["delta_schema"], "array_schema": raw["array_schema"],
        "training_mask_rule_source_bound": True, "partial_rows_retained": True, "field_map": FIELD_MAP,
        "training_mask_review_delegation": p["inputs"]["dataset_saved_review"],
        "acquisition_protocol": raw["protocol"], "source_acquisition_result_pin": p["inputs"]["dataset_report"],
        "family_split": {k: [f["family_id"] for f in plan["families"] if f["split"] == k]
            for k in ("train", "validation", "test")},
        "family_split_frozen_before_acquisition": design["split"]["frozen_before_native"],
        "cross_partition_receipt_control_aliases": raw["cross_partition_alias_groups"],
        "paired_timesteps_same_family": True, "partition_arrays": raw["outputs"],
        "sample_identities": raw["sample_identities"], "source_design": raw["design"],
        "mask_reason_scope": "Audited acquisition contract/source; trainer independently verifies every used row complete and finite, not all discarded warning histories",
        "contact_summary_contract": {"coverage_relation": "pad_normal_N > threshold_N", "coverage_threshold_N": .1,
            "force_phase": "all_substep_solver_PRE", "endpoint_phase": "last_solver_PRE_H_minus_h", "state_target_phase": "POST_H20ms"}}
    require(d.get("state_schema") == "phone_COM_world_state58:1"
        and d.get("physical_delta_schema") == "phone_physical_world_delta52:2"
        and d.get("array_schema") == "phone_episode_H20ms_arrays:1"
        and d.get("training_mask_rule_source_bound") is True
        and d.get("partial_rows_retained") is True and d.get("field_map") == FIELD_MAP,
        "Raw physical dataset, retained partials and explicit complete-H mask required")
    require(d.get("acquisition_protocol") == p["inputs"]["acquisition_protocol"], "Acquisition protocol identity")
    split = d["family_split"]
    require(set(split) == {"train", "validation", "test"}
        and [len(split[k]) for k in ("train", "validation", "test")] == [36, 6, 6]
        and len(set(sum((split[k] for k in split), []))) == 48,
        "Prior48-family split36/6/6; no random row split")
    require(d.get("family_split_frozen_before_acquisition") is True
        and d.get("cross_partition_receipt_control_aliases") == []
        and d.get("paired_timesteps_same_family") is True, "Frozen family/alias separation")
    require(set(d["partition_arrays"]) == set(split), "Three separately pinned partition NPZs")
    for r in d["partition_arrays"].values():
        checked(r)  # hash only, including TEST; never np.load(TEST)
    checked(d["sample_identities"])  # Source masks/reasons remain pinned, not silently discarded.
    audit = read(checked(p["inputs"]["dataset_saved_review"]))
    require(audit.get("schema") == p["dataset_audit_schema"] and audit.get("status") == p["dataset_audit_status"]
        and audit.get("engineering_qualified") is True and audit.get("dataset_report") == p["inputs"]["dataset_report"]
        and audit.get("checks") and all(type(v) is bool and v for v in audit["checks"].values()), "Positive exact saved dataset review")
    c = d["contact_summary_contract"]
    require(c.get("coverage_relation") == "pad_normal_N > threshold_N"
        and type(c.get("coverage_threshold_N")) in (int, float) and math.isfinite(c["coverage_threshold_N"])
        and c["coverage_threshold_N"] == .1
        and c.get("force_phase") == "all_substep_solver_PRE"
        and c.get("endpoint_phase") == "last_solver_PRE_H_minus_h"
        and c.get("state_target_phase") == "POST_H20ms", "Explicit force/state label time scopes")
    return p, d, out


def _adapter():
    import sys
    require(pin(ADAPTER)["sha256"] == ADAPTER_SHA, "Frozen phone adapter")
    spec = importlib.util.spec_from_file_location("phone_training_frozen_adapter", ADAPTER)
    a = importlib.util.module_from_spec(spec); sys.modules[spec.name] = a; spec.loader.exec_module(a)
    return a


def load_partition(record, allowed_families):
    with np.load(checked(record), allow_pickle=False) as z:
        require(set(z.files) == ARRAY_KEYS, "Exact partition array membership")
        raw = {k: z[k].copy() for k in ARRAY_KEYS}
    a = {k: raw[v] for k, v in FIELD_MAP.items()}
    n = len(a["family_id"])
    require(n > 0, "At least one saved interval in this partition")
    for k in ("complete_H_bool", "training_eligible_bool"):
        require(raw[k].shape == (n,) and raw[k].dtype == np.bool_, "Explicit raw completion/eligibility mask")
    for k in ("row_id_i64", "episode_id_i64", "executed_substeps_i64"):
        require(raw[k].shape == (n,) and raw[k].dtype == np.int64, "Stable row/episode/step identities")
    require(len(set(raw["row_id_i64"].tolist())) == n, "Unique raw row IDs")
    require(raw["contact_summary_F64"].shape == (n, 7) and raw["contact_summary_F64"].dtype == np.float64,
        "Raw seven-field contact summary")
    # Preserve all original masks/IDs; apply ONLY the acquisition's audited mask.
    mask = raw["training_eligible_bool"]
    require(bool(np.any(mask)), "No eligible complete interval in this partition")
    require(bool(np.all(raw["complete_H_bool"][mask]))
        and bool(np.all(raw["executed_substeps_i64"][mask] == a["substep_count"][mask])), "Eligible row must span exactly H")
    selection = {"all_raw_row_ids": raw["row_id_i64"].tolist(), "used_row_ids": raw["row_id_i64"][mask].tolist(),
        "excluded_rows": [{"row_id": int(raw["row_id_i64"][i]), "reason": "source_training_eligible_bool_false",
            "complete_H": bool(raw["complete_H_bool"][i]), "executed_substeps": int(raw["executed_substeps_i64"][i])}
            for i in np.flatnonzero(~mask)],
        "reason_scope": "Exact source eligibility flag; detailed warnings/abort/mask reasons remain in pinned sample_identities, delegated to dataset saved review"}
    a.update(pad_coverage2=raw["contact_summary_F64"][:, :2], pad_min_normal_N2=raw["contact_summary_F64"][:, 2:4],
        pad_endpoint_normal_N2=raw["contact_summary_F64"][:, 4:6], phone_nonpad_max_normal_N=raw["contact_summary_F64"][:, 6:7])
    widths = {"state58": 58, "action20": 20, "next_state58": 58, "physical_world_delta52": 52,
        "pad_coverage2": 2, "pad_min_normal_N2": 2, "pad_endpoint_normal_N2": 2, "phone_nonpad_max_normal_N": 1}
    for key, width in widths.items():
        require(a[key].shape == (n, width) and a[key].dtype == np.float64, "Exact F64 raw partition ABI: " + key)
    require(a["native_h_s"].shape == (n,) and a["native_h_s"].dtype == np.float64
        and bool(np.isfinite(a["native_h_s"]).all()), "Native h array")
    for key in ("family_id", "interval_index", "substep_count"):
        require(a[key].shape == (n,) and a[key].dtype == np.int64, "Exact integer metadata: " + key)
    require(set(a["family_id"].tolist()) <= set(allowed_families), "Partition contains foreign family")
    require(bool(np.all((a["interval_index"] >= 0) & (a["interval_index"] < 10))), "At most10 H20ms intervals")
    pairs = set(zip(a["native_h_s"].tolist(), a["substep_count"].tolist()))
    require(pairs <= {(.0003125, 64), (.00015625, 128)}, "Explicit qualified native h/substep combinations")
    ids = list(zip(a["family_id"].tolist(), a["native_h_s"].tolist(), a["interval_index"].tolist()))
    require(len(set(ids)) == n, "Duplicate family/dt/interval row")
    # Targets of partial rows may be NaN by contract; they are never repaired.
    a = {k: v[mask].copy() for k, v in a.items()}
    require(all(bool(np.isfinite(v).all()) for v in a.values()), "Every eligible input/target/contact value finite")
    require(bool(np.all((a["pad_coverage2"] >= 0) & (a["pad_coverage2"] <= 1))), "Coverage fractions in[0,1]")
    require(all(bool(np.all(a[k] >= 0)) for k in ("pad_min_normal_N2", "pad_endpoint_normal_N2", "phone_nonpad_max_normal_N")), "Positive-normal aggregation, no silent label clamp")
    decoder = _adapter()
    decoder.validate_state_numpy(a["state58"], state_schema=decoder.STATE_SCHEMA)
    decoder.validate_state_numpy(a["next_state58"], state_schema=decoder.STATE_SCHEMA)
    x80 = np.column_stack((a["state58"], a["action20"], a["native_h_s"], a["substep_count"]))
    force5 = np.column_stack((a["pad_min_normal_N2"], a["pad_endpoint_normal_N2"], a["phone_nonpad_max_normal_N"]))
    return {**a, "x80": x80, "logforce5": np.log1p(force5), "selection": selection}


def fit_normalizers(train):
    """TRAIN rows only, F64 population statistics; immutable F32 runtime buffers."""
    out = {}
    for key, floor in (("x80", RECIPE["input_std_floor"]), ("physical_world_delta52", RECIPE["physical_delta_std_floor"]),
                       ("logforce5", RECIPE["logforce_std_floor"])):
        out[key + "_mean"] = train[key].mean(0)
        out[key + "_std"] = np.maximum(train[key].std(0, ddof=0), floor)
    out["coverage2_train_mean"] = train["pad_coverage2"].mean(0)
    require(all(bool(np.isfinite(v).all()) for v in out.values()), "Finite train-only normalization")
    return out


def make_model(normalizers, torch):
    class PhoneWM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(80, 128), torch.nn.SiLU(),
                torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 59))
            for name, value in normalizers.items():
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
            with torch.no_grad():
                self.net[-1].weight.zero_(); self.net[-1].bias.zero_()
                prob = self.coverage2_train_mean.clamp(.001, .999)
                self.net[-1].bias[52:54].copy_(torch.logit(prob))
                logforce = self.logforce5_mean.clamp_min(.001)
                self.net[-1].bias[54:59].copy_(logforce + torch.log(-torch.expm1(-logforce)))

        def forward(self, raw80):
            y = self.net((raw80 - self.x80_mean) / self.x80_std)
            return {"normalized_delta52": y[:, :52], "coverage_logits2": y[:, 52:54],
                    "logforce5": torch.nn.functional.softplus(y[:, 54:59])}

        def physical_delta(self, prediction):
            return prediction["normalized_delta52"] * self.physical_world_delta52_std + self.physical_world_delta52_mean
    return PhoneWM()


def tensors(partition, torch):
    return {k: torch.from_numpy(partition[k].astype(np.float32)) for k in
        ("x80", "state58", "next_state58", "physical_world_delta52", "pad_coverage2", "logforce5")}


def loss_terms(model, prediction, batch, torch):
    target = (batch["physical_world_delta52"] - model.physical_world_delta52_mean) / model.physical_world_delta52_std
    delta = (prediction["normalized_delta52"] - target).square().mean()
    coverage = torch.nn.functional.binary_cross_entropy_with_logits(prediction["coverage_logits2"], batch["pad_coverage2"])
    force = ((prediction["logforce5"] - batch["logforce5"]) / model.logforce5_std).square().mean()
    total = delta + coverage + force
    return {"delta": delta, "coverage": coverage, "logforce": force, "total": total}


def rng_state(torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_CPU": torch.get_rng_state().clone()}


def raw_equal(a, b, torch):
    """Compare saved raw values; never use pickle reserialization as an RNG hash."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
            and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
            and a.dtype == b.dtype and a.tobytes() == b.tobytes())
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(raw_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(raw_equal(x, y, torch) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def atomic_checkpoint(path, payload, torch):
    path = Path(path); require(not path.exists(), "Never overwrite a checkpoint")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _finite_parameters_and_adam(model, optimizer, iteration, torch):
    require(len(optimizer.state) == len(list(model.parameters())), "Adam state for every trainable parameter")
    for parameter in model.parameters():
        require(bool(torch.isfinite(parameter).all()), "Finite parameters")
        s = optimizer.state[parameter]
        require(float(s["step"]) == iteration and all(bool(torch.isfinite(s[k]).all()) for k in ("exp_avg", "exp_avg_sq")), "Finite exact Adam step/moments")


def validation_metrics(model, batch, prediction, torch):
    a = _adapter()
    physical = model.physical_delta(prediction)
    state = a.decode_torch(batch["state58"], physical, state_schema=a.STATE_SCHEMA, delta_schema=a.DELTA_SCHEMA)
    e = (state.double() - batch["next_state58"].double())
    result = {k: float(v) for k, v in loss_terms(model, prediction, batch, torch).items()}
    result["physical_delta_RMSE_per_feature"] = (physical.double() - batch["physical_world_delta52"].double()).square().mean(0).sqrt().tolist()
    result["COM_vector_RMS_m"] = float(e[:, 40:43].square().sum(-1).mean().sqrt())
    result["Rbody_raw9_Frobenius_RMS"] = float(e[:, 43:52].square().sum(-1).mean().sqrt())
    result["coverage_Brier_mean"] = float((prediction["coverage_logits2"].sigmoid() - batch["pad_coverage2"]).square().mean())
    require(all(math.isfinite(v) for k, v in result.items() if not isinstance(v, list)), "Finite validation metrics")
    return result


def run_training(protocol_path):
    p, dataset, out = metadata_preflight(protocol_path)
    require(p["CPU_run_authorized"] is True, "Root execution authorization required")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    require(not torch.cuda.is_initialized(), "No CUDA context")
    torch.set_num_threads(4); torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
    seed = RECIPE["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train = load_partition(dataset["partition_arrays"]["train"], dataset["family_split"]["train"])
    validation = load_partition(dataset["partition_arrays"]["validation"], dataset["family_split"]["validation"])
    require(not set(train["selection"]["all_raw_row_ids"]) & set(validation["selection"]["all_raw_row_ids"]), "TRAIN/VALIDATION row identities overlap")
    norm = fit_normalizers(train)
    model = make_model(norm, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
    tr, va = tensors(train, torch), tensors(validation, torch)
    out.mkdir(); write(out/"protocol.json", p); write(out/"dataset_binding.json", dataset)
    write(out/"used_and_excluded_rows.json", {"train": train["selection"], "validation": validation["selection"],
        "source_masks_and_reasons": dataset["sample_identities"], "TEST_NPZ_reads": 0})
    np.savez_compressed(out/"TRAIN_only_normalizers_F64.npz", **norm)
    start = time.perf_counter(); records = []
    counts = {"partition_NPZ_reads": 2, "TEST_NPZ_reads": 0, "model_constructors": 1,
        "training_FW": 0, "backward": 0, "optimizer_steps": 0, "validation_FW": 0,
        "checkpoint_readback_FW": 0, "own_checkpoint_loads": 0, "physics_Env_GPU_old_checkpoint": 0}
    initial_rng = rng_state(torch)
    phase = "training"
    try:
        with (out/"updates.jsonl").open("x", encoding="utf-8") as log:
            for iteration in range(1, 5001):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tr["x80"]); counts["training_FW"] += 1
                terms = loss_terms(model, prediction, tr, torch)
                require(all(bool(torch.isfinite(v)) for v in terms.values()), "Finite training loss")
                terms["total"].backward(); counts["backward"] += 1
                require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()), "Finite every-parameter PRE-step gradient")
                preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
                require(all(bool(torch.isfinite(v.grad).all()) for v in model.parameters()), "Finite clipped gradient")
                optimizer.step(); counts["optimizer_steps"] += 1
                row = {"iteration": iteration, "losses": {k: float(v.detach()) for k, v in terms.items()}, "preclip_norm": float(preclip)}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if iteration not in RECIPE["checkpoint_updates"]:
                    continue
                log.flush(); os.fsync(log.fileno())
                phase = f"checkpoint_{iteration}"
                _finite_parameters_and_adam(model, optimizer, iteration, torch)
                with torch.no_grad():
                    before = model(va["x80"]); counts["validation_FW"] += 1
                    metrics = validation_metrics(model, va, before, torch)
                elapsed = time.perf_counter() - start
                eta = elapsed / iteration * (5000 - iteration)
                payload = {"schema": ARTIFACT_SCHEMA, "iteration": iteration,
                    "input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA, "recipe": RECIPE,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "protocol_pin": pin(protocol_path), "dataset_input_pins": p["inputs"],
                    "TRAIN_family_ids": dataset["family_split"]["train"], "normalizers_fit_partition": "train_only",
                    "normalizers_F64": norm, "counts": dict(counts), "RNG_this_run_initial": initial_rng,
                    "counts_phase": "AFTER_OPTIMIZER_I_AND_VALIDATION_BEFORE_OWN_READBACK",
                    "RNG_at_checkpoint": rng_state(torch), "gradient_phase": f"PRE_OPTIMIZER_{iteration}",
                    "gradient_weights_after_updates": iteration-1,
                    "weights_phase": f"AFTER_OPTIMIZER_{iteration}",
                    "gradients_postclip": {k: v.grad.detach().clone() for k, v in model.named_parameters()},
                    "preclip_norm": preclip.detach().clone(), "validation": metrics,
                    "log_prefix": {**pin(out/"updates.jsonl"), "records": iteration,
                        "scope": "Only first bytes bytes are immutable; the training log may append after this checkpoint"}}
                ckpt = out/f"checkpoint_{iteration:04d}.pt"
                atomic_checkpoint(ckpt, payload, torch)
                loaded = torch.load(ckpt, map_location="cpu", weights_only=False); counts["own_checkpoint_loads"] += 1
                require(loaded["schema"] == ARTIFACT_SCHEMA and loaded["iteration"] == iteration, "Own checkpoint identity")
                require(raw_equal(loaded["model_state"], model.state_dict(), torch)
                    and raw_equal(loaded["optimizer_state"], optimizer.state_dict(), torch)
                    and raw_equal(loaded["RNG_at_checkpoint"], rng_state(torch), torch), "Exact raw weights/Adam/RNG serialization")
                model.load_state_dict(loaded["model_state"], strict=True)
                with torch.no_grad():
                    after = model(va["x80"]); counts["checkpoint_readback_FW"] += 1
                require(all(torch.equal(before[k], after[k]) for k in before), "Exact save/load validation predictions")
                require(raw_equal(initial_rng, rng_state(torch), torch), "Raw RNG unchanged by deterministic fullbatch updates/checkpoint/readback")
                require(not torch.cuda.is_initialized(), "No CUDA context")
                receipt = {"schema": "phone_WM_checkpoint_health:1", "status": "HEALTHY_SAVED_READBACK_NOT_DYNAMICS_QUALIFICATION",
                    "iteration": iteration, "checkpoint": pin(ckpt), "finite_all_parameters_gradients_Adam": True,
                    "exact_save_load_predictions": True, "TEST_arrays_read": 0, "validation": metrics,
                    "raw_RNG_unchanged_from_pre_training": True, "raw_Adam_and_all_state_readback_equal": True,
                    "elapsed_seconds": elapsed, "ETA_remaining_seconds": eta, "counts_after_readback": dict(counts),
                    "elapsed_scope": "After data/model setup through validation, before checkpoint readback; ETA excludes pending I/O and process setup",
                    "first_checkpoint_handoff_required": iteration == 250,
                    "background_continue_to_final": True, "Goal_status_owned_by_Root": True}
                write(out/f"checkpoint_{iteration:04d}_health.json", receipt); records.append(receipt)
                atomic_json(out/"runtime.json", receipt)
                print(json.dumps({"status": receipt["status"], "iteration": iteration, "health": str(out/f"checkpoint_{iteration:04d}_health.json"), "ETA_seconds": eta}), flush=True)
                phase = "training"
        require(counts == {"partition_NPZ_reads": 2, "TEST_NPZ_reads": 0, "model_constructors": 1,
            "training_FW": 5000, "backward": 5000, "optimizer_steps": 5000, "validation_FW": 3,
            "checkpoint_readback_FW": 3, "own_checkpoint_loads": 3, "physics_Env_GPU_old_checkpoint": 0}, "Exact fixed final runtime counts")
        write(out/"report.json", {"schema": "phone_WM_training_completion:1", "status": "COMPLETED_FIXED5000_NOT_POLICY_QUALIFICATION",
            "protocol": pin(protocol_path), "counts": counts, "checkpoints": records,
            "TEST_arrays_read": 0, "training_or_validation_is_ego_generalization": False,
            "scope": "Learned conditional approximation on simulator family rollouts; hidden solver context absent, no holding/Actor/CHORD benefit qualification"})
        return out/"report.json"
    except BaseException as error:
        write(out/"failure.json", {"phase": phase, "error": repr(error), "counts": counts,
            "automatic_retry": False, "checkpoints_preserved": len(records)})
        raise
