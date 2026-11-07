"""Scoped full151 baseline-u150 + native residual EMA, checkpoint ABI105.

This module does not implement resets, finite-horizon survival, physics or a
reward. The companion runtime owns those contracts. CPU checks are not CUDA or
training qualification. Inactive imagined rows return finite u149 placeholders.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
from .metadata import FORMAT_VERSION, MODE, VARIANT, DESCRIPTOR_SCHEMA

MARGIN_RAD = .020
KIND = "FULL151_KNOWNROUGH_BASELINE_EMA_C2"
NFRAME = 151
NINTERVAL = 150
LAST_INTERVAL = 149
PURE_PATH = Path("D:/workspace/supermimic-c2-training-study/src/target_inset.py")
PURE_SHA256 = "2d88859c62d2833633f8b09a3be42d11c943e0c3422095e1d941100cb1bf86f4"


def _canonical(value):
    if type(value) is not dict:
        raise ValueError("Root descriptor must be an ordinary canonical JSON dict")
    def ordinary(x):
        if x is None or type(x) in (str, bool, int, float):
            return
        if type(x) is list:
            for v in x: ordinary(v)
            return
        if type(x) is dict and all(type(k) is str for k in x):
            for v in x.values(): ordinary(v)
            return
        raise ValueError("Descriptor contains a non-JSON value")
    ordinary(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _require(condition, message):
    """CPU ValueError, device-side CUDA assertion; no GPU scalar host read."""
    import torch
    if condition.device.type == "cpu":
        if not bool(condition): raise ValueError(message)
    else:
        torch._assert_async(condition, message)


def _phases(time, *, real):
    import torch
    if time.ndim == 2 and time.shape[-1] == 1: time = time.squeeze(-1)
    if time.ndim != 1: raise ValueError("Expected one local interval per batch row")
    _require(torch.isfinite(time).all() & (time >= 0).all() & (time < 1000000).all()
             & (time == time.floor()).all(), "Nonfinite/negative/noninteger full151 interval")
    if real: _require((time < NINTERVAL).all(), "No real action is permitted at terminal observation150")
    return time.long()


def _zero_bias(bias):
    import torch
    _require(torch.isfinite(bias).all() & (bias == 0).all(), "Rough controller requires frozen encoder bias0")


def _actual_ctrl(env, term):
    if hasattr(env.sim, "wp_data"):
        import warp as wp
        value = wp.to_torch(env.sim.wp_data.ctrl)
    else:
        # Narrow CPU-method fixture and native tensor-view path; never use cache
        # as a substitute for an actual control buffer.
        value = env.sim.data.ctrl
    return value[:, term._position_ctrl_ids]


@contextmanager
def full151_controller(env_class, mdp_class, runner_class, baseline_u150, descriptor):
    """Install before construction, inside full151_runtime's outer context.

    ``descriptor`` is a frozen Root-validated source/init/phase/bundle contract.
    This layer adds baseline/limit/source signatures. From-scratch only. Calling
    the context does not authorize GPU or training and never writes a state.
    """
    import torch
    if getattr(env_class, "_full151_controller_scope_active", False):
        raise ValueError("Nested/double full151 controller installation")
    if (descriptor.get('schema') != DESCRIPTOR_SCHEMA
            or descriptor.get('mode') != MODE or descriptor.get('retarget_variant_id') != VARIANT
            or descriptor.get('physical_reference_qualified') is not False):
        raise ValueError('Exact new known-rough full151 descriptor is required; old prefix descriptors are rejected')
    canonical = _canonical(descriptor)
    frozen_descriptor = json.loads(canonical)
    baseline = torch.as_tensor(baseline_u150).detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if baseline.shape != (NINTERVAL, 20) or not bool(torch.isfinite(baseline).all()):
        raise ValueError("Frozen baseline must have150x20 finite absolute FP32 targets")
    baseline_sha = hashlib.sha256(baseline.numpy().tobytes()).hexdigest()
    if hashlib.sha256(PURE_PATH.read_bytes()).hexdigest() != PURE_SHA256:
        raise ValueError("Frozen C2 pure source changed")
    spec = importlib.util.spec_from_file_location("_full151_frozen_C2", PURE_PATH)
    pure = importlib.util.module_from_spec(spec); spec.loader.exec_module(pure)
    target_inset, validate_domain = pure.target_inset, pure.validate_inset_domain
    originals = {"init": env_class.__init__, "step": env_class.step,
                 "mdp_init": mdp_class.__init__, "control": mdp_class.control_target,
                 "collect": runner_class.collect, "save": runner_class.save,
                 "load": runner_class.load, "model_load": runner_class.load_model_only}
    installed_terms, installed_mdps = [], []

    def env_init(env, *args, **kwargs):
        originals["init"](env, *args, **kwargs)
        if env.cfg.auto_reset: raise ValueError("Rough collection requires explicit/manual reset")
        term = env.action_manager.get_term("joint_pos")
        if term._mask_enabled or term.cfg.alpha != .5 or term.cfg.scale != .5 or term.cfg.residual_clip != (-.6, .6):
            raise ValueError("Original mask0/EMA.5/scale.5/residualclip contract required")
        _zero_bias(term._entity.data.encoder_bias[:, term._target_ids])
        lo, hi = term.target_position_limits
        validate_domain(lo, hi, MARGIN_RAD)
        if not torch.equal(lo, lo[:1].expand_as(lo)) or not torch.equal(hi, hi[:1].expand_as(hi)):
            raise ValueError("Rough imagined limits must be identical across worlds")
        env._full151_controller_lower = lo[0].detach().clone()
        env._full151_controller_upper = hi[0].detach().clone()
        env._full151_controller_baseline = baseline.to(env.device).clone()
        projected = target_inset(env._full151_controller_baseline, env._full151_controller_lower, env._full151_controller_upper, MARGIN_RAD)
        if not torch.equal(projected, env._full151_controller_baseline):
            raise ValueError("Nominal baseline must already be identity under hard/control C2 domain")
        env._full151_controller_active = False
        env._full151_controller_pending = []
        env._full151_controller_process_calls = 0
        native_process = term.process_actions
        installed_terms.append((env, term, native_process))

        def process(action):
            if env._full151_controller_active:
                phase = _phases(env.command_manager.get_term("motion").time_steps, real=True)
                bias = term._entity.data.encoder_bias[:, term._target_ids]
                _zero_bias(bias)
            native_process(action)
            if env._full151_controller_active:
                residual = term.processed_action
                # Reconstruct from the residual, never translate native's
                # already-clamped q_ref + residual absolute target.
                proposed = env._full151_controller_baseline[phase] + residual - bias
                _require(torch.isfinite(proposed).all(), "Nonfinite full151 real target/residual")
                bounded = proposed.clamp(min=env._full151_controller_lower, max=env._full151_controller_upper)
                actual = target_inset(bounded, env._full151_controller_lower, env._full151_controller_upper, MARGIN_RAD)
                env._full151_controller_last = {"phase": phase.detach().clone(),
                    "baseline": env._full151_controller_baseline[phase].detach().clone(),
                    "proposed": proposed.detach().clone(), "residual_before_filter": residual.detach().clone()}
                term._targets.copy_(actual)
                env._full151_controller_process_calls += 1
        term.process_actions = process

    def env_step(env, *args, **kwargs):
        _phases(env.command_manager.get_term("motion").time_steps, real=True)
        count = env._full151_controller_process_calls
        env._full151_controller_active = True
        try:
            result = originals["step"](env, *args, **kwargs)
            if env._full151_controller_process_calls != count + 1:
                raise ValueError("Expected exactly one process call per real interval")
            term = env.action_manager.get_term("joint_pos")
            env._full151_controller_pending.append({**env._full151_controller_last,
                "cached": term._targets.detach().clone(),
                "actual": _actual_ctrl(env, term).detach().clone(),
                "residual": term.processed_action.detach().clone()})
            return result
        finally:
            env._full151_controller_active = False

    def mdp_init(mdp, reader, *args, **kwargs):
        originals["mdp_init"](mdp, reader, *args, **kwargs)
        env = reader.env
        if mdp.mask_cfg.mask_prob != 0: raise ValueError("Rough imagination requires mask0")
        _zero_bias(reader.robot.data.encoder_bias)
        if not torch.equal(mdp.target_lower, env._full151_controller_lower) or not torch.equal(mdp.target_upper, env._full151_controller_upper):
            raise ValueError("Real/imagined hard-control intersections disagree")
        mdp._full151_controller_baseline = env._full151_controller_baseline.detach().clone()
        native_signature = mdp.reference_signature
        metadata = {"schema": "full151_knownrough_baseline_EMA_C2:1", "kind": KIND,
            "mode": MODE, "retarget_variant_id": VARIANT,
            "format_version": FORMAT_VERSION, "exact_resume_supported": False,
            "root_descriptor": copy.deepcopy(frozen_descriptor), "root_descriptor_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "baseline_u150_FP32_sha256": baseline_sha, "baseline_shape": [NINTERVAL, 20],
            "margin_rad": MARGIN_RAD, "target_lower_rad": mdp.target_lower.cpu().tolist(),
            "target_upper_rad": mdp.target_upper.cpu().tolist(),
            "native_reference_signature": native_signature,
            "pure_projection_sha256": PURE_SHA256, "controller_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "active_intervals": [0, NINTERVAL], "terminal_observation": NINTERVAL,
            "inactive_imagination": "finite u149 placeholder; runtime must keep survival0 and mask dead losses",
            "memory": "c1/c2 are native residual EMA only; frozen_target is actual absoluteu",
            "bias": "frozen zero; no current/future physical state read in imagination"}
        metadata["reference_controller_signature"] = _digest(metadata)
        mdp.reference_signature = metadata["reference_controller_signature"]
        mdp.full151_controller_metadata = metadata
        installed_mdps.append((mdp, native_signature))

    def control_target(mdp, action, memory, bias):
        phase = _phases(memory["time"], real=False)
        _zero_bias(bias)
        _require(((memory["end"]-memory["start"]) == NFRAME).all(), "Full151 imagined reference must have151 knots")
        residual, _ = originals["control"](mdp, action, memory, bias)
        _require(torch.isfinite(residual).all(), "Nonfinite full151 imagined residual")
        nominal = mdp._full151_controller_baseline[phase.clamp(max=LAST_INTERVAL)]
        proposed = nominal + residual - bias
        bounded = proposed.clamp(min=mdp.target_lower, max=mdp.target_upper)
        filtered = target_inset(bounded, mdp.target_lower, mdp.target_upper, MARGIN_RAD)
        actual = torch.where((phase < NINTERVAL)[:, None], filtered, nominal)
        memory["frozen_target"] = actual
        return residual, actual

    def collect(runner, steps):
        env = runner.env
        env._full151_controller_pending = []
        result = originals["collect"](runner, steps)
        pending = env._full151_controller_pending
        if len(pending) != steps: raise ValueError("Missing all-batch real control records")
        if not steps: return result
        values = {k: torch.stack([row[k] for row in pending]) for k in pending[0]}
        rows = (runner.buffer.ptr-steps+torch.arange(steps, device=runner.device)) % runner.buffer.capacity
        replay_u, replay_c = runner.buffer.data["u"][rows], runner.buffer.data["c"][rows]
        expected = target_inset(values["proposed"].clamp(min=env._full151_controller_lower, max=env._full151_controller_upper), env._full151_controller_lower, env._full151_controller_upper, MARGIN_RAD)
        errors = {"cache_actual": float((values["cached"]-values["actual"]).abs().max()),
            "cache_expected": float((values["cached"]-expected).abs().max()),
            "residual_before_after": float((values["residual_before_filter"]-values["residual"]).abs().max()),
            "replay_u_actual": float((replay_u-values["actual"]).abs().max()),
            "replay_c_residual": float((replay_c-values["residual"]).abs().max())}
        finite = all(bool(torch.isfinite(v).all()) for v in (*values.values(), replay_u, replay_c))
        if not finite or any(v != 0 for v in errors.values()): raise ValueError(f"Rough actual/replay mismatch: {errors}")
        count = getattr(runner, "_full151_controller_collect_calls", 0)+1
        runner._full151_controller_collect_calls = count
        evidence = {"collect_call": count, "iteration": runner.iteration, "steps": steps,
            "worlds": env.num_envs, "real_transitions": steps*env.num_envs,
            "all_batch_exact": True, "all_finite": finite, "max_abs_errors": errors,
            "phase_min": int(values["phase"].min()), "phase_max": int(values["phase"].max()),
            "phase_histogram": torch.bincount(values["phase"].flatten(), minlength=NINTERVAL).cpu().tolist(),
            "clamped_components": int((values["proposed"] != values["cached"]).sum()),
            "inactive_real_actions": 0}
        runner.full151_actualu_evidence = evidence
        folder = Path(runner.log_dir).parent
        with (folder/"full151_actualu_evidence.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, allow_nan=False)+"\n")
        if count in (1, 2, 3) or count % 100 == 0:
            import numpy as np
            trace = {k: v[-min(4, steps):, :min(4, env.num_envs)].cpu().numpy() for k, v in values.items()}
            trace.update(replay_u=replay_u[-min(4, steps):, :min(4, env.num_envs)].cpu().numpy(), replay_c=replay_c[-min(4, steps):, :min(4, env.num_envs)].cpu().numpy())
            np.savez_compressed(folder/f"full151_actualu_trace_{count:05d}.npz", **trace)
        env._full151_controller_pending = []
        return result

    def save(runner, path, next_iteration=None):
        native_save = torch.save
        metadata = copy.deepcopy(runner.mdp.full151_controller_metadata)
        metadata["latest_actualu_evidence"] = copy.deepcopy(getattr(runner, "full151_actualu_evidence", None))
        def serialize(state, *args, **kwargs):
            if (not isinstance(state, dict) or type(state.get("format_version")) is not int
                    or state.get("format_version") != 3 or "modules" not in state
                    or any(key in state for key in ("full151_controller", "rough_controller", "c2_controller"))):
                raise ValueError("Only fresh native format3 serialization may become full151 format105")
            if state.get("reference_signature") != runner.mdp.reference_signature:
                raise ValueError("Checkpoint reference/controller signature mismatch")
            return native_save({**state, "format_version": FORMAT_VERSION, "full151_controller": metadata}, *args, **kwargs)
        torch.save = serialize
        try: return originals["save"](runner, path, next_iteration)
        finally: torch.save = native_save

    def no_resume(*args, **kwargs):
        raise ValueError("Full151 format105 is from-scratch only; old3/103/104 and any resume/model warm start are rejected")

    env_class._full151_controller_scope_active = True
    env_class.__init__, env_class.step = env_init, env_step
    mdp_class.__init__, mdp_class.control_target = mdp_init, control_target
    runner_class.collect, runner_class.save = collect, save
    runner_class.load, runner_class.load_model_only = no_resume, no_resume
    try:
        yield
    finally:
        for env, term, native_process in reversed(installed_terms):
            term.process_actions = native_process
            for key in list(vars(env)):
                if key.startswith("_full151_controller_"): delattr(env, key)
        for mdp, native_signature in reversed(installed_mdps):
            mdp.reference_signature = native_signature
            for key in ("_full151_controller_baseline", "full151_controller_metadata"):
                if hasattr(mdp, key): delattr(mdp, key)
        env_class.__init__, env_class.step = originals["init"], originals["step"]
        mdp_class.__init__, mdp_class.control_target = originals["mdp_init"], originals["control"]
        runner_class.collect, runner_class.save = originals["collect"], originals["save"]
        runner_class.load, runner_class.load_model_only = originals["load"], originals["model_load"]
        delattr(env_class, "_full151_controller_scope_active")
