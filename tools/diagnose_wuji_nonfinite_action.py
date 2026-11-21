"""Preserve a frozen evaluator's explicit nonfinite-action failure without applying it."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np
import evaluate_flash_wuji_quality as frozen
from supermimic_research.runtime.flash_quality import STATES,COMMANDS


def stats(x):
    a=np.asarray(x)
    mask=np.isfinite(a)
    v=a[mask]
    return dict(shape=list(a.shape),dtype=str(a.dtype),finite=int(mask.sum()),total=int(a.size),
                nan=int(np.isnan(a).sum()),posinf=int(np.isposinf(a).sum()),neginf=int(np.isneginf(a).sum()),
                finite_min=None if not v.size else float(v.min()),finite_max=None if not v.size else float(v.max()))


def evaluate(args):
    start=time.perf_counter()
    try:
        frozen.evaluate(args)
    except ValueError as exc:
        if str(exc)!="Nonfinite actor output": raise
        tb=exc.__traceback__
        local=None
        while tb is not None:
            if tb.tb_frame.f_code is frozen.evaluate.__code__: local=tb.tb_frame.f_locals.copy()
            tb=tb.tb_next
        if local is None:raise
        i=int(local["i"]);rec=local["rec"];env=local["env"];pred=np.asarray(local["action_np"])
        if np.isfinite(pred).all() or len(local["actions"])!=i or len(rec.endpoint_states)!=i+1:
            raise ValueError("Nonfinite-action failure was not verified at the frozen pre-step boundary") from exc
        obs=np.asarray(local["observation"]())
        raw_obs=env.observation.numpy().copy()
        source=local["frames"];state={k:getattr(env.state_0,k).numpy().copy() for k in STATES}
        ref=env.scene.robot_reference.joint_target[i]
        explicit_source=local["truth"][i]
        state_health={k:stats(a) for k,a in state.items()}
        rec.close()
        low={k:np.stack(v) if v else np.empty((0,)) for k,v in rec.commands.items()}
        prefix={k:np.stack([s[k] for s in rec.endpoint_states]) for k in STATES}
        prefix.update({"command_"+k:a for k,a in low.items()})
        rows=np.concatenate(rec.contact_rows) if rec.contact_rows else np.empty((0,10))
        prefix.update(actor_action=np.stack(local["actions"]) if i else np.empty((0,pred.size)),
              contact_rows=rows,substep_contacts=np.asarray(rec.samples),contact_offsets=np.array(rec.contact_offsets),
              raw_object_pose=local['truth'],raw_source_frame_ids=source,completed_source_frames=source[:i+1])
        np.savez_compressed(args.out/"verified_prefix.npz",**prefix)
        np.savez_compressed(args.out/"invalid_action_boundary.npz",input_observation=obs,
             native_observation=raw_obs,output_action=pred,nonfinite_action_mask=~np.isfinite(pred),**state)
        failure=dict(exception_class=type(exc).__name__,message=str(exc),frozen_traceback=traceback.format_exc(),
           attempted_interval_index=i,source_frame=int(source[i]),completed_intervals=i,
           remaining_intervals=len(source)-1-i,executed_physics_substeps=len(rec.samples),
           invalid_action_submitted=False,subsequent_state_observed=False,
           input_observation=stats(obs),native_observation=stats(raw_obs),output_action=stats(pred),
           invalid_action_indices=np.flatnonzero(~np.isfinite(pred)).tolist(),
           source_object_state=stats(explicit_source),source_robot_reference=stats(ref),prior_native_state=state_health)
        duration=time.perf_counter()-start
        ident=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=frozen.file_hash(args.checkpoint),
           parquet_sha256=frozen.file_hash(args.parquet),protocol_sha256=frozen.file_hash(args.protocol),
           frozen_evaluator_sha256=frozen.file_hash(Path(frozen.__file__)),wrapper_sha256=frozen.file_hash(Path(__file__)),
           prefix_sha256=frozen.file_hash(args.out/"verified_prefix.npz"),
           boundary_sha256=frozen.file_hash(args.out/"invalid_action_boundary.npz"))
        frozen.dump(args.out/"failure_boundary.json",dict(**ident,**failure,diagnostic_elapsed_s=duration))
        result=dict(**ident,sequence=local["setup"].reference.metadata.sequence_id,actor_observation_dim=int(obs.shape[-1]),
           shared_geometry_conditioned=bool(local["shared"]),physical_intervals=i,requested_intervals=len(source)-1,
           substeps_per_interval=env.config.sim.substeps,reset="explicit frame 0",auto_reset=False,VOC=0,
           zero_direct_object_force_verified_per_substep=(len(rec.samples)==i*env.config.sim.substeps),
           terminal=dict(failed=True,reason="nonfinite actor output",termination_causes=["invalid_action"],truncated=False),
           status="failed_nonfinite_action",failure_boundary=failure,passed_numeric_development_gates=False,
           physical_repeat=dict(status="failed_original_trajectory",attempts=[],comparisons=[]),
           gates=dict(complete=False),metrics=dict(full_position_rms_mm=None,full_rotation_rms_deg=None,
                  endpoint_error_mm=None),phases={},seconds=duration)
        frozen.dump(args.out/"report.json",result)
        print('VERIFIED FAILED ACTION',i,pred.shape,stats(pred),stats(obs),flush=True)
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint","parquet","protocol","out","warp-cache"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--exercise-short-repeats",action="store_true")
    evaluate(p.parse_args())
