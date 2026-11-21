"""Record only a verified, rejected nonfinite actor proposal as a held-out numerical outcome."""
import argparse
import json
from pathlib import Path

import numpy as np


def stats(x):
    x=np.asarray(x)
    return dict(shape=list(x.shape),finite=int(np.isfinite(x).sum()),nan=int(np.isnan(x).sum()),
                posinf=int(np.isposinf(x).sum()),neginf=int(np.isneginf(x).sum()))


def outcome(args):
    import evaluate_wuji_heldout as heldout
    primary=heldout.primary
    try:return heldout.evaluate(args)
    except ValueError as error:
        if str(error)!="Nonfinite actor output":raise
        raw=outer=None
        trace=error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code is primary.evaluate.__code__:raw=dict(trace.tb_frame.f_locals)
            if trace.tb_frame.f_code is heldout.evaluate.__code__:outer=dict(trace.tb_frame.f_locals)
            trace=trace.tb_next
        if raw is None or outer is None:raise
        i=int(raw["i"]);rec=raw["rec"];env=raw["env"]
        action=np.asarray(raw["action_np"]);inputs=np.asarray(raw["observation"]())
        if (np.isfinite(action).all() or len(raw["actions"])!=i or len(rec.endpoint_states)!=i+1
                or len(rec.samples)!=i*env.config.sim.substeps):
            raise ValueError("Rejected action does not match the unexecuted physical step") from error
        from supermimic_research.runtime.flash_quality import STATES
        current={k:np.asarray(getattr(env.state_0,k).numpy()) for k in STATES}
        finite_scene=all(np.isfinite(x).all() for x in current.values())
        failure=("numerical_actor_output" if np.isfinite(inputs).all() and finite_scene
                 else "numerical_observation_or_physics")
        frames=raw["frames"]
        rec.close()
        rawdir=raw["args"].out
        if rawdir!=args.out/"raw" or (rawdir/"report.json").exists():raise ValueError("Unexpected prefix output") from error
        actions=(np.stack(raw["actions"]) if i else np.empty((0,action.size),dtype=action.dtype))
        prefix=rawdir/"verified_numeric_prefix.npz"
        np.savez_compressed(prefix,source_frames=frames[:i+1],applied_actor_actions=actions,
                            rejected_action=action,observation_at_rejection=inputs,
                            **{k:np.stack([s[k] for s in rec.endpoint_states]) for k in STATES})
        diag=dict(kind=failure,exception=str(error),rejected_source_frame=int(frames[i]),
                  applied_intervals=i,requested_intervals=len(frames)-1,recorded_endpoint_states=i+1,
                  applied_substeps=len(rec.samples),rejected_action_applied=False,nonfinite_input=not np.isfinite(inputs).all(),
                  scene_finite=bool(finite_scene),proposal=stats(action),policy_input=stats(inputs),
                  prefix_sha256=primary.file_hash(prefix))
        primary.dump(rawdir/"report.json",dict(numerical_failure=diag,physical_intervals=i,
                     requested_intervals=len(frames)-1,passed_numeric_development_gates=False,
                     physical_repeat=dict(status="not_eligible_rejected_action",attempts=[])))
        result=dict(status=failure,numerical_heldout_pass=False,realizations=[],numerical_failure=diag,
                    force_validation=outer["force"],legacy_exact_gates=False,legacy_repeat_status="not_eligible_rejected_action",
                    identity=outer["identity"],contract_sha256=args.contract_sha256,source_scorable=bool(outer["meta"]["scorable"]),
                    checkpoint_sha256=primary.file_hash(args.checkpoint),parquet_sha256=primary.file_hash(args.parquet),
                    legacy_report_sha256=primary.file_hash(rawdir/"report.json"))
        primary.dump(args.out/"report.json",result)
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("evaluate",))
    for name in ("checkpoint","parquet","protocol","out","warp-cache","contract","physics","cpu-certificate"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--contract-sha256",required=True);p.add_argument("--gpu-certificate",type=Path)
    p.add_argument("--development-fixture",action="store_true");p.add_argument("--cpu-only-fixture",action="store_true")
    print(json.dumps({"status":outcome(p.parse_args())["status"]}))
