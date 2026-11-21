"""Compare saved FlashSAC actors by per-demo sampling and training-side assistance."""

import argparse
import json
from pathlib import Path

from flash_chord.training.checkpoint import read_checkpoint_metadata


def compare(plan, ledger):
    per_ref_horizon = plan["single_clip_curriculum_horizon"]
    stream_actors, seen = {}, set()
    for segment in ledger:
        nrefs=len(segment["source_sequences"])
        stream=segment["stream"]
        out=stream_actors.setdefault(stream,[])
        segment_final=segment["measured"]["total_steps"]
        before=segment_final-segment["native_added_steps"]
        for path in (Path(segment["measured"]["path"])/"checkpoints").glob("policy_*.safetensors"):
            steps=int(path.stem.removeprefix("policy_"))
            if not before < steps <= segment_final:
                continue
            key=(stream,steps)
            if key in seen:
                raise ValueError(f"Ambiguous saved actor for {key}")
            seen.add(key)
            header=read_checkpoint_metadata(path)
            if int(header["environment_steps"])!=steps:
                raise ValueError(f"Checkpoint/header mismatch: {path}")
            out.append(dict(round=segment["round"], steps=steps,per_ref_steps=steps/nrefs,
                            curriculum_stage=int(header["curriculum_stage"]),target_voc=float(header["voc_scale"]),
                            update_attempts=int(header["global_update_step"]),checkpoint=str(path),
                            originally_scored=(steps==segment_final)))
    results={}
    for stream,actors in stream_actors.items():
        actors.sort(key=lambda x:x["steps"])
        selections={}
        for frac in (.25,.5,1.):
            selections[str(frac)]=next((a for a in actors if a["per_ref_steps"]>=frac*per_ref_horizon),None)
        results[stream]=dict(scored_endpoints=[a for a in actors if a["originally_scored"]],
                             first_saved_at_or_above_fraction=selections,
                             first_saved_at_zero_target_voc=next((a for a in actors if a["target_voc"]==0),None))
    return dict(streams=results,per_reference_horizon=per_ref_horizon,
                scope="Header-based selection from closed segments before looking at quality. "
                      "The saved sample count may overshoot the target fraction. "
                      "Target assistance differs from actual assistance during reset settling. "
                      "Unscored intermediate actors do not change the prospective first verified success time.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result=compare(json.loads((args.run/"plan.json").read_bytes()),json.loads((args.run/"ledger.json").read_bytes()))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
