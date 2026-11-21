"""Extract only world-model parameters and calibration from a matching shared teacher."""
import argparse
from pathlib import Path
import json
import os


def export(actor,output,selection=None,development_fixture=False,record_numerical_unavailability=False):
    import jax
    import numpy as np
    from flax import serialization
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from supermimic_research.runtime.flash_multiref_checkpoint import atomic_json,file_hash,parameter_hash
    from supermimic_research.learning import flash_imagination
    from supermimic_research.learning.heldout_dynamics_transfer import contract,contract_digest

    meta=read_checkpoint_metadata(actor)
    cfg=checkpoint_config(meta)
    step=int(meta["environment_steps"]);link=meta["checkpoint_set_id"]
    if (step<=0 or actor.name!=f"policy_{step}.safetensors" or meta["checkpoint_kind"]!="policy"
            or not cfg.get("hybrid") or len(cfg.get("reference_shards",[]))!=3):
        raise ValueError("Expected the completed, shared three-reference hybrid policy")
    if len({s["sequence"] for s in cfg["reference_shards"]})!=3:raise ValueError("Duplicate source sequence")
    if not development_fixture:
        if selection is None:raise ValueError("Require the signed-off final development checkpoint selection")
        from run_wuji_matched_tranche import verified_phase,read
        freeze=read(selection)
        dev_plan=freeze["dev_plan"]
        if file_hash(Path(dev_plan["path"]))!=dev_plan["sha256"]:raise ValueError("Development plan changed")
        verified_phase(read(Path(dev_plan["path"])))
        if file_hash(actor)!=freeze["teachers"]["shared_wm"]["sha256"] or not cfg.get("physics_identity"):
            raise ValueError("Source actor/physics differ from the prospective final selection")
    elif selection is not None or cfg.get("reference",{}).get("num_frames")!=141:
        raise ValueError("Use an isolated developer fixture or the real frozen selection")
    state=actor.with_name(f"state_{step}.safetensors")
    state_meta=read_checkpoint_metadata(state)
    pair_path=actor.with_name(f"paired_{step}.json");pair=json.loads(pair_path.read_bytes())
    if (state_meta["checkpoint_set_id"]!=link or int(state_meta["environment_steps"])!=step
            or pair["environment_steps"]!=step or pair["checkpoint_set_id"]!=link or not pair["hybrid"]
            or pair["source_fingerprint"]!=cfg["source_fingerprint"] or pair["auxiliary_file"]!=f"world_model_{step}.msgpack"):
        raise ValueError("Native actor, training state and world-model sidecar are not the same checkpoint")
    sidecar=actor.parent/pair["auxiliary_file"]
    if file_hash(sidecar)!=pair["auxiliary_sha256"]:raise ValueError("Training sidecar failed SHA-256 validation")
    raw=serialization.msgpack_restore(sidecar.read_bytes())
    if int(raw["native_environment_steps"])!=step:
        raise ValueError("Auxiliary source checkpoint step differs from native state")
    payload={k:raw["state"][k] for k in ("model_params","scale")}
    source_updates=int(raw["state"]["model_updates"]);del raw
    interface=contract(meta);scale=payload["scale"]
    state_width=interface["policy_schema"]["observation"]["dimension"]+interface["critic_schema"]["context"]["dimension"]
    term_width=len(interface["native_objective_terms"])
    for kind,width in (("state",state_width),("output",state_width+term_width)):
        if any(np.shape(scale[kind+"_"+k])!=(width,) for k in ("center","scale")):
            raise ValueError("Incompatible learned model normalization shape")
    invalid=[dict(leaf=jax.tree_util.keystr(k),nonfinite=int((~np.isfinite(v)).sum())) for k,v
             in jax.tree_util.tree_flatten_with_path(payload)[0] if not np.isfinite(v).all()]
    nonpositive={k:int(np.sum(scale[k]<=0)) for k in ("state_scale","output_scale") if np.any(scale[k]<=0)}
    if invalid or nonpositive or source_updates<1:
        if not record_numerical_unavailability:
            raise ValueError("Selected source dynamics is unfitted or contains invalid parameters/calibration")
        output.mkdir(parents=True,exist_ok=False)
        status=dict(usable=False,status="invalid_source_dynamics" if invalid or nonpositive else "source_dynamics_not_fitted",
                    source_set_id=link,source_training_steps=step,source_model_updates=source_updates,
                    actor_sha256=file_hash(actor),source_pair_sha256=file_hash(pair_path),
                    source_sidecar_sha256=pair["auxiliary_sha256"],nonfinite_leaves=invalid,
                    nonpositive_scale_values=nonpositive,exported_payload_bytes=0)
        atomic_json(output/"unavailable.json",status)
        return status
    output.mkdir(parents=True,exist_ok=False)
    dst=output/"dynamics.msgpack";dst.write_bytes(serialization.msgpack_serialize(payload))
    manifest=dict(format="wuji_frozen_dynamics_v1",source_set_id=link,source_training_steps=step,source_model_updates=source_updates,
        actor_sha256=file_hash(actor),native_state_sha256=file_hash(state),source_pair_sha256=file_hash(pair_path),
        source_sidecar_sha256=pair["auxiliary_sha256"],source_fingerprint=pair["source_fingerprint"],
        source_selection_sha256=file_hash(selection) if selection else None,development_fixture=development_fixture,
        payload_file=dst.name,payload_bytes=dst.stat().st_size,payload_sha256=file_hash(dst),
        model_params_sha256=parameter_hash(payload["model_params"]),scales_sha256=parameter_hash(scale),
        reference_contract=interface,contract_sha256=contract_digest(interface),
        dynamics_code_sha256=file_hash(Path(flash_imagination.__file__)),
        payload_keys=sorted(payload),semantics="Weights and normalization only. No source ring, transition rows, masked target history, optimizer state, critic, or counter is exported.")
    atomic_json(output/"manifest.json",manifest)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--selection",type=Path)
    p.add_argument("--development-fixture",action="store_true")
    p.add_argument("--record-numerical-unavailability",action="store_true")
    a=p.parse_args()
    os.environ["JAX_PLATFORMS"]="cpu";os.environ["CUDA_VISIBLE_DEVICES"]=""
    m=export(a.actor.resolve(),a.output.resolve(),a.selection.resolve() if a.selection else None,
             a.development_fixture,a.record_numerical_unavailability)
    print(json.dumps({k:m[k] for k in ("source_set_id","source_training_steps","source_model_updates","payload_sha256","status") if k in m}))
