"""Read actual Adam applications and scheduled attempts from a native FlashSAC state file."""
import argparse
import json
import os
from pathlib import Path


FIELDS = {
    ".environment_steps":"environment_steps",
    ".global_update_step":"global_optimizer_attempts",
    ".actor.schedule_step":"actor_attempted",
    ".actor.optimizer_state.count":"actor_applied",
    ".critic.schedule_step":"critic_attempted",
    ".critic.optimizer_state.count":"critic_applied",
}


def inspect_state(path):
    """Resolve flat safetensor keys against the tagged learner pytree. Read six scalars only."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_critic_schema, checkpoint_policy_schema
    from flash_chord.training.flash_sac.checkpoint import training_checkpoint
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner

    path=Path(path).resolve()
    with safe_open(str(path),framework="numpy") as f:
        file_meta=f.metadata()
        meta={k.removeprefix("user."):v for k,v in file_meta.items() if k.startswith("user.")}
        if (file_meta["format"]!="learner_state" or file_meta["version"]!="1"
                or meta["checkpoint_kind"]!="training_state" or meta["algorithm"]!="flash_sac"):
            raise ValueError("Expected the tagged FlashSAC learner-state safetensors file")
        cfg=checkpoint_config(meta)
        training=instantiate_typed(OmegaConf.create(cfg["training"]),TrainingConfig)
        policy=checkpoint_policy_schema(meta)
        critic=checkpoint_critic_schema(meta)
        context=tuple(b["name"] for b in critic["context"]["blocks"])
        width=policy["observation"]["dimension"]
        actions=policy["action"]["dimension"]
        cpu=jax.devices("cpu")[0]
        def template():
            # Shape evaluation never allocates the replay ring, runs the simulator,
            # compiles native collection graphs, restores the whole model or updates weights.
            learner,_=create_learner(training,jnp.zeros((training.world_count,width),jnp.float32),
                                     jnp.zeros((training.world_count,len(context)),jnp.float32),
                                     action_dim=actions,objective_term_count=1,
                                     critic_context_names=context,device=cpu)
            return training_checkpoint(learner)
        tree=jax.eval_shape(template)
        template_leaves,_=jax.tree_util.tree_flatten_with_path(tree)
        manifest=json.loads(file_meta["manifest"])
        counters={}
        for (key,leaf),entry in zip(template_leaves,manifest,strict=True):
            # Validate all indexed shapes and dtypes before trusting any scalar index.
            if list(leaf.shape)!=entry["shape"] or np.dtype(leaf.dtype).str!=entry["dtype"]:
                raise ValueError(f"Checkpoint shape or dtype mismatch at {jax.tree_util.keystr(key)}")
            label=FIELDS.get(jax.tree_util.keystr(key))
            if label is None:
                continue
            value=f.get_tensor(entry["key"])
            if leaf.shape!=() or value.size!=1 or value.dtype.kind not in "iu":
                raise ValueError(f"Invalid counter leaf {label}")
            counters[label]=int(value.item())
        if set(counters)!=set(FIELDS.values()):
            raise ValueError("Incomplete path-to-scalar mapping for native state")
        if (counters["environment_steps"]!=int(meta["environment_steps"])
                or counters["global_optimizer_attempts"]!=int(meta["global_update_step"])
                or counters["critic_attempted"]!=counters["global_optimizer_attempts"]):
            raise ValueError("Tensor counters disagree with the native state header")
        for group in ("actor","critic"):
            skipped=counters[group+"_attempted"]-counters[group+"_applied"]
            if skipped<0:raise ValueError("Accepted counter exceeds attempts")
            counters[group+"_skipped"]=skipped
        return dict(path=str(path),checkpoint_set_id=meta["checkpoint_set_id"],
                    source_fingerprint=cfg.get("source_fingerprint"),counts=counters)


def interval_counts(current, previous=None):
    """Return per-segment deltas. The caller supplies the documented parent/checkpoint ancestry."""
    if previous and current["source_fingerprint"]!=previous["source_fingerprint"]:
        raise ValueError("Cannot subtract counts from different reference banks or physics contracts")
    counters={k:v-(previous["counts"][k] if previous else 0) for k,v in current["counts"].items()}
    if min(counters.values())<0:
        raise ValueError("A native optimizer count decreased")
    return counters


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state",type=Path)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    os.environ["JAX_PLATFORMS"]="cpu"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    result=inspect_state(args.state)
    previous=inspect_state(args.previous) if args.previous else None
    report=dict(current=result,previous=previous,delta=interval_counts(result,previous))
    data=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:args.output.write_text(data,encoding="utf8")
    else:print(data,end="")
