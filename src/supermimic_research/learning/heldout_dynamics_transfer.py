"""Held-out-only transfer of dynamics weights and their original normalization."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
from flash_chord.training.checkpoint_metadata import checkpoint_config,checkpoint_policy_schema,checkpoint_critic_schema
from supermimic_research.learning import flash_imagination_hook as native
from supermimic_research.learning import flash_imagination
from supermimic_research.learning.flash_imagination import Scales
from supermimic_research.runtime.flash_multiref_checkpoint import atomic_json,file_hash,parameter_hash


def contract(metadata):
    cfg=checkpoint_config(metadata)
    p,c=checkpoint_policy_schema(metadata),checkpoint_critic_schema(metadata)
    geom=p["observation"]["blocks"][-1]
    if geom["name"]!="source_object_extent_xyz_m" or geom["end"]-geom["start"]!=3:
        raise ValueError("Expected the final XYZ metric scan extent block")
    scene={k:v for k,v in cfg["scene"].items() if k not in ("world_count","object_scale_seed")}
    return dict(policy_schema=p,critic_schema=c,native_objective_terms=list(OBJECTIVE_TERM_NAMES),
                label_semantics=cfg["objective"],geometry=cfg["geometry_source"],
                geometry_block=geom,physics=dict(sim=cfg["sim"],collision=cfg["collision"],
                     embodiment=cfg["embodiment"],scene=scene,identity=cfg.get("physics_identity")))


def contract_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_payload(manifest_path):
    manifest=json.loads(manifest_path.read_bytes())
    if manifest["format"]!="wuji_frozen_dynamics_v1" or manifest["dynamics_code_sha256"]!=file_hash(Path(flash_imagination.__file__)):
        raise ValueError("Unknown transfer format or changed dynamics model implementation")
    if manifest["payload_file"]!="dynamics.msgpack":raise ValueError("Unexpected transfer artifact name")
    path=manifest_path.parent/manifest["payload_file"]
    if file_hash(path)!=manifest["payload_sha256"]:raise ValueError("Changed dynamics transfer payload")
    exported=serialization.msgpack_restore(path.read_bytes())
    if set(exported)!={"model_params","scale"}:raise ValueError("Transfer payload may only carry model parameters and scales")
    if set(exported["scale"])!=set(Scales._fields):raise ValueError("Wrong normalization field schema")
    for key,data in (("model_params_sha256",exported["model_params"]),("scales_sha256",exported["scale"])):
        if parameter_hash(data)!=manifest[key]:raise ValueError(f"Invalid tensor fingerprint {key}")
    for leaf in jax.tree.leaves(exported):
        if not np.isfinite(leaf).all():raise ValueError("Nonfinite transferred value")
    for key in ("state_scale","output_scale"):
        if np.any(exported["scale"][key]<=0):raise ValueError("Nonpositive transferred normalization")
    return manifest,exported


class TransferredImaginationHook(native.ImaginationHook):
    def __init__(self, training, *, manifest_path, target_dir, **options):
        super().__init__(training,**options)
        self.manifest_path=Path(manifest_path)
        self.source,self.payload=load_payload(self.manifest_path)
        self.options["transfer_manifest_sha256"]=file_hash(self.manifest_path)
        self.options["scale_mode"]="locked_source_normalization"
        self.out=target_dir
        self.restoring=False
        self.transfer_scale=Scales(**{k:jnp.asarray(v) for k,v in self.payload["scale"].items()})
        if self.options["history"]<2:raise ValueError("At least two target collections required")

    def validate_native(self,env,metadata,actor_path,device):
        expected=self.source["reference_contract"]
        actual=contract(metadata)
        cfg=checkpoint_config(metadata)
        raw=Path(cfg["reference"]["parquet"]).parent/"export.json"
        exp=json.loads(raw.read_bytes())
        if exp.get("phase_scope")!="frozen_heldout_evaluation":raise ValueError("Expected a frozen held-out reference")
        if self.source["development_fixture"]:
            # Only the historical CPU fixture predates the source-code physics fingerprint.
            if device!="cpu" or expected["physics"]["identity"] is not None:
                raise ValueError("Legacy development fixture is CPU-only")
            if exp["original_sequence"]!="subject_1/20231025_170650" or exp["phase_scope"]!="frozen_heldout_evaluation":
                raise ValueError("Fixture transfer cannot consume a true reserved source")
            from evaluate_flash_wuji_quality import resolved_pin
            if json.loads(resolved_pin(exp["frozen_recipe"]).read_bytes()).get("no_reserved_participant_access") is not True:
                raise ValueError("Missing development-only source recipe marker")
            actual=dict(actual,physics={**actual["physics"],"identity":None})
        elif exp["frozen_recipe"]["sha256"]!=self.source["source_selection_sha256"]:
            raise ValueError("Target reference and source teacher were frozen under different selections")
        if contract_digest(actual)!=self.source["contract_sha256"] or actual!=expected:
            raise ValueError("Target observation, action, critic, objective order/semantics, geometry or physics differ")
        if list(env.objective_term_names)!=expected["native_objective_terms"]:
            raise ValueError("Native objective vector does not match transferred output columns")
        if actor_path and file_hash(actor_path)!=self.source["actor_sha256"]:
            raise ValueError("The starting actor is not the source dynamics checkpoint partner")

    def initialize(self,batch,actor_params):
        # Even the template uses the incoming normalization; no target fit is needed.
        with patch.object(native,"fit_scales",lambda _:self.transfer_scale):
            super().initialize(batch,actor_params)
        if self.restoring:return
        params=serialization.from_state_dict(self.state.model_params,self.payload["model_params"])
        for expected,value in zip(jax.tree.leaves(self.state.model_params),jax.tree.leaves(params),strict=True):
            if expected.shape!=value.shape or expected.dtype!=value.dtype:raise ValueError("Transferred model leaf mismatch")
        params=jax.device_put(params)
        # Initialize Adam against the imported parameters, carrying no source moments.
        fresh_model_opt=self._model_opt.init(params)
        self.state=self.state.replace(model_params=params,model_optimizer=fresh_model_opt,scale=self.transfer_scale)
        s=self.state
        hist_zero=all(not np.any(v) for v in jax.tree.leaves(jax.device_get(s.history)))
        opt_zero=all(not np.any(v) for v in jax.tree.leaves(jax.device_get(s.model_optimizer)))
        if not hist_zero or not opt_zero or self.collected or int(s.model_updates) or int(s.actor_updates):
            raise ValueError("Fresh target must have an empty history and optimizer")
        scalehash=parameter_hash(serialization.to_state_dict(s.scale))
        modelhash=parameter_hash(serialization.to_state_dict(s.model_params))
        if scalehash!=self.source["scales_sha256"] or modelhash!=self.source["model_params_sha256"]:
            raise ValueError("Warm dynamics initialization did not exactly reproduce source tensors")
        atomic_json(self.out/"warm_dynamics_start.json",dict(scope="fresh",empty_target_history=hist_zero,
                    fresh_model_optimizer=opt_zero,model_updates=0,imagined_actor_updates=0,target_collections=0,
                    model_params_sha256=modelhash,scales_sha256=scalehash,source_set_id=self.source["source_set_id"],
                    source_actor_sha256=self.source["actor_sha256"],source_manifest_sha256=file_hash(self.manifest_path)))

    def __call__(self,consume,learner,tr,weights,dt,*args):
        # At precisely the first full target bank, the base hook would replace the
        # state/output calibration used by the transferred model. Keep it unchanged.
        if self.collected+1==self.options["history"]:
            with patch.object(native,"fit_scales",lambda _:self.transfer_scale):
                answer=super().__call__(consume,learner,tr,weights,dt,*args)
            actual=parameter_hash(serialization.to_state_dict(self.state.scale))
            if actual!=self.source["scales_sha256"]:raise ValueError("Target ring overwrote transfer normalization")
            atomic_json(self.out/"warm_dynamics_scale_lock.json",dict(target_collections=self.collected,
                         source_set_id=self.source["source_set_id"],scales_sha256=actual))
            return answer
        return super().__call__(consume,learner,tr,weights,dt,*args)

    def restore(self,path,batch_template,actor_params,*,native_environment_steps,continuous_solver_state=False):
        self.restoring=True
        try:
            super().restore(path,batch_template,actor_params,native_environment_steps=native_environment_steps,
                            continuous_solver_state=continuous_solver_state)
        finally:self.restoring=False
        scalehash=parameter_hash(serialization.to_state_dict(self.state.scale))
        if scalehash!=self.source["scales_sha256"]:raise ValueError("Resumed target models must use the same source scales")
        atomic_json(self.out/"warm_dynamics_start.json",dict(scope="target_resume",restored_native_steps=native_environment_steps,
                    target_collections=self.collected,model_updates=int(self.state.model_updates),imagined_actor_updates=int(self.state.actor_updates),
                    model_params_sha256=parameter_hash(serialization.to_state_dict(self.state.model_params)),
                    model_optimizer_sha256=parameter_hash(serialization.to_state_dict(self.state.model_optimizer)),
                    actor_optimizer_sha256=parameter_hash(serialization.to_state_dict(self.state.actor_optimizer)),
                    target_history_sha256=parameter_hash(serialization.to_state_dict(self.state.history)),
                    scales_sha256=scalehash,source_set_id=self.source["source_set_id"],source_manifest_sha256=file_hash(self.manifest_path)))
