"""Select a deployable actor by saved step and tensor finiteness on one actual native lineage."""
import hashlib
import json
from pathlib import Path
import time

from heldout_teacher_health import inspect_actor


def _read(path):return json.loads(Path(path).read_bytes())


def _identity(meta):
    cfg=json.loads(meta["resolved_config_json"])
    refs=[(r["sequence"],r["sha256"],r["worlds"],r["reset_seed"]) for r in cfg["reference_shards"]]
    encoded=dict(source=cfg["source_fingerprint"],references=refs,hybrid=cfg["hybrid"],
                 physics=cfg["physics_identity"],actor=json.loads(meta["policy_schema_json"]),
                 critic=json.loads(meta["critic_schema_json"]),native_network=cfg["training"]["network"],
                 native_learner_seed=cfg["training"]["seed"],discount=cfg["training"]["discount"])
    return hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _paired_headers(folder,step,signature,method,*,require_resume_files=False):
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    pair_file=folder/f"paired_{step}.json"
    if pair_file.exists():
        pair=_read(pair_file)
    elif require_resume_files:
        raise FileNotFoundError(pair_file)
    else:
        actor=read_checkpoint_metadata(folder/f"policy_{step}.safetensors")
        config=json.loads(actor["resolved_config_json"])
        pair=dict(environment_steps=int(actor["environment_steps"]),hybrid=config["hybrid"],
                  checkpoint_set_id=actor["checkpoint_set_id"],source_fingerprint=config["source_fingerprint"])
    if pair["environment_steps"]!=step or pair["hybrid"]!=(method=="shared_wm"):raise ValueError("Wrong source checkpoint/set method")
    files={"policy":"policy","state":"training_state","replay":"replay"};headers={}
    for basename,expected in files.items():
        path=folder/f"{basename}_{step}.safetensors"
        if not path.exists() and basename!="policy" and not require_resume_files:continue
        m=read_checkpoint_metadata(path)
        if (m["checkpoint_kind"]!=expected or m["algorithm"]!="flash_sac" or int(m["environment_steps"])!=step
                or m["checkpoint_set_id"]!=pair["checkpoint_set_id"] or _identity(m)!=signature):
            raise ValueError("Native actor/state/replay or physics/source identity mismatch")
        cfg=json.loads(m["resolved_config_json"])
        if cfg["source_fingerprint"]!=pair["source_fingerprint"]:
            raise ValueError("Native checkpoint and pair manifest have different source/physics fingerprints")
        if Path(cfg["output_dir"]).resolve()!=folder.resolve():
            raise ValueError("Checkpoint is not from this recorded native training segment")
        headers[basename]=m
    return pair,headers


def select_recovery(final_actor,*,method,expected_init=None,allow_development_fixture=False):
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    start=time.perf_counter();final_actor=Path(final_actor).resolve()
    raw=read_checkpoint_metadata(final_actor);cfg=json.loads(raw["resolved_config_json"])
    final=inspect_actor(final_actor);last=final["teacher_environment_steps"]
    if final_actor.name!=f"policy_{last}.safetensors":raise ValueError("Final endpoint filename/budget mismatch")
    if method not in ("shared_mf","shared_wm") or bool(cfg["hybrid"])!=(method=="shared_wm"):raise ValueError("Wrong recovery method")
    if not allow_development_fixture:
        refids=[s["sequence"] for s in cfg["reference_shards"]]
        if refids!=["subject1_G04_2","subject3_G10_3","subject4_G16_4"]:
            raise ValueError("Recovery requires the declared development bank only")
        if not cfg.get("physics_identity") or expected_init is None:raise ValueError("Missing original canonical initialization or physics")
    signature=_identity(raw)
    out=dict(final_actor=str(final_actor),fixed_final=final,method=method,
             lineage_semantics_sha256=signature,inspected=[],ancestry=[],selected_actor=None,selected_step=None)
    def finish(status):
        out.update(status=status,selection_cpu_wall_s=time.perf_counter()-start)
        return out
    if final["usable"]:return finish("not_triggered_finite_final")
    current=final_actor;seen=set();segments=[]
    while True:
        if current in seen:raise ValueError("Cycle in native resume lineage")
        seen.add(current)
        header=read_checkpoint_metadata(current);step=int(header["environment_steps"]);folder=current.parent
        pair,headers=_paired_headers(folder,step,signature,method)
        local=json.loads(headers["policy"]["resolved_config_json"])
        training=local["training"];lower=0;ancestor=None
        if training["resume"]:
            state=Path(training["checkpoint_path"]).resolve()
            parentmeta=read_checkpoint_metadata(state);lower=int(parentmeta["environment_steps"])
            if not 0<lower<step or state.name!=f"state_{lower}.safetensors":raise ValueError("Invalid parent step")
            if _identity(parentmeta)!=signature:raise ValueError("Resume parent belongs to a different training/physics lineage")
            replay=training["checkpoint"].get("load_replay_path")
            if replay is None or Path(replay).resolve()!=state.with_name(f"replay_{lower}.safetensors"):
                raise ValueError("Parent replay is absent or from another checkpoint")
            parent_pair,_=_paired_headers(state.parent,lower,signature,method,require_resume_files=True)
            if parent_pair["checkpoint_set_id"]!=parentmeta["checkpoint_set_id"]:raise ValueError("Parent native set mismatch")
            ancestor=state.with_name(f"policy_{lower}.safetensors")
        else:
            init_path=local.get("shared_initial_state")
            if expected_init:
                initial=Path(expected_init).resolve()
                if not init_path or Path(init_path).resolve()!=initial:raise ValueError("Root episode has different canonical zero-state")
                observed=_read(folder.parent/"shared_initial_loader.json")["full_training_tree_sha256"]
                if observed!=_read(initial/"initial.json")["full_training_tree_sha256"]:raise ValueError("Root canonical tensor witness mismatch")
                out["canonical_initial_tree_sha256"]=observed
        segments.append((folder,lower,step))
        out["ancestry"].append(dict(run=str(folder.parent),excluded_lower_step=lower,inclusive_upper_step=step,
                                    upper_set_id=pair["checkpoint_set_id"],actual_parent_actor=str(ancestor) if ancestor else None))
        if ancestor is None:break
        current=ancestor
    for folder,lower,upper in segments:
        candidates=[]
        for actor in folder.glob("policy_*.safetensors"):
            n=int(actor.stem.split("_")[-1])
            if lower<n<=upper and n<last:candidates.append((n,actor))
        for n,actor in sorted(candidates,reverse=True):
            pair,_=_paired_headers(folder,n,signature,method)
            health=inspect_actor(actor)
            out["inspected"].append(dict(step=n,path=str(actor),usable=health["usable"],
                sha256=health["checkpoint_sha256"],set_id=health["checkpoint_set_id"],
                nonfinite_leaf_count=len(health["nonfinite_leaves"])))
            if health["usable"]:
                out.update(selected_actor=str(actor),selected_step=n,selected_sha256=health["checkpoint_sha256"],
                           selected_checkpoint_set_id=pair["checkpoint_set_id"])
                if method=="shared_wm":
                    raw_path=pair.get("auxiliary_file")
                    aux=folder/raw_path if raw_path==f"world_model_{n}.msgpack" else None
                    out["paired_dynamics"]=dict(required_source_set_id=pair["checkpoint_set_id"],required_source_step=n,
                        path=str(aux) if aux and aux.is_file() else None,declared_sha256=pair.get("auxiliary_sha256"),
                        state=str(folder/f"state_{n}.safetensors"),pair=str(folder/f"paired_{n}.json"),
                        status="must_verify_payload_before_transfer" if aux and aux.is_file() else "unavailable_same_set_sidecar")
                return finish("selected_latest_finite_actor")
    return finish("no_finite_actor_on_lineage")
