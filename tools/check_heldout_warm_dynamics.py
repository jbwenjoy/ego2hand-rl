"""CPU-only integration test of dynamics transfer using the development carton fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Constrain this test and all its child processes; the physical study may be using CUDA.
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["JAX_PLATFORMS"]="cpu"
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flash_chord.training.checkpoint import read_checkpoint_metadata
from safetensors import safe_open
from supermimic_research.learning.flash_imagination import Dynamics, Scales, predictions


def read(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def load_aux(path):
    return serialization.msgpack_restore(path.read_bytes())


def same_tree(a,b):
    return (jax.tree.structure(a)==jax.tree.structure(b) and
            all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True)))


def zeros(tree):
    return all(np.all(np.asarray(v)==0) for v in jax.tree.leaves(tree))


def evaluate_model(fields, batch):
    scales=Scales(**jax.tree.map(jnp.asarray,fields["scale"]))
    obs, ctx, action=(batch[k][0] for k in ("observation","context","action"))
    state=jnp.concatenate((obs,ctx),axis=-1)
    net=Dynamics(scales.output_center.shape[0])
    return tuple(np.asarray(x) for x in predictions(net,fields["model_params"],scales,state,action))


def command(args, label, argv, *, reject=False):
    log=args.out/f"{label}.log"
    clock=time.monotonic()
    with log.open("x") as f:
        proc=subprocess.run([sys.executable,"-u",*(str(x) for x in argv)],cwd=args.repo,
                            env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
    if (proc.returncode!=0) != reject:
        raise RuntimeError(f"Unexpected exit {proc.returncode} for {label}. Inspect {log}.")
    return time.monotonic()-clock


def test(args):
    # All source paths intentionally resolve to participant 1's development clip or the
    # small CPU shared-learner fixture. No study split manifest or reserved ZIP is read.
    repo=args.repo
    state_dir=repo/"runs/wuji_shared_seed_cpu_v1/multi/checkpoints"
    source_actor=state_dir/"policy_30.safetensors"
    package=repo/"runs/hocap_heldout_driver_fixture_v1"
    exp_path,=package.glob("inputs/assets/human_motion_data/hocap/processed/*/*/export.json")
    exp=read(exp_path)
    if (exp["original_sequence"]!="subject_1/20231025_170650" or exp["object_id"]!="G04_2"
            or read(package/"freeze.json").get("no_reserved_participant_access") is not True):
        raise ValueError("Fixture inputs must be the labelled development-only S1 carton.")
    if jax.default_backend()!="cpu":raise RuntimeError("CPU-only validation required")
    args.out.mkdir(parents=True,exist_ok=False)
    bundle=args.out/"export"; target=args.out/"new_target"; resumed=args.out/"resumed_target"
    source_aux=state_dir/"world_model_30.msgpack"
    source_pair=read(state_dir/"paired_30.json")
    source_meta=[read_checkpoint_metadata(state_dir/f"{kind}_30.safetensors") for kind in ("policy","state","replay")]
    assert {m["checkpoint_set_id"] for m in source_meta}=={source_pair["checkpoint_set_id"]}
    assert all(int(m["environment_steps"])==30 for m in source_meta)
    assert source_pair["hybrid"] and hash_file(source_aux)==source_pair["auxiliary_sha256"]
    source=load_aux(source_aux)
    assert source["native_environment_steps"]==30 and source["collected"]==10 and not zeros(source["state"]["history"])

    export_args=[repo/"tools/export_wuji_heldout_dynamics.py","--actor",source_actor,"--output",bundle,"--development-fixture"]
    seconds={"export":command(args,"export",export_args)}
    manifest_path=bundle/"manifest.json"; manifest=read(manifest_path)
    payload_path=bundle/manifest["payload_file"]
    assert manifest["development_fixture"] and manifest["source_set_id"]==source_pair["checkpoint_set_id"]
    assert manifest["actor_sha256"]==hash_file(source_actor) and manifest["payload_sha256"]==hash_file(payload_path)
    payload=load_aux(payload_path)
    assert set(payload)=={"model_params","scale"}
    assert same_tree(payload["model_params"],source["state"]["model_params"])
    assert same_tree(payload["scale"],source["state"]["scale"])

    launcher=repo/"tools/train_wuji_heldout_warm_dynamics.py"
    options=[launcher,"train","--dynamics-transfer",manifest_path,"--upstream",args.upstream,
             "--assets",repo/"runs/flashchord_wuji_asset_v1","--parquet",exp_path.with_name("data.parquet"),
             "--device","cpu","--worlds-per-reference","3","--steps","30","--checkpoint-every","3",
             "--evaluation-adaptation","--hybrid","--wm-history","4","--wm-every","1","--wm-warmup","2",
             "--wm-batch","6","--wm-fit-horizon","3","--horizon","3",
             "--override","training.replay.minimum_size=9","--override","training.replay.batch_size=6",
             "--override","training.replay.capacity=48","--override","training.network.actor_hidden_dim=32",
             "--override","training.network.critic_hidden_dim=32","--override","training.mixed_precision=false"]
    init=["--shared-initial-state",repo/"runs/wuji_shared_seed_cpu_v1/initial"]
    seconds["reject_unpaired_actor"]=command(args,"reject_unpaired_actor",[
        *options,"--out",args.out/"wrong_actor",*init,"--adapt-from-actor",state_dir/"policy_18.safetensors"],reject=True)
    seconds["fresh_adaptation"]=command(args,"fresh_adaptation",[*options,"--out",target,*init,"--adapt-from-actor",source_actor])

    target_set=target/"checkpoints"
    fresh_receipt=read(target/"warm_dynamics_start.json")
    assert fresh_receipt["scope"]=="fresh" and fresh_receipt["empty_target_history"] and fresh_receipt["fresh_model_optimizer"]
    assert fresh_receipt["target_collections"]==fresh_receipt["model_updates"]==fresh_receipt["imagined_actor_updates"]==0
    assert fresh_receipt["scales_sha256"]==manifest["scales_sha256"] and fresh_receipt["model_params_sha256"]==manifest["model_params_sha256"]
    start=read(target/"adaptation_start.json"); native_start=read(target/"initial_learner.json")
    canonical=read(repo/"runs/wuji_shared_seed_cpu_v1/initial/initial.json")
    assert start["source_sha256"]==hash_file(source_actor) and start["target_native_steps"]==0 and start["target_replay_size"]==0
    assert native_start["critic_sha256"]==canonical["critic_sha256"] and native_start["actor_sha256"]==start["actor_sha256"]
    assert native_start["actor_schedule"]==native_start["critic_schedule"]==0
    # In this fixture no optimizer can update before the third collection. Compare
    # every saved actor leaf, including batch statistics, against the shared policy.
    with safe_open(str(source_actor),framework="numpy") as before_actor, safe_open(
            str(target_set/"policy_3.safetensors"),framework="numpy") as target_actor:
        assert before_actor.keys()==target_actor.keys()
        for key in before_actor.keys():np.testing.assert_array_equal(before_actor.get_tensor(key),target_actor.get_tensor(key))
    early=load_aux(target_set/"world_model_3.msgpack"); state=early["state"]
    assert early["native_environment_steps"]==3 and early["collected"]==1
    assert same_tree(state["scale"],source["state"]["scale"]) and same_tree(state["model_params"],source["state"]["model_params"])
    assert int(state["model_updates"])==int(state["actor_updates"])==0 and zeros(state["model_optimizer"]) and zeros(state["actor_optimizer"])
    assert all(np.all(np.asarray(v)[1:]==0) for v in state["history"].values())
    assert np.allclose(state["history"]["observation"][0,:,-3:], [.0972691,.2014467,.0807090],atol=1e-7)
    assert not zeros(state["history"]["action"][0]) and not zeros(source["state"]["model_optimizer"])
    expected=evaluate_model(source["state"],state["history"])
    before=evaluate_model(state,state["history"])
    assert len(expected)==3 and all(np.array_equal(x,y) for x,y in zip(expected,before,strict=True))
    # Source scales must be distinctly different from a target history recalibration.
    ring=load_aux(target_set/"world_model_12.msgpack")["state"]["history"]
    target_features=np.concatenate([ring["observation"],ring["context"]],axis=-1).reshape(-1,286)
    recalibration_gap=float(np.max(abs(target_features.mean(axis=0)-state["scale"]["state_center"])))
    assert recalibration_gap>1.e-6
    for step in (9,12,15,30):
        target_state=load_aux(target_set/f"world_model_{step}.msgpack")
        assert target_state["collected"]==step//3 and same_tree(target_state["state"]["scale"],source["state"]["scale"])
    final=load_aux(target_set/"world_model_30.msgpack")["state"]
    assert not same_tree(final["model_params"],source["state"]["model_params"])
    assert int(final["model_updates"])>0 and int(final["actor_updates"])>0
    assert not zeros(final["model_optimizer"]) and not zeros(final["actor_optimizer"])
    source_again=load_aux(source_aux)
    assert same_tree(source_again["state"],source["state"]) and hash_file(source_aux)==source_pair["auxiliary_sha256"]

    seconds["resume_before_history_fill"]=command(args,"resume",[*options,"--out",resumed,
        "--resume-state",target_set/"state_9.safetensors","--resume-replay",target_set/"replay_9.safetensors"])
    resume0=read(resumed/"initial_learner.json"); restart=load_aux(resumed/"checkpoints/world_model_12.msgpack")
    assert resume0["native_steps"]==9 and restart["native_environment_steps"]==12 and restart["collected"]==4
    assert restart["state"]["cut_after"][2]  # native worlds restart after the final saved interval
    old_history=load_aux(target_set/"world_model_9.msgpack")["state"]["history"]
    assert all(np.array_equal(v[:3],old_history[k][:3]) for k,v in restart["state"]["history"].items())
    assert same_tree(restart["state"]["scale"],source["state"]["scale"])
    tail=load_aux(resumed/"checkpoints/world_model_30.msgpack")["state"]
    assert same_tree(tail["scale"],source["state"]["scale"])
    assert int(tail["model_updates"])>0 and int(tail["actor_updates"])>0
    for folder in (target,resumed):
        result=read(folder/"result.json")
        lock=read(folder/"warm_dynamics_scale_lock.json")
        assert lock["target_collections"]==4 and lock["scales_sha256"]==manifest["scales_sha256"]
        p=read(folder/"checkpoints/paired_30.json")
        assert result["learner_steps"]==30 and result["auxiliary"]["aux_skipped"]==0
        assert p["auxiliary_sha256"]==hash_file(folder/"checkpoints/world_model_30.msgpack")
        saved=load_aux(folder/"checkpoints/world_model_30.msgpack")
        assert saved["options"]["transfer_manifest_sha256"]==hash_file(manifest_path)
    resume_receipt=read(resumed/"warm_dynamics_start.json")
    assert resume_receipt["scope"]=="target_resume" and resume_receipt["restored_native_steps"]==9
    # A second restart after optimization checks the learned target state, as well
    # as the earlier restart across the source-scale / first-target-buffer boundary.
    later=args.out/"resumed_after_updates"
    seconds["resume_after_target_updates"]=command(args,"resume_after_updates",[*options,"--out",later,
        "--resume-state",target_set/"state_18.safetensors","--resume-replay",target_set/"replay_18.safetensors"])
    mid=load_aux(target_set/"world_model_18.msgpack")["state"]; after=read(later/"warm_dynamics_start.json")
    assert int(mid["model_updates"])>0 and int(mid["actor_updates"])>0
    assert after["restored_native_steps"]==18 and after["target_collections"]==6
    assert after["model_updates"]==int(mid["model_updates"]) and after["imagined_actor_updates"]==int(mid["actor_updates"])
    assert not same_tree(mid["model_params"],source["state"]["model_params"])
    from supermimic_research.runtime.flash_multiref_checkpoint import parameter_hash
    for key,receipt_field in (("model_params","model_params_sha256"),("model_optimizer","model_optimizer_sha256"),
                              ("actor_optimizer","actor_optimizer_sha256"),("history","target_history_sha256")):
        assert after[receipt_field]==parameter_hash(mid[key])
    resumed_state=load_aux(later/"checkpoints/world_model_30.msgpack")["state"]
    assert same_tree(resumed_state["scale"],source["state"]["scale"])
    assert int(resumed_state["model_updates"])>int(mid["model_updates"]) and int(resumed_state["actor_updates"])>int(mid["actor_updates"])
    return dict(scope="Development-only CPU integration on a known S1 clip. This verifies transfer plumbing and numerical identity, not held-out performance.",
                used_gpu=False,opened_reserved_subjects=False,native_steps=30,warm_source_step=30,
                initial_model_params_and_scale_bitwise_equal=True,same_zero_shot_actor_and_batch_stats=True,
                source_checkpoint_unchanged=True,
                target_critic_optimizer_and_replay_fresh=True,empty_unfilled_target_history_slots=True,
                pre_update_prediction_max_abs=[float(np.max(abs(a-b))) for a,b in zip(expected,before,strict=True)],
                counterfactual_target_refit_vs_transferred_center_max_abs=recalibration_gap,
                fixed_scale_across_history_fill_and_resume=True,target_resume_native_steps=[9,18],
                restored_adapted_parameters_and_optimizer_moments_exact=True,
                post_resume_model_updates=int(tail["model_updates"]),post_resume_imagined_actor_updates=int(tail["actor_updates"]),
                transfer_manifest_sha256=hash_file(manifest_path),source_actor_sha256=hash_file(source_actor),
                source_pair_id=source_pair["checkpoint_set_id"],unpaired_actor_rejected=True,measured_cpu_process_seconds=seconds)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","upstream","out"):
        p.add_argument("--"+key,type=lambda x:Path(x).resolve(),required=True)
    args=p.parse_args()
    result=test(args)
    (args.out/"test_report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result,allow_nan=False))
