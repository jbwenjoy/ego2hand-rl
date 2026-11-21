"""CPU-only checks of intermediate-target checkpoint failure routing; development data only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["JAX_PLATFORMS"]="cpu"

from flax import serialization
import numpy as np


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def copy_small_set(src,dst):
    dst.mkdir()
    names=[f"{kind}_30.safetensors" for kind in ("policy","state","replay")]
    names+=["world_model_30.msgpack","paired_30.json"]
    if sum((src/name).stat().st_size for name in names)>16*1024**2:
        raise ValueError("Only the small three-world CPU development fixture may be copied")
    for name in names:shutil.copyfile(src/name,dst/name)


def poison_replay(path, *, kind="committed"):
    # One IEEE Inf in a committed float replay row; preserve the entire checkpoint-set header.
    with path.open("r+b") as f:
        header_length=struct.unpack("<Q",f.read(8))[0]
        header=json.loads(f.read(header_length))
        for key,tensor in header.items():
            if key=="__metadata__" or tensor["dtype"]!="F32":
                continue
            shape=tensor["shape"]
            ring=len(shape)==2 and shape[0]==48
            pending=len(shape)==3 and shape[:2]==[2,3]
            if (kind=="pending" and not pending) or (kind!="pending" and not ring):continue
            # This short fixture has committed 24/48 rows. The last physical row is unused.
            index=(shape[0]-1)*shape[1] if kind=="unused" else 0
            offset=8+header_length+tensor["data_offsets"][0]+4*index
            f.seek(offset);old=struct.unpack("<f",f.read(4))[0]
            assert np.isfinite(old)
            f.seek(offset);f.write(struct.pack("<f",np.inf))
            return key
    raise AssertionError("Expected a 48-row Float32 ring in the CPU fixture")


def poison_auxiliary(path, *, diagnostic_only=False):
    raw=serialization.msgpack_restore(path.read_bytes())
    def replace_one(tree,trail=()):
        for key,value in tree.items():
            if isinstance(value,dict):
                changed=replace_one(value,(*trail,key))
                if changed:return changed
            elif isinstance(value,np.ndarray) and value.size and value.dtype.kind=="f":
                assert np.isfinite(value).all()
                copy=value.copy()
                copy.flat[0]=np.nan
                tree[key]=copy
                return [*trail,key]
        return None
    if diagnostic_only:
        raw["state"]["last_model_loss"]=np.asarray(np.nan,dtype=np.float32)
        changed=["last_model_loss"]
    else:
        changed=replace_one(raw["state"]["model_optimizer"])
        if not changed: raise ValueError("No floating auxiliary Adam moment in the fixture")
    path.write_bytes(serialization.msgpack_serialize(raw))
    pair=path.parent/"paired_30.json"
    meta=json.loads(pair.read_bytes())
    meta.update(auxiliary_bytes=path.stat().st_size,auxiliary_sha256=file_hash(path))
    pair.write_text(json.dumps(meta,indent=2)+"\n")
    return ".".join(changed)


def assert_state(report,cls,actor,resume):
    assert report["classification"]==cls,(cls,report)
    assert report["actor_usable"] is actor and report["resume_usable"] is resume,report


def count_invalid(leaves):
    return sum(item[key] for item in leaves for key in ("nan","posinf","neginf"))


def routes(report, step):
    from heldout_adaptation_health import allow_after_stop
    event={**report,"step":step}
    output={}
    for kind in ("adapt","evaluate","disturbance"):
        for at in (step,step+3):
            allowed=allow_after_stop(event,kind,at)
            output[f"{kind}_{at}"]=bool(allowed)
            expected=(at==step and kind in ("evaluate","disturbance") and report["actor_usable"])
            assert bool(allowed)==expected,(kind,at,expected,allowed)
    return output


def cost_checks(actor_failure,resume_failure):
    from heldout_wuji_cost import summarize
    plan={"subjects":[{"subject":"development_fixture"}],"budget":30,"steps":[10,20,30],
          "protocol":{"adaptation":["fixture_arm"],"primary":"development_fixture","supplementary":[]},
          "upfront_exclusive_card_s":{"fixture_arm":4.}}
    def entry(kind,step,health=None,passed=None):
        d=dict(subject="development_fixture",method="fixture_arm",kind=kind,steps=step,
               known_wall_s=2.0,card=True,unknown_cost=False,outcome=None)
        if health is not None:
            d["checkpoint_health"]={k:health[k] for k in ("classification","actor_usable","resume_usable")}
            d["checkpoint_health"]["step"]=step
        if passed is not None:d["outcome"]={"status":"passed" if passed else "failed","full_repeated":passed}
        return d
    output={}
    for name,broken,has_earlier_pass,stop,at_stop_pass in (
            ("bad_actor_at_interim",actor_failure,True,20,False),
            ("finite_actor_nonresumable",resume_failure,False,20,True),
            ("finite_actor_at_final_nonresumable",resume_failure,False,30,True),
            ("bad_actor_at_final",actor_failure,False,30,False)):
        events=[entry("adapt",10),entry("evaluate",10,passed=has_earlier_pass)]
        events.append(entry("adapt",stop,health=broken))
        if broken["actor_usable"]:events.append(entry("evaluate",stop,passed=at_stop_pass))
        summary=summarize(plan,events,{"development_fixture":"scorable"})["methods"]["fixture_arm"]
        row=summary["subjects"]["development_fixture"];halt=row["adaptation_stop"]
        assert summary["assigned"]==1 and summary["target_branches_accounted"] and row["source_status"]=="scorable"
        assert summary["numerically_stopped_targets"]==1 and summary["through_final_budget"]==(stop==30)
        assert row["qualified"]==(has_earlier_pass or at_stop_pass)
        assert row["target_s"]==2.*len(events)
        assert halt["stopped_step"]==stop and halt["unreached_budget_steps"]==[i for i in plan["steps"] if i>stop]
        assert halt["actor_evaluation_at_stop"]==("passed" if at_stop_pass else "not_run_invalid_target_actor")
        expect=("unreachable_after_"+broken["classification"]) if stop<30 else ("passed" if at_stop_pass else broken["classification"])
        assert row["final_budget_status"]==expect and summary["final_budget_verified"]==int(expect=="passed")
        passed_steps=[r["steps"] for r in row["scored"] if r["status"]=="passed"]
        assert passed_steps==([10] if has_earlier_pass else ([stop] if at_stop_pass else []))
        output[name]=dict(final_status=expect,charged_target_s=row["target_s"],earned_pass_steps=passed_steps,
                          assigned=summary["assigned"],stop=halt)
    return output


def test(repo,formal):
    from heldout_adaptation_health import inspect_target_set
    if formal.name!="checkpoints" or formal.parent.name!="r2_single2":
        raise ValueError("Expected the read-only stage-2 development single2 snapshots")
    tiny=repo/"runs/wuji_shared_seed_cpu_v1/multi/checkpoints"
    real={}
    for step,classification,ok in ((456960,"healthy",True),(548352,"nonfinite_target_actor",False)):
        result=inspect_target_set(formal,step,False)
        assert_state(result,classification,ok,ok)
        real[str(step)]=result
    clean=inspect_target_set(tiny,30,True)
    assert_state(clean,"healthy",True,True)
    base_actor=file_hash(tiny/"policy_30.safetensors")
    source_files={p.name:file_hash(p) for p in tiny.iterdir() if p.is_file() and "_30." in p.name}
    cases={}
    with tempfile.TemporaryDirectory(prefix="heldout_target_health_") as t:
        root=Path(t)
        for kind,fn,name in (("replay",poison_replay,"replay_30.safetensors"),
                             ("auxiliary",poison_auxiliary,"world_model_30.msgpack")):
            target=root/kind;copy_small_set(tiny,target)
            changed=fn(target/name)
            result=inspect_target_set(target,30,True)
            assert_state(result,"nonfinite_target_resume",True,False)
            assert file_hash(target/"policy_30.safetensors")==base_actor
            invalid=result[kind]["unusable_values"]
            assert count_invalid(invalid)==1 and len(invalid)==1,invalid
            assert not result["native_state"]["unusable_values"]
            cases[kind]={"patched_leaf":changed,"expected_nonfinite_elements":1,
                         "record":result,"routing":routes(result,30)}
        for kind in ("unused","pending","last_loss_only"):
            target=root/kind;copy_small_set(tiny,target)
            changed=(poison_auxiliary(target/"world_model_30.msgpack",diagnostic_only=True) if kind=="last_loss_only"
                     else poison_replay(target/"replay_30.safetensors",kind=kind))
            result=inspect_target_set(target,30,True)
            assert_state(result,"healthy",True,True)
            ignored=(result["auxiliary"]["ignored_telemetry_or_unfilled_history"] if kind=="last_loss_only"
                     else result["replay"]["discarded_on_resume_or_unused"])
            assert count_invalid(ignored)==1,ignored
            cases[kind]=dict(patched_leaf=changed,record=result)
        identity_guards={}
        for kind in ("pair_manifest","swapped_actor"):
            target=root/kind;copy_small_set(tiny,target)
            if kind=="pair_manifest":
                p=target/"paired_30.json";pair=json.loads(p.read_bytes())
                pair["checkpoint_set_id"]="wrong_development_pair_id"
                p.write_text(json.dumps(pair))
            else:
                alternate=repo/"runs/wuji_shared_seed_cpu_v1/single/checkpoints/policy_30.safetensors"
                if alternate.stat().st_size>2*1024**2:raise ValueError("Unexpected development actor size")
                shutil.copyfile(alternate,target/"policy_30.safetensors")
            try:inspect_target_set(target,30,True)
            except (ValueError,RuntimeError) as e:identity_guards[kind]=str(e)
            else:raise AssertionError("Mismatched checkpoint identity accepted as a numerical method result")
    assert all(file_hash(tiny/k)==v for k,v in source_files.items())
    return {"scope":"Development-only CPU inspection of original r2 tensors and two one-value local corruptions. Synthetic corruption is not a method result.",
            "used_gpu":False,"opened_reserved_source":False,"large_checkpoint_copies":0,
            "real_development":real,"r2_failed_actor_routes":routes(real["548352"],548352),
            "tiny_healthy_set":clean,"tiny_cases":cases,"identity_mismatch_errors":identity_guards,
            "tiny_source_hashes_unchanged":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","development-checkpoints","output"):
        parser.add_argument("--"+key,required=True,type=lambda v:Path(v).resolve())
    parser.add_argument("--health-only",action="store_true")
    args=parser.parse_args()
    result=test(args.repo,args.development_checkpoints)
    if not args.health_only:
        result["synthetic_cost_routing"]=cost_checks(result["real_development"]["548352"],
                                                    result["tiny_cases"]["replay"]["record"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print({n:r["classification"] for n,r in result["real_development"].items()})
