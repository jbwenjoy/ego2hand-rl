"""Sealed-source, held-out-only evaluation and independent per-object force certificates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

import evaluate_flash_wuji_quality as primary
import evaluate_wuji_disturbance as stress
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from supermimic_research.ego.source_stage_contract import load_frozen_stages,score_frozen_source_stages
from supermimic_research.runtime.flash_quality import COMMANDS
from validate_flash_wuji_contact_force import check as passive_object_check


ROOT=Path(__file__).resolve().parents[1]
RULES=ROOT/"configs/hocap_holdout_source_qualification_v1.json"
RULES_SHA="3f3445e7bc6f38f2ad6502bd753faac2e56e22519d856b13b176b3221bdf6497"
PHYSICS_SCHEME="movable_dof_mean_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_stage(args):
    export_path=args.parquet.parent/"export.json"
    exp=read_json(export_path)
    if not args.development_fixture and exp["phase_scope"]!="frozen_heldout_evaluation":
        raise ValueError("Production held-out wrapper expects a post-freeze held-out reference")
    fixture_recipe=False
    if "frozen_recipe" in exp:
        recipe=read_json(primary.resolved_pin(exp["frozen_recipe"]))
        fixture_recipe=recipe.get("no_reserved_participant_access") is True
    if args.development_fixture and (not fixture_recipe or exp["sequence"]!="subject1_G04_2"):
        raise ValueError("Software fixture is limited to the explicitly labelled historical development carton")
    if not args.development_fixture and fixture_recipe:
        raise ValueError("An explicitly labelled development fixture cannot enter held-out outcomes")
    truth,human,frames=primary.source_truth(args.parquet)
    ar,meta=load_frozen_stages(args.contract,contract_sha256=args.contract_sha256,
                       raw_source_report_sha256=exp["raw_source"]["sha256"],frames=frames,truth=truth)
    if (meta["config_sha256"]!=RULES_SHA or primary.file_hash(RULES)!=RULES_SHA or
            meta["config"]!=read_json(RULES)):
        raise ValueError("Stage thresholds differ from the prospectively pinned source-stage rules")
    if not args.development_fixture and exp["frozen_recipe"]["sha256"]!=meta["recipe_freeze_sha256"]:
        raise ValueError("Source stage contract and exported reference refer to different recipe freezes")
    cfg=read_json(args.physics)
    for pin in cfg["evidence"]+cfg["code"]:
        if primary.file_hash(ROOT/pin["path"])!=pin["sha256"]:raise ValueError(f"Physics validation evidence changed: {pin['path']}")
    if cfg["scheme"]!=PHYSICS_SCHEME or not cfg["validated"]:raise ValueError("Corrected verified contact physics required")
    for pin in [*exp["asset_pins"],exp["support"]]:primary.resolved_pin(pin)
    identity=dict(parquet_sha256=primary.file_hash(args.parquet),export_sha256=primary.file_hash(export_path),
                  raw_source_report_sha256=exp["raw_source"]["sha256"],sequence=exp["sequence"],
                  object_id=exp["object_id"],mass_kg=float(exp["physics"]["mass_kg"]),
                  source_contract_sha256=args.contract_sha256,source_rules_sha256=RULES_SHA,
                  physics_sha256=primary.file_hash(args.physics),scope=exp["phase_scope"],
                  recipe_freeze_sha256=meta.get("recipe_freeze_sha256"))
    saved=check_checkpoint_physics(args.checkpoint,cfg)
    scenario={k:v for k,v in saved["scene"].items() if k!="world_count"}
    physical=dict(embodiment=saved["embodiment"],scene=scenario,collision=saved["collision"],sim=saved["env"]["sim"])
    identity["scene_configuration_sha256"]=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
    return ar,meta,identity,truth,human,frames


def check_checkpoint_physics(checkpoint,manifest):
    config=checkpoint_config(read_checkpoint_metadata(checkpoint))
    # The saved resolved env.sim is what the learner and evaluator actually instantiate.
    for text in manifest["overrides"]:
        names,expected=text.split("=",1)
        keys=names.split(".")
        matrices=[config]
        if keys[0]=="sim":matrices.append(config["env"])
        for values in matrices:
            actual=values
            for key in keys: actual=actual[key]
            equal=(float(actual)==float(expected) if type(actual) in (int,float) else str(actual).lower()==expected.lower())
            if not equal:raise ValueError(f"Checkpoint physical configuration mismatch {names}: {actual}")
    return config


def certify(a):
    a.out.mkdir(parents=True,exist_ok=False)
    _,_,identity,_,_,_=checked_stage(a);physics=read_json(a.physics)
    check_checkpoint_physics(a.checkpoint,physics)
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device(a.device)
    rows=[];states={}
    for height in (0.,.05):
        row,state=passive_object_check(a.parquet,a.checkpoint,True,height,.8)
        if abs(row["mass_kg"]-identity["mass_kg"])>1e-6:raise ValueError("Imported mass differs from source asset mass hypothesis")
        rows.append(row);states[f"height_{int(1000*height)}mm"]=state
    trace=a.out/"passive_tests.npz";np.savez_compressed(trace,**states)
    result=dict(identity=identity,device=a.device,checkpoint_used_to_reconstruct_settings=primary.file_hash(a.checkpoint),
                tests=rows,passed=all(r["passed"] for r in rows),state_file=trace.name,state_file_sha256=primary.file_hash(trace),
                passive_tool_sha256=primary.file_hash(ROOT/"tools/validate_flash_wuji_contact_force.py"),
                wrapper_sha256=primary.file_hash(Path(__file__)),
                fixture=bool(a.development_fixture))
    primary.dump(a.out/"certificate.json",result)
    return result


def check_certificates(cpu,gpu,identity,*,allow_missing_gpu=False):
    ok=True;checks=[]
    for path,device in ((cpu,"cpu"),(gpu,"cuda:0")):
        if path is None:
            if device=="cuda:0" and allow_missing_gpu:
                ok=False
                checks.append(dict(device=device,passed=None,reason="CPU software fixture only"))
                continue
            raise ValueError("Production held-out evaluation requires both CPU and GPU force certificates")
        cert=read_json(path)
        if cert["identity"]!=identity or cert["device"]!=device:raise ValueError(f"Certificate identity mismatch: {device}")
        if (cert["wrapper_sha256"]!=primary.file_hash(Path(__file__)) or
            cert["passive_tool_sha256"]!=primary.file_hash(ROOT/"tools/validate_flash_wuji_contact_force.py") or
            cert["state_file"]!="passive_tests.npz" or primary.file_hash(path.parent/cert["state_file"])!=cert["state_file_sha256"]):
            raise ValueError("Passive calibration implementation or raw record changed")
        rows=cert["tests"]
        if len(rows)!=2 or {r["drop_m"] for r in rows}!={0.,.05}:raise ValueError("Require rest and drop tests")
        result=all(r["conditioned"] and r["scheme"]==PHYSICS_SCHEME and
                 abs(r["mass_kg"]-identity["mass_kg"])<1e-6 and r["passed"] and all(r["gates"].values()) for r in rows)
        if bool(cert["passed"])!=bool(result):raise ValueError("Inconsistent force certificate result")
        checks.append(dict(device=device,path=str(path.resolve()),sha256=primary.file_hash(path),passed=result))
        ok&=result
    return dict(validated=bool(ok),identity=identity,certificates=checks)


def initial_object_body(trace,pose):
    b=trace["body_q"][0]
    delta=np.linalg.norm(b[:,:3]-pose[:3,3],axis=1)
    angle=(Rotation.from_quat(b[:,3:]).inv()*Rotation.from_matrix(pose[:3,:3])).magnitude()
    ids=np.flatnonzero((delta<2e-6)&(angle<1e-5))
    if len(ids)!=1:raise ValueError("Could not uniquely locate the recorded free object at the raw source pose")
    return int(ids[0])


def one(tr,gates,terminal,ar,meta,identity,body,mass,substeps):
    pos=tr["body_q"][:,body]
    n=len(pos);samples=tr["substep_contacts"].reshape(n-1,substeps,6)
    if not np.isfinite(samples).all() or not np.isfinite(pos).all():
        raise ValueError("Nonfinite physical trace")
    mats=np.broadcast_to(np.eye(4),(n,4,4)).copy();mats[:,:3,3]=pos[:,:3]
    mats[:,:3,:3]=Rotation.from_quat(pos[:,3:]).as_matrix()
    loads=samples[:,:,:2].mean(axis=1)/(mass*9.81)
    counts=samples[:,:,2].max(axis=1)
    extras,metrics=score_frozen_source_stages(ar,meta,meta["config"],mats,loads[:,0],loads[:,1],counts)
    retained={k:v for k,v in gates.items() if k not in {"recorded_command_replay","hand_support_in_air","beats_fixed_pose","release"}}
    if set(extras)&set(retained):raise ValueError("Stage logic unexpectedly replaced an absolute/base criterion")
    effective=dict(**retained,**extras)
    return dict(source_intervals=n-1,terminal=terminal,original_legacy_gates=gates,heldout_gates=effective,
                stage_metrics=metrics,numerical_pass=bool(all(effective.values())),force_required=True)


def adjudicate(rawdir,ar,meta,identity,force,truth,frames):
    raw=read_json(rawdir/"report.json")
    if raw["parquet_sha256"]!=identity["parquet_sha256"] or raw["sequence"]!=identity["sequence"]:
        raise ValueError("Raw policy output belongs to another source")
    if not (raw["reset"]=="explicit frame 0" and not raw["auto_reset"] and raw["VOC"]==0 and raw["zero_direct_object_force_verified_per_substep"]):
        raise ValueError("Raw actor trajectory was assisted, reset or started mid-reference")
    if len(raw["physical_repeat"]["attempts"])!=2:raise ValueError("Missing two independent recorded-command replays")
    refs=[];realizations=[];mass=identity["mass_kg"];body=None
    for k in range(3):
        fname="trace.npz" if k==0 else f"independent_physical_{k}.npz"
        path=rawdir/fname;ref=raw if not k else raw["physical_repeat"]["attempts"][k-1]
        with np.load(path,allow_pickle=False) as tr:
            n=len(tr["body_q"])
            if k==0:
                np.testing.assert_array_equal(tr["source_frames"],frames[:n]);np.testing.assert_array_equal(tr["raw_object_pose"],truth)
                body=initial_object_body(tr,truth[0])
                refs=[tr["command_"+name] for name in COMMANDS]
            else:
                if not ref["identical_supplied_commands"] or body!=initial_object_body(tr,truth[0]):
                    raise ValueError("Fresh replay identity or start frame differs")
                for name,original in zip(COMMANDS,refs):
                    actual=tr["command_"+name];np.testing.assert_array_equal(actual,original[:len(actual)])
            entry=one(tr,ref["gates"],ref["terminal"],ar,meta,identity,body,mass,raw["substeps_per_interval"])
            entry.update(id=["actor","independent_1","independent_2"][k],trace_sha256=primary.file_hash(path))
            realizations.append(entry)
    numerical=all(r["numerical_pass"] for r in realizations)
    # A missing or failed load calibration cannot turn a trajectory into a qualified result.
    load_dependent={"hand_support_in_air","release","contact_during_demonstrated_motion"}
    conclusive_failure=any(not value for r in realizations for k,value in r["heldout_gates"].items()
                           if k not in load_dependent)
    if not meta["scorable"]:status="unscorable_source"
    elif conclusive_failure:status="failed"
    elif not force["validated"]:status="provisional_unknown_contact_forces"
    else:status="passed" if numerical else "failed"
    return dict(status=status,numerical_heldout_pass=numerical,realizations=realizations,
         force_validation=force,conclusive_failure_independent_of_load=conclusive_failure,
         legacy_exact_gates=raw["passed_numeric_development_gates"],
         legacy_repeat_status=raw["physical_repeat"]["status"],legacy_report_sha256=primary.file_hash(rawdir/"report.json"),
         unchanged_absolute_physical_thresholds=True,physical_repeat_comparisons=raw["physical_repeat"]["comparisons"])


def evaluate(a):
    a.out.mkdir(parents=True,exist_ok=False)
    ar,meta,identity,truth,_,frames=checked_stage(a)
    check_checkpoint_physics(a.checkpoint,read_json(a.physics))
    fixture=bool(a.development_fixture and a.cpu_only_fixture)
    if a.cpu_only_fixture and not fixture:raise ValueError("CPU test routing is restricted to labelled development fixtures")
    force=check_certificates(a.cpu_certificate,a.gpu_certificate,identity,allow_missing_gpu=fixture)
    if not meta["scorable"]:
        result=dict(status="unscorable_source",realizations=[],force_validation=force,source_eligibility=meta["eligibility"])
    else:
        rawdir=a.out/"raw"
        primary_args=SimpleNamespace(out=rawdir,parquet=a.parquet,checkpoint=a.checkpoint,
                    protocol=a.protocol,warp_cache=a.warp_cache,exercise_short_repeats=True)
        if fixture:
            # Exercise the complete algorithm in the isolated CPU test without editing the
            # frozen evaluator. Only its explicit hardware selection is redirected.
            import jax
            class CpuWarpView:
                def set_device(self,_):return wp.set_device("cpu")
                def __getattr__(self,k):return getattr(wp,k)
            with patch.object(primary,"wp",CpuWarpView()),patch.object(primary,"resolve_jax_device",lambda _:jax.devices("cpu")[0]):
                primary.evaluate(primary_args)
        else:
            primary.evaluate(primary_args)
        result=adjudicate(rawdir,ar,meta,identity,force,truth,frames)
    result.update(identity=identity,contract_sha256=a.contract_sha256,contract=str(a.contract.resolve()),
                  checkpoint_sha256=primary.file_hash(a.checkpoint),parquet_sha256=identity["parquet_sha256"],
                  physics_manifest=str(a.physics.resolve()),fixture=bool(a.development_fixture),
                  base_protocol_sha256=primary.file_hash(a.protocol),source_scorable=bool(meta["scorable"]))
    primary.dump(a.out/"report.json",result)
    return result


def disturb(a):
    held=read_json(a.heldout_score)
    if held["fixture"] or held["status"] in ("unscorable_source","provisional_unknown_contact_forces"):
        raise ValueError("Disturbance attribution requires a scored, physically validated held-out source")
    if primary.file_hash(a.checkpoint)!=held["checkpoint_sha256"] or primary.file_hash(a.parquet)!=held["parquet_sha256"]:
        raise ValueError("Held-out disturbance inputs differ from the primary scored instance")
    raw_path=a.heldout_score.parent/"raw"/"report.json"
    if primary.file_hash(raw_path)!=held["legacy_report_sha256"]:raise ValueError("Changed original primary effect record")
    checks=held["force_validation"]["certificates"]
    for cf in checks:
        if primary.file_hash(Path(cf["path"]))!=cf["sha256"]:raise ValueError("Changed per-object force certificate")
    force=check_certificates(Path(checks[0]["path"]),Path(checks[1]["path"]),held["identity"])
    if not force["validated"]:raise ValueError("Per-object calibration did not pass")
    # Replace only the development-registry lookup in the isolated stress invocation.
    def validate(cfg,environment,seq):
        expected=held["identity"]
        mass=float(environment.model.body_mass.numpy()[environment.scene.objects[0].bodies[0].body_id])
        if seq!=expected["sequence"] or abs(mass-expected["mass_kg"])>1e-6 or not getattr(environment.solver,"wuji_conditioning",None):
            raise ValueError("Stress scene mismatches calibrated held-out physics")
        return force
    with patch.object(stress,"validation_status",validate):
        stress.run(SimpleNamespace(checkpoint=a.checkpoint,parquet=a.parquet,primary=raw_path,protocol=a.protocol,
                                   warp_cache=a.warp_cache,out=a.out))
    primary.dump(a.out/"heldout_pairing.json",dict(heldout_report_sha256=primary.file_hash(a.heldout_score),
                heldout_status=held["status"],stress_report_sha256=primary.file_hash(a.out/"report.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    for mode in ("certify","evaluate","disturb"):
        p=sub.add_parser(mode)
        for name in ("checkpoint","parquet","warp-cache","out"):p.add_argument("--"+name,type=Path,required=True)
        if mode!="disturb":
            for name in ("physics","contract"):p.add_argument("--"+name,type=Path,required=True)
            p.add_argument("--contract-sha256",required=True);p.add_argument("--development-fixture",action="store_true")
        if mode!="certify":p.add_argument("--protocol",type=Path,required=True)
        if mode=="certify":p.add_argument("--device",choices=("cpu","cuda:0"),required=True)
        if mode=="evaluate":
            p.add_argument("--cpu-certificate",type=Path,required=True);p.add_argument("--gpu-certificate",type=Path)
            p.add_argument("--cpu-only-fixture",action="store_true")
        if mode=="disturb":p.add_argument("--heldout-score",type=Path,required=True)
    args=parser.parse_args()
    if args.mode=="certify":certify(args)
    elif args.mode=="evaluate":evaluate(args)
    else:disturb(args)
