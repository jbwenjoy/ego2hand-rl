"""Run the separately registered held-out hybrid arm without altering the frozen trainer."""
import argparse
from pathlib import Path
import os
import json
import runpy
import sys
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--dynamics-transfer",type=Path,required=True)
    a,remaining=p.parse_known_args()
    probe=argparse.ArgumentParser(add_help=False)
    for name in ("out","device","adapt-from-actor","resume-state","resume-replay","shared-initial-state"):
        probe.add_argument("--"+name)
    probe.add_argument("--parquet",action="append",required=True)
    run,_=probe.parse_known_args(remaining)
    if not remaining or remaining[0]!="train" or "--hybrid" not in remaining or "--evaluation-adaptation" not in remaining:
        raise ValueError("Transferred dynamics is a single-target held-out training arm")
    if len(run.parquet)!=1 or not run.out:raise ValueError("Supply one evaluation reference and output")
    if run.resume_state:
        if not run.resume_replay or run.adapt_from_actor or run.shared_initial_state:raise ValueError("Resume only the complete target checkpoint")
    elif not run.adapt_from_actor or not run.shared_initial_state:
        raise ValueError("Need the original shared actor and a fresh canonical target critic")
    if run.device in (None,"cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        os.environ["JAX_PLATFORMS"]="cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from supermimic_research.learning import flash_imagination_hook
    from supermimic_research.learning.heldout_dynamics_transfer import TransferredImaginationHook, contract_digest
    from supermimic_research.runtime import flash_multiref
    from supermimic_research.runtime.flash_multiref_checkpoint import atomic_json, file_hash
    from flash_chord.training.checkpoint_metadata import checkpoint_config,CONFIG_METADATA_KEY
    out=Path(run.out);manifest=a.dynamics_transfer.resolve()
    native=flash_multiref.run_shared_training
    def build(training,**options):
        return TransferredImaginationHook(training,manifest_path=manifest,target_dir=out,**options)
    def bound(env,training,**kwargs):
        aux=kwargs["transition_hook"].aux
        if not isinstance(aux,TransferredImaginationHook):raise TypeError("Missing heldout-specific auxiliary hook")
        meta=dict(kwargs["checkpoint_metadata"])
        aux.validate_native(env,meta,Path(run.adapt_from_actor) if run.adapt_from_actor else None,run.device)
        recorded=checkpoint_config(meta)
        binding=dict(arm="shared_actor_and_dynamics",manifest_sha256=file_hash(manifest),
                     source_set_id=aux.source["source_set_id"],reference_contract_sha256=aux.source["contract_sha256"],
                     target_source_fingerprint=recorded["source_fingerprint"],
                     target_physics_identity=recorded.get("physics_identity"),
                     options=aux.options,policy_schema_sha256=contract_digest(env.policy_schema))
        atomic_json(out/"warm_dynamics_binding.json",binding)
        recorded["heldout_dynamics_transfer"]=binding
        meta[CONFIG_METADATA_KEY]=json.dumps(recorded,sort_keys=True,separators=(",",":"))
        kwargs["checkpoint_metadata"]=meta
        return native(env,training,**kwargs)
    driver=Path(__file__).with_name("run_wuji_multiref_flashsac.py")
    sys.argv=[str(driver),*remaining]
    with patch.object(flash_imagination_hook,"ImaginationHook",build),patch.object(flash_multiref,"run_shared_training",bound):
        runpy.run_path(str(driver),run_name="__main__")


if __name__=="__main__":main()
