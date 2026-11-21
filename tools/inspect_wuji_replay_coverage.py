"""Measure per-source phase and contact coverage in committed FlashSAC replay rows."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import jax
import numpy as np
from safetensors import safe_open

from flash_chord.training.flash_sac.replay import ReplayState, ReplayStorage, PendingStorage


def block(schema,name):
    match, = [r for r in schema["blocks"] if r["name"]==name]
    return slice(match["start"],match["end"])


def counts(mask,active,voc,indices):
    return dict(entries=int(mask.sum()), distinct_start_frames=int(np.unique(indices[mask]).size),
                contact_signal_entries=int(np.count_nonzero(active & mask)),
                zero_applied_voc_entries=int(np.count_nonzero((voc==0)&mask)),
                contact_at_zero_applied_voc=int(np.count_nonzero(active&(voc==0)&mask)))


def inspect(file):
    shards=json.loads((file.parent.parent/"allocated.json").read_bytes())
    cols=[r["material_extents_m"] for r in shards]
    storage=ReplayStorage(*([0]*len(fields(ReplayStorage))))
    pending=PendingStorage(*([0]*len(fields(PendingStorage))))
    structure=ReplayState(storage,pending,0,0,None)
    paths,_=jax.tree_util.tree_flatten_with_path(structure)
    with safe_open(str(file),framework="numpy") as handle:
        md=handle.metadata()
        if md["user.checkpoint_kind"]!="replay" or md["version"]!="1":
            raise ValueError("Incorrect replay file kind/version")
        manifest=json.loads(md["manifest"])
        keys={jax.tree_util.keystr(path):row["key"] for (path,_),row in zip(paths,manifest,strict=True)}
        n=int(handle.get_tensor(keys[".size"]).item())
        obs=handle.get_slice(keys[".storage.observation"])
        ctx=handle.get_slice(keys[".storage.critic_context"])
        if not 0<n<=obs.get_shape()[0]:
            raise ValueError("Invalid committed replay size")
        ps=json.loads(md["user.policy_schema_json"])["observation"]
        context=json.loads(md["user.critic_schema_json"])["context"]
        extent=block(ps,"source_object_extent_xyz_m")
        contact=block(ps,"contact_right_force_direction_wrist_unit")
        phase=block(context,"normalized_reference_phase")
        assist=block(context,"applied_voc_scale")
        xs=[]
        for first in range(0,n,4096):
            rows=slice(first,min(n,first+4096))
            sizes=obs[rows,extent]
            matched=np.max(abs(sizes[:,None,:]-np.array(cols)[None,:,:]),axis=-1)<1e-6
            if not np.all(matched.sum(1)==1):
                raise ValueError("Replay geometry feature does not identify a unique planned shard")
            force_dir=obs[rows,contact].reshape(len(sizes),-1,3)
            active=(np.linalg.norm(force_dir,axis=-1)>.5).any(axis=1)
            xs.append((np.argmax(matched,axis=1),np.ravel(ctx[rows,phase]),np.ravel(ctx[rows,assist]),active))
        ids,phases,voc,active=(np.concatenate(a) for a in zip(*xs,strict=True))
        if np.any((phases<0)|(phases>1.00001)) or not np.isfinite(voc).all():
            raise ValueError("Invalid reference phase or assistance context")
        results={}
        for i,shard in enumerate(shards):
            match=ids==i
            frames=np.rint(phases*(shard["frames"]-1)).astype(int)
            masks=dict(all=match, first_half=match&(phases<.5),second_half=match&(phases>=.5),
                       last_quarter=match&(phases>=.75))
            if shard["sequence"]=="subject1_G04_2":
                masks.update(acquisition_26_35=match&(frames>=26)&(frames<=35),
                             early_lift_36_46=match&(frames>=36)&(frames<=46))
            results[shard["sequence"]]={k:counts(mask,active,voc,frames) for k,mask in masks.items()}
        return dict(checkpoint=str(file),set_id=md["user.checkpoint_set_id"],retained_rows=n,
                    environment_steps=int(md["user.environment_steps"]),by_source=results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("ledger",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    data={}
    for segment in json.loads(a.ledger.read_bytes()):
        path=Path(segment["checkpoint"])
        replay=path.with_name(path.name.replace("policy_","replay_",1))
        data[f'{segment["stream"]}_r{segment["round"]}']=inspect(replay)
    a.output.write_text(json.dumps(dict(checkpoints=data,scope="Retained committed n-step starting observations only. "
        "Rows include arbitrary reference resets, not necessarily reachable from frame zero. "
        "Contact means a nonzero reported hand contact-force direction, not a force or a successful lift."),
        indent=2,allow_nan=False),encoding="utf8")
