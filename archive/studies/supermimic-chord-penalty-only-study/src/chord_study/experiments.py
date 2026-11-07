"""Controlled reference perturbations, preserving independent targets and source split."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .reference import Reference, assign_split


def perturb_reference(reference, *, joint_bias_rad=0., time_offset_s=0., seed=0):
    reference.validate()
    if not np.isfinite(joint_bias_rad) or not np.isfinite(time_offset_s):
        raise ValueError("perturbations must be finite")
    result=copy.deepcopy(reference)
    a,m=result.arrays,result.meta
    source_q=reference.arrays["joint_pos_rad"]
    rng=np.random.default_rng(seed)
    signs=rng.choice([-1.,1.],size=20)
    q=np.stack([np.interp(a["time_s"]+time_offset_s,a["time_s"],source_q[:,j]) for j in range(20)],axis=-1)
    a["joint_pos_rad"]=q+joint_bias_rad*signs
    parent_id=m["source_id"]
    spec={"joint_bias_rad":float(joint_bias_rad),"joint_signs":signs.tolist(),"time_offset_s":float(time_offset_s),"seed":seed,
          "time_boundary":"edge_hold","object_goal":"unchanged","contact_target":"unchanged_independent_parent_labels"}
    suffix=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    m["source_id"]=f"{parent_id}/perturbation-{suffix}"
    m["provenance"]={"parent_source_id":parent_id,"parent_provenance":m["provenance"],"controlled_perturbation":spec,
                     "warning":"No joint clipping or feasibility claim; recheck targets and evaluate in physics."}
    m["split"]=reference.meta.get("split",assign_split(m["split_group"]))
    result.validate()
    return result


def save_reference(reference,path):
    reference.validate()
    path=Path(path)
    if path.suffix != ".npz": raise ValueError("reference extension must be .npz")
    if path.exists() or path.with_suffix(".json").exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**reference.arrays)
    meta=copy.deepcopy(reference.meta)
    meta["npz_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
