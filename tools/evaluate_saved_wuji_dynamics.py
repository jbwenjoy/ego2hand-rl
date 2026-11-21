"""Compare actual learned dynamics checkpoints on separately saved, unassisted native transitions."""

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin
from supermimic_research.learning.flash_imagination import Dynamics, Scales, predictions


def schema_block(schema,name):
    b,=[v for v in schema["blocks"] if v["name"]==name]
    return slice(b["start"],b["end"])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def physical_error(predicted,truth,pos,quat):
    translation=1000*np.linalg.norm(predicted[:,pos]-truth[:,pos],axis=1)
    qp=predicted[:,quat]
    if not np.isfinite(predicted).all() or np.any(np.linalg.norm(qp,axis=1)<1e-8):
        return dict(numeric_failure=True)
    angle=(Rotation.from_quat(qp,scalar_first=True).inv()*Rotation.from_quat(truth[:,quat],scalar_first=True)).magnitude()
    return dict(position_rms_mm=rms(translation),rotation_rms_deg=rms(np.rad2deg(angle)),
                p95_mm=float(np.quantile(translation,.95)),numeric_failure=False)


def evaluate_saved(model_path,data,receipt):
    saved=serialization.msgpack_restore(model_path.read_bytes())
    st=saved["state"]
    model=Dynamics(len(st["scale"]["output_center"]))
    scale=Scales(**{k:jnp.asarray(v) for k,v in st["scale"].items()})
    params=jax.device_put(st["model_params"])
    predict=jax.jit(lambda x,a:predictions(model,params,scale,x,a))
    obs=data["obs"]
    frames,worlds,obs_dim=obs.shape
    schema=receipt["policy_schema"]["observation"]
    pos=schema_block(schema,"object_position_e_m")
    quat=schema_block(schema,"object_orientation_e_wxyz")
    contact=schema_block(schema,"contact_right_force_direction_wrist_unit")
    def in_contact(x):
        return (np.linalg.norm(x[:,contact].reshape(len(x),-1,3),axis=2)>.5).any(axis=1)
    metrics={}
    for horizon in (1,3):
        length=frames-horizon+1
        def window(key,offset=0):
            x=data[key]
            return x[offset:offset+length].reshape((-1,*x.shape[2:]))
        x0=np.column_stack((window("obs"),window("critic_context")))
        x=jnp.asarray(x0)
        valid=np.ones(length*worlds,dtype=bool)
        has_contact=in_contact(window("obs"))
        switches=np.zeros_like(valid)
        for k in range(horizon):
            x,terms,_=predict(x,jnp.asarray(window("action",k)))
            if saved["options"]["geometry_tail"]:
                start=obs_dim-saved["options"]["geometry_tail"]
                x=x.at[:,start:obs_dim].set(x0[:,start:obs_dim])
            contact1=in_contact(window("next_obs",k))
            switches|=contact1!=in_contact(window("obs",k))
            has_contact|=contact1
            if k<horizon-1:
                valid &= ~np.logical_or(window("terminated",k),window("truncated",k))
        pred=np.asarray(x)
        truth=window("next_obs",horizon-1)
        term_id=receipt["objective_names"].index("contact_wrench_support")
        term_pred=np.asarray(terms)[:,term_id]
        term_true=window("objective_terms",horizon-1)[:,term_id]
        groups={}
        for name,mask in (("all",valid),("without_reported_contact",valid&~has_contact),
                          ("any_reported_contact",valid&has_contact),("contact_changed",valid&switches)):
            groups[name]=dict(windows=int(mask.sum()))
            if mask.any():
                groups[name].update(model=physical_error(pred[mask],truth[mask],pos,quat),
                                    stationary=physical_error(x0[mask],truth[mask],pos,quat),
                                    support_term_rmse=rms(term_pred[mask]-term_true[mask]))
        metrics[str(horizon)]=groups
    return dict(checkpoint=file_pin(model_path),collected_environment_steps=int(saved["native_environment_steps"]),
                model_updates=int(st["model_updates"]),imagined_actor_updates=int(st["actor_updates"]),metrics=metrics)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("ledger",type=Path)
    p.add_argument("batch_dir",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    receipt=json.loads((args.batch_dir/"report.json").read_bytes())
    with np.load(args.batch_dir/"transitions.npz",allow_pickle=False) as z:data=dict(z)
    if np.any(data["applied_voc"]!=0):raise ValueError("Expected physically unaided labels")
    result={}
    for row in json.loads(args.ledger.read_bytes()):
        if row["stream"]!="shared_wm":continue
        ckpt=Path(row["checkpoint"])
        path=ckpt.with_name(ckpt.name.replace("policy_","world_model_").replace(".safetensors",".msgpack"))
        result[str(row["round"])]=evaluate_saved(path,data,receipt)
    args.output.write_text(json.dumps(dict(checkpoints=result,validation=file_pin(args.batch_dir/"transitions.npz"),
       tool=file_pin(__file__),scope="Independent development-clip diagnostics on actual native actions. "
       "Reset-crossing windows omitted. Contact signal is from the native hand-contact observation. "
       "No parameter fit on this batch in the evaluated policy training or during this diagnostic."),
       indent=2,allow_nan=False),encoding="utf8")
