"""Predict held physical rollout windows using frozen WM weights, CPU only.

This is state-prediction evidence, not proof of policy-gradient reliability.
Actual numerical dynamics come from the same Warp backend as training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.world_model import FrozenPenWorldModel,rollout_palm_states


def errors(predicted,true):
    p=true[...,40:43]-predicted[...,40:43]
    r=predicted[...,43:52].reshape(-1,3,3);rt=true[...,43:52].reshape(-1,3,3)
    axis=np.rad2deg(np.arccos(np.clip((r[:,:,0]*rt[:,:,0]).sum(-1),-1,1)))
    return {"joint_rmse_rad":float(np.sqrt(np.mean((predicted[...,:20]-true[...,:20])**2))),
            "joint_velocity_rmse_rad_s":float(np.sqrt(np.mean((predicted[...,20:40]-true[...,20:40])**2))),
            "object_position_error_mean_m":float(np.linalg.norm(p,axis=-1).mean()),
            "object_position_error_p95_m":float(np.quantile(np.linalg.norm(p,axis=-1),.95)),
            "shaft_error_mean_deg":float(axis.mean()),"shaft_error_p95_deg":float(np.quantile(axis,.95)),
            "object_velocity_rmse_m_s":float(np.sqrt(np.mean((predicted[...,52:55]-true[...,52:55])**2)))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--context",type=Path,required=True)
    p.add_argument("--rollout",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    wm=FrozenPenWorldModel(a.checkpoint,a.context,device="cpu")
    with np.load(a.rollout,allow_pickle=False) as d:log={k:d[k] for k in d.files}
    meta=json.loads(a.rollout.with_suffix(".json").read_text(encoding="utf-8-sig"))
    if meta["physics_engine"]!="mujoco_warp" or meta["external_disturbances"] or meta["domain_randomization"]:
        raise ValueError("use same-backend unperturbed physical logs")
    if meta["joint_names"]!=wm.joint_names:raise ValueError("joint order mismatch")
    states=rollout_palm_states(log)
    results={}
    for horizon in (1,4,8):
        starts=np.arange(0,len(states)-horizon,4)
        predicted=torch.as_tensor(states[starts]).clone()
        with torch.no_grad():
            for j in range(horizon):
                # Row k+1 records the actual target held during interval k->k+1.
                targets=torch.as_tensor(log["joint_target_rad"][starts+j+1],dtype=torch.float32)
                predicted,_=wm(predicted,targets)
        truth=states[starts+horizon]
        results[str(horizon)]={"windows":len(starts),"horizon_s":horizon*.02,
                               "world_model":errors(predicted.numpy(),truth),
                               "persistence_baseline":errors(states[starts],truth)}
    # Verify the wrapper preserves action derivatives; this is an algebraic
    # check only. Simulator intervention agreement is a separate later probe.
    x=torch.as_tensor(states[:1]);u=torch.as_tensor(log["joint_target_rad"][1:2],dtype=torch.float32).requires_grad_(True)
    y,_=wm(x,u);gradient=torch.autograd.grad(y[:,40:43].square().sum(),u)[0]
    result={"schema":"supermimic_chord_wm_prediction_diagnostic:1","model":wm.metadata,"rollout_sha256":hashlib.sha256(a.rollout.read_bytes()).hexdigest(),
            "scope":"fixed A0 trajectory windows, same Warp backend; not independent source/video generalization or causal-gradient validation",
            "windows":results,"wrapper_action_gradient_finite":bool(torch.isfinite(gradient).all()),
            "wrapper_action_gradient_l2":float(gradient.norm()),"physical_directional_gradient_validated":False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
