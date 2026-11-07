"""Compare independently logged actual contacts to frozen geometric intent.

Auxiliary diagnostic only: geometry intent is not a measured force target,
known no-contact is not a command to forbid stabilizing extra robot contacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.contacts import support_reward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout",type=Path,required=True);p.add_argument("--targets",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.rollout,allow_pickle=False) as d:run={k:d[k] for k in d.files}
    with np.load(a.targets,allow_pickle=False) as d:target={k:d[k] for k in d.files}
    rm=json.loads(a.rollout.with_suffix(".json").read_text(encoding="utf-8-sig"))
    manifest=json.loads((a.targets.parent/"manifest.json").read_text(encoding="utf-8-sig"))
    if hashlib.sha256(a.targets.read_bytes()).hexdigest()!=manifest["target_npz_sha256"]:raise ValueError("fixed target hash mismatch")
    if rm["model_sha256"]!=manifest["model_sha256"]:raise ValueError("physical geometry/model differs from fixed contact intent")
    refpath=a.rollout.parent/"reference.npz"
    with np.load(refpath,allow_pickle=False) as ref:
        if not np.allclose(ref["object_pose_w"],target["input_object_pose_w"],atol=1e-7) or not np.allclose(ref["wrist_pose_w"],target["input_wrist_pose_w"],atol=1e-7):
            raise ValueError("object or wrist reference differs from frozen intent")
    if not np.allclose(run["support_basis"],target["support_basis"],atol=1e-7):raise ValueError("support bases differ")
    n=len(run["time_s"])
    if not np.allclose(run["time_s"],target["time_s"][:n],atol=1e-5):raise ValueError("contact reference phase differs")
    actual=torch.as_tensor(run["live_support512"],dtype=torch.float64)[:,None,None,:]
    fixed=torch.as_tensor(target["support512"][:n],dtype=torch.float64)
    score=support_reward(actual,fixed).numpy()
    phase=target["phase"][:n];known=target["contact_known"][:n].astype(bool)
    weights=np.where(np.isin(phase,[2,3]),1.,np.where(phase==4,.5,0.))*known
    match=float((score*weights).sum()/weights.sum()) if weights.sum()>0 else None
    result={"schema":"supermimic_chord_physical_contact_diagnostic:1","rollout_sha256":hashlib.sha256(a.rollout.read_bytes()).hexdigest(),
            "fixed_target_sha256":manifest["target_npz_sha256"],"frames":n,"weighted_actual_support_match512":match,
            "known_contact_frames":int((weights>0).sum()),"phase_mean_scores":{str(i):float(score[phase==i].mean()) for i in np.unique(phase)},
            "actual_solver_contact_frames":int(run["live_contact_active"].any(-1).sum()),
            "reference_geometry_known_no_contact_frames":int((target["known_no_contact"][:n]).sum()),
            "score_framewise":score.tolist(),"scope":"auxiliary physical contact vs fixed kinematic geometry intent, not actual force matching or task success",
            "no_finger_identity_or_extra_contact_failure":True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="score_framewise"},indent=2))


if __name__=="__main__":main()
