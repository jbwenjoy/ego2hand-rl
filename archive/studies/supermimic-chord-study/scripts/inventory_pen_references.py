"""Inspect source catalog without declaring kinematic clips physically feasible."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.geometry import rotation_error


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clips-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[]
    for index_path in sorted(args.clips_root.glob("*/index.json")):
        index=json.loads(index_path.read_text(encoding="utf-8-sig"))
        for entry in index["clips"]:
            path=index_path.parent/entry["file"]
            if not path.is_file(): raise ValueError(f"missing {path}")
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry["sha256"]: raise ValueError(f"hash mismatch: {path}")
            with np.load(path,allow_pickle=False) as d:
                arrays={k:d[k] for k in d.files}
            q,wrist,obj=(arrays[k] for k in ("reference_qpos","wrist_pose_w","object_pose_w"))
            dt=index["frame_dt"]
            pos=np.linalg.norm(wrist[:,:3]-wrist[0,:3],axis=-1)
            angle=np.rad2deg(rotation_error(wrist[:,3:],wrist[0,3:]))
            reports.append({"clip_id":f"{index['group_id']}/{entry['clip_id']}","sha256":digest,
                            "frames":len(q),"duration_s":float((len(q)-1)*dt),"frame_dt":dt,
                            "shapes":{k:list(v.shape) for k,v in arrays.items()},
                            "all_finite":all(np.isfinite(v).all().item() for v in arrays.values()),
                            "source_wrist_position_from_start_max_m":float(pos.max()),
                            "source_wrist_angle_from_start_max_deg":float(angle.max()),
                            "source_wrist_near_fixed_5mm_5deg":bool(pos.max()<=.005 and angle.max()<=5),
                            "joint_velocity_abs_p95_rad_s":float(np.quantile(np.abs(np.diff(q,axis=0)/dt),.95)),
                            "object_quaternion_norm_error_max":float(np.abs(np.linalg.norm(obj[:,3:],axis=-1)-1).max()),
                            "contact_labels_available":False,"original_video_parent_id":"unavailable_in_catalog",
                            "physical_feasibility":"unverified"})
    result={"schema":"supermimic_chord_inventory:1","scope":"source kinematics only; not a physical or ego result",
            "clips":reports,"count":len(reports),"independent_raw_ego_demos_verified":0,
            "note":"Catalog clips are not established as independent original videos; raw source lineage must precede video-level generalization claims."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"count":len(reports),"near_fixed_source_wrist_count":sum(x["source_wrist_near_fixed_5mm_5deg"] for x in reports),"output":str(args.output)},ensure_ascii=False))


if __name__=="__main__":main()
