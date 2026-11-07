"""Select a genuinely rotating training reference among passed initial grips.

No learned-policy outcome enters this filter. Static hold remains a control,
not a replacement for the requested rotating manipulation experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.acceptance import fit_reference_projected_rotation
from select_initial_grasp import transform_source,export_selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.scan=a.scan.resolve(strict=True);a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    gate={"scope":"training initialization/reference selection before formal A1/A2; no heldout claim",
          "parent_scan_sha256":hashlib.sha256(a.scan.read_bytes()).hexdigest(),"requires_source_fd_hold_pass":True,
          "reference_projected_progress_min_deg":45.,"shaft_excursion_min_deg":60.,"shaft_path_min_deg":90.,
          "dominant_axis_energy_min":.60,"progress_definition":"integral of minimal shaft increments projected on a fixed reference-only dominant axis; not roll/winding count",
          "ranking":["hand_object_penetration","hold_position_error","hold_axis_error","candidate_id"]}
    (a.output/"motion_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    scan=json.loads(a.scan.read_text(encoding="utf-8-sig"))
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FIXED_WRIST_POSE_W,RIGHT_FINGER_JOINT_NAMES
    fixed=np.asarray(RIGHT_FIXED_WRIST_POSE_W)
    sources={};checks=[];eligible=[]
    for r in scan["records"]:
        if not r["passed"] or r["velocity_group"]!="source_fd":continue
        path=Path(r["source_file"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r["source_sha256"]:raise ValueError("source hash changed")
        with np.load(path,allow_pickle=False) as d:arrays={k:d[k].copy() for k in d.files}
        sources[r["source_clip_id"].split("/")[1]]=arrays
        lo,hi=r["source_start_frame"],r["source_stop_exclusive"]
        pose=transform_source(arrays["object_pose_w"][lo:hi],arrays["wrist_pose_w"][lo:hi],fixed)
        try:
            frame=fit_reference_projected_rotation(pose)
            checks.append({"candidate_id":r["candidate_id"],"eligible":True,"progress_frame":frame});eligible.append(r)
        except ValueError as error:
            checks.append({"candidate_id":r["candidate_id"],"eligible":False,"reason":str(error)})
    eligible.sort(key=lambda r:(r["max_penetration_m"]["hand_object"],r["max_position_error_m"],r["max_axis_error_deg"],r["candidate_id"]))
    selected=eligible[0] if eligible else None
    result={"schema":"supermimic_chord_rotating_reference_selection:1","gate":gate,"checks":checks,"eligible_count":len(eligible),
            "selected_candidate_id":selected["candidate_id"] if selected else None,"parent_scan":str(a.scan.resolve())}
    (a.output/"selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if selected:
        model=mujoco.MjModel.from_binary_path(scan["model_file"])
        source_root=Path(selected["source_file"]).parents[1]
        export_selected(a.output,selected,sources,source_root,model,model.body("object/cube").id,fixed,list(RIGHT_FINGER_JOINT_NAMES))
        path=a.output/"selected_reference.json"
        meta=json.loads(path.read_text(encoding="utf-8"))
        meta["provenance"]["additional_motion_qualification"]={"motion_gate_file":"motion_gate.json","selection_file":"selection.json",
                                                                  "claim":"qualified short shaft reorientation target; no full revolution or physical 3s feasibility claim"}
        path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"eligible_count":len(eligible),"selected_candidate_id":result["selected_candidate_id"],"output":str(a.output)},ensure_ascii=False))


if __name__=="__main__":main()
