"""Freeze source-only held-out stage evidence before loading a policy or scoring a replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import source_stage_contract as stage
from supermimic_research.strict_json import write_json


def support_evidence(source_path,config_path,data_root,visible,output):
    from audit_hocap_carton_support import backproject, fit_table
    source=json.loads(source_path.read_bytes())
    cfg=json.loads(config_path.read_bytes())
    if source.get("role")!="evaluation after recipe freeze":
        raise ValueError("Reserve the source and freeze the policy recipe first")
    pins=source.get("endpoint_image_pins",[])
    if len(pins)!=2: raise ValueError("The intake must preserve original first/last RGB for source-only review")
    for pin in pins: verify_file_pin(pin)
    first,last=source["frames"]; planned=[first,(first+last)//2,last]
    seq=data_root/source["sequence"]
    seq_meta=seq/"meta.yaml"
    rows=[];missing=[]
    if seq_meta.is_file():
        ext_path=data_root/"calibration/extrinsics"/yaml.safe_load(seq_meta.read_text())["extrinsics"]
    else:
        ext_path=None
    for serial in cfg["depth_camera_serials"]:
        int_path=data_root/"calibration/intrinsics"/f"{serial}.yaml"
        for i in planned:
            path=seq/serial/f"depth_{i:06}.png"
            if ext_path is None or not all(p.is_file() for p in (path,int_path,ext_path)):
                missing.append(dict(camera=serial,frame=i,reason="missing selected source depth or sensor calibration"))
                continue
            ext=yaml.safe_load(ext_path.read_bytes())["extrinsics"]
            mat=lambda k:np.vstack((np.asarray(ext[k]).reshape(3,4),[0,0,0,1]))
            points=backproject(path,yaml.safe_load(int_path.read_bytes())["color"],np.linalg.solve(mat("tag_1"),mat(serial)))
            row=dict(camera=serial,frame=i,depth=file_pin(path),calibration=[file_pin(int_path),file_pin(ext_path)])
            if len(points)<1000:
                rows.append(dict(**row,used=False,reason="insufficient tabletop visibility"))
                continue
            try:
                plane,stats=fit_table(points)
            except RuntimeError as error:
                rows.append(dict(**row,used=False,reason=str(error)))
                continue
            mask=abs(points[:,2]-(points[:,:2]@plane[:2]+plane[2]))<.006
            support_bounds=np.quantile(points[mask,:2],[.05,.95],axis=0)
            spans=np.ptp(support_bounds,axis=0)
            rows.append(dict(**row,used=bool(stats["inliers"]>=800 and min(spans)>.18),
                             inlier_xy_bounds_m=support_bounds.tolist(),**stats))
    output.mkdir(parents=True,exist_ok=False)
    depth=output/"depth.json"
    identity=dict(sequence=source["sequence"],source_report_sha256=file_pin(source_path)["sha256"])
    write_json(depth,dict(**identity,plane_fits=rows,missing=missing,planned_frames=planned,planned_cameras=cfg["depth_camera_serials"]))
    evidence=output/"evidence.json"
    write_json(evidence,dict(**identity,frames=source["frames"],endpoint_image_pins=pins,table_seen_at_both_ends=visible,
               source_depth_fit_report=file_pin(depth),source_review="Endpoint support classification based only on original source RGB before any policy scoring."))
    return evidence


def freeze(source_path, evidence_path, config_path, output):
    source=json.loads(source_path.read_bytes())
    evidence=json.loads(evidence_path.read_bytes())
    cfg=json.loads(config_path.read_bytes())
    if source.get("role")!="evaluation after recipe freeze" or not source.get("freeze"):
        raise ValueError("Expected the source-only output of frozen evaluation intake")
    if (evidence.get("source_report_sha256")!=file_pin(source_path)["sha256"]
            or evidence.get("sequence")!=source["sequence"] or evidence.get("frames")!=source["frames"]
            or type(evidence.get("table_seen_at_both_ends")) is not bool):
        raise ValueError("Source scene, window or visible support evidence mismatch")
    if not evidence.get("endpoint_image_pins"):
        raise ValueError("Identify the source frames used to verify endpoint support")
    for pin in evidence["endpoint_image_pins"]:
        verify_file_pin(pin)
    depth_rows=[]
    depth_pin=evidence.get("source_depth_fit_report")
    if depth_pin is not None:
        report=json.loads(verify_file_pin(depth_pin).read_bytes())
        if report.get("sequence")!=source["sequence"] or report.get("source_report_sha256")!=file_pin(source_path)["sha256"]:
            raise ValueError("Depth evidence is not bound to this source capture")
        first,last=source["frames"]
        planned={(c,i) for c in cfg["depth_camera_serials"] for i in (first,(first+last)//2,last)}
        listed=[(r["camera"],r["frame"]) for r in [*report["plane_fits"],*report["missing"]]]
        if set(listed)!=planned or len(listed)!=len(planned):
            raise ValueError("Depth evidence misses or duplicates a predetermined view/frame")
        depth_rows=report["plane_fits"]
    elif not evidence.get("reason_no_independent_depth"):
        raise ValueError("Record the source-specific reason that the endpoint envelope is needed")
    mesh=trimesh.load_mesh(verify_file_pin(source["geometry"]["source"]),process=False)
    with np.load(verify_file_pin(source["source_poses"]),allow_pickle=False) as src:
        frames=src["frames"]
        arrays,meta=stage.derive_stages(src["world_from_carton"],src["world_right_hand21"][:,[4,8,12,16,20]],
                                       mesh,cfg,depth_rows,table_seen_at_both_ends=evidence["table_seen_at_both_ends"])
    output.mkdir(parents=True,exist_ok=False)
    file=output/"source_intervals.npz"
    np.savez_compressed(file,source_frame_ids=frames,**arrays)
    info=dict(sequence=source["sequence"],object=source["object_id"],config=cfg,config_sha256=file_pin(config_path)["sha256"],
              recipe_freeze_sha256=source["freeze"]["sha256"],raw_source_report_sha256=file_pin(source_path)["sha256"],
              source_evidence=file_pin(evidence_path),logic=file_pin(stage.__file__),tool=file_pin(__file__),
              intervals_file=file.name,intervals_sha256=file_pin(file)["sha256"],
              frozen_at=datetime.now(timezone.utc).isoformat(),**meta)
    dest=output/"contract.json"
    write_json(dest,info)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    collect=sub.add_parser("evidence")
    collect.add_argument("source_report",type=Path)
    collect.add_argument("config",type=Path)
    collect.add_argument("dataset_root",type=Path)
    collect.add_argument("output",type=Path)
    collect.add_argument("--table-seen-at-both-ends",choices=("yes","no"),required=True)
    seal=sub.add_parser("freeze")
    for name in ("source_report","source_evidence","config","output"):
        seal.add_argument(name,type=Path)
    a=p.parse_args()
    if a.mode=="evidence":
        print(support_evidence(a.source_report.resolve(),a.config.resolve(),a.dataset_root.resolve(),
                               a.table_seen_at_both_ends=="yes",a.output.resolve()))
    else:
        print(freeze(a.source_report.resolve(),a.source_evidence.resolve(),a.config.resolve(),a.output.resolve()))
