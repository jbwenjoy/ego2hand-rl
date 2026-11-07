"""Per-reference independent target bindings; missing targets never become zeros."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .reader import _checked_file,_json,_pin


class MissingTargetBundleError(RuntimeError): pass


class TargetRegistry:
    def __init__(self,reference_registry,path,*,expected_sha256):
        self.reference_registry=reference_registry
        self.path=_checked_file(path,expected_sha256);self.manifest=_json(self.path)
        if self.manifest.get("schema")!="shared_independent_target_registry:1" or self.manifest["reference_registry_sha256"]!=reference_registry.sha256:
            raise ValueError("Independenttarget/reference registry identity mismatch")
        rows=self.manifest["bindings"]
        if len(rows)!=len(reference_registry.source_ids) or len({r["source_id"] for r in rows})!=len(rows): raise ValueError("Targetregistry coverage/duplicate")
        self.bindings={r["source_id"]:r for r in rows}
        self.arrays={}
        for sid in reference_registry.source_ids:
            r=self.bindings[sid];e=reference_registry.entry(sid)
            if r["native_reference_sha256"]!=e["canonical_native"]["sha256"] or r["source_frame_range_half_open"]!=list(e["source_frame_range_half_open"]) or r["frames"]!=e["frames"]:
                raise ValueError("Targetbundle belongs to anotherreference/window")
            if r["target_defined_from_current_policy"] or not r["no_force_or_teacher_claim"]: raise ValueError("Onpolicy/force-defined targets not allowed")
            if r["status"]=="MISSING_INDEPENDENT_BUNDLE":
                if r["target"] is not None: raise ValueError("Pendingtarget cannot contain a fabricatedpayload")
                continue
            if r["status"]!="FROZEN_KINEMATIC_REFERENCE_GEOMETRY": raise ValueError("Unqualified/unknown targetstatus")
            path=_pin(r["target"])
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            required={"time_s","support_basis","support512","contact_known","known_no_contact","phase","input_joint_pos_rad","input_object_pose_w","input_wrist_pose_w"}
            if set(arrays)!=required or not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError("Targetpayload layout/nonfinite")
            n=e["frames"]
            if arrays["support512"].shape!=(n,1,1,512) or np.any(arrays["support512"]<0) or arrays["support_basis"].shape!=(512,6): raise ValueError("Targetsupport basis/layout")
            if not np.allclose(np.linalg.norm(arrays["support_basis"],axis=-1),1,atol=1e-6): raise ValueError("Targetbasis units")
            basis_sha=hashlib.sha256(np.ascontiguousarray(arrays["support_basis"]).tobytes()).hexdigest()
            if basis_sha!=self.manifest["support_basis_sha256"]: raise ValueError("Eachtrajectory must use the identical frozenwrench basis")
            if not np.array_equal(arrays["time_s"],np.arange(n)*.02): raise ValueError("Targetlocalclock")
            with np.load(e["canonical_native"]["path"],allow_pickle=False) as native:
                for k,j in (("input_joint_pos_rad","reference_qpos"),("input_object_pose_w","object_pose_w"),("input_wrist_pose_w","wrist_pose_w")):
                    if not np.array_equal(arrays[k],native[j]): raise ValueError("Targetgeometry is not the frozenreference")
            for key in ("contact_known","known_no_contact"):
                if arrays[key].shape!=(n,) or arrays[key].dtype.kind not in "biu" or not np.isin(arrays[key],[0,1]).all(): raise ValueError("Known/absence mask")
            if arrays["phase"].shape!=(n,) or not np.isin(arrays["phase"],[0,1,2,3,4]).all() or np.any(arrays["known_no_contact"].astype(bool)&~arrays["contact_known"].astype(bool)): raise ValueError("Phase/known layout")
            self.arrays[sid]=arrays

    def requests(self,track_name,*,memory_start,memory_end,memory_time):
        t=self.reference_registry.track(track_name)
        start,end,time=(np.asarray(v).reshape(-1) for v in (memory_start,memory_end,memory_time))
        if not(start.shape==end.shape==time.shape) or not all(np.isfinite(x).all() and np.array_equal(x,np.rint(x)) for x in (start,end,time)): raise ValueError("Targetmemory indices")
        result=[]
        for a,b,c in zip(start.astype(int),end.astype(int),time.astype(int)):
            matches=[i for i,(lo,hi) in enumerate(zip(t["trajectory_starts"],t["trajectory_ends"])) if a==lo and b==hi]
            if len(matches)!=1: raise ValueError("Targetprefix not native trackprefix")
            sid=t["source_order"][matches[0]];local=min(max(c,0),b-a-1)
            result.append({"source_id":sid,"local_frame":local,"native_global_frame":a+local,"status":self.bindings[sid]["status"]})
        return result

    def gather(self,track_name,**memory):
        requests=self.requests(track_name,**memory)
        missing=sorted({r["source_id"] for r in requests if r["source_id"] not in self.arrays})
        if missing: raise MissingTargetBundleError("Independent fixedtarget bundles missing; no zero/localtime fallback: "+",".join(missing))
        return {k:np.stack([self.arrays[r["source_id"]][k][r["local_frame"]] for r in requests]) for k in ("support512","contact_known","known_no_contact","phase")}
