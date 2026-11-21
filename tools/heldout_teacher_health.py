"""Check only the inference tensors of a fixed terminal FlashSAC actor on CPU."""
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


def inspect_actor(path):
    """Malformed files are provenance errors; verified NaN/Inf is a numerical method outcome."""
    path=Path(path);start=time.perf_counter()
    types={"F64":"<f8","F32":"<f4","F16":"<f2","I64":"<i8","I32":"<i4","I16":"<i2","I8":"i1",
           "U64":"<u8","U32":"<u4","U16":"<u2","U8":"u1","BOOL":"?"}
    invalid=[];elements=0;leaves=0
    with path.open("rb") as f:
        length=struct.unpack("<Q",f.read(8))[0]
        if length<=0 or length>16<<20:raise ValueError("Unrecognized policy header size")
        hdr=json.loads(f.read(length));meta=hdr.pop("__metadata__")
        if meta.get("user.checkpoint_kind")!="policy" or meta.get("user.algorithm")!="flash_sac":
            raise ValueError("Expected a native FlashSAC policy checkpoint")
        rows=json.loads(meta["manifest"])
        if {r["key"] for r in rows}!=set(hdr) or len(rows)!=len(hdr):
            raise ValueError("Policy leaf manifest does not match the safetensors header")
        data_offset=f.tell();file_size=path.stat().st_size
        for row in rows:
            key=row["key"];h=hdr[key];lo,hi=h["data_offsets"];kind=h["dtype"]
            if lo<0 or hi<lo or data_offset+hi>file_size:raise ValueError("Invalid policy tensor bounds")
            is_bf16=kind=="BF16";dtype=np.dtype("<u2" if is_bf16 else types[kind])
            expected_storage_shape=row["shape"] if row["shape"] else [1]
            if (expected_storage_shape!=h["shape"] or (not is_bf16 and np.dtype(row["dtype"])!=dtype)
                    or int(np.prod(h["shape"],dtype=np.int64))*dtype.itemsize!=hi-lo
                    or int(np.prod(row["shape"],dtype=np.int64))*dtype.itemsize!=hi-lo):
                raise ValueError("Policy tensor byte length and shape disagree")
            f.seek(data_offset+lo);arr=np.frombuffer(f.read(hi-lo),dtype=dtype)
            if is_bf16:arr=(arr.astype(np.uint32)<<16).view(np.float32)
            if arr.dtype.kind=="f":
                nan=int(np.isnan(arr).sum());pos=int(np.isposinf(arr).sum());neg=int(np.isneginf(arr).sum())
                if nan+pos+neg:invalid.append(dict(key=key,shape=row["shape"],nan=nan,posinf=pos,neginf=neg))
            elements+=arr.size;leaves+=1
    if not leaves or not elements:raise ValueError("Empty native policy checkpoint")
    with path.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(usable=not invalid,status="finite_inference_tensors" if not invalid else "nonfinite_frozen_teacher",
                checkpoint_sha256=sha,checkpoint_set_id=meta["user.checkpoint_set_id"],
                teacher_environment_steps=int(meta["user.environment_steps"]),inference_leaves=leaves,
                inference_elements=elements,nonfinite_leaves=invalid,inspection_cpu_wall_s=time.perf_counter()-start,
                scope="Inference tensor health only. A finite actor may still fail on a particular physical reference.")
