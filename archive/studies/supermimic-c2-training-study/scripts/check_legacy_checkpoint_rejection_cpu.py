"""Execute the old native load/model-only/play guards on actual C2 smokes."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import torch
from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
from dex_supermimic.rl.dex_supermimic import cli

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for arm in ("a1l","p_only","a2"):
    path=ROOT/f"outputs/c2_{arm}_10iter_seed0_engineering_v1/compiled/train/model_final.pt"
    state=torch.load(path,map_location="cpu",weights_only=False)
    row={"arm":arm,"checkpoint":str(path),"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
         "format103":state["format_version"]==103,"guards":{}}
    dummy=object.__new__(DexSuperMimicRunner)
    dummy.device="cpu";dummy.is_pen=True
    dummy.cfg=SimpleNamespace(task="hand2-pen-spin",algorithm="world_model",control_hz=50)
    dummy.mdp=SimpleNamespace(reference_signature=state["reference_signature"])
    dummy.reader=SimpleNamespace(robot=SimpleNamespace(joint_names=state["joint_names"]),context_names=state["context_names"])
    for name in ("load","load_model_only"):
        try:getattr(dummy,name)(path)
        except ValueError as error:row["guards"][name]={"rejected":True,"error":str(error)}
        else:row["guards"][name]={"rejected":False}
    config=SimpleNamespace(viewer=False,viser=False,steps=1,wm_horizon=8,wm_opacity=.4,checkpoint=path,device="cpu")
    with patch.object(cli.tyro,"cli",return_value=config):
        try:cli.play()
        except ValueError as error:row["guards"]["native_play"]={"rejected":True,"error":str(error)}
        else:row["guards"]["native_play"]={"rejected":False}
    rows.append(row)
ok=all(r["format103"] and all(g["rejected"] for g in r["guards"].values()) for r in rows) and not torch.cuda.is_initialized()
out=ROOT/"outputs/legacy_checkpoint_rejection_cpu_v1";out.mkdir(parents=True,exist_ok=True)
record={"status":"passed" if ok else "failed","scope":"Actual frozen native load, model-only, and public play functions on actual final10 format103 files; CLI parser supplied CPU args, no environment constructed",
        "rows":rows,"CUDA_initialized":torch.cuda.is_initialized(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/"checks.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,indent=2));raise SystemExit(0 if ok else 1)
