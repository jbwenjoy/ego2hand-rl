"""Freeze control-sampled simulation task criteria before formal A1/A2 runs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.reference import Reference
from chord_study.acceptance import make_pilot_protocol,protocol_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--progress-mode",choices=["azimuth","projected_shaft_rotation"],default="azimuth")
    a=p.parse_args()
    a.reference=a.reference.resolve(strict=True);a.output=a.output.resolve()
    ref=Reference.load(a.reference);ref.validate()
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(a.reference.read_bytes()).hexdigest()
    protocol=make_pilot_protocol(ref,digest,progress_mode=a.progress_mode)
    ref.meta["task_acceptance"]=protocol;ref.meta["task_acceptance_sha256"]=protocol_digest(protocol)
    ref.meta["protocol_frozen_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    shutil.copyfile(a.reference,a.output/"reference.npz")
    (a.output/"reference.json").write_text(json.dumps(ref.meta,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (a.output/"acceptance_protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps({"reference":str(a.output/"reference.npz"),"protocol_sha256":protocol_digest(protocol),"progress_frame":protocol["progress_frame"]},ensure_ascii=False))


if __name__=="__main__":main()
