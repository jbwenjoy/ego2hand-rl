"""Run a bounded experiment with exact argv and wall-clock accounting, no shell."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",required=True,choices=["data_acquisition","reconstruction","retargeting","teacher_training","teacher_data","cold_compile","simulation_sampling","world_model_fit","policy_update","evaluation","retry","mixed_training"])
    p.add_argument("--cache",choices=["cold","warm","unknown"],required=True)
    p.add_argument("--timeout-s",type=float,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--gpu-count",type=int,default=1)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==["--"] else a.command
    if not cmd or a.timeout_s<=0 or a.gpu_count<0: p.error("command, positive timeout, and valid gpu-count required")
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    record={"schema":"supermimic_chord_cost:1","stage":a.stage,"cache":a.cache,"argv":cmd,"cwd":str(Path.cwd()),
            "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"gpu_count_reserved":a.gpu_count,
            "gpu_hour_definition":"wall time multiplied by reserved GPU count; not sampled utilization or training-only time"}
    if a.config:
        record["config_path"]=str(a.config.resolve());record["config_sha256"]=hashlib.sha256(a.config.read_bytes()).hexdigest()
    begin=time.perf_counter()
    with (a.output/"process.log").open("w",encoding="utf-8") as log:
        try:
            process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_s,check=False)
            record["exit_code"]=process.returncode;record["timed_out"]=False
        except subprocess.TimeoutExpired:
            record["exit_code"]=None;record["timed_out"]=True
        except OSError as error:
            record["exit_code"]=None;record["launch_error"]=str(error)
    record["wall_s"]=time.perf_counter()-begin
    record["reserved_gpu_hours"]=record["wall_s"]*a.gpu_count/3600
    record["end_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output/"cost.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False))
    raise SystemExit(record.get("exit_code") if record.get("exit_code") is not None else 1)


if __name__=="__main__": main()
