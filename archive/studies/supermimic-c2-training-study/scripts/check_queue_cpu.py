"""CPU-only lifecycle/fail-closed tests for the prospective queue. No trainer runs."""
from __future__ import annotations
import argparse
import importlib.util
import json
import pathlib
import sys
import threading
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("prospective_queue",ROOT/"scripts/run_matched_training_queue.py")
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)

class FakeOwnedProcess:
    def __init__(self):self.returncode=None
    def poll(self):return self.returncode

def run():
    checks={}
    # A sleeping main thread represents an audit/CIM call. The independent timer
    # must request owned cleanup without waiting for this thread to poll.
    p=FakeOwnedProcess();stopped=threading.Event()
    def stop(child):
        assert child is p;child.returncode=-9;stopped.set()
        return {"owned_tree_termination_confirmed":True,"root_reaped":True}
    d=queue.OwnedDeadline(.04,stop);d.start();d.attach(p)
    time.sleep(.12)
    report=d.finish()
    checks["deadline_runs_while_main_thread_blocked"]=stopped.is_set() and report["expired"] and not report["cleanup_unverified"]
    # A slow Popen returns after the deadline. Attach must stop that exact handle.
    p=FakeOwnedProcess();stopped.clear();d=queue.OwnedDeadline(.03,stop);d.start();time.sleep(.08);d.attach(p);report=d.finish()
    checks["launch_is_inside_deadline_late_attach_stopped"]=stopped.is_set() and report["details"]["launch_in_progress_at_deadline"]
    # Successful process exit cancels timer, and never requests a kill.
    calls=[];p=FakeOwnedProcess();d=queue.OwnedDeadline(.1,lambda child:calls.append(child));d.start();d.attach(p);p.returncode=0;report=d.finish();time.sleep(.12)
    checks["completed_owned_child_not_killed"]=not calls and not report["expired"]
    # Unknown descendants are preserved as unknown, not treated as safe to retry.
    p=FakeOwnedProcess();d=queue.OwnedDeadline(.03,lambda child:{"owned_tree_termination_confirmed":False,"root_reaped":True});d.start();d.attach(p);time.sleep(.08);report=d.finish()
    checks["cleanup_unknown_retained"]=report["expired"] and report["cleanup_unverified"]
    # A failing cleanup primitive is bounded by its own reviewed timeout; its
    # exception is reported and cannot turn into a successful cleanup assertion.
    def broken(child):raise RuntimeError("synthetic cleanup failure")
    p=FakeOwnedProcess();d=queue.OwnedDeadline(.03,broken);d.start();d.attach(p);time.sleep(.08);report=d.finish()
    checks["cleanup_exception_retained"]=report["cleanup_unverified"] and "cleanup_error" in report["details"]
    p=FakeOwnedProcess();calls=[]
    def error_stop(child):calls.append(child);child.returncode=-9;return {"owned_tree_termination_confirmed":True}
    d=queue.OwnedDeadline(1,error_stop);d.start();d.attach(p);report=d.finish(on_error=True)
    checks["unexpected_failure_stops_only_retained_handle"]=calls==[p] and not report["expired"] and not report["cleanup_unverified"]
    pending={"schema":"C2_020_prospective_training_queue:1","status":"draft","caps_frozen_from_seed0_smoke":False}
    checks["pending_validate_never_executable"]=queue.validate(pending)["status"]=="PENDING_NOT_EXECUTABLE"
    try:queue.validate(pending,for_execute=True)
    except AssertionError:checks["pending_execute_rejected"]=True
    else:checks["pending_execute_rejected"]=False
    ready={"schema":"C2_020_prospective_training_queue:1","status":"frozen_ready_for_authorization","caps_frozen_from_seed0_smoke":True,"formal_training_authorized":False}
    try:queue.validate(ready,for_execute=True)
    except AssertionError as error:checks["unauthorized_ready_execute_rejected"]=str(error)=="Formal queue not authorized"
    else:checks["unauthorized_ready_execute_rejected"]=False
    checks["reviewed_cleanup_source_still_exact"]=queue.digest(queue.LIFECYCLE)==queue.LIFECYCLE_SHA
    checks["no_torch_Warp_import_or_training_process"]="torch" not in sys.modules and "warp" not in sys.modules
    return {"status":"passed" if all(checks.values()) else "failed","checks":checks,"GPU_started":False,"training_process_started":False,"scope":"Real Python timer threads with fake owned process handles; no GPU, trainer or task-quality test.","queue_sha256":queue.digest(ROOT/"scripts/run_matched_training_queue.py"),"checker_sha256":queue.digest(__file__)}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=pathlib.Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True);result=run();queue.write(args.output_dir/"checks.json",result);print(json.dumps(result,indent=2));raise SystemExit(result["status"]!="passed")
