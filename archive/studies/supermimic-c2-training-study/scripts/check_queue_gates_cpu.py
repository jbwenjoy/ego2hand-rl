"""Adversarial CPU gate tests using measured seed0 artifacts, without training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("prospective_queue",ROOT/"scripts/run_matched_training_queue.py")
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)

def main():
    manifest=queue.read(ROOT/"configs/confirmatory_queue_v1.json");checks={}
    checks["all_seven_measured_gates_current_sources_nine_configs_pass"]=queue.validate(manifest)["status"]=="passed_CPU_frozen_queue_contract"
    for key in ["adapter","pure_projection"]:
        bad=copy.deepcopy(manifest);bad["source_bindings"][key]["sha256"]="0"*64
        try:queue.validate_gate_evidence(bad)
        except AssertionError:checks[key+"_source_mismatch_rejected"]=True
        else:checks[key+"_source_mismatch_rejected"]=False
    with tempfile.TemporaryDirectory() as temp:
        for name,gate,modify in [
            ("WM_only_actor0_not_H8_smoke","smoke_A2",lambda d:d["actual_optimizer_steps"].update(actor=0,critic=0)),
            ("stale_auditor_rejected","smoke_A1L",lambda d:d.update(audit_script_sha256="0"*64)),
            ("format3_rejected","smoke_P_ONLY",lambda d:d.update(checkpoint_format=3)),
            ("pending_actualu_rejected","smoke_A2",lambda d:d["C2_actual_u_evidence"].update(status="PENDING")),
            ("failed_native_group_rejected","cuda_real_imag",lambda d:d["warp"]["groups"][0].update(passed=False)),
            ("failed_legacy_route_rejected","legacy_format103_rejection",lambda d:d["rows"][0]["guards"]["native_play"].update(rejected=False)),
            ("incomplete_pure_CUDA_rejected","pure_projection_AMP",lambda d:d["cuda"].update(passed=False))]:
            bad=copy.deepcopy(manifest);proof=queue.read(bad["gate_evidence"][gate]["path"]);modify(proof)
            file=Path(temp)/(name+".json");queue.write(file,proof)
            bad["gate_evidence"][gate].update(path=str(file),sha256=queue.digest(file))
            try:queue.validate_gate_evidence(bad)
            except AssertionError:checks[name]=True
            else:checks[name]=False
    try:queue.validate(manifest,for_execute=True)
    except AssertionError as error:checks["formal_authorization_false_prevents_launch"]=str(error)=="Formal queue not authorized"
    else:checks["formal_authorization_false_prevents_launch"]=False
    checks["no_torch_Warp_import_or_training_started"]="torch" not in sys.modules and "warp" not in sys.modules
    report={"status":"passed" if all(checks.values()) else "failed","checks":checks,"queue_source_sha256":queue.digest(ROOT/"scripts/run_matched_training_queue.py"),"auditor_sha256":manifest["audit_script_sha256"],"review_manifest_sha256":queue.digest(ROOT/"configs/confirmatory_queue_v1.json"),"checker_sha256":queue.digest(__file__),"GPU_started":False,"training_started":False,"scope":"Negative evidence mutations in temporary files only; original evidence and checkpoint sources remain unchanged."}
    output=ROOT/"outputs/queue_cpu_gates_v1";output.mkdir(exist_ok=True);queue.write(output/"checks.json",report);print(json.dumps(report,indent=2));raise SystemExit(report["status"]!="passed")

if __name__=="__main__":main()
