"""Freeze the CPU-only factory/reference building blocks and explicit pending scope."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_ray_trace_factory_v1 import generate
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_ray_trace_factory_cpu_qualification_v1'
    if out.exists(): raise ValueError('Fresh CPU factory qualification required')
    auditpin=file_pin(ROOT/'runs/pen_linesearch_endpoint_audit_v1/report.json'); audit=json.loads(verify_file_pin(auditpin).read_text())
    regpin=file_pin(ROOT/'runs/pen_termination_comparator_registration_v1/registration.json'); reg=json.loads(verify_file_pin(regpin).read_text())
    tests=ROOT/'runs/pen_ray_trace_tests610_v1.log'; text=tests.read_text(encoding='utf-8-sig')
    reference=ROOT/'runs/pen_ray_reference_v1_tests.log'; rt=reference.read_text(encoding='utf-8-sig')
    checks=dict(source_epoch_audit=audit['passed']==audit['total']==15 and all(audit['checks'].values()),
        full610='Ran 610 tests in 10.969s' in text and '\nOK\n' in text,
        actual_cpu_original='Module _linesearch_iterative_kernel__locals__kernel' in text and "device 'cpu'" in text,
        actual_cpu_instrumented='Module _linesearch_iterative_kernel_ray_trace_v1__locals__kernel' in text,
        independent_ray_reference='Ran 3 tests' in rt and '\nOK\n' in rt)
    if not all(checks.values()): raise ValueError('CPU qualification prerequisites differ')
    out.mkdir(); manifest=generate(reg['inputs']['installed_solver'],out/'generated_factory.py')
    checks.update(manifest['source_proof'])
    write_json(out/'generated_manifest.json',manifest)
    docs=[ROOT/'docs'/n for n in ['pen-paired-ray-trace-protocol-v1.md','pen-linesearch-epoch-audit-results-2026-11-17.md','research-plan-v1.53.md']]
    test_sources=[ROOT/'tests'/n for n in ['test_pen_ray_trace_factory_v1.py','test_pen_ray_reference_v1.py','test_pen_ray_epoch_audit_v1.py']]
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),out/'generated_factory.py']+docs+test_sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    identity=dict(schema='pen_paired_ray_factory_cpu_prototype:1',generated_manifest=file_pin(out/'generated_manifest.json'),source_epoch_audit=auditpin,protocol=file_pin(docs[0]))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    logs=[ROOT/'runs'/n for n in ['pen_ray_trace_factory_v1_tests.log','pen_ray_trace_factory_v1_tests_v2.log','pen_ray_trace_tests607_v1.log','pen_ray_trace_tests607_v2.log']]
    result=dict(schema='pen_ray_trace_factory_cpu_qualification:1',prior_comparator=file_pin(ROOT/'provenance/pen_termination_comparator_v1/ledger.json'),
        source_receipt=file_pin(out/'source_receipt.json'),epoch_audit=auditpin,prototype_identity=identity,prototype_id=digest,tests=file_pin(tests),
        reference_tests=file_pin(reference),retained_stage_logs=[file_pin(p) for p in logs],documents=[file_pin(p) for p in docs],
        checks=checks,passed=sum(checks.values()),total=len(checks),cpu_factory_fixtures=7,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        cuda_qualified=False,post_update_copier_implemented=False,coupled_observer_implemented=False,native_worker_implemented=False,complete_saved_reader_implemented=False,
        prospective_queries=240,prospective_native_cap=1920,native_trace_registered=False,
        scope='Exact physical AST and actual CPU primitive fixtures plus independent fixed-ray arithmetic only. Permanent generated source is canonical;CPU tests use same generator on temporary copies. '
              'Not a coupled live observer,CUDA/source-transfer/physical qualification,native registration,action-gradient or task/ego/speed admission.')
    verify_source_snapshot(receipt,live=True); verify_file_pin(manifest['installed_solver']); verify_file_pin(auditpin); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],prototype_id=digest,native=0)))


if __name__=='__main__': main()
