"""Pin the completed CPU kernel qualification without admitting native replay."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.pen_termination_rule import implementation_identity
from supermimic_research.strict_json import write_json


def main():
    out = ROOT / 'runs/pen_termination_rule_cpu_qualification_v1'
    if out.exists():
        raise ValueError('Fresh CPU qualification required')
    old = ROOT / 'runs/pen_solve_exit_replay_registration_v2/registration.json'
    registration = json.loads(old.read_text())
    installed = registration['inputs']['installed_solver']
    backend_pin = registration['inputs']['backend_manifest']
    backend = json.loads(verify_file_pin(backend_pin).read_text())
    identity = implementation_identity(backend['backend_id'], installed)
    log = ROOT / 'runs/pen_termination_rule_cpu597_v1.log'
    text = log.read_text(encoding='utf-8-sig')
    checks = dict(full597='Ran 597 tests in 11.245s' in text and '\nOK\n' in text,
                  actual_cpu_original_kernel="Module mujoco_warp._src.solver" in text and "device 'cpu'" in text,
                  actual_cpu_candidate_kernel='Module supermimic_research.runtime.pen_termination_rule_kernels' in text,
                  only_one_expression=identity['identity']['single_expression_change'],
                  distinct_identity=identity['backend_id'] != backend['backend_id'],
                  no_physical_admission=identity['physically_qualified'] is False,
                  no_source_transfer_admission=identity['old_source_transfer_qualified'] is False)
    if not all(checks.values()):
        raise ValueError('CPU qualification evidence discrepancy')
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), ROOT / 'tests/test_pen_termination_rule.py',
        ROOT / 'docs/research-plan-v1.51.md', ROOT / 'docs/pen-termination-rule-comparator-protocol-v1.md']
    out.mkdir()
    receipt = freeze_sources(sources, out / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(out / 'source_receipt.json', receipt)
    report = dict(schema='pen_termination_rule_cpu_qualification:1', prior_registration=file_pin(old),
        prior_results=file_pin(ROOT / 'provenance/pen_full54_solve_exit_v1/ledger.json'),
        source_receipt=file_pin(out / 'source_receipt.json'), identity=identity, tests=file_pin(log),
        retained_initial_five_tests=file_pin(ROOT / 'runs/pen_termination_rule_cpu_v1.log'),
        protocol=file_pin(ROOT / 'docs/pen-termination-rule-comparator-protocol-v1.md'),
        plan=file_pin(ROOT / 'docs/research-plan-v1.51.md'),
        checks=checks, passed=sum(checks.values()), total=len(checks),
        declared_future_queries=240, declared_future_native_cap=1920,
        actual_native_integrations=0, optimizer_steps=0, normalizer_fits=0,
        native_worker_implemented=False, native_reader_implemented=False, source_import_implemented=False,
        scope='Actual CPU kernel branch/strict-boundary/cleanup and full597 tests only. '
              'Not native graph/source-transfer/physical qualification or a completed registered comparator. '
              'Child CUDA visibility deliberately disabled; no installed sources/drivers/CUDA edited.')
    verify_source_snapshot(receipt, live=True)
    write_json(out / 'report.json', report)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[sum(checks.values()), len(checks)],
                          candidate_backend_id=identity['backend_id'], native=0)))


if __name__ == '__main__':
    main()
