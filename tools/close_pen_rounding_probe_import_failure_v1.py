"""Account the retained pre-launch vector-type import failure;never rerun V1."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    folder=ROOT/'runs/pen_friction_rounding_probe_v1'; dest=folder/'failure_closure.json'
    if dest.exists() or (folder/'report.json').exists() or (folder/'outputs.npz').exists(): raise ValueError('Only the original closed import failure may be accounted')
    log=file_pin(ROOT/'runs/pen_friction_rounding_probe_v1.log'); text=verify_file_pin(log).read_text(encoding='utf-8-sig')
    receipt=file_pin(folder/'source_receipt.json'); verify_source_snapshot(json.loads(verify_file_pin(receipt).read_text()),live=True)
    source=file_pin(ROOT/'tools/probe_pen_friction_rounding_v1.py'); code=verify_file_pin(source).read_text()
    checks=dict(vector_import_failure="has no attribute 'vec'" in text and 'line 7, in <module>' in text,
        failed_at_module_import='line 39, in main' in text and 'spec.loader.exec_module(module)' in text,
        launch_follows_failed_import=code.index('spec.loader.exec_module(module)')<code.index('wp.init()')<code.index('wp.launch('),
        no_completed_GPU_launch='initialized:' not in text and 'load on device' not in text)
    if not all(checks.values()): raise ValueError('Pre-launch failure evidence differs')
    write_json(dest,dict(schema='pen_rounding_probe_closed_import_failure:1',source=source,accounting_source=file_pin(__file__),traceback=log,
        source_receipt=receipt,intent=file_pin(folder/'intent.json'),checks=checks,passed=sum(checks.values()),total=len(checks),
        actual_GPU_kernel_launches=0,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,resumable=False,
        correction='Separate V2 uses wp.types.vector and passed CPU/actual GPU checks;original files and failed output are retained.',
        scope='A diagnostic import API error before wp.init/any wp.launch;not a native or numerical qualification failure and not a resumed run.'))
    print(json.dumps(dict(report=file_pin(dest),checks=[sum(checks.values()),len(checks)],actual_GPU_kernel_launches=0)),flush=True)


if __name__=='__main__': main()
