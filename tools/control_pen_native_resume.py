"""Same-process control for diagnosing portable versus native repeat differences."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.pen_portable_state import capture,restore
from supermimic_research.runtime.pen_resume_probe import advance,substep_record
from review_pen_native_resume import compare_packets


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());gp=(cp.parent/cfg['gates']).resolve();gates=tomllib.loads(gp.read_text())
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['partial_reset_at_boundary']:raise ValueError('Fresh uninterrupted control output')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/review_pen_native_resume.py',cp,gp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;record=False;raw=[];adapter=None
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
        if record:raw.append(substep_record(adapter))
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),a.worlds)
            for k in range(8):advance(adapter,k,partial_reset_at_boundary=False)
            state=capture(adapter,receipt);torch.save(state,out/'native_state.pt')
            def continuation():
                nonlocal raw,record
                raw=[];record=True
                transitions=[advance(adapter,k,partial_reset_at_boundary=False) for k in range(8,16)]
                record=False
                return dict(substeps=raw,transitions=transitions)
            A=continuation();immediate=restore(adapter,state);B=continuation()
            torch.save(dict(original=A,replayed=B),out/'continuations.pt')
            result=compare_packets(A,B,gates)
            result['checks'].update(immediate_array_bytes=immediate['all_array_bytes_equal'],actual_native_budget=observed==192*a.worlds)
            result.update(schema='pen_native_same_process_restore_control:1',config=file_pin(cp),gates=file_pin(gp),
                source_receipt=file_pin(out/'source_receipt.json'),checkpoint=file_pin(out/'native_state.pt'),
                continuations=file_pin(out/'continuations.pt'),immediate=immediate,actual_native_integrations=observed,
                passed=int(sum(result['checks'].values())),total=len(result['checks']),actual_optimizer_updates=0,
                worlds=a.worlds,scope='Declared-world same-process full-array restore control, uninterrupted boundary. Not a fresh-process or policy/task success qualification.')
            verify_source_snapshot(receipt,live=True);write(out/'report.json',result)
            print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],
                first_failed_substep=result['first_failed_substep'],maximum_state_errors=result['maximum_state_errors'])))
            if not all(result['checks'].values()):raise ValueError('Same-process native repeat also unqualified; retain failure')
    finally:
        Simulation.step=original
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds',type=int,choices=[1,32],required=True);main(p.parse_args())
