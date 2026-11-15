"""One fresh process for producing or restoring a controlled native checkpoint."""
import argparse
import json
import random
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.pen_portable_state import capture,restore
from supermimic_research.runtime.pen_resume_probe import advance,substep_record


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def cpu(v):return v.detach().cpu().numpy().copy()


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh process output required')
    if cfg['prefix_controls']!=8 or cfg['continuation_controls']!=8 or a.worlds not in cfg['worlds']:raise ValueError('Declared resume protocol')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study']
        for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,(cp.parent/cfg['adapter_config']).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    oldstep=Simulation.step;observed=0;adapter=None;raw=[];record=False
    def step(sim):
        nonlocal observed
        oldstep(sim);observed+=sim.num_envs
        if record:
            raw.append(substep_record(adapter))
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),a.worlds)
            phase='initial';payload=[];immediate=None;checkpoint=None
            def control(k):
                payload.append(advance(adapter,k,partial_reset_at_boundary=cfg['partial_reset_at_boundary']))
            if a.mode=='producer':
                phase='prefix'
                for k in range(8):control(k)
                phase='capture';state=capture(adapter,receipt)
                checkpoint=out/'native_state.pt';temporary=out/'native_state.tmp'
                torch.save(state,temporary);temporary.replace(checkpoint)
                write(out/'python_inventory.json',state['python']['description'])
                print(json.dumps(dict(checkpoint=file_pin(checkpoint),observed_integrations=observed)),flush=True)
            else:
                phase='load';state=torch.load(a.checkpoint.resolve(),map_location='cpu',weights_only=False)
                checkpoint=a.checkpoint.resolve()
                phase='restore';immediate=restore(adapter,state);write(out/'immediate_restore.json',immediate)
                if not immediate['all_array_bytes_equal']:raise ValueError('Immediate native/Python array restore differs')
            witness=dict(python=[random.random() for _ in range(3)],numpy=np.random.rand(3).tolist(),
                torch=torch.rand(3).tolist(),cuda=torch.rand(3,device=adapter.device).cpu().tolist())
            write(out/'rng_witness.json',witness)
            payload=[];record=True;phase='continuation'
            for k in range(8,16):control(k)
            record=False
            torch.save(dict(transitions=payload,substeps=raw),out/'continuation.pt')
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='pen_native_portable_worker:1',mode=a.mode,worlds=a.worlds,checkpoint=file_pin(checkpoint),
                source_receipt=file_pin(out/'source_receipt.json'),continuation=file_pin(out/'continuation.pt'),
                rng_witness=file_pin(out/'rng_witness.json'),immediate=immediate,
                observed_integrations=observed,logical_adapter_integrations=adapter.physics_integrations,
                config=file_pin(cp),partial_reset_at_boundary=cfg['partial_reset_at_boundary'],
                actual_optimizer_updates=0,scope='Completed worker, independent comparison still required; not portable replay/task qualification.')
            write(out/'report.json',result);print(json.dumps(result),flush=True)
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),observed_integrations=observed,
            mode=a.mode,worlds=a.worlds,phase=locals().get('phase','construction')))
        raise
    finally:
        Simulation.step=oldstep
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['producer','restorer'],required=True);p.add_argument('--worlds',type=int,required=True)
    p.add_argument('--checkpoint',type=Path);main(p.parse_args())
