"""Verify alternating controls and task-local resets for the paired W32 rollout."""
from pathlib import Path
import json,sys
import torch
from benchmark_pen_scaled import ROOT,COLLISION,OBJECTIVE,factory,backend_scope
from evaluate_pen_dev_clip import config
from supermimic_research.runtime.pen_multiclip_batch import multiclip_w32
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json

def main(out):
    out.mkdir(exist_ok=False);obj,ref=config(out,'clip_003',ROOT/'runs/pen_refit_dev_preload_v1/report.json')
    write_json(out/'intent.json',dict(source=file_pin(Path(__file__)),clip3=ref['source'],objective001=file_pin(OBJECTIVE),objective003=file_pin(obj),
        scope='Two separate W32 tasks, same guarded solver and collision scope, canonical reference residuals. Step or reset one task; check all opposite-task qpos/qvel/clock/phase values remain exact. No optimization.'))
    rows=[]
    def snap(rt):
        return [getattr(rt.env.sim.data,k).clone() for k in ('qpos','qvel','time')]+[rt.motion.time_steps.clone()]
    with torch.jit.optimized_execution(False),torch.no_grad(),backend_scope(COLLISION),multiclip_w32(OBJECTIVE,obj,factory()) as tasks:
        for step in range(3):
            for name,other in [('clip_001','clip_003'),('clip_003','clip_001')]:
                rt=tasks[name];before=snap(tasks[other]);phase=rt.motion.time_steps.clone()
                raw=rt.initial_raw[None].expand(32,-1);obs,r,term,trunc,info=rt.step(raw)
                assert torch.equal(info['reference_phase'],phase) and torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
                terminal=(term|trunc).nonzero().flatten()
                if len(terminal):rt.reset(terminal)
                if step==1:
                    ids=torch.tensor([0,7,31],device=rt.device);keep=torch.ones(32,dtype=torch.bool,device=rt.device);keep[ids]=False
                    pre=snap(rt);rt.reset(ids)
                    assert all(torch.equal(a[keep],b[keep]) for a,b in zip(pre,snap(rt)))
                    assert (rt.motion.time_steps[ids]==1).all()
                assert all(torch.equal(a,b) for a,b in zip(before,snap(tasks[other])))
                rows.append(dict(clip=name,round=step,world_steps=rt.owned_native_integrations,terminal_worlds=len(terminal),observation_shape=list(obs['policy'].shape)))
        assert all(rt.owned_native_integrations==3*8*32 for rt in tasks.values())
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,total_native=1536,optimizer_updates=0));print(rows)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
