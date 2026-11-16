"""Actual factory CPU allocations from ten preserved source layout envelopes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import warp as wp
from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.runtime.pen_solve_exit_observer_v2 import expected_layout
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_context_layout_v2_preflight';prep_path=ROOT/'runs/pen_solver_tick_preparation_v1/report.json';prep=json.loads(prep_path.read_text())
    if out.exists():raise ValueError('Fresh layout preflight required')
    rows=[];checks={}
    for entry in prep['rows']:
        s=load_shared(entry['case']['source']);ms=s['native']['model']['scalars'];ds=s['native']['data']['scalars'];nv,nvp,sv,W=ms['nv'],ms['nv_pad'],ms['opt.solver'],s['worlds']
        # Only the factory's numeric allocation envelope is provided; no
        # native Model/Data or solver/forward/step is constructed or executed.
        with wp.ScopedDevice('cpu'):
            ctx=solver._create_solver_context(SimpleNamespace(nv=nv,nv_pad=nvp,opt=SimpleNamespace(solver=sv)),SimpleNamespace(nworld=W,njmax=ds['njmax']))
        shape={k:list(getattr(ctx,k).shape) for k in ['grad','Mgrad','search']};wanted=expected_layout(nv,nvp,sv,W)
        local=dict(actual_factory_layout=all(tuple(shape[k])==v for k,v in wanted.items()),gradient_float32=ctx.grad.dtype==wp.float32,zero_padded_gradient=bool(np.all(ctx.grad.numpy()[0,nv:]==0)),v1_gradient_defect=tuple(shape['grad'])!=(1,26),v1_Mgrad_defect=tuple(shape['Mgrad'])!=(1,26))
        checks.update({entry['name']+'_'+k:bool(v) for k,v in local.items()});rows.append(dict(name=entry['name'],source=entry['case']['source'],nv=nv,nv_pad=nvp,solver=sv,factory_shapes=shape,checks=local))
    checks['all10']=len(rows)==10;checks['no_v1_replica_outputs']=all(not (ROOT/'runs/pen_solve_exit_replay_v1'/r).exists() for r in ['A','B'])
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tests/test_pen_context_layout_v2.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='pen_context_layout_CPU_preflight:2',preparation=file_pin(prep_path),installed_solver=file_pin(solver.__file__),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,CUDA_graph_qualified=False,scope='Real pinned factory allocation code on CPU with saved numeric envelopes only. No native Model/Data/solve/forward/step. V1 shape defect confirmed;v2 graph/replay remains unqualified.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Layout preflight discrepancy preserved')


if __name__=='__main__':main()
