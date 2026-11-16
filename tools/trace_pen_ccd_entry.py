"""Trace distinct scratch/geometry exits while preserving reference results."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import mujoco
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import collision_gjk, collision_convex, collision_core, types
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.ccd_diagnostic_sources import traced_gjk, convex_source
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    start = time.perf_counter(); cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if (cfg['study'], cfg['max_pair_queries'], cfg['horizons'], cfg['face_factor']) != ('CCD_ENTRY_AND_FACE_CAPACITY_TRACE', 4, [24,48], 5):
        raise ValueError('Fixed four-query trace required')
    pp, tp = [(cp.parent / cfg[k]).resolve() for k in ['pair_report','translation_report']]
    pair, trans = [json.loads(p.read_text()) for p in [pp,tp]]
    for r in [pair,trans]: verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()),live=True)
    pr, tr = [json.loads(verify_file_pin(p).read_text()) for p in [pair['capacity_results'], trans['rows']]]
    def select(rows, **wanted):
        found = [r for r in rows if all(r.get(k)==v for k,v in wanted.items())]
        if len(found)!=1: raise ValueError('Unique pinned case required')
        return found[0]
    cases = [('BASE24',24,select(pr,scene='INCIDENT_BASE',capacity=24)),
        ('BASE48',48,select(pr,scene='INCIDENT_BASE',capacity=48)),
        ('LOCAL24',24,select(tr,scene='QUAT3_ULP-1',placement='GEOM1_ORIGIN')),
        ('PARENT24',24,select(pr,scene='PARENT_RAW',capacity=24))]
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'): raise ValueError('Fresh trace required')
    out.mkdir()
    receipt = freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py'],out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(out/'source_receipt.json',receipt)
    engine = [file_pin(Path(m.__file__)) for m in [collision_gjk,collision_convex]]
    gjk_text, edits = traced_gjk(Path(collision_gjk.__file__).read_text())
    write(out/'protocol.json',dict(config=file_pin(cp),inputs=[file_pin(pp),file_pin(tp)],cases=[dict(label=l,horizon=h,reference=r) for l,h,r in cases],
        max_pair_queries=4,installed_sources=engine,source_edits=edits,native_integrations=0,optimizer_steps=0,
        scope='printf-only distinct face-capacity/projection/GJK/EPA-initialization observations. Require raw flag/count reference equality.'))
    attempted=completed=0; rows=[]
    try:
        folder=out/'isolated_sources';folder.mkdir();path=folder/'collision_gjk.py';path.write_text(gjk_text,encoding='utf-8',newline='\n')
        imported('isolated_pen_entry_gjk',path); variants=[file_pin(path)]; modules={}
        for horizon in [24,48]:
            text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=horizon,gjk_module='isolated_pen_entry_gjk')
            path=folder/f'convex{horizon}.py';path.write_text(text,encoding='utf-8',newline='\n')
            modules[horizon]=imported(f'isolated_pen_entry_convex{horizon}',path);variants.append(file_pin(path))
            write(folder/f'edits{horizon}.json',edits)
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(pair['inputs']['parent']),map_location='cpu',weights_only=False)
            host=mujoco.MjModel.from_binary_path(str(verify_file_pin(pair['inputs']['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            ctx=collision_core.create_collision_context(data.naconmax);p=np.zeros((data.naconmax,2),np.int32);p[0]=[1,55]
            ctx.collision_pair.assign(p);ctx.collision_pairid.fill_(-1);ctx.collision_worldid.fill_(31)
            for label,horizon,reference in cases:
                scene=np.load(verify_file_pin(reference['artifact']));data.geom_xpos.assign(scene['geom_xpos']);data.geom_xmat.assign(scene['geom_xmat'])
                data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();print('TRACE_CASE_BEGIN '+label,flush=True)
                attempted+=1;modules[horizon].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)])
                wp.synchronize();completed+=1
                row=dict(label=label,raw_mask=data.overflow.numpy().tolist(),nacon=int(data.nacon.numpy()[0]))
                row['reference_equal']=row['raw_mask']==reference['raw_mask'] and row['nacon']==reference['nacon'];rows.append(row)
                write(out/'rows.json',rows);print('TRACE_CASE_END '+json.dumps(row),flush=True)
                if not row['reference_equal']:raise ValueError('Instrumentation changes reference')
        verify_source_snapshot(receipt,live=True)
        for p in engine:verify_file_pin(p)
        report=dict(schema='isolated_CCD_entry_trace:1',config=file_pin(cp),protocol=file_pin(out/'protocol.json'),source_receipt=file_pin(out/'source_receipt.json'),
            installed_sources=engine,source_variants=variants,rows=rows,attempted_pair_queries=attempted,actual_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,wall_s=time.perf_counter()-start,
            scope='Derived fixtures, printf-only trace;old attach-face return0 label was ambiguous. No physical repair/backend admission.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_pair_queries=attempted,actual_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
