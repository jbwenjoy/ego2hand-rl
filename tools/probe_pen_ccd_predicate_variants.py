"""Same44 scenes; isolated scratch and selective tetra-predicate precision."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
import mujoco
import warp as wp
from mujoco_warp._src import collision_gjk,collision_convex,collision_core,types
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.ccd_diagnostic_sources import convex_source
from supermimic_research.runtime.ccd_predicate_precision import predicate_source
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['max_scenes'],cfg['max_pair_queries'],cfg['horizon'],cfg['face_factor'],cfg['variants'])!=('ISOLATED_TETRA_PREDICATE_PRECISION',44,220,96,20,['INSTALLED','SCRATCH20','SIDE64','SIMPLEX64','BOTH64']):raise ValueError('Fixed predicate scope required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['source_report','protocol','registration']}
    prior=json.loads(paths['source_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prior['source_receipt']).read_text()),live=True)
    if len(prior['scenes'])!=44:raise ValueError('All44 frozen scenes required')
    inputs=dict(source_report=file_pin(paths['source_report']),protocol=file_pin(paths['protocol']),parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'])
    engine=[file_pin(Path(m.__file__)) for m in [collision_gjk,collision_convex,collision_core]]
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='CCD_predicate_variant_registration:1',config=file_pin(cp),inputs=inputs,installed_sources=engine,source_receipt=file_pin(out/'source_receipt.json'),max_pair_queries=220))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['installed_sources']!=engine:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh probe required')
    out.mkdir();receipt=freeze(out);attempted=completed=0;rows=[]
    try:
        folder=out/'isolated_sources';folder.mkdir();modules={'INSTALLED':collision_convex};variants=[]
        for name,side,simplex in [('SCRATCH20',False,False),('SIDE64',True,False),('SIMPLEX64',False,True),('BOTH64',True,True)]:
            gjk_module=None
            if side or simplex:
                text,edits=predicate_source(Path(collision_gjk.__file__).read_text(),side=side,simplex=simplex)
                path=folder/(name+'_gjk.py');path.write_text(text,encoding='utf-8',newline='\n');gjk_module='isolated_pen_predicate_'+name.lower()
                imported(gjk_module,path);variants.append(file_pin(path));write(folder/(name+'_gjk_edits.json'),edits)
            text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=96,face_factor=20,gjk_module=gjk_module)
            path=folder/(name+'_convex.py');path.write_text(text,encoding='utf-8',newline='\n')
            modules[name]=imported('isolated_pen_convex_'+name.lower(),path);variants.append(file_pin(path));write(folder/(name+'_convex_edits.json'),edits)
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            ctx=collision_core.create_collision_context(data.naconmax);p=np.zeros((data.naconmax,2),np.int32);p[0]=[1,55]
            ctx.collision_pair.assign(p);ctx.collision_pairid.fill_(-1);ctx.collision_worldid.fill_(31)
            for scene in prior['scenes']:
                f=np.load(verify_file_pin(scene['geometry']));data.geom_xpos.assign(f['geom_xpos']);data.geom_xmat.assign(f['geom_xmat'])
                for name in cfg['variants']:
                    data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                    modules[name].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist()
                    contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    artifact=out/(scene['name']+'_'+name+'.npz');np.savez_compressed(artifact,overflow=np.asarray(mask,np.int32),**contact)
                    row=dict(scene=scene['name'],variant=name,reference_scene=scene['geometry'],native_distance_m=scene['native_distance_m'],boundary_scope=scene['boundary_scope'],
                        raw_mask=mask,nacon=count,finite=bool(all(np.isfinite(v).all() for v in contact.values())),GPU_distances_m=contact['dist'].tolist(),artifact=file_pin(artifact))
                    rows.append(row);write(out/'rows.json',rows)
                print(json.dumps(dict(scene=scene['name'],flags=[r['raw_mask'][31] for r in rows[-5:]],contacts=[r['nacon'] for r in rows[-5:]])),flush=True)
            if completed!=220:raise ValueError('Fixed query accounting mismatch')
        verify_source_snapshot(receipt,live=True)
        for p in engine:verify_file_pin(p)
        report=dict(schema='isolated_CCD_predicate_variants:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),installed_sources=engine,
            variants=variants,results=file_pin(out/'rows.json'),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,new_native_distance_queries=0,FK_builds=0,wall_s=time.perf_counter()-started,
            scope='Same44 derived scenes and oracle inputs; selective FP64 tetra predicates only. No broad backend/state/training qualification or performance claim.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
