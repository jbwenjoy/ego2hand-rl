"""Fixed paired source/geometry controls for scratch and tetra recovery."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
import mujoco
import warp as wp
from mujoco_warp._src import collision_gjk,collision_convex,collision_core,types
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.ccd_diagnostic_sources import convex_source
from supermimic_research.runtime.ccd_simplex_recovery import retry_inconsistent_tetrahedron
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    expected=[('INSTALLED',24,5,False),('SCRATCH20',96,20,False),('SCRATCH40',96,40,False),('RETRY20',96,20,True),('RETRY40',96,40,True)]
    if (cfg['study'],cfg['max_scenes'],cfg['max_pair_queries'],cfg['max_native_distance_queries'])!=('ISOLATED_SCRATCH_AND_TETRA_RECOVERY',44,220,44) or [(v['name'],v['horizon'],v['face_factor'],v['retry_tetra']) for v in cfg['variants']]!=expected:
        raise ValueError('Fixed variant and query budget required')
    if cfg['separation_epsilons_m']!=[.00000025,.000002,.00001] or cfg['separation_extra_steps_m']!=[-.00005,.00005]:raise ValueError('Fixed challenge offsets required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['pair_report','translation_report','trace_report','protocol','registration']}
    pair=json.loads(paths['pair_report'].read_text());trans=json.loads(paths['translation_report'].read_text())
    for p in [pair,trans]:verify_source_snapshot(json.loads(verify_file_pin(p['source_receipt']).read_text()),live=True)
    source_rows=json.loads(verify_file_pin(trans['rows']).read_text())
    if len(source_rows)!=36:raise ValueError('All36 retained translation controls required')
    inputs={k:file_pin(paths[k]) for k in ['pair_report','translation_report','trace_report','protocol']}
    inputs.update(parent=pair['inputs']['parent'],host_model=pair['inputs']['host_model'])
    engine=[file_pin(Path(m.__file__)) for m in [collision_gjk,collision_convex,collision_core]]
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='isolated_CCD_recovery_registration:1',config=file_pin(cp),inputs=inputs,
            installed_sources=engine,source_scene_rows=trans['rows'],source_receipt=file_pin(out/'source_receipt.json'),max_GPU_pair_queries=220,max_native_distance_queries=44))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['installed_sources']!=engine:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh recovery probe required')
    out.mkdir();receipt=freeze(out);attempted=completed=cpu_attempted=cpu_completed=0;rows=[];scenes=[]
    try:
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False)
            host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            native=mujoco.MjData(host);ms=parent['native']['native']['model']['arrays']
            host.geom_size[:]=ms['geom_size'][31%ms['geom_size'].shape[0]].astype(np.float64)
            host.opt.ccd_iterations=50;host.opt.ccd_tolerance=float(ms['opt.ccd_tolerance'][0])
            for r in source_rows:
                f=np.load(verify_file_pin(r['artifact']));scenes.append(dict(name=r['scene']+'_'+r['placement'],source=r['artifact'],
                    geom_xpos=f['geom_xpos'].copy(),geom_xmat=f['geom_xmat'].copy()))
            base=next(s for s in scenes if s['name']=='INCIDENT_BASE_WORLD')
            # Reuse the pinned prior witness and depth; scene construction itself
            # performs no uncounted oracle query.
            prior=next(r for r in source_rows if r['scene']=='INCIDENT_BASE' and r['placement']=='WORLD')
            f=np.load(verify_file_pin(prior['artifact']));witness=f['native_fromto'];depth=abs(prior['native_signed_distance_m'])
            normal=witness[3:]-witness[:3];normal/=np.linalg.norm(normal)
            shifts=[sign*(depth+eps) for eps in cfg['separation_epsilons_m'] for sign in [-1,1]]+cfg['separation_extra_steps_m']
            object_body=host.body('object/cube').id;gids=np.flatnonzero(host.geom_bodyid==object_body)
            for i,offset in enumerate(shifts):
                positions=base['geom_xpos'].copy();positions[31,gids]=(positions[31,gids].astype(np.float64)+offset*normal).astype(np.float32)
                scenes.append(dict(name=f'SEPARATION_{i}',source=prior['artifact'],geom_xpos=positions,geom_xmat=base['geom_xmat'].copy(),offset_m=offset))
            modules={'INSTALLED':collision_convex};variants=[];folder=out/'isolated_sources';folder.mkdir()
            recovered,edit=retry_inconsistent_tetrahedron(Path(collision_gjk.__file__).read_text())
            path=folder/'collision_gjk_retry.py';path.write_text(recovered,encoding='utf-8',newline='\n')
            imported('isolated_pen_tetra_retry_gjk',path);variants.append(file_pin(path));write(folder/'gjk_edit.json',edit)
            for v in cfg['variants'][1:]:
                text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=v['horizon'],face_factor=v['face_factor'],
                    gjk_module='isolated_pen_tetra_retry_gjk' if v['retry_tetra'] else None)
                path=folder/(v['name']+'.py');path.write_text(text,encoding='utf-8',newline='\n')
                modules[v['name']]=imported('isolated_pen_variant_'+v['name'].lower(),path);variants.append(file_pin(path));write(folder/(v['name']+'_edits.json'),edits)
            ctx=collision_core.create_collision_context(data.naconmax);p=np.zeros((data.naconmax,2),np.int32);p[0]=[1,55]
            ctx.collision_pair.assign(p);ctx.collision_pairid.fill_(-1);ctx.collision_worldid.fill_(31)
            scene_pins=[]
            for scene in scenes:
                native.geom_xpos[:]=scene['geom_xpos'][31].astype(np.float64);native.geom_xmat[:]=scene['geom_xmat'][31].reshape(host.ngeom,9).astype(np.float64)
                witness=np.full(6,np.nan);cpu_attempted+=1
                distance=float(mujoco.mj_geomDistance(host,native,1,55,1.,witness));cpu_completed+=1
                path=out/(scene['name']+'_geometry.npz');np.savez_compressed(path,geom_xpos=scene['geom_xpos'],geom_xmat=scene['geom_xmat'],native_fromto=witness)
                scene_pins.append(dict(name=scene['name'],source=scene['source'],geometry=file_pin(path),native_distance_m=distance,
                    boundary_scope='OVERLAP' if distance < -1e-6 else 'SEPARATED' if distance>1e-6 else 'WITHIN_1UM_BOUNDARY'))
                data.geom_xpos.assign(scene['geom_xpos']);data.geom_xmat.assign(scene['geom_xmat'])
                for v in cfg['variants']:
                    data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                    modules[v['name']].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)])
                    wp.synchronize();completed+=1
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist()
                    contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    artifact=out/(scene['name']+'_'+v['name']+'.npz');np.savez_compressed(artifact,overflow=np.asarray(mask,np.int32),**contact)
                    row=dict(scene=scene['name'],variant=v['name'],native_distance_m=distance,raw_mask=mask,nacon=count,
                        contact_finite=bool(all(np.isfinite(a).all() for a in contact.values())),GPU_distances_m=contact['dist'].tolist(),artifact=file_pin(artifact))
                    rows.append(row);write(out/'rows.json',rows)
                print(json.dumps(dict(scene=scene['name'],reference=distance,flags=[r['raw_mask'][31] for r in rows[-5:]],contacts=[r['nacon'] for r in rows[-5:]])),flush=True)
            if (completed,cpu_completed)!=(220,44):raise ValueError('Fixed query accounting mismatch')
            verify_source_snapshot(receipt,live=True)
            for p in engine:verify_file_pin(p)
            report=dict(schema='isolated_CCD_recovery_variants:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,
                source_receipt=file_pin(out/'source_receipt.json'),installed_sources=engine,source_variants=variants,
                construction=file_pin(out/'construction.json'),scenes=scene_pins,results=file_pin(out/'rows.json'),
                attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,attempted_native_distance_queries=cpu_attempted,
                actual_native_distance_queries=cpu_completed,actual_native_integrations=0,optimizer_steps=0,FK_builds=0,wall_s=time.perf_counter()-started,
                scope='Paired derived collision fixture challenge, including separation/boundary cases. Isolated scratch and geometric retry variants only;no production backend/state/gradient/training admission.')
            write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),GPU=completed,CPU=cpu_completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,
            attempted_native_distance_queries=cpu_attempted,actual_native_distance_queries=cpu_completed,actual_native_integrations=0,optimizer_steps=0,
            source_receipt=file_pin(out/'source_receipt.json'),wall_s=time.perf_counter()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true')
    a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
