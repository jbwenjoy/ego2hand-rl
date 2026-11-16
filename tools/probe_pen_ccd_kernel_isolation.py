"""Same44 inputs, eight demonstrably distinct compiled CCD variants."""
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
from mujoco_warp._src.math import upper_trid_index
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.ccd_diagnostic_sources import convex_source
from supermimic_research.runtime.ccd_simplex_recovery import retry_inconsistent_tetrahedron
from supermimic_research.runtime.ccd_predicate_precision import predicate_source
from supermimic_research.runtime.ccd_kernel_namespace import namespace_builders
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    names=['INSTALLED','SCRATCH20','SCRATCH40','RETRY20','RETRY40','SIDE64','SIMPLEX64','BOTH64']
    if (cfg['study'],cfg['max_scenes'],cfg['max_pair_queries'],cfg['variants'])!=('DISTINCT_KERNEL_ALGORITHM_VARIANTS',44,352,names):raise ValueError('Fixed eight-variant budget required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['source_report','prior_predicate_report','protocol','registration']}
    prior=json.loads(paths['source_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prior['source_receipt']).read_text()),live=True)
    inputs={k:file_pin(paths[k]) for k in ['source_report','prior_predicate_report','protocol']};inputs.update(parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'])
    engine=[file_pin(Path(m.__file__)) for m in [collision_gjk,collision_convex,collision_core]]
    from mujoco_warp._src import warp_util
    engine.append(file_pin(Path(warp_util.__file__)))
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='distinct_CCD_kernel_registration:1',config=file_pin(cp),inputs=inputs,installed_sources=engine,source_receipt=file_pin(out/'source_receipt.json'),max_pair_queries=352))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['installed_sources']!=engine:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh isolated kernel output required')
    out.mkdir();receipt=freeze(out);attempted=completed=0;rows=[]
    try:
        folder=out/'isolated_sources';folder.mkdir();modules={'INSTALLED':collision_convex};factories={'INSTALLED':'ccd_kernel_builder'};sources=[]
        for name in names[1:]:
            gjk_module=None;factor=40 if name.endswith('40') else 20
            if name.startswith('RETRY'):
                text,edit=retry_inconsistent_tetrahedron(Path(collision_gjk.__file__).read_text());edits=[edit]
            elif name in ['SIDE64','SIMPLEX64','BOTH64']:
                text,edits=predicate_source(Path(collision_gjk.__file__).read_text(),side=name!='SIMPLEX64',simplex=name!='SIDE64')
            else:text=None
            if text is not None:
                gjk_module='isolated_proven_gjk_'+name.lower();p=folder/(name+'_gjk.py');p.write_text(text,encoding='utf-8',newline='\n')
                imported(gjk_module,p);sources.append(file_pin(p));write(folder/(name+'_gjk_edits.json'),edits)
            text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=96,face_factor=factor,gjk_module=gjk_module)
            text,mapping=namespace_builders(text,'proven_'+name.lower());p=folder/(name+'_convex.py');p.write_text(text,encoding='utf-8',newline='\n')
            modules[name]=imported('isolated_proven_convex_'+name.lower(),p);factories[name]=mapping['ccd_kernel_builder'];sources.append(file_pin(p))
            write(folder/(name+'_edits.json'),dict(mathematics_and_scratch=edits,kernel_namespace=mapping))
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            # Same arguments as production convex_narrowphase for cylinder/mesh.
            geomgeomid=int(upper_trid_index(len(types.GeomType),int(types.GeomType.CYLINDER),int(types.GeomType.MESH)))
            args=(int(types.GeomType.CYLINDER),int(types.GeomType.MESH),model.opt.ccd_iterations,model.opt.ccd_iterations,
                not bool(model.opt.disableflags & types.DisableBit.MULTICCD),geomgeomid,model.block_dim.convex_ccd,bool(model.opt.warn_overflow))
            kernels={name:getattr(modules[name],factories[name])(*args) for name in names}
            identities=[]
            for name in names:
                kernel=kernels[name];factory=getattr(modules[name],factories[name])
                identities.append(dict(variant=name,factory_name=factory.__wrapped__.__name__,kernel_object_id=id(kernel),
                    kernel_key=str(kernel.key),kernel_module=str(kernel.module.name),builder_module=factory.__wrapped__.__module__,
                    bound_gjk_module=str(getattr(modules[name].gjk_phase,'module',None))))
            write(out/'kernel_identities.json',dict(args=list(args),records=identities))
            if len({id(k) for k in kernels.values()})!=8 or len({v['factory_name'] for v in identities})!=8:
                raise ValueError('Algorithm variants still alias a cached kernel')
            ctx=collision_core.create_collision_context(data.naconmax);p=np.zeros((data.naconmax,2),np.int32);p[0]=[1,55]
            ctx.collision_pair.assign(p);ctx.collision_pairid.fill_(-1);ctx.collision_worldid.fill_(31)
            for scene in prior['scenes']:
                f=np.load(verify_file_pin(scene['geometry']));data.geom_xpos.assign(f['geom_xpos']);data.geom_xmat.assign(f['geom_xmat'])
                for name in names:
                    data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                    modules[name].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                    # Re-request verifies that this module's actual cache entry
                    # is the preflight object; other variants cannot replace it.
                    if getattr(modules[name],factories[name])(*args) is not kernels[name]:raise ValueError('Kernel identity changed during query')
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist();contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    artifact=out/(scene['name']+'_'+name+'.npz');np.savez_compressed(artifact,overflow=np.asarray(mask,np.int32),**contact)
                    rows.append(dict(scene=scene['name'],variant=name,reference_scene=scene['geometry'],native_distance_m=scene['native_distance_m'],boundary_scope=scene['boundary_scope'],
                        raw_mask=mask,nacon=count,finite=bool(all(np.isfinite(v).all() for v in contact.values())),GPU_distances_m=contact['dist'].tolist(),artifact=file_pin(artifact)))
                    write(out/'rows.json',rows)
                print(json.dumps(dict(scene=scene['name'],flags=[r['raw_mask'][31] for r in rows[-8:]],contacts=[r['nacon'] for r in rows[-8:]])),flush=True)
        if completed!=352:raise ValueError('Fixed query budget mismatch')
        verify_source_snapshot(receipt,live=True)
        for p in engine:verify_file_pin(p)
        report=dict(schema='distinct_CCD_kernel_variants:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),installed_sources=engine,
            source_variants=sources,kernel_identities=file_pin(out/'kernel_identities.json'),construction=file_pin(out/'construction.json'),results=file_pin(out/'rows.json'),
            attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,actual_native_integrations=0,optimizer_steps=0,new_native_distance_queries=0,FK_builds=0,wall_s=time.perf_counter()-start,
            scope='Eight independently keyed CCD kernel objects on same44 derived inputs. Prior same-name algorithm comparisons were cache-confounded;physical/backend/state qualification remains separate.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json'),wall_s=time.perf_counter()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
