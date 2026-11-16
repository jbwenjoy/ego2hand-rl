"""Two distinct cylinder/mesh kernels on all44 fixed geometry challenge cases."""
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
from mujoco_warp._src import collision_convex,collision_core,types
from mujoco_warp._src.math import upper_trid_index
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.ccd_diagnostic_sources import convex_source
from supermimic_research.runtime.ccd_kernel_namespace import namespace_builders
from supermimic_research.runtime.ccd_axis_retry import axis_retry_source
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['max_scenes'],cfg['max_pair_queries'],cfg['max_extra_CCD_attempts'],cfg['horizon'],cfg['face_factor'])!=('BOUNDED_GPU_CCD_AXIS_RETRY',44,88,132,96,20):raise ValueError('Fixed retry scope required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['source_report','kernel_isolation_report','protocol','registration']}
    prior=json.loads(paths['source_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prior['source_receipt']).read_text()),live=True)
    if len(prior['scenes'])!=44:raise ValueError('All44 frozen scenes required')
    inputs={k:file_pin(paths[k]) for k in ['source_report','kernel_isolation_report','protocol']};inputs.update(parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'])
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    engine=file_pin(Path(collision_convex.__file__))
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='CCD_axis_retry_registration:1',config=file_pin(cp),inputs=inputs,installed_convex=engine,source_receipt=file_pin(out/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['installed_convex']!=engine:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh retry output required')
    out.mkdir();receipt=freeze(out);attempted=completed=0;rows=[]
    try:
        folder=out/'isolated_sources';folder.mkdir();modules={};factories={};sources=[]
        for name in ['CONTROL','AXIS_RETRY']:
            text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=96,face_factor=20)
            if name=='AXIS_RETRY':text,edit=axis_retry_source(text);edits.append(edit)
            text,mapping=namespace_builders(text,'axis_study_'+name.lower());p=folder/(name+'.py');p.write_text(text,encoding='utf-8',newline='\n')
            modules[name]=imported('isolated_axis_study_'+name.lower(),p);factories[name]=mapping['ccd_kernel_builder'];sources.append(file_pin(p));write(folder/(name+'_edits.json'),dict(edits=edits,mapping=mapping))
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            idx=int(upper_trid_index(len(types.GeomType),5,7));args=(5,7,50,50,True,idx,model.block_dim.convex_ccd,bool(model.opt.warn_overflow))
            kernels={n:getattr(modules[n],factories[n])(*args) for n in modules}
            if len({id(k) for k in kernels.values()})!=2:raise ValueError('Cache identity aliases')
            write(out/'kernel_identities.json',[dict(variant=n,factory=factories[n],kernel_id=id(k),module=k.module.name,key=k.key) for n,k in kernels.items()])
            ctx=collision_core.create_collision_context(data.naconmax);p=np.zeros((data.naconmax,2),np.int32);p[0]=[1,55]
            ctx.collision_pair.assign(p);ctx.collision_pairid.fill_(-1);ctx.collision_worldid.fill_(31)
            for scene in prior['scenes']:
                f=np.load(verify_file_pin(scene['geometry']));data.geom_xpos.assign(f['geom_xpos']);data.geom_xmat.assign(f['geom_xmat'])
                for name in ['CONTROL','AXIS_RETRY']:
                    print('QUERY_BEGIN '+scene['name']+' '+name,flush=True)
                    data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                    modules[name].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                    if getattr(modules[name],factories[name])(*args) is not kernels[name]:raise ValueError('Kernel cache changed')
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist();contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    path=out/(scene['name']+'_'+name+'.npz');np.savez_compressed(path,overflow=np.asarray(mask,np.int32),**contact)
                    row=dict(scene=scene['name'],variant=name,native_distance_m=scene['native_distance_m'],boundary_scope=scene['boundary_scope'],reference_scene=scene['geometry'],raw_mask=mask,
                        nacon=count,finite=bool(all(np.isfinite(v).all() for v in contact.values())),GPU_distances_m=contact['dist'].tolist(),artifact=file_pin(path))
                    rows.append(row);write(out/'rows.json',rows);print('QUERY_END '+json.dumps(dict(scene=scene['name'],variant=name,raw=mask[31],nacon=count)),flush=True)
        if completed!=88:raise ValueError('Query accounting mismatch')
        verify_source_snapshot(receipt,live=True);verify_file_pin(engine)
        report=dict(schema='bounded_CCD_axis_retry_probe:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,installed_convex=engine,
            source_receipt=file_pin(out/'source_receipt.json'),variants=sources,construction=file_pin(out/'construction.json'),kernel_identities=file_pin(out/'kernel_identities.json'),results=file_pin(out/'rows.json'),
            attempted_pair_queries=attempted,actual_pair_queries=completed,max_extra_CCD_attempts=132,actual_native_integrations=0,optimizer_steps=0,FK_builds=0,new_native_distance_queries=0,wall_s=time.perf_counter()-start,
            scope='Same44 derived fixtures, bounded GPU-internal retry only. Actual extra attempts require printf-log review. No production/state/task/performance/ego admission.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_pair_queries=attempted,actual_pair_queries=completed,actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
