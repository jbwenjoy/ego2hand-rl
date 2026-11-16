"""Same484 pairs; normal and depth from a consistent EPA face."""
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
from supermimic_research.runtime.ccd_axis_retry import axis_retry_source
from supermimic_research.runtime.ccd_kernel_namespace import namespace_builders
from supermimic_research.runtime.ccd_epa_face_normal import retain_cylinder_mesh_face,face_frame_source
from supermimic_research.mechanics.ccd_contact_validation import evaluate
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['max_cases'],cfg['max_pair_queries'],cfg['ccd_tolerance'])!=('SAME_EPA_FACE_NORMAL_VALIDATION',484,968,1e-6):raise ValueError('Fixed face-normal scope required')
    limits={k:cfg[k] for k in ['distance_error_m','normal_angle_deg','midpoint_error_m','boundary_m']}
    if list(limits.values())!=[5e-6,5.,1e-4,1e-6]:raise ValueError('Original criteria required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['validation_report','tolerance_report','protocol','registration']}
    prior=json.loads(paths['validation_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prior['source_receipt']).read_text()),live=True)
    cases=[r for r in json.loads(verify_file_pin(prior['results']).read_text()) if r['variant']=='AXIS_RETRY']
    if len(cases)!=484:raise ValueError('All484 cases required')
    scene_report=json.loads(verify_file_pin(prior['inputs']['source_report']).read_text());scenes={s['name']:s for s in scene_report['scenes']}
    inputs={k:file_pin(paths[k]) for k in ['validation_report','tolerance_report','protocol']};inputs.update(parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'],control=prior['backends']['AXIS_RETRY'])
    engine=[file_pin(Path(m.__file__)) for m in [collision_gjk,collision_convex]]
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='EPA_face_normal_registration:1',config=file_pin(cp),inputs=inputs,installed_sources=engine,source_receipt=file_pin(out/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['installed_sources']!=engine:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh face-normal output required')
    out.mkdir();receipt=freeze(out);attempted=completed=0;rows=[]
    try:
        folder=out/'isolated_sources';folder.mkdir();text,edit=retain_cylinder_mesh_face(Path(collision_gjk.__file__).read_text())
        path=folder/'collision_gjk.py';path.write_text(text,encoding='utf-8',newline='\n');imported('isolated_pen_face_normal_gjk',path)
        variants=[file_pin(path)];write(folder/'gjk_edit.json',edit)
        text,edits=convex_source(Path(collision_convex.__file__).read_text(),horizon=96,face_factor=20,gjk_module='isolated_pen_face_normal_gjk')
        text,edit=axis_retry_source(text);edits.append(edit);text,edit=face_frame_source(text);edits.append(edit)
        text,mapping=namespace_builders(text,'face_normal_v1');path=folder/'convex.py';path.write_text(text,encoding='utf-8',newline='\n')
        modified=imported('isolated_pen_face_normal_convex',path);variants.append(file_pin(path));write(folder/'convex_edits.json',dict(edits=edits,mapping=mapping))
        modules=dict(CONTROL=imported('isolated_axis_study_axis_retry',verify_file_pin(inputs['control'])),FACE_NORMAL=modified)
        factories=dict(CONTROL='ccd_kernel_builder_axis_study_axis_retry',FACE_NORMAL=mapping['ccd_kernel_builder'])
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            args=(5,7,50,50,True,int(upper_trid_index(len(types.GeomType),5,7)),model.block_dim.convex_ccd,bool(model.opt.warn_overflow))
            kernels={n:getattr(modules[n],factories[n])(*args) for n in modules}
            if len({id(k) for k in kernels.values()})!=2:raise ValueError('Cache alias')
            write(out/'kernel_identities.json',[dict(variant=n,factory=factories[n],kernel_id=id(k),key=k.key,module=k.module.name) for n,k in kernels.items()])
            ctx=collision_core.create_collision_context(data.naconmax);ctx.collision_worldid.fill_(31);ctx.collision_pairid.fill_(-1);p=np.zeros((data.naconmax,2),np.int32)
            for case in cases:
                frame=np.load(verify_file_pin(scenes[case['scene']]['geometry']));old=np.load(verify_file_pin(case['artifact']));witness=old['native_witness']
                data.geom_xpos.assign(frame['geom_xpos']);data.geom_xmat.assign(frame['geom_xmat']);p[0]=case['geoms'];ctx.collision_pair.assign(p)
                for name in ['CONTROL','FACE_NORMAL']:
                    print('QUERY_BEGIN '+case['scene']+' '+str(case['geoms'])+' '+name,flush=True)
                    data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                    modules[name].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                    if getattr(modules[name],factories[name])(*args) is not kernels[name]:raise ValueError('Kernel identity changed')
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist();contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    checked=evaluate(contact,mask,case['native_distance_m'],witness,case['geoms'],limits)
                    equal=mask==old['overflow'].tolist() and all(contact[k].dtype==old[k].dtype and contact[k].shape==old[k].shape and contact[k].tobytes()==old[k].tobytes() for k in contact)
                    path=out/(case['scene']+f"_pair{case['geoms'][0]}_{case['geoms'][1]}_"+name+'.npz');np.savez_compressed(path,native_witness=witness,overflow=np.asarray(mask,np.int32),**contact)
                    rows.append(dict(scene=case['scene'],geoms=case['geoms'],variant=name,native_distance_m=case['native_distance_m'],reference=case['artifact'],raw_mask=mask,nacon=count,**checked,prior_bytes_equal=equal,artifact=file_pin(path)))
                    write(out/'rows.json',rows);print('QUERY_END '+json.dumps(dict(scene=case['scene'],pair=case['geoms'],variant=name,passed=checked['passed'],total=checked['total'])),flush=True)
        if completed!=968:raise ValueError('Fixed query accounting mismatch')
        verify_source_snapshot(receipt,live=True)
        for p in engine:verify_file_pin(p)
        report=dict(schema='same_EPA_face_normal_validation:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),installed_sources=engine,source_variants=variants,
            construction=file_pin(out/'construction.json'),kernel_identities=file_pin(out/'kernel_identities.json'),results=file_pin(out/'rows.json'),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,
            actual_native_integrations=0,optimizer_steps=0,FK_builds=0,new_native_distance_queries=0,wall_s=time.perf_counter()-start,
            scope='Same484 fixed criteria with consistent EPA face normal;candidate fixture validation only, no integrated backend/state/gradient/training admission.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_pair_queries=attempted,actual_pair_queries=completed,actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
