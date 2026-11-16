"""Same484 pair/geometry/oracle cases; only explicit CCD numerical precision changes."""
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
from mujoco_warp._src import collision_core,types
from mujoco_warp._src.math import upper_trid_index
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.ccd_contact_validation import evaluate
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['ccd_tolerances'],cfg['max_cases'],cfg['max_pair_queries'],cfg['max_extra_CCD_attempts'])!=('PINNED_RETRY_TOLERANCE_VALIDATION',[1e-6,1e-7,1e-8],484,1452,4356):raise ValueError('Fixed precision budget required')
    limits={k:cfg[k] for k in ['distance_error_m','normal_angle_deg','midpoint_error_m','boundary_m']}
    if list(limits.values())!=[5e-6,5.,1e-4,1e-6]:raise ValueError('Original criteria required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['validation_report','protocol','registration']}
    prior=json.loads(paths['validation_report'].read_text());verify_source_snapshot(json.loads(verify_file_pin(prior['source_receipt']).read_text()),live=True)
    oldrows=json.loads(verify_file_pin(prior['results']).read_text());cases=[r for r in oldrows if r['variant']=='AXIS_RETRY']
    if len(cases)!=484:raise ValueError('All484 prior cases required')
    scene_report=json.loads(verify_file_pin(prior['inputs']['source_report']).read_text());scenes={s['name']:s for s in scene_report['scenes']}
    inputs=dict(validation_report=file_pin(paths['validation_report']),protocol=file_pin(paths['protocol']),backend=prior['backends']['AXIS_RETRY'],
        parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'],scene_report=prior['inputs']['source_report'])
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='retry_CCD_tolerance_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh precision output required')
    out.mkdir();receipt=freeze(out);attempted=completed=0;rows=[]
    try:
        module=imported('isolated_axis_study_axis_retry',verify_file_pin(inputs['backend']));wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            before=model.opt.ccd_tolerance.numpy().copy();write(out/'numerical_changes.json',dict(field='opt.ccd_tolerance',before=before.tolist(),values=cfg['ccd_tolerances'],all_other_Model_fields_unchanged_by_driver=True))
            idx=int(upper_trid_index(len(types.GeomType),5,7));args=(5,7,50,50,True,idx,model.block_dim.convex_ccd,bool(model.opt.warn_overflow));factory=module.ccd_kernel_builder_axis_study_axis_retry;kernel=factory(*args)
            write(out/'kernel_identity.json',dict(factory=factory.__wrapped__.__name__,kernel_id=id(kernel),key=kernel.key,module=kernel.module.name))
            ctx=collision_core.create_collision_context(data.naconmax);ctx.collision_worldid.fill_(31);ctx.collision_pairid.fill_(-1)
            p=np.zeros((data.naconmax,2),np.int32)
            for case in cases:
                scene=scenes[case['scene']];frame=np.load(verify_file_pin(scene['geometry']));old=np.load(verify_file_pin(case['artifact']));witness=old['native_witness']
                data.geom_xpos.assign(frame['geom_xpos']);data.geom_xmat.assign(frame['geom_xmat']);p[0]=case['geoms'];ctx.collision_pair.assign(p)
                for tolerance in cfg['ccd_tolerances']:
                    model.opt.ccd_tolerance.fill_(tolerance);data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_()
                    print('QUERY_BEGIN '+case['scene']+' '+str(case['geoms'])+' '+str(tolerance),flush=True)
                    attempted+=1;module.convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                    if factory(*args) is not kernel:raise ValueError('Kernel identity changed')
                    count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist();contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                    checked=evaluate(contact,mask,case['native_distance_m'],witness,case['geoms'],limits)
                    path=out/(case['scene']+f"_pair{case['geoms'][0]}_{case['geoms'][1]}_tol{tolerance}.npz");np.savez_compressed(path,native_witness=witness,overflow=np.asarray(mask,np.int32),**contact)
                    equal=mask==old['overflow'].tolist() and all(contact[k].dtype==old[k].dtype and contact[k].shape==old[k].shape and contact[k].tobytes()==old[k].tobytes() for k in contact)
                    row=dict(scene=case['scene'],geoms=case['geoms'],tolerance=tolerance,native_distance_m=case['native_distance_m'],reference=case['artifact'],raw_mask=mask,nacon=count,
                        **checked,prior_bytes_equal=equal,artifact=file_pin(path));rows.append(row)
                    write(out/'rows.json',rows);print('QUERY_END '+json.dumps(dict(scene=case['scene'],pair=case['geoms'],tolerance=tolerance,passed=row['passed'],total=row['total'])),flush=True)
            if completed!=1452:raise ValueError('Fixed precision budget mismatch')
        verify_source_snapshot(receipt,live=True);verify_file_pin(inputs['backend'])
        report=dict(schema='pinned_CCD_retry_tolerance_validation:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
            construction=file_pin(out/'construction.json'),numerical_changes=file_pin(out/'numerical_changes.json'),kernel_identity=file_pin(out/'kernel_identity.json'),results=file_pin(out/'rows.json'),
            attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,new_native_distance_queries=0,actual_native_integrations=0,optimizer_steps=0,FK_builds=0,wall_s=time.perf_counter()-start,
            scope='Same484 immutable pair/geometry/oracle cases and fixed criteria;only explicit numerical CCD tolerance changes. Not production state/backend or integrated control admission.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),queries=completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
