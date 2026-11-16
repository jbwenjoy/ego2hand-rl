"""All eleven filtered cylinder/mesh pairs, same frozen geometry and retry code."""
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
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture
from trace_pen_epa_degenerate_exit import imported


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['study'],cfg['max_scenes'],cfg['expected_pairs'],cfg['max_pair_queries'],cfg['max_native_distance_queries'],cfg['max_extra_CCD_attempts'])!=('PINNED_AXIS_RETRY_MULTIFINGER_VALIDATION',44,11,968,484,1452):raise ValueError('Fixed all-pair budget required')
    if [cfg[k] for k in ['distance_error_m','normal_angle_deg','midpoint_error_m','boundary_m']]!=[5e-6,5.,1e-4,1e-6]:raise ValueError('Predeclared criteria required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['source_report','retry_report','protocol','registration']}
    prior,retry=[json.loads(paths[k].read_text()) for k in ['source_report','retry_report']]
    for report in [prior,retry]:verify_source_snapshot(json.loads(verify_file_pin(report['source_receipt']).read_text()),live=True)
    inputs={k:file_pin(paths[k]) for k in ['source_report','retry_report','protocol']};inputs.update(parent=prior['inputs']['parent'],host_model=prior['inputs']['host_model'])
    if len(prior['scenes'])!=44 or len(retry['variants'])!=2:raise ValueError('Pinned inputs required')
    backends=dict(zip(['CONTROL','AXIS_RETRY'],retry['variants']))
    for name,pin in backends.items():
        expected=paths['retry_report'].parent/'isolated_sources'/(name+'.py')
        if Path(pin['path'])!=expected:raise ValueError('Exact backend artifact path required')
        verify_file_pin(pin)
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/probe_retained_pen_collision.py',ROOT/'tools/trace_pen_epa_degenerate_exit.py']
    def freeze(out):
        receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='multifinger_CCD_validation_registration:1',config=file_pin(cp),inputs=inputs,backends=backends,source_receipt=file_pin(out/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=json.loads(paths['registration'].read_text())
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['backends']!=backends:raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()),live=True)
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh validation output required')
    out.mkdir();receipt=freeze(out);attempted=completed=cpu_attempted=cpu_completed=0;rows=[]
    try:
        modules={n:imported('isolated_axis_study_'+n.lower(),verify_file_pin(p)) for n,p in backends.items()}
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent=torch.load(verify_file_pin(inputs['parent']),map_location='cpu',weights_only=False);host=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model,data,construction=build_fixture(host,parent['native']['native']);write(out/'construction.json',construction)
            ma=parent['native']['native']['model']['arrays'];pairs=ma['nxn_geom_pair_filtered'];pairids=ma['nxn_pairid_filtered'];gt=ma['geom_type']
            selected=[i for i,(g,h) in enumerate(pairs) if (int(gt[g]),int(gt[h]))==(5,7)]
            if len(selected)!=11:raise ValueError('Exactly all11 filtered ordered pairs required')
            write(out/'pairs.json',[dict(index=i,geoms=pairs[i].tolist(),pairids=pairids[i].tolist()) for i in selected])
            native=mujoco.MjData(host)
            if not np.array_equal(host.mesh_vert,ma['mesh_vert']) or not np.array_equal(host.geom_dataid,ma['geom_dataid'][31%ma['geom_dataid'].shape[0]]):raise ValueError('Oracle mesh mismatch')
            host.geom_size[:]=ma['geom_size'][31%ma['geom_size'].shape[0]].astype(np.float64);host.opt.ccd_iterations=50;host.opt.ccd_tolerance=float(ma['opt.ccd_tolerance'][0])
            factory={n:'ccd_kernel_builder_axis_study_'+n.lower() for n in modules};idx=int(upper_trid_index(len(types.GeomType),5,7));args=(5,7,50,50,True,idx,model.block_dim.convex_ccd,bool(model.opt.warn_overflow))
            kernels={n:getattr(modules[n],factory[n])(*args) for n in modules}
            if len({id(v) for v in kernels.values()})!=2:raise ValueError('Backend kernel alias')
            write(out/'kernel_identities.json',[dict(variant=n,kernel_id=id(k),factory=factory[n],module=k.module.name,key=k.key) for n,k in kernels.items()])
            ctx=collision_core.create_collision_context(data.naconmax);ctx.collision_worldid.fill_(31)
            pairbuf=np.zeros((data.naconmax,2),np.int32);idbuf=np.full((data.naconmax,2),-1,np.int32)
            for scene in prior['scenes']:
                f=np.load(verify_file_pin(scene['geometry']));data.geom_xpos.assign(f['geom_xpos']);data.geom_xmat.assign(f['geom_xmat'])
                native.geom_xpos[:]=f['geom_xpos'][31].astype(np.float64);native.geom_xmat[:]=f['geom_xmat'][31].reshape(host.ngeom,9).astype(np.float64)
                for index in selected:
                    g,h=map(int,pairs[index]);witness=np.full(6,np.nan);cpu_attempted+=1
                    distance=float(mujoco.mj_geomDistance(host,native,g,h,1.,witness));cpu_completed+=1
                    pairbuf[0]=[g,h];idbuf[0]=pairids[index];ctx.collision_pair.assign(pairbuf);ctx.collision_pairid.assign(idbuf)
                    scope='OVERLAP' if distance < -cfg['boundary_m'] else 'SEPARATED' if distance>cfg['boundary_m'] else 'BOUNDARY'
                    for name in ['CONTROL','AXIS_RETRY']:
                        print('QUERY_BEGIN '+scene['name']+f' pair{g}_{h} '+name,flush=True)
                        data.nacon.zero_();data.ncollision.fill_(1);data.overflow.zero_();attempted+=1
                        modules[name].convex_narrowphase(model,data,ctx,[(types.GeomType.CYLINDER,types.GeomType.MESH)]);wp.synchronize();completed+=1
                        if getattr(modules[name],factory[name])(*args) is not kernels[name]:raise ValueError('Kernel identity changed')
                        count=int(data.nacon.numpy()[0]);mask=data.overflow.numpy().tolist();contact={k:getattr(data.contact,k).numpy()[:count].copy() for k in ['worldid','geom','dist','pos','frame','dim']}
                        finite=bool(np.isfinite(distance) and np.isfinite(witness).all() and all(np.isfinite(v).all() for v in contact.values()))
                        metrics={}
                        if count==1 and abs(distance)>1e-6:
                            normal=contact['frame'][0,0].astype(np.float64);reference=(witness[3:]-witness[:3])/distance
                            cosine=np.dot(normal,reference)/(np.linalg.norm(normal)*np.linalg.norm(reference))
                            metrics=dict(distance_error_m=float(abs(float(contact['dist'][0])-distance)),normal_angle_deg=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
                                midpoint_error_m=float(np.linalg.norm(contact['pos'][0].astype(np.float64)-.5*(witness[:3]+witness[3:]))),
                                frame_orthogonality=float(np.max(np.abs(contact['frame'][0]@contact['frame'][0].T-np.eye(3)))))
                        checks=dict(no_overflow=not any(mask),finite=finite,one_contact_max=count<=1,
                            overlap_has_contact=scope!='OVERLAP' or count==1,separated_no_contact=scope!='SEPARATED' or count==0,
                            correct_contact_identity=count==0 or (contact['worldid'].tolist()==[31] and contact['geom'].tolist()==[[g,h]]),
                            distance=scope!='OVERLAP' or (count==1 and metrics.get('distance_error_m',float('inf'))<=cfg['distance_error_m']),
                            normal=scope!='OVERLAP' or (count==1 and metrics.get('normal_angle_deg',float('inf'))<=cfg['normal_angle_deg']),
                            midpoint=scope!='OVERLAP' or (count==1 and metrics.get('midpoint_error_m',float('inf'))<=cfg['midpoint_error_m']),
                            frame=count==0 or metrics.get('frame_orthogonality',0.)<=1e-4)
                        path=out/(scene['name']+f'_pair{g}_{h}_'+name+'.npz');np.savez_compressed(path,native_witness=witness,overflow=np.asarray(mask,np.int32),**contact)
                        row=dict(scene=scene['name'],geoms=[g,h],variant=name,native_distance_m=distance,boundary_scope=scope,raw_mask=mask,nacon=count,metrics=metrics,checks=checks,passed=sum(checks.values()),total=len(checks),artifact=file_pin(path))
                        rows.append(row);write(out/'rows.json',rows);print('QUERY_END '+json.dumps(dict(scene=scene['name'],pair=[g,h],variant=name,passed=row['passed'],total=row['total'])),flush=True)
            if (completed,cpu_completed)!=(968,484):raise ValueError('Fixed validation query accounting mismatch')
        verify_source_snapshot(receipt,live=True)
        for pin in backends.values():verify_file_pin(pin)
        report=dict(schema='pinned_axis_retry_multifinger_validation:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,backends=backends,source_receipt=file_pin(out/'source_receipt.json'),
            construction=file_pin(out/'construction.json'),kernel_identities=file_pin(out/'kernel_identities.json'),pairs=file_pin(out/'pairs.json'),results=file_pin(out/'rows.json'),
            attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,attempted_native_distance_queries=cpu_attempted,actual_native_distance_queries=cpu_completed,
            actual_native_integrations=0,optimizer_steps=0,FK_builds=0,wall_s=time.perf_counter()-started,
            scope='All11 filtered cylinder/mesh pairs on44 derived scenes, fixed numerical/witness criteria. Not independent ego data or full dynamic/backend/state/gradient/training admission;extra retry attempts require console review.')
        write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),GPU=completed,CPU=cpu_completed)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),attempted_GPU_pair_queries=attempted,actual_GPU_pair_queries=completed,attempted_native_distance_queries=cpu_attempted,
            actual_native_distance_queries=cpu_completed,actual_native_integrations=0,optimizer_steps=0,source_receipt=file_pin(out/'source_receipt.json')));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--register',action='store_true');a=p.parse_args()
    if not a.register and a.output is None:p.error('--output required')
    main(a)
