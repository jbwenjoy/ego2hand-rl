"""Expose native broadphase pairs and separate installed GJK/EPA phase outputs."""
import argparse
import gc
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write
from supermimic_research.runtime.saved_native_plant import saved_native_plant
from supermimic_research.runtime.saved_dynamic_arrays import restore_dynamic_arrays
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.contact_phase_probe import ContactPhaseProbe


def main(a):
    import mujoco_warp as mw
    import warp as wp
    from mujoco_warp._src import collision_driver as cd
    from mujoco_warp._src.collision_core import create_collision_context
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['branch']).resolve();branch=json.loads(bp.read_text())
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or cfg['pair']!=[1,29] or cfg['poses']!=[p['name'] for p in branch['poses']]
            or cfg.get('calibration_pairs')!=['BOX_BOX','CYLINDER_MESH_BOX'] or cfg['repeats']!=5 or cfg['phase_scratch']!='ZERO_BEFORE_EACH_QUERY' or cfg['calibration_center_separations_m']!=[.5,.15,.05]
            or cfg['calibration_box_halfsize_m']!=.1 or cfg['calibration_distance_tolerance_m']!=1e-5):raise ValueError('Fixed pair-stage diagnostic and calibrations')
    wp.init();wp.set_device('cuda:0');stream=wp.Stream(device='cuda:0');a.output.mkdir()
    receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__)],a.output/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(a.output/'source_receipt.json',receipt)
    package=Path(mw.__file__).resolve().parent
    installed=freeze_sources([package/'_src'/name for name in ['collision_core.py','collision_driver.py','collision_gjk.py','collision_convex.py','smooth.py','io.py','warp_util.py','types.py','math.py']],
        a.output/'installed_source_snapshot',package,'installed-package-source-not-git');write(a.output/'installed_source_receipt.json',installed)
    calibrations=[];keepers=[]
    with wp.ScopedStream(stream):
        for fixture,separation in [(f,s) for f in cfg['calibration_pairs'] for s in cfg['calibration_center_separations_m']]:
            vertices=' '.join(str(v) for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1] for v in [x,y,z])
            asset=f'<asset><mesh name="box" vertex="{vertices}"/></asset>' if fixture=='CYLINDER_MESH_BOX' else ''
            first='type="cylinder" size=".1 .1"' if fixture=='CYLINDER_MESH_BOX' else 'type="box" size=".1 .1 .1"'
            second='type="mesh" mesh="box"' if fixture=='CYLINDER_MESH_BOX' else 'type="box" size=".1 .1 .1"'
            m=mujoco.MjModel.from_xml_string(f'<mujoco><option ccd_iterations="50"/>{asset}<worldbody><body><freejoint/><geom {first}/></body><body pos="{separation} 0 0"><freejoint/><geom {second}/></body></worldbody></mujoco>')
            dm=mw.put_model(m);data=mw.make_data(m,nworld=1,naconmax=32,njmax=128);data.qpos.assign(m.qpos0.astype(np.float32)[None])
            mw.kinematics(dm,data);mw.collision(dm,data);wp.synchronize_stream(stream)
            probe=ContactPhaseProbe(dm,data);result,packet=probe.query(0,1);wp.synchronize_stream(stream)
            contact_count=int(data.nacon.numpy()[0]);present=separation<.2
            checks=dict(native_contact_presence=(contact_count>0)==present,phase_contact_presence=result['pre_multiccd_emit']==present,
                no_overflow=not result['overflow'] and not bool(data.overflow.numpy()[0]),
                analytic_penetration=(not present) or abs(result['final_surface_distance_m']-(separation-.2))<=1e-5)
            path=a.output/f'calibration_{fixture}_{separation:g}.npz';np.savez_compressed(path,**packet)
            calibrations.append(dict(fixture=fixture,separation_m=separation,expected_signed_distance_m=separation-.2,native_contacts=contact_count,phase=result,checks=checks,trace=file_pin(path)))
            keepers.append((dm,data,probe))
        write(a.output/'calibration.json',dict(cases=calibrations))
        if not all(all(c['checks'].values()) for c in calibrations):raise ValueError('Analytic wrapper calibration failed; source queries not run')
    replay=json.loads(verify_file_pin(branch['replay']).read_text());original=json.loads(verify_file_pin(replay['original_run']).read_text())
    c=next(c for c in original['cases'] if c['name']=='STATE_PLUS_VELOCITY_SUPPORT' and c['dt']==.005)
    source=dict(np.load(verify_file_pin(c['planning'][9]['source'])))
    with wp.ScopedStream(stream):
        m,dm,data,reconstruction=saved_native_plant(c,source)
        if int(dm.opt.broadphase)!=0 or m.nflex or dm.callback.contactfilter is not None or dm.opt.enableflags or m.npair:
            raise ValueError('Exact original NXN, no flex/sleep/callback/explicit-pair scene')
        ctx=create_collision_context(data.naconmax);probe=ContactPhaseProbe(dm,data)
        was_gc=gc.isenabled();gc.disable()
        try:
            with wp.ScopedCapture(stream=stream) as capture:
                mw.kinematics(dm,data);data.ncollision.zero_();data.nacon.zero_()
                cd.nxn_broadphase(dm,data,ctx);cd._narrowphase(dm,data,ctx)
        finally:
            if was_gc:gc.enable()
    _,before_scalars,before_hashes=capture_model(dm);records=[];g,h=cfg['pair']
    for pose in branch['poses']:
        folder=a.output/pose['name'];folder.mkdir();p=np.load(verify_file_pin(pose['input_values']))
        with wp.ScopedStream(stream):
            restore_dynamic_arrays(data,source);data.qpos.assign(p['qpos'].astype(np.float32)[None]);data.qvel.assign(p['qvel'].astype(np.float32)[None]);initial=ResidentDataSnapshot(data)
            for repeat in range(5):
                initial.restore(data);before=initial.audit(data);wp.capture_launch(capture.graph,stream=stream);wp.synchronize_stream(stream)
                ncollision,nacon=int(data.ncollision.numpy()[0]),int(data.nacon.numpy()[0])
                if ncollision>data.naconmax or nacon>data.naconmax:raise ValueError('Capacity overflow in stage query')
                pairs=ctx.collision_pair.numpy()[:ncollision];pairids=ctx.collision_pairid.numpy()[:ncollision];worldids=ctx.collision_worldid.numpy()[:ncollision]
                ids=np.flatnonzero(np.all(np.sort(pairs,axis=1)==[g,h],axis=1))
                contacts=data.contact.geom.numpy()[:nacon];distances=data.contact.dist.numpy()[:nacon]
                contact_ids=np.flatnonzero(np.all(np.sort(contacts,axis=1)==[g,h],axis=1))
                if any(pairids[i,1]>=0 for i in ids):raise ValueError('Collision-sensor stage needs a different declared cutoff')
                # A culled pair is evidence, not an abort: direct geometry query
                # then bypasses broadphase explicitly, under this no-explicit-pair scene.
                order=pairs[ids[0]] if len(ids) else np.array([g,h],dtype=np.int32)
                pairid=pairids[ids[0]] if len(ids) else np.array([-1,-1],dtype=np.int32)
                phase,packet=probe.query(int(order[0]),int(order[1]),int(pairid[0]));wp.synchronize_stream(stream)
                path=folder/f'stages_R{repeat}.npz';np.savez_compressed(path,**packet,broadphase_pairs=pairs,broadphase_pairids=pairids,broadphase_worldids=worldids,
                    contact_geoms=contacts,contact_distances=distances,geom_xpos=data.geom_xpos.numpy()[0,[g,h]],geom_xmat=data.geom_xmat.numpy()[0,[g,h]])
                old=next(e for e in branch['records'] if e['pose']==pose['name'] and e['repeat']==repeat)
                checks=dict(public_source_exact=before['all_array_bytes_equal'] and before['scalar_equal'],
                    original_contact_presence=(len(contact_ids)>0)==(old['pair_contacts']>0),no_overflow=not phase['overflow'] and not bool(data.overflow.numpy()[0]),
                    staged_distances_exact=np.array_equal(np.sort(distances[contact_ids]),np.sort(np.asarray(old['pair_distances_m'],dtype=np.float32))))
                records.append(dict(pose=pose['name'],repeat=repeat,ncollision=ncollision,nacon=nacon,broadphase_pair_rows=ids.tolist(),
                    broadphase_pair_order=order.astype(int).tolist(),broadphase_pairids=pairid.astype(int).tolist(),direct_query_bypasses_culled_pair=not len(ids),
                    pair_contacts=len(contact_ids),pair_distances_m=distances[contact_ids].astype(float).tolist(),phase=phase,checks=checks,trace=file_pin(path)))
                write(a.output/'progress.json',dict(calibrations=calibrations,records=records))
    _,after_scalars,after_hashes=capture_model(dm);model_unchanged=before_scalars==after_scalars and before_hashes==after_hashes
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(installed,live=True)
    result=dict(schema='pen_contact_stage_probe:1',config=file_pin(cp),branch=file_pin(bp),source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),
        calibration=file_pin(a.output/'calibration.json'),reconstruction=reconstruction,model_unchanged=model_unchanged,
        original_model_arrays=c['fixed_model_arrays'],original_model_scalars=c['fixed_model_scalars'],poses=branch['poses'],records=records,
        contracts=[sum(sum(e['checks'].values()) for e in records),sum(len(e['checks']) for e in records)],native_integrations=0,neural_updates=0,stage_queries=len(records),
        scope='Original recorded inputs, unchanged native model; exposed native broad/narrowphase and independent installed GJK/EPA wrapper with separate zeroed scratch. Compare native outputs before causal interpretation; no constraint solve/integration, no original force/iteration labels invented, no measured ego/hardware claim.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=result['contracts'],model_unchanged=model_unchanged,
        records=[dict(pose=e['pose'],repeat=e['repeat'],broadphase=e['broadphase_pair_rows'],contact=e['pair_contacts'],phase=e['phase']) for e in records])))
    if not model_unchanged or result['contracts'][0]!=result['contracts'][1]:raise ValueError('Retain stage/native-output mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
