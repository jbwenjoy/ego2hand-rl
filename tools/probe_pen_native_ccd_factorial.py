"""Native-only2x2 CCD numerics; geometry/material/dynamics parameters unchanged."""
import argparse
import gc
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write
from supermimic_research.runtime.saved_native_plant import saved_native_plant
from supermimic_research.runtime.saved_dynamic_arrays import restore_dynamic_arrays
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot


def main(a):
    import mujoco_warp as mw
    import warp as wp
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['branch']).resolve();branch=json.loads(bp.read_text())
    if (a.output.exists() or cfg['ccd_iterations']!=[50,100] or cfg['ccd_tolerances']!=[1e-6,1e-7] or cfg['pair']!=[1,29]
            or cfg['poses']!=[p['name'] for p in branch['poses']] or cfg['repeats']!=5 or cfg['pipeline']!='UNMODIFIED_NATIVE_KINEMATICS_COLLISION_ONLY'):
        raise ValueError('Fixed native-only CCD numerical factorial')
    a.output.mkdir();wp.init();wp.set_device('cuda:0');stream=wp.Stream(device='cuda:0')
    receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__)],a.output/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(a.output/'source_receipt.json',receipt)
    package=Path(mw.__file__).resolve().parent
    installed=freeze_sources([package/'_src'/p for p in ['collision_core.py','collision_driver.py','collision_gjk.py','collision_convex.py','smooth.py','io.py','warp_util.py','types.py','math.py']],a.output/'installed_source_snapshot',package,'installed-package-source-not-git')
    write(a.output/'installed_source_receipt.json',installed)
    replay=json.loads(verify_file_pin(branch['replay']).read_text());original=json.loads(verify_file_pin(replay['original_run']).read_text())
    c=next(c for c in original['cases'] if c['name']=='STATE_PLUS_VELOCITY_SUPPORT' and c['dt']==.005)
    source=dict(np.load(verify_file_pin(c['planning'][9]['source'])));g,h=cfg['pair'];cells=[];keepers=[]
    for iterations in cfg['ccd_iterations']:
        for tolerance in cfg['ccd_tolerances']:
            folder=a.output/f'I{iterations}_T{tolerance:g}';folder.mkdir()
            with wp.ScopedStream(stream):
                m,dm,data,reconstruction=saved_native_plant(c,source)
                before,old_scalars,old_hashes=capture_model(dm)
                dm.opt.ccd_iterations=iterations;dm.opt.ccd_tolerance.assign(np.full_like(before['opt.ccd_tolerance'],tolerance))
                arrays,scalars,hashes=capture_model(dm)
                changed_arrays=sorted(k for k in old_hashes if old_hashes[k]!=hashes[k]);changed_scalars=sorted(k for k in old_scalars if old_scalars[k]!=scalars[k])
                if set(changed_arrays)-{'opt.ccd_tolerance'} or set(changed_scalars)-{'opt.ccd_iterations'}:raise ValueError('Unexpected physical/numerical model change')
                np.savez_compressed(folder/'fixed_model_arrays.npz',**arrays);write(folder/'fixed_model_scalars.json',scalars)
                was_gc=gc.isenabled();gc.disable()
                try:
                    with wp.ScopedCapture(stream=stream) as capture:
                        mw.kinematics(dm,data);mw.collision(dm,data)
                finally:
                    if was_gc:gc.enable()
            keepers.append((m,dm,data,capture.graph));records=[]
            for pose in branch['poses']:
                p=np.load(verify_file_pin(pose['input_values']))
                with wp.ScopedStream(stream):
                    restore_dynamic_arrays(data,source);data.qpos.assign(p['qpos'].astype(np.float32)[None]);data.qvel.assign(p['qvel'].astype(np.float32)[None]);initial=ResidentDataSnapshot(data)
                    for repeat in range(5):
                        initial.restore(data);audit=initial.audit(data);wp.capture_launch(capture.graph,stream=stream);wp.synchronize_stream(stream)
                        nacon,ncollision=int(data.nacon.numpy()[0]),int(data.ncollision.numpy()[0])
                        if nacon>data.naconmax or ncollision>data.naconmax:raise ValueError('Collision capacity overflow')
                        geoms=data.contact.geom.numpy()[:nacon];dist=data.contact.dist.numpy()[:nacon]
                        mask=np.all(np.sort(geoms,axis=1)==[g,h],axis=1);old=next(e for e in branch['records'] if e['pose']==pose['name'] and e['repeat']==repeat)
                        path=folder/f"{pose['name']}_R{repeat}.npz";np.savez_compressed(path,geom=geoms,dist=dist,pos=data.contact.pos.numpy()[:nacon],frame=data.contact.frame.numpy()[:nacon])
                        baseline=iterations==50 and tolerance==1e-6
                        checks=dict(exact_public_restore=audit['all_array_bytes_equal'] and audit['scalar_equal'],capacity=nacon<=data.naconmax and ncollision<=data.naconmax,
                            no_overflow=not bool(data.overflow.numpy()[0]),
                            baseline_matches_original=(not baseline) or (int(mask.sum())==old['pair_contacts'] and np.array_equal(np.sort(dist[mask]),np.sort(np.array(old['pair_distances_m'],np.float32)))))
                        records.append(dict(pose=pose['name'],repeat=repeat,nacon=nacon,ncollision=ncollision,pair_contacts=int(mask.sum()),pair_distances_m=dist[mask].astype(float).tolist(),checks=checks,trace=file_pin(path)))
            _,after_scalars,after_hashes=capture_model(dm)
            cell=dict(iterations=iterations,requested_tolerance=tolerance,observed_tolerance=float(arrays['opt.ccd_tolerance'][0]),
                before_tolerance=float(before['opt.ccd_tolerance'][0]),before_iterations=int(old_scalars['opt.ccd_iterations']),
                changed_arrays=changed_arrays,changed_scalars=changed_scalars,model_unchanged_after_queries=after_scalars==scalars and after_hashes==hashes,
                fixed_model_arrays=file_pin(folder/'fixed_model_arrays.npz'),fixed_model_scalars=file_pin(folder/'fixed_model_scalars.json'),reconstruction=reconstruction,records=records)
            cells.append(cell);write(a.output/'progress.json',dict(cells=cells));print(json.dumps(dict(iterations=iterations,tolerance=tolerance,
                results={p['name']:[r['pair_contacts'] for r in records if r['pose']==p['name']] for p in branch['poses']})),flush=True)
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(installed,live=True)
    result=dict(schema='pen_native_ccd_numerical_factorial:1',config=file_pin(cp),branch=file_pin(bp),source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),
        original_model_arrays=c['fixed_model_arrays'],original_model_scalars=c['fixed_model_scalars'],cells=cells,native_collision_queries=sum(len(c['records']) for c in cells),native_integrations=0,neural_updates=0,
        scope='Native-only numerical sensitivity, original geometry/material/mass/inertia/solver settings unchanged. Only declared CCD iteration/tolerance fields vary. Baseline exact native-output gate retained. Four input labels/two pose images, not independent trials or complete dynamics/gradient/task/ego qualification.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),queries=result['native_collision_queries'])))
    if not all(c['model_unchanged_after_queries'] and all(all(r['checks'].values()) for r in c['records']) for c in cells):raise ValueError('Retain factorial integrity failure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
