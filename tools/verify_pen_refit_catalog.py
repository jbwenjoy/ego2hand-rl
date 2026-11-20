"""Refit only pen collision bounds on a fresh copy, then repeat previous catalog and holds."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from screen_pen_reference_catalog import main as screen,ROOT
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())


def run(out):
    from supermimic_research.mechanics.pen_body_bvh import copy_refit_pen_bvh
    out.mkdir(exist_ok=False);runs=ROOT/'runs'
    old=file_pin(runs/'pen_catalog_screen_v2/report.json');gpu=file_pin(runs/'pen_gpu_catalog_v1/report.json')
    old_ref=read(old);gpu_ref=read(gpu)
    c=read(read(old_ref['intent'])['case']);body=c['model_layout']['object_body']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
    write_json(out/'intent.json',dict(source=c['model'],cpu_catalog=old,gpu_catalog=gpu,scope='Unchanged physical parameters and servo targets, replace only derived object BVH boxes on a fresh copy. Recheck all clips and identical fixed-target holds.'))
    corrected,proof=copy_refit_pen_bvh(model,body)
    new_model=out/'refit_pen.mjb';mujoco.mj_saveModel(corrected,str(new_model));fresh=mujoco.MjModel.from_binary_path(str(new_model))
    np.testing.assert_array_equal(fresh.bvh_aabb,corrected.bvh_aabb)
    write_json(out/'refit.json',dict(source=c['model'],model=file_pin(new_model),proof=proof))
    screen(out/'catalog',new_model)
    new=read(file_pin(out/'catalog/report.json'));cat={}
    for k,n in new['results'].items():
        o=old_ref['results'][k];g=gpu_ref['results'][k]
        with np.load(verify_file_pin(n['series'])) as a,np.load(verify_file_pin(o['series'])) as b,np.load(verify_file_pin(g['series'])) as target:
            cat[k]=dict(frames=n['frames'],geometry_mismatches_before=int(np.sum(b['metrics'][:,5]!=target['metrics'][:,4])),geometry_mismatches_after=int(np.sum(a['metrics'][:,5]!=target['metrics'][:,4])),
                geometry_pass=n['geometry_pass_frames'],feasible=n['geometry_and_cone'],gpu_feasible=g['both'],unresolved=len(n['unresolved']))
    pr=read(file_pin(runs/'pen_dev_projection_v1/report.json'));controls=read(file_pin(runs/'pen_dev_static_preload_v1/report.json'));gpu_hold=read(file_pin(runs/'pen_dev_gpu_hold_v1/report.json'));holds=[]
    with np.load(verify_file_pin(c['trace'])) as f:mp=f['initial_mocap_pos'][0];mq=f['initial_mocap_quat'][0]
    qids=c['qids'];qa=c['model_layout']['object_qposadr']
    for row in controls['rows']:
        if 'holds' not in row:continue
        clip=row['clip'];v=next(a for a in pr['rows'] if a['clip']==clip)
        with np.load(verify_file_pin(v['variants']['projection']['trace'])) as f:q=f['initial_hand_q'];pos=f['object_pos'];quat=f['object_quat']
        for arm,p in row['holds'].items():
            with np.load(verify_file_pin(p['trace'])) as f:ctrl=f['ctrl']
            d=mujoco.MjData(fresh);d.mocap_pos[:]=mp;d.mocap_quat[:]=mq;d.qpos[qids]=q;d.qpos[qa:qa+7]=np.r_[pos,quat];d.ctrl[c['aids']]=ctrl;mujoco.mj_forward(fresh,d)
            traj=[]
            for i in range(80):
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                mujoco.mj_step(fresh,d);traj.append(d.qpos.copy())
            traj=np.array(traj);R=Rotation.from_quat(traj[:,qa+np.array([4,5,6,3])])
            dest=out/f'{clip}_{arm}.npz';np.savez_compressed(dest,qpos=traj)
            holds.append(dict(clip=clip,arm=arm,old_mm=p['max_mm'],old_deg=p['max_deg'],corrected_mm=1000*float(np.linalg.norm(traj[:,qa:qa+3]-pos,axis=1).max()),
                corrected_deg=float(np.rad2deg((R*Rotation.from_quat(quat[[1,2,3,0]]).inv()).magnitude()).max()),
                gpu_mm=[r['max_mm'] for r in gpu_hold['rows'] if r['clip']==clip and r['arm']==arm],trace=file_pin(dest)))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),refit=file_pin(out/'refit.json'),catalog=file_pin(out/'catalog/report.json'),rows=cat,holds=holds))
    print('mismatches',sum(x['geometry_mismatches_before'] for x in cat.values()),sum(x['geometry_mismatches_after'] for x in cat.values()),'hold',[(x['clip'],x['arm'],x['corrected_mm'],x['corrected_deg']) for x in holds])
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
