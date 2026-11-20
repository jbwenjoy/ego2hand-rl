"""Independently compare endpoint-momentum labels to per-contact loads on held-out branches."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from compare_hot3d_mug_wrench import physics,file_pin,verify_file_pin,write_json,MugContactImpulse


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    pin=file_pin(root/'mug_feasible_paths_v2/report.json');bank=physics.read_json(pin)
    cfg=physics.read_json(physics.read_json(bank['intent'])['source']);inputs=cfg['inputs']
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    sensor=MugContactImpulse(m,physics.read_json(inputs['binding']));d=mujoco.MjData(m)
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(bank=pin,code=file_pin(__file__),scope='Seed8312 width1. All control states, slots 0, 15, 31 and 63 fixed before direct force extraction. Compare endpoint momentum to mean world-axis hand contact wrench at moving COM.'))
    rows=[]
    for row in bank['rows']:
        if row['seed']!=8312 or row['width']!=1:continue
        with np.load(verify_file_pin(row['bank'])) as f:z=dict(f)
        for t in range(len(z['source'])):
            for a in (0,15,31,63):
                mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,z['source'][t,a],sig)
                d.qacc_warmstart[:]=z['warmstart'][t,a];d.ctrl[:]=z['cmd'][t,a]
                total=np.zeros(6);peak=0.;err=0.
                for _ in range(64):
                    mujoco.mj_step(m,d)
                    w,table,e=sensor.hand_wrench(m,d,verify=True)
                    total+=w;peak=max(peak,table);err=max(err,e)
                exact=np.empty_like(z['end'][t,a]);mujoco.mj_getState(m,d,exact,sig)
                np.testing.assert_array_equal(exact,z['end'][t,a])
                rows.append(dict(clip=row['clip'],phase=t+1,id=a,table=peak,contact_decoder_error=err,
                    momentum=z['net_wrench'][t,a].tolist(),mean_hand=(total/64).tolist()))
    diff=np.array([np.array(r['momentum'])-r['mean_hand'] for r in rows])
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,mean_force_vector_error_N=np.linalg.norm(diff[:,:3],axis=1).mean(),
        mean_torque_vector_error_Nm=np.linalg.norm(diff[:,3:],axis=1).mean(),max_table_N=max(r['table'] for r in rows),
        max_decoder_error=max(r['contact_decoder_error'] for r in rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
