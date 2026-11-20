"""Re-examine from-start mug sequences for solver geometry, hardware demand and parameter sensitivity."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from collect_mug_handoff_labels import h3


def geometry(model_path,binding,z,base,cmds,expected):
    m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m)
    d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base;mujoco.mj_forward(m,d)
    hand=np.isin(np.arange(m.ngeom),[m.geom(n).id for n in binding['hand_collision_names']])
    mug=m.geom_bodyid==binding['mug_body_id'];j=m.actuator_trnid[:,0];lo,hi=m.jnt_range[j].T
    assert np.all(m.jnt_limited[j]) and not m.neq and not m.body_gravcomp.any()
    rows=[];local=np.empty(6)
    for i,cmd in enumerate(cmds):
        d.ctrl[:]=cmd;peaks=np.zeros(6)
        assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
        for _ in range(64):
            mujoco.mj_step(m,d);a,b=d.contact.geom.T
            fingers=(hand[a]&mug[b])|(hand[b]&mug[a])
            self_contact=hand[a]&hand[b]
            depth=-1000*np.asarray(d.contact.dist);q=d.qpos[m.jnt_qposadr[j]]
            normal=0.
            for ic in np.flatnonzero(fingers):
                mujoco.mj_contactForce(m,d,ic,local);normal+=max(0.,local[0])
            metrics=[depth[fingers].max(initial=0.),depth[self_contact].max(initial=0.),
                     max(0.,(lo-q).max(),(q-hi).max()),normal,abs(d.qvel[m.jnt_dofadr[j]]).max(),
                     abs(d.actuator_force).max()]
            peaks=np.maximum(peaks,metrics)
        np.testing.assert_array_equal(d.qpos,expected[i])
        rows.append(peaks)
    return np.asarray(rows)


def run(search,out):
    root=Path(__file__).resolve().parents[1]/'runs'
    source=h3.file_pin(search/'report.json');opt=h3.physics.read_json(source)
    original=h3.physics.read_json(h3.physics.read_json(opt['intent'])['source'])
    basecfg=h3.physics.read_json(h3.file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    path=h3.verify_file_pin(basecfg['model']);binding=h3.physics.read_json(basecfg['binding']);old=h3.physics.load_arrays(basecfg['reference'])
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=source,source_asset=basecfg['model'],code=h3.file_pin(__file__),
        columns=['max_hand_mug_mm','max_hand_self_mm','max_joint_excursion_rad','peak_total_hand_normal_N','peak_hand_velocity_rad_s','peak_actuator_force'],
        scope='Recompute every nominal native contact and endpoint, with extra diagnostic limits 2mm pen-style hand/object, 0.2mm hand/self, 0.001rad joint excursion. These are additional diagnostics, not fit criteria or measured hardware limits. Fixed commands also tested with 1.15x mass/inertia and 0.8x hand/mug friction, together and separately.'))
    rows=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=h3.physics.read_json(h3.file_pin(root/name/'report.json'))['rows'];clip=item['clip']
        ref=h3.physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=old['feedforward_PD_target20'][0];ik=ref['q'][1:]+base-old['initial_qpos27'][:20]
        p=next(r for r in original['rows'] if r['clip']==clip and r['mode']=='baseline' and r['seed']==8441)
        prior=h3.physics.load_arrays(p['trace']);m=mujoco.MjModel.from_binary_path(str(path))
        nominal=np.clip(ik,*m.actuator_ctrlrange.T);nominal[:len(prior['ctrl'])]=prior['ctrl']
        arms={'start':nominal}
        for r in opt['rows']:
            if r['clip']==clip:arms[str(r['seed'])]=h3.physics.load_arrays(r['best'])['commands']
        for arm,cmds in arms.items():
            for label,mass,mu in [('nominal',1.,1.),('heavy',1.15,1.),('low_mu',1.,.8),('heavy_low',1.15,.8)]:
                detail,tr=h3.physics.replay(path,binding,z,base,cmds,mass,mu)
                case=dict(clip=clip,arm=arm,condition=label,valid_s=detail['valid_s'],full=detail['right_censored'],
                          first_failure=detail['first_breach'],solver_mu=detail['active_solver_mu'])
                if label=='nominal':
                    audit=geometry(path,binding,z,base,cmds,tr['qpos'])
                    passed=int(round(detail['valid_s']/.02))
                    over=np.flatnonzero((audit[:,:3]>[2.,.2,.001]).any(1))
                    case.update(passing_with_geometry_s=.02*min(passed,int(over[0]) if len(over) else len(cmds)),
                        maxima_over_pose_passing=audit[:passed].max(0).tolist() if passed else None)
                    file=out/f'{clip}_{arm}.npz';np.savez_compressed(file,geometry=audit,**tr);case['trace']=h3.file_pin(file)
                rows.append(case)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('search',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.search.resolve(),a.out.resolve())
