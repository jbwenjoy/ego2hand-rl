"""Register the annotated hand and mug together with one yaw and translation."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from retarget_adjacent_mugs import physics, file_pin, verify_file_pin, write_json, read_hand_window
from supermimic_research.ego.hand_frames import palm_frame
from supermimic_research.ego.trajectory_clock import resample_body_poses
from supermimic_research.mechanics.retargeting import HandTipIK


def tilt(rotations):
    return np.degrees(np.arccos(np.clip(rotations[..., 2, 1], -1, 1)))


def main(out):
    root = Path(__file__).resolve().parents[1] / 'runs'
    inputs = physics.read_json(file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    bind = physics.read_json(inputs['binding'])
    old = physics.load_arrays(inputs['reference'])
    with np.load(verify_file_pin(bind['reference'])) as f:
        robot_palm = f['T_robot_world_from_robot_palm']
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json', dict(model=inputs['model'], binding=inputs['binding'], code=file_pin(__file__),
        scope='Conditional on annotation-world +Z being up. Align anatomical palm origins at frame zero and fit yaw only. Apply the same constant rigid transform to hand and mug for the entire original clock; no fingertip offsets. Baseline IK and unassisted first-control test from rest.'))
    results = []
    for clip, group in [(1884, 'mug_annotated_windows_original_v1'), (1888, 'mug_unilateral_interval_v1')]:
        source = file_pin(root/group/'report.json')
        item, = physics.read_json(source)['rows']
        f = physics.load_arrays(item['reference'])
        data = read_hand_window(verify_file_pin(item['archive']), list(map(int, f['frames'])), 'right', 9, item['window']['object_uid'])
        anchors = np.asarray(data['profile']['landmark_rest_positions'])[[8,11,14,17]]*.001*[-1,1,1]
        palm = f['source_wrist'] @ palm_frame(np.zeros(3), anchors)
        relative = robot_palm[:3,:3] @ palm[0,:3,:3].T
        yaw = np.arctan2(relative[1,0]-relative[0,1],relative[0,0]+relative[1,1])
        reg = np.eye(4)
        reg[:3,:3] = Rotation.from_euler('z',yaw).as_matrix()
        reg[:3,3] = robot_palm[:3,3] - reg[:3,:3]@palm[0,:3,3]
        body = reg @ f['source_obj']
        sampled = resample_body_poses(f['source_time'],body,old['COM_BODY_m'])
        ix = sampled['source_left_index']; a = sampled['source_interval_alpha'][:,None,None]
        tip_local = (1-a)*f['source_tip_in_body'][ix]+a*f['source_tip_in_body'][ix+1]
        target_body = sampled['T_world_from_BODY']
        targets = np.einsum('nij,nkj->nki',target_body[:,:3,:3],tip_local)+target_body[:,None,:3,3]
        np.testing.assert_allclose(tilt(body[:,:3,:3]),tilt(f['source_obj'][:,:3,:3]),atol=1e-9)
        source_sample = resample_body_poses(f['source_time'], f['source_obj'], old['COM_BODY_m'])
        np.testing.assert_allclose(reg @ source_sample['T_world_from_BODY'],target_body,atol=1e-10)
        np.testing.assert_allclose((targets-target_body[:,None,:3,3])@target_body[:,:3,:3],tip_local,atol=1e-10)
        q0 = old['initial_qpos27'].copy()
        qa = bind['mug_free_qpos']
        q0[qa:qa+3] = target_body[0,:3,3]
        q0[qa+3:qa+7] = sampled['quaternion_wxyz'][0]
        ik = HandTipIK(model,q0,targets[0],regularization=0.)
        prev = np.clip(q0[:20],ik.lo,ik.hi); joints=[]; actual=[]; converged=[]
        for target in targets:
            ik.targets = target
            fit = least_squares(lambda q: np.r_[ik.residual(q)[:15], .05*(q-prev)],
                prev,jac=lambda q:np.vstack([ik.jacobian(q)[:15],.05*np.eye(20)]),
                bounds=(ik.lo,ik.hi),max_nfev=120)
            prev=fit.x.copy();joints.append(prev);actual.append(ik.state(prev));converged.append(fit.success)
        d=mujoco.MjData(model);d.qpos[:]=q0;d.qpos[:20]=joints[0];d.ctrl[:]=joints[0];mujoco.mj_forward(model,d)
        mug=model.geom_bodyid==bind['mug_body_id']
        hand=np.isin(np.arange(model.ngeom),[model.geom(n).id for n in bind['hand_collision_names']])
        table=model.geom('mug_task_table').id
        def depths():
            g,h=d.contact.geom.T;dep=-1000*d.contact.dist
            return [dep[(mug[g]&hand[h])|(hand[g]&mug[h])].max(initial=0.),
                    dep[hand[g]&hand[h]].max(initial=0.),dep[(mug[g]&(h==table))|(mug[h]&(g==table))].max(initial=0.)]
        entry=np.array(depths())
        first_ctrl=None
        if (entry<=[2.,.2,0.]).all():
            d.ctrl[:]=joints[1];peak=entry.copy()
            for _ in range(64):
                assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
                mujoco.mj_step(model,d);peak=np.maximum(peak,depths())
            R=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])])
            com=d.qpos[qa:qa+3]+R.apply(old['COM_BODY_m'])
            first_ctrl=dict(COM_mm=1000*np.linalg.norm(com-sampled['COM_world_m'][1]),
                            SO3_deg=np.rad2deg((R*Rotation.from_matrix(target_body[1,:3,:3]).inv()).magnitude()),peak_mm=peak.tolist())
        err=1000*np.linalg.norm(np.array(actual)-targets,axis=-1)
        dest=out/f'{clip}_source_faithful.npz'
        np.savez_compressed(dest,transform=reg,body=target_body,t=sampled['elapsed_s'],q=joints,
                            targets=targets,fitted=actual,source_frames=f['frames'])
        results.append(dict(clip=clip,source=source,reference=file_pin(dest),
            tilt_deg=[float(tilt(target_body[:,:3,:3]).min()),float(tilt(target_body[:,:3,:3]).max())],
            old_initial_tilt_deg=tilt(f['body'][0,:3,:3]),start_tilt_deg=tilt(target_body[0,:3,:3]),
            source_wrist_travel_from_start_mm=1000*np.linalg.norm(f['source_wrist'][:,:3,3]-f['source_wrist'][0,:3,3],axis=1).max(),
            start_five_tip_rms_mm=np.sqrt(np.mean(err[0]**2)),five_tip_rms_mm=np.sqrt(np.mean(err**2)),
            ik_converged=sum(converged),samples=len(targets),entry_hand_mug_self_table_mm=entry.tolist(),first_control=first_ctrl))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
