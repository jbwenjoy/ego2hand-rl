"""Separate object tracking, finger retargeting and load demands for the two offline witnesses."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    inputs=physics.read_json(file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    bind=physics.read_json(inputs['binding']);b=bind['mug_body_id'];d=mujoco.MjData(model)
    sites=[model.site(f'right_finger{i}_tip').id for i in range(1,6)];j=model.actuator_trnid[:,0];ids=model.jnt_qposadr[j]
    out.mkdir(exist_ok=False)
    sources={c:file_pin(root/f'mug_strict_extend_{c}_v1/report.json') for c in (1884,1888)}
    write_json(out/'intent.json',dict(sources=sources,inputs=inputs,code=file_pin(__file__),
        scope='Using the already frozen strict trajectories, quantify object and five-tip errors at each passed control endpoint. Tip reference is the anchored source-relative displacement target; source RGB and target initial robot grip differ. Report joint offsets from IK, commanded target slew and digit load summaries.'))
    results=[]
    for c,name in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
        ref=physics.load_arrays(item['reference'])
        result=physics.read_json(sources[c]);trace=physics.load_arrays(result['trace']);n=result['strict_controls']
        targets=ref['target'][1:n+1];T=ref['body'][1:n+1];actual=[];local=[];normal=[];vel=[]
        old=physics.load_arrays(inputs['reference']);mujoco.mj_resetData(model,d)
        d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26']
        d.ctrl[:]=old['feedforward_PD_target20'][0];mujoco.mj_forward(model,d)
        mug=model.geom_bodyid==b;groups=np.full(model.ngeom,-1,int)
        for g in bind['hand_collision_names']:
            groups[model.geom(g).id]=next((i for i in range(1,6) if g.startswith(f'right_finger{i}_')),0)
        wrench=np.empty(6)
        for i in range(n):
            d.ctrl[:]=trace['ctrl'][i]
            for _ in range(64):
                mujoco.mj_step(model,d)
                a,h=d.contact.geom.T;load=np.zeros(6)
                eligible=(mug[a]&(groups[h]>=0))|(mug[h]&(groups[a]>=0))
                for k in np.flatnonzero(eligible):
                    digit=groups[h[k] if mug[a[k]] else a[k]]
                    mujoco.mj_contactForce(model,d,k,wrench);load[digit]+=max(0.,wrench[0])
                normal.append(load);vel.append(np.abs(d.qvel[model.jnt_dofadr[j]]).max())
            np.testing.assert_array_equal(d.qpos,trace['qpos'][i])
            mujoco.mj_kinematics(model,d)
            actual.append(d.site_xpos[sites].copy())
            local.append((d.site_xpos[sites]-d.xpos[b])@d.xmat[b].reshape(3,3))
        local_target=(targets-T[:,None,:3,3])@T[:,:3,:3]
        errors=1000*np.linalg.norm(actual-targets,axis=2)
        body_errors=1000*np.linalg.norm(local-local_target,axis=2)
        delta=trace['qpos'][:n,ids]-ref['q'][1:n+1]
        loads=np.asarray(normal)
        series=out/f'{c}_series.npz';np.savez_compressed(series,tip_mm=errors,body_tip_mm=body_errors,joint_delta=delta,
            hand_normal_N=loads,peak_joint_speed=np.asarray(vel))
        peak=trace['strict_prefix_metrics'][:n].max(0)
        results.append(dict(clip=c,controls=n,ref_controls=len(ref['q'])-1,full=n==len(ref['q'])-1,
            object_COM_RMS_mm=np.sqrt(np.mean(trace['com_mm'][:n]**2)),object_SO3_RMS_deg=np.sqrt(np.mean(trace['so3_deg'][:n]**2)),
            strict_max=peak.tolist(),tip_RMS_mm=np.sqrt(np.mean(errors**2)),body_tip_RMS_mm=np.sqrt(np.mean(body_errors**2)),
            per_digit_tip_RMS_mm=np.sqrt(np.mean(errors**2,axis=0)).tolist(),hand_joint_RMS_rad=np.sqrt(np.mean(delta**2)),
            hand_joint_max_abs_rad=abs(delta).max(),peak_total_hand_normal_N=loads.sum(1).max(),mean_total_hand_normal_N=loads.sum(1).mean(),
            peak_normal_palm_digits_N=loads.max(0).tolist(),peak_joint_speed_rad_s=max(vel),series=file_pin(series)))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
