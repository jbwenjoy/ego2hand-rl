"""One saved-trace reviewer for all new support/placement/initialization trials."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def check_case(case,expected_q,expected_u,model,bind,groups):
    import mujoco
    verify_file_pin(case['trace']);z=np.load(case['trace']['path']);summary=case['summary'];n=len(z['post_qpos']);dt=float(z['native_dt_s'])
    pq,pre=z['post_qpos'],z['pre_qpos'];quats=pq[:,23:27];prequats=pre[:,23:27]
    unit_error=float(max(np.abs(np.linalg.norm(quats,axis=1)-1).max(),np.abs(np.linalg.norm(prequats,axis=1)-1).max()))
    if unit_error>1e-12:raise ValueError('Raw quaternion unit gate before descriptor Rotation conversion')
    R=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix();preR=Rotation.from_quat(prequats[:,[1,2,3,0]]).as_matrix()
    com=model.body_ipos[bind['mug_body_id']];postCOM=pq[:,20:23]+np.einsum('nij,j->ni',R,com)
    preCOM=pre[:,20:23]+np.einsum('nij,j->ni',preR,com);parts=set(bind['mug_part_geom_ids'])
    raw=z['raw_contact_tick_plus23'];wrench_error=normal_error=0.
    for tick in z['raw_sample_ticks']:
        force=np.zeros(3);torque=np.zeros(3);normal=np.zeros(3)
        for row in raw[raw[:,0]==tick]:
            a,b=map(int,row[1:3]);oa,ob=a in parts,b in parts
            if oa==ob:continue
            sign=1 if ob else -1;other=a if ob else b;frame=row[9:18].reshape(3,3)
            F=sign*(frame.T@row[18:21]);T=sign*(frame.T@row[21:24])
            force+=F;torque+=T+np.cross(row[6:9]-preCOM[tick],F)
            if other in groups:normal[groups[other]]+=max(row[18],0.)
        body=np.r_[preR[tick].T@force,preR[tick].T@torque]
        wrench_error=max(wrench_error,float(np.abs(body-z['pre_wrench_BODY_COM'][tick]).max()))
        normal_error=max(normal_error,float(np.abs(normal-z['pre_group_normal_N'][tick]).max()))
    data=mujoco.MjData(model);data.qpos[:]=pq[-1];data.qvel[:]=z['post_qvel'][-1]
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
    native=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bind['mug_body_id'],native,0)
    velocity_error=float(np.abs(native-np.r_[z['post_omega_world'][-1],z['post_COM_velocity'][-1]]).max())
    com_error=float(max(np.abs(postCOM-z['post_COM']).max(),np.abs(preCOM-z['pre_object_COM']).max()))
    drift=float(np.linalg.norm(postCOM[-1]-z['initial_COM']));angle=float(2*np.arccos(np.clip(abs(quats[-1]@z['initial_quat']),0,1)))
    limits=np.maximum(np.maximum(model.jnt_range[:20,0]-pq[:,:20],pq[:,:20]-model.jnt_range[:20,1]),0).max(1)
    checks=dict(initial_qpos_exact=np.array_equal(pre[0],expected_q),constant_command_exact=np.array_equal(z['ctrl20'],expected_u),
        state_POST_to_PRE_exact=np.array_equal(pre[1:],pq[:-1]),velocity_POST_to_PRE_exact=np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
        full_saved_arrays_finite=all(np.isfinite(z[k]).all() for k in z.files),native_clock=summary['completed_steps']==n and summary['actual_duration_s']==n*dt,
        every_PRE_POST_COM_reconstructed=com_error<1e-12,sampled_contact_wrench_reconstructed=wrench_error<1e-10,
        sampled_three_group_normal_reconstructed=normal_error<1e-10,endpoint_velocity_native_check=velocity_error<1e-10,
        limits_recomputed=np.max(np.abs(limits-z['post_limit_excursion_rad']))<1e-12,
        final_pose_metrics_recomputed=abs(drift-summary['metrics']['final_COM_drift_m'])<1e-12 and abs(angle-summary['metrics']['final_angle_rad'])<1e-12,
        partials_are_failed=summary['actual_duration_s']<2 and not summary['gates']['duration_complete'] and not summary['native_holding_gates_pass'])
    return dict(trace=case['trace'],checks={k:bool(v) for k,v in checks.items()},errors=dict(raw_quaternion_unit=unit_error,
        all_COM_m=com_error,sampled_BODY_wrench=wrench_error,sampled_group_normal_N=normal_error,endpoint_native_velocity=velocity_error))


def main(args):
    import mujoco
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    reference=args.reference.resolve();bind=json.loads((reference/'binding.json').read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    groups={model.geom(f'right_finger{i}_{s}').id:i-1 for i in range(1,4) for s in ['link3_col','link4_col','tip_sensor_col']}
    records=[];inputs=[]
    for path,kind in [(args.support_native,'support'),(args.equilibrium,'equilibrium'),(args.gravity_hold,'gravity')]:
        folder=path.resolve();report=json.loads((folder/'report.json').read_text());intent=json.loads((folder/'intent.json').read_text());inputs.append(file_pin(folder/'report.json'))
        for pin in report['sources']:verify_file_pin(pin)
        if kind=='support':
            q=np.load(verify_file_pin(report['initial_state']))['qpos27']
            for case in report['cases']:records.append(check_case(case,q,np.array(intent['targets'][case['target_index']]['u']),model,bind,groups))
        elif kind=='gravity':
            q=np.load(verify_file_pin(intent['qualified_candidate']))['qpos27']
            for case in report['cases']:
                index=intent['geometric_overclosure_m'].index(case['overclosure_m'])
                records.append(check_case(case,q,np.array(intent['targets'][index]),model,bind,groups))
        else:
            for variant in report['results']:
                state=np.load(verify_file_pin(variant['state']))
                for case in variant['native_cases']:records.append(check_case(case,state['qpos27'],state['ctrl20'],model,bind,groups))
    checks=[v for r in records for v in r['checks'].values()]
    out.mkdir(parents=True);result=dict(schema='mug_support_trials_saved_review:1',source=file_pin(__file__),inputs=inputs,
        cases=len(records),checks_passed=sum(checks),checks_total=len(checks),records=records,neural_updates=0,integration_steps=0,
        native_endpoint_velocity_checks=len(records),
        scope='All saved state continuity/clock/COM/limits and endpoints plus10ms-sampled full contact sums. No independent proof of unsaved applied-force arrays, unsampled contacts, continuous convex geometry or dynamic success. All evaluated cases remain partial failures.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),cases=len(records),checks=[sum(checks),len(checks)])))
    if not all(checks) or len(records)!=20:raise ValueError('Retain failed support-trials review')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','support-native','equilibrium','gravity-hold','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
