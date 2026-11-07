"""Independently review saved native hold traces and sampled contact wrench sums."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    import mujoco
    run,reference,out=args.run.resolve(),args.reference.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());intent=json.loads((run/'intent.json').read_text())
    bind=json.loads((reference/'binding.json').read_text())
    for pin in report['sources']+[intent['qualified_candidate'],intent['qualified_geometry']]:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));oid=bind['mug_body_id'];com=model.body_ipos[oid]
    parts=set(bind['mug_part_geom_ids']);table=model.geom('mug_task_table').id
    groups={int(g):int(k)-1 for k,v in intent['pad_groups'].items() for g in v}
    initial=np.load(intent['qualified_candidate']['path'])['qpos27'];reviews=[];all_checks=[]
    for case in report['cases']:
        verify_file_pin(case['case']);verify_file_pin(case['trace']);summary=case['summary'];z=np.load(case['trace']['path'])
        n=len(z['post_qpos']);dt=float(z['native_dt_s']);pq=z['post_qpos'];preq=z['pre_qpos'];qadr=bind['mug_free_qpos'];dadr=bind['mug_free_dof']
        quats=pq[:,qadr+3:qadr+7]
        if np.max(np.abs(np.linalg.norm(quats,axis=1)-1))>1e-12:raise ValueError('Raw unit quaternion gate before descriptor Rotation conversion')
        R=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix();centers=pq[:,qadr:qadr+3]+np.einsum('nij,j->ni',R,com)
        center_error=float(np.abs(centers-z['post_COM']).max())
        preR=Rotation.from_quat(preq[:,qadr+3:qadr+7][:,[1,2,3,0]]).as_matrix()
        precenters=preq[:,qadr:qadr+3]+np.einsum('nij,j->ni',preR,com)
        force_error=group_error=0.
        rows=z['raw_contact_tick_plus23']
        for tick in z['raw_sample_ticks']:
            force=np.zeros(3);torque=np.zeros(3);normal=np.zeros(3)
            for row in rows[rows[:,0]==tick]:
                a,b=map(int,row[1:3]);is_a,is_b=a in parts,b in parts
                if is_a==is_b:continue
                sign=1 if is_b else -1;other=a if is_b else b
                frame=row[9:18].reshape(3,3);F=sign*(frame.T@row[18:21]);T=sign*(frame.T@row[21:24])
                force+=F;torque+=T+np.cross(row[6:9]-precenters[tick],F)
                if other in groups:normal[groups[other]]+=max(row[18],0)
            body=np.r_[preR[tick].T@force,preR[tick].T@torque]
            force_error=max(force_error,float(np.abs(body-z['pre_wrench_BODY_COM'][tick]).max()))
            group_error=max(group_error,float(np.abs(normal-z['pre_group_normal_N'][tick]).max()))
        data=mujoco.MjData(model);data.qpos[:]=pq[-1];data.qvel[:]=z['post_qvel'][-1]
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
        native=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,oid,native,0)
        velocity_error=float(np.abs(native-np.r_[z['post_omega_world'][-1],z['post_COM_velocity'][-1]]).max())
        drift=float(np.linalg.norm(centers[-1]-z['initial_COM']));angle=float(2*np.arccos(np.clip(abs(quats[-1]@z['initial_quat']),0,1)))
        selected=intent['geometric_overclosure_m'].index(case['overclosure_m'])
        checks=dict(pin_verified=True,initial_qpos_byte_equal=np.array_equal(preq[0],initial),
            next_PRE_equals_previous_POST=np.array_equal(preq[1:],pq[:-1]),
            next_PRE_velocity_equals_previous_POST=np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            raw_constant_target_matches_intent=np.array_equal(z['ctrl20'],intent['targets'][selected]),
            native_clock_matches_completed_steps=summary['actual_duration_s']==n*dt,
            BODY_to_COM_all_POST_reconstructed=center_error<1e-12,
            PRE_COM_reconstructed=float(np.abs(precenters-z['pre_object_COM']).max())<1e-12,
            sampled_contact_BODY_wrench_reconstructed=force_error<1e-10,
            sampled_canonical_normal_force_reconstructed=group_error<1e-10,
            saved_endpoint_COM_velocity_matches_native=velocity_error<1e-10,
            sampled_native_external_wrench_matches=float(z['rne_external_wrench_errors'].max())<1e-8,
            final_metrics_reconstructed=abs(drift-summary['metrics']['final_COM_drift_m'])<1e-12 and abs(angle-summary['metrics']['final_angle_rad'])<1e-12,
            partial_duration_not_reported_success=not summary['native_holding_gates_pass'] and not summary['gates']['duration_complete'])
        checks={k:bool(v) for k,v in checks.items()};all_checks.extend(checks.values())
        reviews.append(dict(case=case['case'],checks=checks,max_errors=dict(COM_m=center_error,body_wrench=force_error,
            canonical_normal_N=group_error,native_endpoint_velocity=velocity_error)))
    checks=dict(all_case_checks=all(all_checks),exact10_cases=len(reviews)==10,
        no_partial_pair_claimed_consistent=all(not x['dt_pair_consistency_pass'] and not x['qualified'] for x in report['dt_pairs']),
        source_model_no_mocap_equalities_gravitycomp=model.neq==model.nmocap==0 and bool(np.all(model.body_gravcomp==0)),
        actual_native_step_total=sum(x['summary']['completed_steps'] for x in report['cases'])==report['integration_steps']==9343)
    checks={k:bool(v) for k,v in checks.items()};out.mkdir(parents=True)
    result=dict(schema='mug_hold_saved_review:1',source=file_pin(__file__),run=file_pin(run/'report.json'),
        checks=checks,passed=sum(checks.values()),total=len(checks),cases=reviews,
        case_checks_passed=sum(all_checks),case_checks_total=len(all_checks),native_endpoint_velocity_checks=10,
        integration_steps=0,neural_updates=0,
        scope='Saved state/clock/endpoint velocities, all POST COM and10ms-sampled complete contact sums. Does not independently certify unrecorded applied-force arrays, unsampled contact details or continuous ideal-convex clearance. All cases remain failed native holding.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,case_checks=[sum(all_checks),len(all_checks)])))
    if not all(checks.values()):raise RuntimeError('Saved holding review failed; retain output')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
