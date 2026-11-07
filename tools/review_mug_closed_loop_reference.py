"""Independent Rodrigues/finite-difference readback of the anchored50Hz input."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def pose_at(t,T,query):
    left=min(int(np.searchsorted(t,query,side='right')-1),len(t)-2)
    if query<t[0] or query>t[-1]:raise ValueError('No source extrapolation')
    alpha=(int(query)-int(t[left]))/(int(t[left+1])-int(t[left]));R0,R1=T[left,:3,:3],T[left+1,:3,:3];relative=R0.T@R1
    angle=np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))
    if angle>.3:raise ValueError('This independent small-arc reviewer is scoped to this clip')
    coefficient=1+angle**2/6+7*angle**4/360 if angle<1e-4 else angle/np.sin(angle)
    v=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])*.5*coefficient
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);magnitude=np.linalg.norm(v)
    R=R0@(np.eye(3)+alpha*np.sinc(alpha*magnitude/np.pi)*K+.5*alpha**2*np.sinc(alpha*magnitude/(2*np.pi))**2*(K@K))
    p=T[left,:3,3]+alpha*(T[left+1,:3,3]-T[left,:3,3]);return R,p


def main(args):
    import mujoco
    run,out=args.run.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text())
    for pin in report['sources']:verify_file_pin(pin)
    z=np.load(verify_file_pin(report['output']));contract=json.loads(verify_file_pin(report['controller_contract']).read_text())
    t=z['source_timestamps_ns'];T=z['source_anchored_BODY_poses61'];COM=z['COM_BODY_m'];pose_errors=[];velocity_errors=[];omega_errors=[];quaternion_errors=[]
    for i,query in enumerate(z['timestamps_ns']):
        R,p=pose_at(t,T,int(query));pose_errors.append(max(float(abs(R-z['T_world_from_BODY'][i,:3,:3]).max()),float(abs(p-z['T_world_from_BODY'][i,:3,3]).max())))
        native=np.empty(9);mujoco.mju_quat2Mat(native,z['quaternion_wxyz'][i]);quaternion_errors.append(float(abs(native.reshape(3,3)-R).max()))
        low=int(query)-1000 if i else int(query);high=int(query)+1000;Rl,pl=pose_at(t,T,low);Rh,ph=pose_at(t,T,high);dt=(high-low)*1e-9
        linear=(ph+Rh@COM-pl-Rl@COM)/dt;dR=(Rh-Rl)/dt;skew=dR@R.T;omega=np.array([skew[2,1]-skew[1,2],skew[0,2]-skew[2,0],skew[1,0]-skew[0,1]])/2
        velocity_errors.append(float(abs(linear-z['COM_velocity_world_m_s'][i]).max()));omega_errors.append(float(abs(omega-z['angular_velocity_world_rad_s'][i]).max()))
    relative_error=max(float(abs(np.linalg.solve(a,b)-np.linalg.solve(c,d)).max()) for a,b,c,d in zip(z['source_original_BODY_poses61'][:-1],z['source_original_BODY_poses61'][1:],T[:-1],T[1:]))
    checks=dict(exact_101_integer_clock=np.array_equal(z['timestamps_ns']-t[0],np.arange(101,dtype=np.int64)*20_000_000),
        source_clock_not_stretched=int(t[-1]-z['timestamps_ns'][-1])==4434 and report['time']['excluded_tail_ns']==4434 and not report['time']['time_stretched'],
        independent_rigid_interpolation=max(pose_errors)<1e-12 and max(quaternion_errors)<1e-12,
        actual_COM_velocity_and_angular_derivative=max(velocity_errors)<1e-6 and max(omega_errors)<1e-6,
        source_relative_motion_preserved=relative_error<1e-12,
        constant_grasp_and_distinct_feedforward=np.array_equal(z['grasp_q20_reference'],np.tile(z['initial_qpos27'][:20],(101,1))) and np.array_equal(z['feedforward_PD_target20'],np.tile(z['feedforward_PD_target20'][0],(101,1))),
        no_training_or_contact_truth_claim=not contract['learning_enabled'] and contract['dataset_role']=='INPUT_CONTRACT_ONLY_NOT_TRAIN' and contract['contact_ground_truth'].startswith('UNKNOWN') and not contract['executable_controller_and_native_observation_adapter_validated'])
    checks={k:bool(v) for k,v in checks.items()};out.mkdir(parents=True);result=dict(schema='mug_closed_loop_reference_saved_review:1',source=file_pin(__file__),run=file_pin(run/'report.json'),checks=checks,passed=sum(checks.values()),total=len(checks),
        max_pose_error=max(pose_errors),max_native_quaternion_error=max(quaternion_errors),max_COM_velocity_error_m_s=max(velocity_errors),max_angular_velocity_error_rad_s=max(omega_errors),max_adjacent_relative_pose_error=relative_error,
        derivative_queries=202,derivative_step_ns=1000,first_sample_one_sided=True,integration_steps=0,neural_updates=0,
        scope='Independent small-arc Rodrigues/native-quaternion and finite-difference clock/COM reference readback. Not native trajectory execution, hand retargeting, source gravity calibration, measured contact labels, a completed controller or learning validation.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Retain failed anchored reference review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
