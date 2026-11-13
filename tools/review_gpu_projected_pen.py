"""Independent GPU projected initialization, control and cone certificates."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.support import static_feasibility


def checked_certificate(W,target,certificate):
    if certificate is None or not certificate['verified']:return False
    y=np.array(certificate['witness_dual_to_F_N_T_Nm'])
    return bool(np.isfinite(y).all() and np.min(W.T@y)>=-1e-9 and abs(target@y+1)<1e-9)


def main(args):
    if args.output.exists():raise ValueError('Fresh GPU projected review')
    r=json.loads(args.report.read_text());p=json.loads(verify_file_pin(r['projection']).read_text());proposal=np.load(verify_file_pin(p['proposal']));source=np.load(verify_file_pin(p['case']['trace']))
    import tomllib
    pc=verify_file_pin(p['config']);config=tomllib.loads(pc.read_text());qref=np.load((pc.parent/config['source_clip']).resolve())['reference_qpos']
    for pin in r['sources']+r['task_sources']:verify_file_pin(pin)
    expected={(name,dt) for name in ['ORIGINAL_ZERO_RESIDUAL','PROJECTED_CONSTANT_RESIDUAL'] for dt in [.005,.0025]}
    if {(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=4:raise ValueError('Fixed four GPU cases')
    args.output.mkdir();items=[];tables=[]
    for c in r['cases']:
        z=np.load(verify_file_pin(c['trace']));init=c['initialization'];snap=np.load(verify_file_pin(init['snapshot']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));physical=np.load(verify_file_pin(c['GPU_physical_context']));tables.append(dict(physical))
        projected=c['name']=='PROJECTED_CONSTANT_RESIDUAL';delta=proposal['delta20'].astype(np.float32) if projected else np.zeros(20,dtype=np.float32);expectedq=(proposal['qpos'] if projected else source['initial_qpos']).astype(np.float32).astype(np.float64)
        names=[f'robot/right_finger{i}_joint{j}' for i in range(1,6) for j in range(1,5)];jids=np.array([m.joint(name).id for name in names]);qids=m.jnt_qposadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        lo=np.maximum(physical['jnt_range'][0,jids,0],physical['actuator_ctrlrange'][0,aids,0]);hi=np.minimum(physical['jnt_range'][0,jids,1],physical['actuator_ctrlrange'][0,aids,1]);n=round(.02/c['dt']);N=len(z['post_qpos']);K=len(z['actions20'])
        memory=delta.copy();target=[]
        for phase,action in zip(z['pre_action_phase'],z['actions20']):
            memory=np.float32(.5)*np.clip(np.float32(.5)*action,-.6,.6)+np.float32(.5)*memory
            target.append(np.clip(qref[int(phase)]+memory,lo,hi))
        control_error=float(abs(np.repeat(target,n,axis=0)-z['post_ctrl'][:,aids]).max())
        checks=dict(initial_qpos_exact=np.array_equal(z['initial_qpos'],expectedq),source_velocity_exact=np.array_equal(z['initial_qvel'],source['initial_qvel']),
            static_pose_same=np.array_equal(snap['static_qpos'],z['initial_qpos']),static_velocity_zero=not np.any(snap['static_qvel']),
            no_nonjoint_correction=np.array_equal(np.delete(z['initial_qpos'],qids),np.delete(source['initial_qpos'],qids)),
            source_clock=np.array_equal(z['pre_action_phase'],np.arange(1,K+1)) and N==K*n,
            pose_carry=np.array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1]),velocity_carry=np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            constant_action=np.array_equal(z['actions20'],np.tile(2*delta,(K,1))),actual_PD_targets=control_error<2e-7,
            first_terminal=not np.any(z['terminated_truncated'][:-1]) and not np.any(z['termination_flags'][:-1]) and bool(z['terminated_truncated'][-1,0]),
            no_applied_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']))
        native=mujoco.MjData(m);mujoco.mj_setState(m,native,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        parent=mujoco.MjData(m);mujoco.mj_setState(m,parent,source['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        checks['mocap_warmstart_preserved']=np.array_equal(native.mocap_pos,parent.mocap_pos) and np.array_equal(native.mocap_quat,parent.mocap_quat) and np.array_equal(native.qacc_warmstart,parent.qacc_warmstart)
        body=c['model_layout']['object_body'];R=snap['BODY_rotation_world'];COM=snap['COM_world'];W=[];active=[]
        for k,(a,b) in enumerate(snap['GPU_geom']):
            oa,ob=m.geom_bodyid[a]==body,m.geom_bodyid[b]==body
            if oa==ob or snap['GPU_efc_addresses'][k,0]<0:continue
            sign=1. if ob else -1.;frame=snap['GPU_frame'][k].reshape(3,3);dim=int(snap['GPU_dim'][k]);mu=snap['GPU_friction'][k]
            active.append((int(a),int(b)))
            for direction in range(dim-1):
                for side in [1.,-1.]:
                    ray=np.zeros(6);ray[0]=1.;ray[direction+1]=side*mu[direction]
                    F=sign*(frame.T@ray[:3]);T=sign*(frame.T@ray[3:])+np.cross(snap['GPU_position'][k]-COM,F)
                    W.append(np.r_[R.T@F,R.T@T])
        W=np.array(W).T;Werror=float(abs(W-snap['wrench_BODY']).max());checks['actual_GPU_wrench_basis']=Werror<1e-10
        static_target=np.r_[R.T@(-float(snap['mass_kg'])*snap['gravity_world']),np.zeros(3)]
        checks['static_target_frame']=np.array_equal(static_target,snap['target_static_BODY'])
        pure=static_feasibility(W,snap['target_static_BODY'],radius=.125);dynamic=static_feasibility(W,snap['target_dynamic_BODY'],radius=.125)
        bounded=static_feasibility(W,snap['target_static_BODY'],loads=snap['loads20'],offset=snap['torque_offset'],lower=snap['torque_lower'],upper=snap['torque_upper'],radius=.125)
        checks['frozen_LP_readback']=pure['success']==init['static_wrench_cone_feasible'] and dynamic['success']==init['dynamic_wrench_cone_feasible'] and bounded['success']==init['static_support_with_PD_bounds']
        checks['static_separation']=pure['success'] or checked_certificate(W,snap['target_static_BODY'],init['static_separating_certificate'])
        checks['dynamic_separation']=dynamic['success'] or checked_certificate(W,snap['target_dynamic_BODY'],init['dynamic_separating_certificate'])
        qa=c['model_layout']['object_qposadr'];times=(np.arange(N)+1)*c['dt'];clock=z['time_s'];positions=np.stack([np.interp(times,clock,z['COM_world_m'][:,j]) for j in range(3)],axis=1);dist=np.linalg.norm(z['post_qpos'][:,qa:qa+3]-positions,axis=1)
        actualR=Rotation.from_quat(z['post_qpos'][:,qa+3:qa+7][:,[1,2,3,0]]);goal=Slerp(clock,Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(np.minimum(times,clock[-1]));angle=(actualR*goal.inv()).magnitude();metric=c['summary']['metrics']
        merr=max(abs(np.sqrt(np.mean(dist**2))-metric['COM_RMS_m']),abs(dist.max()-metric['COM_max_m']),abs(np.sqrt(np.mean(angle**2))-metric['angle_RMS_rad']),abs(angle.max()-metric['angle_max_rad']))
        checks['pose_metrics']=merr<1e-10
        items.append(dict(name=c['name'],dt=c['dt'],checks={k:bool(v) for k,v in checks.items()},control_error_rad=control_error,wrench_basis_error=Werror,pose_metric_error=merr,active_support_contacts=active,qualified=c['summary']['qualified']))
    model_parameters_same=all(all(np.array_equal(tables[0][key],v[key]) for key in tables[0]) for v in tables[1:])
    contacts=review_saved_warp_contacts(args.report,args.output/'EFC',expected)
    checks=dict(all_initialization_control_and_support=all(all(c['checks'].values()) for c in items),physical_tables_equal=model_parameters_same,all_EFC_checks=contacts['passed']==contacts['total'])
    result=dict(schema='GPU_projected_pen_saved_review:1',source=file_pin(__file__),run=file_pin(args.report),checks=checks,cases=items,
        passed=sum(sum(c['checks'].values()) for c in items),total=sum(len(c['checks']) for c in items),EFC_review=file_pin(args.output/'EFC/report.json'),EFC_checks=[contacts['passed'],contacts['total']],
        neural_updates=0,native_integration_steps=0,scope='Independent actual GPU cone/dual, LP readback, initialization/velocity/warmstart/control/clock and EFC force receipts. Not manipulation success or derivative/generalization certificate.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Preserve failed GPU projected review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
