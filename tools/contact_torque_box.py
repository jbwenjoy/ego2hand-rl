"""Frozen-reset contact wrench/PD torque-box diagnostic; no integration/fit.

The LP freezes material points, normals, Jacobians and observed non-phone
constraint loads at reset. Sampled ideal friction rays and freely chosen
normal forces omit native compliance. Neither feasibility nor infeasibility
is a dynamic control certificate. Hypothetical force/torque witnesses are
never applied to the simulator.
"""
from pathlib import Path
import argparse,json,os,sys,time,traceback
import numpy as np
from evaluate_online import preflight

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime import native_evaluation as E


def pd_bounds(gain,bias,control_lower,control_upper,actuator_bounds,joint_bounds):
    C.need(all(np.asarray(x).shape==(20,) for x in (gain,bias,control_lower,control_upper)),'Explicit20-axis affine PD map')
    C.need((gain>0).all() and (control_lower<=control_upper).all(),'Positive fixed gain and legal target interval')
    force_lower=np.maximum(actuator_bounds[:,0],joint_bounds[:,0]);force_upper=np.minimum(actuator_bounds[:,1],joint_bounds[:,1])
    C.need((force_lower<=force_upper).all(),'Nonempty actual force limits')
    # These are the plant's fixed force saturations, not action/control repair.
    return (np.minimum(force_upper,np.maximum(force_lower,gain*control_lower+bias)),
            np.minimum(force_upper,np.maximum(force_lower,gain*control_upper+bias)))


def solve_box(wrench,joint_load,target,offset,lower,upper,radius):
    from scipy.optimize import linprog
    scaled=wrench.copy();scaled[3:]/=radius;rhs=target.copy();rhs[3:]/=radius
    A=np.r_[joint_load,-joint_load];b=np.r_[upper-offset,offset-lower]
    result=linprog(np.ones(wrench.shape[1]),A_eq=scaled,b_eq=rhs,A_ub=A,b_ub=b,bounds=(0,None),method='highs')
    if not result.success:return dict(success=False,status=int(result.status),message=result.message,normal_coefficients=None,required_joint_torque_Nm=None,minimum_sum_normal_N=None,force_residual_N=None,moment_residual_Nm=None,minimum_torque_box_margin_Nm=None)
    tau=offset+joint_load@result.x;error=wrench@result.x-target
    return dict(success=True,status=int(result.status),message=result.message,normal_coefficients=result.x.tolist(),required_joint_torque_Nm=tau.tolist(),
        minimum_sum_normal_N=float(result.fun),force_residual_N=error[:3].tolist(),moment_residual_Nm=error[3:].tolist(),
        minimum_torque_box_margin_Nm=float(np.min(np.r_[tau-lower,upper-tau])))


def run(args):
    path,c,registry,root,final,health,zero,binding,out=preflight(args)
    authority=C.read(args.authority_report)
    C.need(authority['status']=='ALL_FIXED_SCHEDULES_RECORDED_NOT_PROOF_OF_IMPOSSIBILITY' and C.positive(authority),'Completed fixed authority diagnostic')
    C.need(authority['data_role']=='NO_FITTING_CONTROL_AUTHORITY_DIAGNOSTIC','Non-fitting physical diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco
    out.mkdir(parents=True);tick=time.perf_counter();phase='load'
    counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,jacobian_point_reads=0,integration_receipt_reads=0,LP_calls=0,NN_FW=0,learning_updates=0,GPU_calls=0)
    C.write(out/'intent.json',dict(tool=file_pin(__file__),authority_report=file_pin(args.authority_report),task_bundle=registry.pin(c['artifacts']['task']),model=registry.pin(c['artifacts']['model']),
        source_pins=C.source_pins(),dt_s=E.DTS,domains=['first_H_EMA_half_scale','steady_residual_scale','full_actuator_target_range_analytical_only'],
        friction_rays=[4,16],mu=.7,maximum_native_forward=2,maximum_integrations=0,maximum_LP_calls=12,
        role='NO_FITTING_RESET_STATIC_APPROXIMATION',no_forces_or_witnesses_applied=True,automatic_retry=False))
    try:
        model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));phone,pg,table,pads=E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np)
        bundle=C.read(registry.resolve(c['artifacts']['task']));q0=np.array(bundle['initial']['qpos27_F64']);u0=np.array(bundle['initial']['ctrl20_F64']);scale=np.array(bundle['action']['symmetric_residual_scale20_rad']);radius=bundle['support']['radius_m'];records=[];checks={}
        for dt in E.DTS:
            phase='reset_'+str(dt);model.opt.timestep=dt;data=mujoco.MjData(model);counts['MjData']+=1
            data.qpos[:]=q0;data.qvel[:]=0;data.ctrl[:]=u0;data.qacc_warmstart[:]=0;mujoco.mj_forward(model,data);counts['initial_forward']+=1
            C.need(not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied) and not np.any(data.warning.number),'Original no-aid reset')
            snapshot=E.contact_snapshot(model,data,pg,table,pads,mujoco,np,counts,-1);rows=np.array(snapshot[-1]).reshape(-1,24)
            baseline=next(case for case in authority['cases'] if case['schedule']=='zero' and case['dt_s']==dt)
            expected=np.array(baseline['native_summary']['initial_native']['contact_rows24'])
            checks[str(dt)+'_reset_contact_bytes']=rows.dtype==expected.dtype and rows.shape==expected.shape and rows.tobytes()==expected.tobytes()
            C.need(checks[str(dt)+'_reset_contact_bytes'],'Same recorded reset contacts before Jacobian reads')
            phone_rows=rows[(rows[:,1]==pg)|(rows[:,2]==pg)]
            C.need(len(phone_rows)==3 and set(np.where(phone_rows[:,1]==pg,phone_rows[:,2],phone_rows[:,1]).astype(int))==set(pads) and np.all(phone_rows[:,3]==3),'Exactly three declared condim3 phone contacts')
            signature=mujoco.mjtState.mjSTATE_INTEGRATION;before=np.empty(300);after=np.empty(300)
            mujoco.mj_getState(model,data,before,signature);counts['integration_receipt_reads']+=1
            jacobians=[];actual_phone_hand_torque=np.zeros(20)
            for row in phone_rows:
                g1,g2=map(int,row[1:3]);other=g1 if g2==pg else g2;sign=1 if g2==pg else -1;frame=row[9:18].reshape(3,3)
                jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,row[6:9].copy(),int(model.geom_bodyid[other]));counts['jacobian_point_reads']+=1
                jacobians.append(jac[:,:20].copy());force=sign*frame.T@row[18:21];actual_phone_hand_torque-=jac[:,:20].T@force
            mujoco.mj_getState(model,data,after,signature);counts['integration_receipt_reads']+=1
            checks[str(dt)+'_jacobian_reads_preserve300']=before.tobytes()==after.tobytes()
            other_constraint=data.qfrc_constraint[:20]-actual_phone_hand_torque
            offset=data.qfrc_bias[:20]-data.qfrc_passive[:20]-other_constraint
            gain=model.actuator_gainprm[:,0].copy()
            affine=(model.actuator_biasprm[:,0]+model.actuator_biasprm[:,1]*data.actuator_length+model.actuator_biasprm[:,2]*data.actuator_velocity).copy()
            lo,hi=pd_bounds(gain,affine,u0,u0,model.actuator_forcerange,model.jnt_actfrcrange[:20])
            checks[str(dt)+'_PD_base_matches_actual']=bool(np.allclose(lo,data.qfrc_actuator[:20],rtol=1e-13,atol=1e-13))
            domains={}
            for name,a,b in [('first_H_EMA_half_scale',u0-.5*scale,u0+.5*scale),('steady_residual_scale',u0-scale,u0+scale),('full_actuator_target_range_analytical_only',model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])]:
                C.need(np.all(a>=model.actuator_ctrlrange[:,0]) and np.all(b<=model.actuator_ctrlrange[:,1]),'Actual legal control intervals, no clipping')
                low,high=pd_bounds(gain,affine,a,b,model.actuator_forcerange,model.jnt_actfrcrange[:20]);domains[name]=dict(control_lower=a.copy(),control_upper=b.copy(),torque_lower=low,torque_upper=high)
            COM=data.xipos[phone].copy();lp=[]
            for rays in (4,16):
                columns=[];joint_columns=[]
                for row,jac in zip(phone_rows,jacobians):
                    sign=1 if int(row[2])==pg else -1;frame=row[9:18].reshape(3,3)
                    for theta in np.arange(rays)*2*np.pi/rays:
                        force=sign*(frame[0]+.7*(np.cos(theta)*frame[1]+np.sin(theta)*frame[2]))
                        columns.append(np.r_[force,np.cross(row[6:9]-COM,force)]);joint_columns.append(jac.T@force)
                wrench=np.array(columns).T;loads=np.array(joint_columns).T;target=np.r_[-model.body_mass[phone]*model.opt.gravity,np.zeros(3)]
                for name,domain in domains.items():
                    answer=solve_box(wrench,loads,target,offset,domain['torque_lower'],domain['torque_upper'],radius);counts['LP_calls']+=1
                    lp.append(dict(domain=name,rays_per_contact=rays,answer=answer))
            arrays=dict(qpos27=q0,ctrl20=u0,body_COM_world_m=COM,contact_rows24=rows,contact_jacobians3x3x20=np.array(jacobians),
                qfrc_bias20=data.qfrc_bias[:20].copy(),qfrc_passive20=data.qfrc_passive[:20].copy(),qfrc_constraint20=data.qfrc_constraint[:20].copy(),
                actual_phone_hand_torque20=actual_phone_hand_torque,observed_other_constraint_torque20=other_constraint,required_static_torque_offset20=offset,
                actual_actuator_torque20=data.qfrc_actuator[:20].copy(),affine_PD_gain20=gain,affine_PD_bias20=affine,
                integration300_before_jacobians=before,integration300_after_jacobians=after)
            for name,domain in domains.items():
                for field,array in domain.items():arrays[name+'__'+field]=array
            raw=out/('reset_'+str(dt)+'.npz')
            with raw.open('xb') as stream:np.savez_compressed(stream,**arrays)
            records.append(dict(dt_s=dt,raw=file_pin(raw),other_constraint_torque_max_abs_Nm=float(abs(other_constraint).max()),LPs=lp))
        checks['same_model_except_dt']=digest==E.model_digest_except_dt(model,np)
        checks['declared_native_and_LP_budget']=counts['initial_forward']==2 and counts['integration_steps']==0 and counts['jacobian_point_reads']==6 and counts['LP_calls']==12
        checks['no_Torch_or_neural_import']='torch' not in sys.modules
        C.need(all(checks.values()),'Reset-static diagnostic integrity failure')
        report=dict(schema='supermimic_research.reset_contact_torque_box:1',checks=checks,passed=sum(checks.values()),total=len(checks),records=records,counts=counts,
            elapsed_s=time.perf_counter()-tick,own_checkpoint_loads=0,
            scope='Frozen reset, material points and normals. Ideal sampled Coulomb point forces, observed non-phone constraint load held fixed, affine unit-gear PD and actual force saturations. No native compliance, evolving contact geometry or dynamic reachability certificate. Full actuator target interval is analytical only and was never executed.')
        C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=report['passed'],total=report['total'],counts=counts)),flush=True);return report
    except BaseException as error:C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--zero-report',type=Path,required=True);p.add_argument('--authority-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run(a)
    else:
        *_,out=preflight(a);print(json.dumps(dict(status='METADATA_ONLY_RESET_STATIC_READY',output=str(out),Torch_imported='torch' in sys.modules,maximum_integrations=0)))
