"""Exactly2 primary LPs on saved v4 operands; remove only PD reachability.

No native model, FK, control command, force-recovery LP or physics.
"""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def need(x,label):
    if not x:raise ValueError(label)
def cross(a,b):return np.cross(a,b)


def main():
    cfg=read(HERE/'preregistration.json')
    need(pin(__file__)==cfg['source'],'source freeze')
    for v in cfg['input_pins'].values():need(pin(v['path'])==v,'input pin')
    r=read(cfg['input_pins']['v4_report']['path']);audit=read(cfg['input_pins']['saved9']['path'])
    need(audit['passed']==audit['total']==9 and audit['dynamic_diagnostic_eligible_count']==0,'existing saved audit')
    need([s['seed'] for s in r['seed_results']]==[0,1],'exact two fixed seeds')
    with np.load(cfg['input_pins']['raw_original_actuator_ranges']['path'],allow_pickle=False) as z:
        act=z['actuator_forcerange'].copy();joint=z['joint_actfrcrange'].copy()
    ideal_lo=np.maximum(act[:,0],joint[:,0]);ideal_hi=np.minimum(act[:,1],joint[:,1]);results=[];calls=0
    for s in r['seed_results']:
        state=s['verification_state'];op=state['support_operands'];old=s['support_LP']
        need(not old['feasible_force_balance'],'original force-hard failed, not recomputed')
        rays=np.array(old['rays_force_world_per_normal_N']);p=np.array(op['contact_points_world']);COM=np.array(op['object_COM']);J=np.array(op['J_robot_contact_3x20']);bias=np.array(op['robot_gravity_bias_tau'])
        need(rays.shape==(2,8,3) and old['normal_cap_N']==[3.,3.] and old['mu_actual']==[.7,.7],'original fixed cone/cap')
        # Use identical saved rays and operands. Only the40 torque bound RHS
        # entries change relative to the frozen primary residual formulation.
        F=np.zeros((6,16));S=np.zeros((2,16))
        for i in range(2):F[3*i:3*i+3,8*i:8*i+8]=rays[i].T;S[i,8*i:8*i+8]=1
        B=np.c_[rays[0].T,rays[1].T]
        M=np.c_[np.cross(p[0]-COM,rays[0]).T,np.cross(p[1]-COM,rays[1]).T]
        JT=J.transpose(2,0,1).reshape(20,6);T=JT@F
        Aeq=np.c_[B,np.zeros(3)];beq=-op['mass']*np.array(op['gravity_world'])
        Aub=np.vstack([np.c_[M,-np.ones(3)],np.c_[-M,-np.ones(3)],np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(2)]])
        bub=np.r_[np.zeros(6),ideal_hi-bias,bias-ideal_lo,[3.,3.]]
        result=linprog(np.r_[np.zeros(16),1.],A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*17,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});calls+=1
        item={'seed':s['seed'],'original_reported_force_hard_feasible':False,'original_recovery_force_infinity_N':old['force_recovery']['force_infinity_N'],
          'ideal_solver_status':int(result.status),'ideal_solver_message':result.message,'ideal_torque_force_balance_feasible':False,
          'moment_infinity_Nm':None,'original_PD_intersection_lower_Nm':op['torque_lower'],'original_PD_intersection_upper_Nm':op['torque_upper'],
          'ideal_native_actuator_joint_lower_Nm':ideal_lo.tolist(),'ideal_native_actuator_joint_upper_Nm':ideal_hi.tolist(),
          'matrix_unchanged_except_torque_bounds':True,'current_controller_legal':False,'may_enter_controller_or_holding':False,'holding_qualified':False}
        if result.status==0:
            lam=result.x[:16];forces=(F@lam).reshape(2,3);fr=B@lam-beq;moment=M@lam;tau=bias+T@lam;fn=S@lam
            valid=bool(lam.min()>=-1e-9 and max(abs(fr))<=1e-8 and min(tau-ideal_lo)>=-1e-9 and min(ideal_hi-tau)>=-1e-9 and max(fn)<=3+1e-9)
            original_lo=np.array(op['torque_lower']);original_hi=np.array(op['torque_upper'])
            orig_violation=np.maximum(np.maximum(original_lo-tau,tau-original_hi),0)
            interpretation='Removing the position-PD reachable interval permits force balance in this saved point/cone model; it is not a legal current position controller or a unique-joint/real-physics attribution.' if valid and max(orig_violation)>1e-9 else 'Numerical or source inconsistency needs review; no attribution.'
            item.update(ideal_torque_force_balance_feasible=valid,lambda_normal_N=lam.reshape(2,8).tolist(),normal_force_N=fn.tolist(),
              force_world_N=forces.tolist(),force_balance_residual_world_N=fr.tolist(),moment_residual_world_Nm=moment.tolist(),moment_infinity_Nm=float(max(abs(moment))),
              ideal_tau_required_Nm=tau.tolist(),ideal_tau_slack_lower_Nm=(tau-ideal_lo).tolist(),ideal_tau_slack_upper_Nm=(ideal_hi-tau).tolist(),
              original_PD_intersection_violation_Nm=orig_violation.tolist(),original_PD_intersection_max_violation_Nm=float(max(orig_violation)),
              solver_slack_t_Nm=float(result.x[-1]),interpretation=interpretation,
              eqlin_marginals=result.eqlin.marginals.tolist(),ineqlin_marginals=result.ineqlin.marginals.tolist())
        else:
            item['interpretation']='Removing only the PD reachable intervals did not produce force-hard feasibility; failure cannot be assigned solely to that constraint family. No recovery LP or further parameter change was run.'
        results.append(item)
    need(calls==2,'exactly2 primary LPs')
    need(not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp']),'no native model imports')
    out={'schema':'saved_v4_ideal_torque_actuation_counterfactual:1','status':'COMPLETED_EXACT_TWO_SAVED_LP_COUNTERFACTUALS_NOT_CONTROLLER_QUALIFICATION',
        'preregistration':pin(HERE/'preregistration.json'),'source':pin(__file__),'input_pins':cfg['input_pins'],
        'counts':{'primary_LP_calls':2,'force_recovery_LP_calls':0,'formal_v4_result_recomputed':0,'model_load_FK_physics_NN':0},
        'results':results,'scope':'One preregistered constraint-family intervention. Original v4 force-hard results are referenced, not rerun. Actual saved contacts/J/normal rays/bias/mass/gravity/mu/caps unchanged. Ideal torque capacity is not available current PD control. Neither finding establishes unique causes, physical holding, calibrated parameters or v5 qualification.'}
    dest=HERE/'report.json'
    with dest.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(dest),'cases':[{'seed':x['seed'],'force_balance_feasible':x['ideal_torque_force_balance_feasible'],'moment_infinity_Nm':x['moment_infinity_Nm']} for x in results]}))


if __name__=='__main__':main()
