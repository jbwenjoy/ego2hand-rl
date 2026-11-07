"""Saved-only relaxed COM-relocation planner; no native solve or applied forces.

Contacts/Jacobians/hand pose are frozen. The object's COM may move horizontally
without enforcing collider consistency: an optimistic proposal, not a new grasp.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin
from supermimic_research.learning import online_contract as C


def minimum_relocation(wrench,loads,offset,lower,upper,friction,*,weight=1.962,trust=.035):
    n=wrench.shape[1];target=np.array([0.,0.,weight,0.,0.,0.]);eq=np.zeros((6,n+3));eq[:,:n]=wrench
    # sum (p_i-COM0)xF_i - deltaCOM x sum F_i = 0; force sum=[0,0,w].
    eq[3:,n:n+2]=np.array([[0.,-weight],[weight,0.],[0.,0.]])
    eq[3:]/=.1;rhs=target.copy();rhs[3:]/=.1
    torque=np.r_[np.c_[loads,np.zeros((20,3))],np.c_[-loads,np.zeros((20,3))]]
    bound=np.r_[upper+friction-offset,offset-lower+friction]
    radius=np.zeros((4,n+3));radius[:2,n:n+2]=np.eye(2);radius[2:,n:n+2]=-np.eye(2);radius[:,-1]=-1
    # Inscribed regular16-gon: all feasible horizontal shifts are <=trust in L2.
    circle=np.zeros((16,n+3));theta=np.arange(16)*2*np.pi/16;circle[:,n:n+2]=np.c_[np.cos(theta),np.sin(theta)]
    A=np.r_[torque,radius,circle];b=np.r_[bound,np.zeros(4),np.full(16,trust*np.cos(np.pi/16))]
    objective=np.zeros(n+3);objective[-1]=1
    result=linprog(objective,A_eq=eq,b_eq=rhs,A_ub=A,b_ub=b,bounds=[(0,None)]*n+[(None,None)]*2+[(0,None)],method='highs')
    answer=dict(success=bool(result.success),status=int(result.status),message=result.message,witness=None)
    if result.success:
        coefficients=result.x[:n];delta=np.r_[result.x[n:n+2],0.];F=target[:3]
        error=wrench@coefficients-target;error[3:]-=np.cross(delta,F);tau=offset+loads@coefficients
        margin=float(np.minimum(tau-(lower-friction),(upper+friction)-tau).min())
        C.need(abs(error[:3]).max()<1e-8 and abs(error[3:]).max()<1e-9 and margin>-1e-8 and np.linalg.norm(delta)<=trust+1e-10,'Checked relaxed COM plan primal')
        answer['witness']=dict(delta_COM_world_m=delta.tolist(),horizontal_Linf_m=float(result.x[-1]),horizontal_L2_m=float(np.linalg.norm(delta)),ray_coefficients_N=coefficients.tolist(),
            force_residual_N=error[:3].tolist(),moment_residual_Nm=error[3:].tolist(),required_torque_without_friction20_Nm=tau.tolist(),minimum_relaxed_torque_margin_Nm=margin)
    return answer


def maximum_force_ignoring_object_moment(wrench,loads,offset,lower,upper,friction):
    n=wrench.shape[1];eq=np.c_[wrench[:3],-np.array([0.,0.,1.])]
    A=np.r_[np.c_[loads,np.zeros(20)],np.c_[-loads,np.zeros(20)]];b=np.r_[upper+friction-offset,offset-lower+friction]
    objective=np.zeros(n+1);objective[-1]=-1
    result=linprog(objective,A_eq=eq,b_eq=np.zeros(3),A_ub=A,b_ub=b,bounds=(0,None),method='highs')
    answer=dict(success=bool(result.success),status=int(result.status),message=result.message,witness=None)
    if result.success:
        coefficients=result.x[:n];force=wrench[:3]@coefficients;tau=offset+loads@coefficients
        margin=float(np.minimum(tau-(lower-friction),(upper+friction)-tau).min())
        C.need(abs(force-[0.,0.,result.x[-1]]).max()<1e-8 and margin>-1e-8,'Checked moment-free force primal')
        answer['witness']=dict(maximum_vertical_force_N=float(result.x[-1]),force_world_N=force.tolist(),unconstrained_moment_about_original_COM_Nm=(wrench[3:]@coefficients).tolist(),
            ray_coefficients_N=coefficients.tolist(),minimum_relaxed_torque_margin_Nm=margin)
    return answer


def run(args):
    registry=ArtifactRegistry(ROOT/'configs/artifacts.toml');torque=C.read(args.torque_report);review=C.read(args.authority_review)
    C.need(C.positive(torque) and C.positive(review) and review['torque_report']==file_pin(args.torque_report),'Reviewed actual frozen-reset data')
    out=args.output.resolve();C.need(out.is_relative_to(ROOT/'runs') and not out.exists(),'Fresh saved-only output')
    records=[]
    for rec in torque['records']:
        with np.load(registry.resolve_pin(rec['raw']),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        rows=raw['contact_rows24'];rows=rows[(rows[:,1]==71)|(rows[:,2]==71)];columns=[];loads=[]
        for row,jac in zip(rows,raw['contact_jacobians3x3x20']):
            gid=int(row[1] if row[2]==71 else row[2]);basis=next(x for x in review['details']['native_pyramid_generators'] if x['dt_s']==rec['dt_s'] and x['geom_id']==gid)
            for local in basis['local_normal_tangent3_rays']:
                force=(1 if int(row[2])==71 else -1)*row[9:18].reshape(3,3).T@np.array(local)
                columns.append(np.r_[force,np.cross(row[6:9]-raw['body_COM_world_m'],force)]);loads.append(jac.T@force)
        for domain in ('steady_residual_scale','full_actuator_target_range_analytical_only'):
            operands=(np.array(columns).T,np.array(loads).T,raw['qfrc_bias20']-raw['qfrc_passive20'],raw[domain+'__torque_lower'],raw[domain+'__torque_upper'],np.array(review['details']['dof_dry_friction_limits_Nm']))
            answer=minimum_relocation(*operands);force_only=maximum_force_ignoring_object_moment(*operands)
            records.append(dict(dt_s=rec['dt_s'],domain=domain,answer=answer,moment_free_force_bound=force_only))
    out.mkdir(parents=True);report=dict(schema='supermimic_research.relaxed_COM_relocation:1',sources=dict(planner=file_pin(__file__),torque=file_pin(args.torque_report),authority_review=file_pin(args.authority_review)),
        records=records,LP_calls=8,native_forward=0,integration_steps=0,NN_FW=0,GPU_calls=0,role='NO_FITTING_RELAXED_COM_RELOCATION_PLAN',
        scope='Frozen original hand/contact points/normals/Jacobians. Native pyramid and actual torque limits with freely varied dry friction; object COM translated independently of its collider. Minimum horizontal Linf within an inscribed16-gon of35mm; additional force bound drops all object moment balance. Numerical proposal only; no reachable-pose/geometry/native-force/holding certificate and no global lower bound over moving contacts.')
    C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),records=records)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('torque-report','authority-review','output'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())
