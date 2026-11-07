"""One conditional COM-axis LP; fixed world contacts are a design approximation."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.optimize import linprog


def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true')
    a=ap.parse_args();p=read(a.protocol)
    if p['schema']!='conditional_saved_contact_COM_axis_LP_protocol:1':raise ValueError('new schema')
    if p['delta_bounds_m']!=[0.,.075] or p['normal_cap_per_finger_N']!=3. or p['LP_calls']!=1:raise ValueError('fixed design')
    for rec in [*p['inputs'].values(),*p['source_pins'].values()]:
        if pin(rec['path'])!=rec:raise ValueError('changed source/input')
    if p['source_pins']['designer']!=pin(__file__):raise ValueError('own source')
    out=Path(p['output'])
    if out.exists():raise FileExistsError(out)
    if not a.run:
        print(json.dumps({'status':'METADATA_ONLY_NO_LP_NATIVE','LP_calls':0}));return
    if p['saved_LP_run_authorized'] is not True:raise ValueError('explicit saved LP authorization')
    contract=read(p['inputs']['native_contract']['path']);probe=read(p['inputs']['probe0']['path'])
    parent=read(p['inputs']['original_v6_report']['path'])
    if (contract['status']!='PASS_SAME_FORWARD_CONTACT_JACOBIAN_AND_PD_EXPORT_NOT_HOLDING'
        or contract['passed']!=contract['total'] or not all(v is True for v in contract['checks'].values())
        or contract['outputs']['probe0.json']!=p['inputs']['probe0']):raise ValueError('native contact/PD contract')
    old=next(s for s in parent['seed_results'] if s['seed']==0)['shortlist'][0]
    if old['rank']!=0 or probe['unknown'] or parent['model_audit']['passed'] is not True:raise ValueError('fixed rank0')
    for key in ('qpos27','ctrl20','phone_COM_world_m','phone_body_R_world'):
        if not np.array_equal(probe['raw'][key],old['native']['raw'][key]):raise ValueError('same v6 input '+key)
    cd=probe['contact_design'];contacts=cd['phone_hand_contacts'];pd=cd['PD_raw'];N=len(contacts);K=4*N
    if not cd['available'] or cd['positive_normal_force_filter_applied'] or N!=3:raise ValueError('all three contacts')
    if not pd['direct_unit_joint_mapping_verified_by_frozen_factory'] or pd['native_disabled_actuator_group_bitmask']!=0:raise ValueError('direct active unit actuator map')
    for key in ('actuator_ctrllimited20','actuator_forcelimited20','joint_actfrclimited20'):
        if not np.all(pd[key]):raise ValueError('explicit native force/control limits')
    C0=np.array(cd['phone_COM_world_m']);R=np.array(probe['raw']['phone_body_R_world']);axis=R[:,2]
    if abs(axis@axis-1)>1e-10:raise ValueError('unit original phone bodyZ world axis')
    gravity=np.array([0.,0.,-9.81]);mass=probe['model_configuration']['phone_mass_kg'];target=-mass*gravity
    if mass!=.2:raise ValueError('same nominal mass')
    points=np.array([c['position_world_m'] for c in contacts]);groups=[c['finger_group'] for c in contacts]
    names=list(dict.fromkeys(groups));rays=[]
    if set(names)!={'finger1','finger2'}:raise ValueError('original two finger groups')
    for c in contacts:
        frame=np.array(c['frame_rows_normal_t1_t2_world']);mu=np.array(c['native_friction5']);sgn=c['force_on_object_sign']
        if c['native_condim']!=3 or c['native_exclude']!=0 or c['native_efc_address']<0 or sgn not in (-1.,1.):raise ValueError('original native active condim3')
        if np.max(abs(frame@frame.T-np.eye(3)))>1e-10 or np.any(mu[:2]<0):raise ValueError('native frame/friction')
        rays.append(sgn*np.array([frame[0]-mu[0]*frame[1],frame[0]+mu[0]*frame[1],frame[0]-mu[1]*frame[2],frame[0]+mu[1]*frame[2]]))
    rays=np.array(rays);F=np.zeros((3*N,K));S=np.zeros((len(names),K))
    for i in range(N):F[3*i:3*i+3,4*i:4*i+4]=rays[i].T;S[names.index(groups[i]),4*i:4*i+4]=1.
    B=np.concatenate([r.T for r in rays],axis=1)
    M=np.concatenate([np.cross(points[i]-C0,rays[i]).T for i in range(N)],axis=1)
    J=np.array([c['hand_Jp3x20_world'] for c in contacts]);T=J.transpose(2,0,1).reshape(20,3*N)@F
    bias=np.array(pd['qfrc_bias26'])[:20]-np.array(pd['qfrc_passive26'])[:20]
    gain=np.array(pd['PD_gain20']);offset=np.array(pd['PD_offset20']);cr=np.array(pd['actuator_ctrlrange20x2'])
    ar=np.array(pd['actuator_forcerange20x2']);jr=np.array(pd['joint_actfrcrange20x2'])
    if np.any(gain<=0):raise ValueError('positive affine gain')
    lo=np.maximum.reduce([gain*cr[:,0]+offset,ar[:,0],jr[:,0]])
    hi=np.minimum.reduce([gain*cr[:,1]+offset,ar[:,1],jr[:,1]])
    if np.any(lo>hi):raise ValueError('empty legal unsaturated inverse-control intersection')
    # Hard sum F=target makes the COM-shift contribution linear, not bilinear.
    Aeq=np.vstack([np.c_[B,np.zeros(3)],np.c_[M,-np.cross(axis,target)]])
    beq=np.r_[target,np.zeros(3)];scale=np.r_[np.full(3,1.962),np.full(3,.151074)]
    Aub=np.vstack([np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(len(names))]])
    bub=np.r_[hi-bias,bias-lo,np.full(len(names),3.)]
    out.mkdir()
    sol=linprog(np.r_[np.zeros(K),1.],A_ub=Aub,b_ub=bub,A_eq=Aeq/scale[:,None],b_eq=beq/scale,
        bounds=[(0,None)]*K+[(0.,.075)],method='highs',options=p['linprog_options'])
    result={'status':'INFEASIBLE_FIXED_CONTACT_CONDITIONAL_LP' if sol.status==2 else 'UNKNOWN_LP',
        'delta_bodyZ_m':None,'proposed_ctrl20':None,'solver_status':int(sol.status),'solver_message':sol.message,
        'native_geometry_or_equilibrium_or_holding_qualified':False}
    if sol.status==0:
        lam=sol.x[:K];delta=float(sol.x[-1]);Cnew=C0+axis*delta;forces=(F@lam).reshape(N,3)
        netforce=B@lam;moment=M@lam-np.cross(axis,target)*delta
        directmoment=np.sum(np.cross(points-Cnew,forces),axis=0);tau=bias+T@lam;u=(tau-offset)/gain
        result.update(status='OPTIMAL_CONDITIONAL_FIXED_CONTACT_COM_AXIS_DESIGN',delta_bodyZ_m=delta,
            proposed_ctrl20=u.tolist(),lambda4_per_contact_N=lam.reshape(N,4).tolist(),forces_on_object_world_N=forces.tolist(),
            net_force_world_N=netforce.tolist(),force_residual_world_N=(netforce-target).tolist(),
            moment_about_new_COM_world_Nm=moment.tolist(),direct_moment_about_new_COM_world_Nm=directmoment.tolist(),
            equality_scaled_max_abs=float(np.max(abs((Aeq@sol.x-beq)/scale))),
            max_inequality_violation=float(max(0.,np.max(Aub@sol.x-bub))),
            torque_required20_Nm=tau.tolist(),torque_slack_lower20_Nm=(tau-lo).tolist(),torque_slack_upper20_Nm=(hi-tau).tolist(),
            ctrl_lower_slack20_rad=(u-cr[:,0]).tolist(),ctrl_upper_slack20_rad=(cr[:,1]-u).tolist(),
            group_normal_load_N=dict(zip(names,(S@lam).tolist())),COM_proposed_world_m=Cnew.tolist(),
            delta_world_m=(axis*delta).tolist(),all_solution_values_finite=bool(np.isfinite(sol.x).all()))
    line=points[1:].mean(0)-points[0];line/=np.linalg.norm(line)
    axial=float(line@np.cross(C0-points[0],mass*gravity));slope=float(line@np.cross(axis,mass*gravity))
    report={'schema':'conditional_saved_contact_COM_axis_LP:1','protocol':pin(a.protocol),'source_pins':p['source_pins'],
        'inputs':p['inputs'],'result':result,'LP_calls':1,'new_native_model_forward_step_NN_GPU_calls':0,
        'comparison_two_point_line_zero_axial_delta_m':-axial/slope,
        'fixed_inputs':{'points_world_m':points.tolist(),'frames_frictions_J_PD':cd,'COM0_world_m':C0.tolist(),
            'axis_bodyZ_world':axis.tolist(),'target_force_world_N':target.tolist(),'rays_world_per_normal_N':rays.tolist(),
            'force_matrix3xK':B.tolist(),'moment_matrix3xK_about_COM0':M.tolist(),'torque_matrix20xK':T.tolist(),
            'affine_bias20':bias.tolist(),'torque_lo20':lo.tolist(),'torque_hi20':hi.tolist(),
            'hard_Aeq':Aeq.tolist(),'hard_beq':beq.tolist(),'Aub':Aub.tolist(),'bub':bub.tolist()},
        'scope':'Fixed world points/normals/J and same robot q20/PD context only. COM axis variable is a conditional linear designer, not new actual contact, native soft equilibrium or holding. Root must verify new pose geometry/contacts and recompute its current LP; old lambda/u are not a new-contact solution.'}
    if any(n in sys.modules for n in ('mujoco','torch','warp','mujoco_warp')):raise ValueError('no native/framework')
    dest=out/'report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(dest),'result':{k:result[k] for k in ('status','delta_bodyZ_m','proposed_ctrl20')},'LP_calls':1}))


if __name__=='__main__':main()
